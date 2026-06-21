"""Tests for the pure command-builders in :mod:`holmes.fleet`.

The orchestration functions (run_up/run_down) touch real infrastructure and are not exercised here;
these lock the argv each one shells out, which is where mistakes would actually bite.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from holmes import cli, fleet

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_box_name_and_fleet_names_round_trip() -> None:
    names = [fleet.box_name(i) for i in range(3)]
    assert names == ["holmes-box-0", "holmes-box-1", "holmes-box-2"]
    # Filtering ignores unrelated servers and sorts for deterministic teardown.
    mixed = ["db-1", "holmes-box-2", "holmes-box-0", "web", "holmes-box-1"]
    assert fleet.fleet_names(mixed) == ["holmes-box-0", "holmes-box-1", "holmes-box-2"]


def test_create_command_pins_the_hcloud_invocation() -> None:
    command = fleet.create_command(
        "holmes-box-0",
        server_type="cpx62",
        image="ubuntu-24.04",
        location="nbg1",
        ssh_key="my-key",
    )
    assert command == [
        "hcloud", "server", "create",
        "--name", "holmes-box-0",
        "--type", "cpx62",
        "--image", "ubuntu-24.04",
        "--location", "nbg1",
        "--ssh-key", "my-key",
        "--user-data-from-file", "-",
    ]  # fmt: skip


def test_delete_and_ip_and_list_commands() -> None:
    assert fleet.delete_command("holmes-box-0") == ["hcloud", "server", "delete", "holmes-box-0"]
    assert fleet.ip_command("holmes-box-0") == ["hcloud", "server", "ip", "holmes-box-0"]
    assert fleet.list_names_command()[:3] == ["hcloud", "server", "list"]


def test_render_user_data_fills_repo_and_branch() -> None:
    rendered = fleet.render_user_data("git@github.com:Data-Simply/holmes.git", "feature-x")
    assert "git clone --branch feature-x git@github.com:Data-Simply/holmes.git /opt/holmes" in rendered
    # No template placeholders should survive substitution.
    assert "${REPO_URL}" not in rendered
    assert "${BRANCH}" not in rendered


def test_ssh_command_includes_identity_and_options() -> None:
    command = fleet.ssh_command("203.0.113.5", "cloud-init status --wait", user="root", identity="/k/id")
    assert command[0] == "ssh"
    assert "-i" in command
    assert "/k/id" in command
    assert command[-2:] == ["root@203.0.113.5", "cloud-init status --wait"]


def test_rsync_push_and_pull_directions() -> None:
    push = fleet.rsync_push("data/processed/", "203.0.113.5", "/opt/holmes/data/processed/", user="root")
    assert push[0] == "rsync"
    assert push[-2:] == ["data/processed/", "root@203.0.113.5:/opt/holmes/data/processed/"]

    pull = fleet.rsync_pull("203.0.113.5", "/opt/holmes/results/", "results/", user="root")
    assert pull[-2:] == ["root@203.0.113.5:/opt/holmes/results/", "results/"]
    # The ssh transport carries our non-interactive options through rsync's -e.
    transport = pull[pull.index("-e") + 1]
    assert "StrictHostKeyChecking=accept-new" in transport


class _FakeRun:
    """Records every ``subprocess.run`` argv and returns canned output per hcloud query.

    ``hcloud server ip`` yields a fixed address; ``hcloud server list`` yields a fleet plus an
    unrelated server; everything else (creates, ssh, rsync) just succeeds. This lets the
    orchestration of run_up/run_down be pinned without touching real infrastructure.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command, *args, **kwargs):
        self.calls.append(list(command))
        stdout = ""
        if command[:3] == ["hcloud", "server", "ip"]:
            # Distinct IP per box (holmes-box-N -> 203.0.113.N) so tests can catch a per-box
            # ship-to-the-wrong-host mix-up, not just the right call counts.
            stdout = f"203.0.113.{command[3].rsplit('-', 1)[-1]}"
        elif command[:3] == ["hcloud", "server", "list"]:
            stdout = "holmes-box-0\nholmes-box-1\nweb\n"
        return SimpleNamespace(returncode=0, stdout=stdout)


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> _FakeRun:
    fake = _FakeRun()
    monkeypatch.setattr("holmes.fleet.subprocess.run", fake)
    monkeypatch.setattr("holmes.fleet.time.sleep", lambda _seconds: None)
    args = cli._build_parser().parse_args(argv)
    cli._COMMANDS[args.command](args)
    return fake


def test_run_up_provisions_then_ships_and_starts_each_active_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2 grid cells (fit seeds 0,1) across 2 boxes -> both boxes active.
    fake = _run(
        monkeypatch,
        [
            "dispatch", "up", "--boxes", "2",
            "--repo-url", "git@github.com:Data-Simply/holmes.git", "--ssh-key", "my-key",
            "--strategies", "grid", "--categories", "Books", "--fit-seeds", "0", "1",
            "--plan-dir", str(tmp_path / "plans"), "--results-dir", str(tmp_path / "results"),
        ],
    )  # fmt: skip

    creates = [c for c in fake.calls if c[:3] == ["hcloud", "server", "create"]]
    rsyncs = [c for c in fake.calls if c[0] == "rsync"]
    starts = [c for c in fake.calls if any("nohup bash box.sh" in part for part in c)]
    assert [c[c.index("--name") + 1] for c in creates] == ["holmes-box-0", "holmes-box-1"]
    assert len(starts) == 2  # one run started per box
    assert len(rsyncs) == 4  # data dir + box.sh, per box
    # All boxes are provisioned before any are set up (create -> wait -> ship -> start).
    assert max(fake.calls.index(c) for c in creates) < min(fake.calls.index(r) for r in rsyncs)
    # Each box's script is shipped to that box's own host, not a shared/mixed-up one.
    script_pushes = {c[-2].rsplit("/", 1)[-1]: c[-1] for c in rsyncs if c[-1].endswith("/box.sh")}
    assert script_pushes["box-0.sh"] == "root@203.0.113.0:/opt/holmes/box.sh"
    assert script_pushes["box-1.sh"] == "root@203.0.113.1:/opt/holmes/box.sh"


def test_run_up_skips_boxes_with_no_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1 cell but 4 boxes: only the one box with work should be provisioned (no paid empty servers).
    fake = _run(
        monkeypatch,
        [
            "dispatch", "up", "--boxes", "4",
            "--repo-url", "git@github.com:Data-Simply/holmes.git", "--ssh-key", "my-key",
            "--strategies", "grid", "--categories", "Books", "--fit-seeds", "0",
            "--plan-dir", str(tmp_path / "plans"), "--results-dir", str(tmp_path / "results"),
        ],
    )  # fmt: skip
    creates = [c for c in fake.calls if c[:3] == ["hcloud", "server", "create"]]
    assert len(creates) == 1


def test_run_down_fetches_results_then_deletes_only_the_fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _run(monkeypatch, ["dispatch", "down", "--results-dir", str(tmp_path / "results")])

    pulls = [c for c in fake.calls if c[0] == "rsync"]
    deletes = [c[-1] for c in fake.calls if c[:3] == ["hcloud", "server", "delete"]]
    assert len(pulls) == 2  # results fetched from each fleet box before deletion
    # Only the holmes-box-* servers are deleted; the unrelated 'web' server is left alone.
    assert deletes == ["holmes-box-0", "holmes-box-1"]
    # Fetch happens before delete, so teardown never races ahead of result retrieval.
    assert max(fake.calls.index(p) for p in pulls) < min(
        i for i, c in enumerate(fake.calls) if c[:3] == ["hcloud", "server", "delete"]
    )


def test_run_down_no_fetch_skips_rsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _run(monkeypatch, ["dispatch", "down", "--no-fetch", "--results-dir", str(tmp_path / "results")])
    assert not [c for c in fake.calls if c[0] == "rsync"]
    assert [c[-1] for c in fake.calls if c[:3] == ["hcloud", "server", "delete"]] == ["holmes-box-0", "holmes-box-1"]
