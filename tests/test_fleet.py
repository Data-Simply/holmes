"""Tests for the cloud-fleet lifecycle in :mod:`holmes.fleet`.

The Hetzner API is faked (a stand-in SDK client) and ssh/rsync are mocked, so the orchestration of
run_up/run_down is pinned without touching real infrastructure; the ssh/rsync command-builders are
pinned directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from holmes import cli, fleet

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _server(name: str, ip: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, public_net=SimpleNamespace(ipv4=SimpleNamespace(ip=ip), primary_ipv4=None))


class _FakeServersClient:
    """Stand-in for client.servers: records creates/deletes, hands back servers with distinct IPs."""

    def __init__(self, existing: list[SimpleNamespace] | None = None) -> None:
        self.created: list[str] = []
        self.create_kwargs: list[dict] = []
        self.deleted: list[str] = []
        self._existing = existing or []

    def create(self, *, name, server_type, image, location, ssh_keys, user_data):
        self.created.append(name)
        self.create_kwargs.append(
            {"server_type": server_type.name, "image": image.name, "location": location.name, "user_data": user_data}
        )
        # Distinct IP per box (holmes-box-N -> 203.0.113.N) so a ship-to-wrong-host bug is catchable.
        server = _server(name, f"203.0.113.{name.rsplit('-', 1)[-1]}")
        return SimpleNamespace(server=server, action=SimpleNamespace(wait_until_finished=lambda: None), next_actions=[])

    def get_all(self) -> list[SimpleNamespace]:
        return list(self._existing)

    def delete(self, server: SimpleNamespace) -> None:
        self.deleted.append(server.name)


def _fake_client(existing: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        servers=_FakeServersClient(existing),
        ssh_keys=SimpleNamespace(get_by_name=lambda name: SimpleNamespace(name=name)),
    )


class _Run:
    """Records every ssh/rsync argv and reports success."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command, *args, **kwargs):
        self.calls.append(list(command))
        return SimpleNamespace(returncode=0, stdout="")


def _drive(monkeypatch: pytest.MonkeyPatch, argv: list[str], *, client: SimpleNamespace) -> _Run:
    runs = _Run()
    monkeypatch.setattr("holmes.fleet._client", lambda: client)
    monkeypatch.setattr("holmes.fleet.subprocess.run", runs)
    monkeypatch.setattr("holmes.fleet.time.sleep", lambda _seconds: None)
    args = cli._build_parser().parse_args(argv)
    cli._COMMANDS[args.command](args)
    return runs


def test_box_name() -> None:
    assert [fleet.box_name(i) for i in range(3)] == ["holmes-box-0", "holmes-box-1", "holmes-box-2"]


def test_fleet_servers_filters_by_prefix_and_sorts() -> None:
    client = _fake_client([_server("holmes-box-1", "a"), _server("web", "b"), _server("holmes-box-0", "c")])
    assert [s.name for s in fleet.fleet_servers(client)] == ["holmes-box-0", "holmes-box-1"]


def test_server_ip_falls_back_to_primary_ipv4() -> None:
    legacy = _server("holmes-box-0", "203.0.113.5")
    assert fleet._server_ip(legacy) == "203.0.113.5"
    primary = SimpleNamespace(public_net=SimpleNamespace(ipv4=None, primary_ipv4=SimpleNamespace(ip="203.0.113.6")))
    assert fleet._server_ip(primary) == "203.0.113.6"


def test_ssh_command_includes_identity_and_options() -> None:
    command = fleet.ssh_command("203.0.113.5", "cloud-init status --wait", user="root", identity="/k/id")
    assert command[0] == "ssh"
    assert "-i" in command
    assert "/k/id" in command
    assert command[-2:] == ["root@203.0.113.5", "cloud-init status --wait"]


def test_rsync_push_carries_excludes_and_pull_direction() -> None:
    push = fleet.rsync_push("/repo/", "203.0.113.5", "/opt/holmes/", excludes=(".git", "data"))
    assert push[0] == "rsync"
    assert push[push.index("--exclude") + 1] == ".git"
    assert push.count("--exclude") == 2
    assert push[-2:] == ["/repo/", "root@203.0.113.5:/opt/holmes/"]

    pull = fleet.rsync_pull("203.0.113.5", "/opt/holmes/results/", "results/", user="root")
    assert pull[-2:] == ["root@203.0.113.5:/opt/holmes/results/", "results/"]
    transport = pull[pull.index("-e") + 1]
    assert "StrictHostKeyChecking=accept-new" in transport


def test_run_up_provisions_then_ships_code_data_script_and_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _fake_client()
    runs = _drive(
        monkeypatch,
        [
            "dispatch", "up", "--boxes", "2", "--ssh-key", "my-key",
            "--strategies", "grid", "--categories", "Books", "--fit-seeds", "0", "1",
            "--plan-dir", str(tmp_path / "plans"), "--results-dir", str(tmp_path / "results"),
        ],
        client=client,
    )  # fmt: skip

    assert client.servers.created == ["holmes-box-0", "holmes-box-1"]
    # The provisioning contract: each box created as the requested type/image/location with cloud-init.
    first = client.servers.create_kwargs[0]
    assert first["server_type"] == "cpx62"
    assert first["image"] == "ubuntu-24.04"
    assert first["location"] == "nbg1"
    assert "uv" in first["user_data"]  # cloud-init user-data was passed through
    rsyncs = [c for c in runs.calls if c[0] == "rsync"]
    assert len(rsyncs) == 2 * 3  # code + data + script, per box
    # Each box's script ships to that box's own host (not a shared/mixed-up one).
    script_pushes = {c[-2].rsplit("/", 1)[-1]: c[-1] for c in rsyncs if c[-1].endswith("/box.sh")}
    assert script_pushes["box-0.sh"] == "root@203.0.113.0:/opt/holmes/box.sh"
    assert script_pushes["box-1.sh"] == "root@203.0.113.1:/opt/holmes/box.sh"
    # uv sync precedes the run, once per box.
    assert sum("uv sync" in part for c in runs.calls for part in c) == 2
    assert sum("nohup bash box.sh" in part for c in runs.calls for part in c) == 2


def test_run_up_skips_boxes_with_no_cells(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client()
    _drive(
        monkeypatch,
        [
            "dispatch", "up", "--boxes", "4", "--ssh-key", "my-key",
            "--strategies", "grid", "--categories", "Books", "--fit-seeds", "0",
            "--plan-dir", str(tmp_path / "plans"), "--results-dir", str(tmp_path / "results"),
        ],
        client=client,
    )  # fmt: skip
    assert client.servers.created == ["holmes-box-0"]


def test_run_down_fetches_then_deletes_only_the_fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client(
        [_server("holmes-box-0", "203.0.113.0"), _server("holmes-box-1", "203.0.113.1"), _server("web", "198.51.100.7")]
    )
    runs = _drive(monkeypatch, ["dispatch", "down", "--results-dir", str(tmp_path / "results")], client=client)

    pulls = [c for c in runs.calls if c[0] == "rsync"]
    assert len(pulls) == len(client.servers.deleted)  # one fetch per box deleted
    # Only the holmes-box-* servers are deleted; the unrelated 'web' server is left alone.
    assert client.servers.deleted == ["holmes-box-0", "holmes-box-1"]


def test_run_down_no_fetch_skips_rsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _fake_client([_server("holmes-box-0", "203.0.113.0"), _server("holmes-box-1", "203.0.113.1")])
    runs = _drive(
        monkeypatch, ["dispatch", "down", "--no-fetch", "--results-dir", str(tmp_path / "results")], client=client
    )
    assert not [c for c in runs.calls if c[0] == "rsync"]
    assert client.servers.deleted == ["holmes-box-0", "holmes-box-1"]
