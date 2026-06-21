"""Tests for the pure command-builders in :mod:`holmes.fleet`.

The orchestration functions (run_up/run_down) touch real infrastructure and are not exercised here;
these lock the argv each one shells out, which is where mistakes would actually bite.
"""

from __future__ import annotations

from holmes import fleet


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
