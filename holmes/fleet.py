"""Cloud-fleet lifecycle for the baseline fan-out: provision, ship+run, and tear down boxes.

Drives the ``holmes dispatch up`` / ``holmes dispatch down`` lifecycle around the ``dispatch`` plan.
``up`` provisions N Hetzner CPX62 boxes via the ``hcloud`` Python SDK, waits for cloud-init (which
just installs uv), rsyncs the local checkout + datasets + that box's plan script, runs ``uv sync``,
and starts the run. ``down`` fetches results back and deletes the fleet.

The Hetzner API goes through the SDK (structured objects, native action waiting), so the only
``subprocess`` calls are ssh/rsync -- binaries with no Python-native equivalent. Shipping the local
code (rather than cloning from GitHub) means a box runs exactly the working-tree code, with no repo
URL, branch, or push to coordinate.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from hcloud import Client
from hcloud.images import Image
from hcloud.locations import Location
from hcloud.server_types import ServerType

from holmes.config import PROJECT_ROOT
from holmes.dispatch import DEFAULT_RESULTS_DIR, add_plan_arguments, plan_boxes, write_box_scripts

if TYPE_CHECKING:
    import argparse

    from hcloud.servers.domain import Server
    from hcloud.ssh_keys import BoundSSHKey

# Hetzner's European datacenters (Nuremberg, Falkenstein, Helsinki). Restricting --location to these
# keeps the whole fleet in a European region, and on one CPU generation, by construction.
EU_LOCATIONS = ("nbg1", "fsn1", "hel1")
CLOUD_INIT = PROJECT_ROOT / "deploy" / "cloud-init.yaml"
REMOTE_DIR = "/opt/holmes"
NAME_PREFIX = "holmes-box-"

# rsync excludes when shipping the local checkout as the box's code: drop the local venv, git
# history, and generated dirs. Datasets (data/processed) are shipped separately, so all of data/
# is excluded here.
_CODE_EXCLUDES = (".git", ".venv", "data", "results", "plans", "__pycache__", ".pytest_cache", "*.egg-info")

# Non-interactive SSH: accept a new host key (boxes are freshly created) and fail fast if unreachable
# so the boot wait-loop can retry rather than hang.
_SSH_OPTS = ("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10")
_BOOT_RETRIES = 40
_BOOT_DELAY_SECONDS = 15
_WAIT_TIMEOUT_SECONDS = 600  # cap on one `cloud-init status --wait` so a wedged box can't hang the run
_SSH_UNREACHABLE = 255  # ssh's own exit code when it cannot establish the connection (vs. a remote rc)


def box_name(index: int) -> str:
    """Return the server name for box ``index``."""
    return f"{NAME_PREFIX}{index}"


def _client() -> Client:
    """Build an authenticated hcloud client from ``HCLOUD_TOKEN``.

    Raises:
        SystemExit: If ``HCLOUD_TOKEN`` is not set.
    """
    token = os.environ.get("HCLOUD_TOKEN")
    if not token:
        msg = "Set HCLOUD_TOKEN to your Hetzner Cloud API token (Console -> Security -> API Tokens)."
        raise SystemExit(msg)
    return Client(token=token, application_name="holmes-dispatch")


def _server_ip(server: Server) -> str:
    """Return a server's public IPv4, preferring the legacy attached IP, then a primary IP.

    Raises:
        RuntimeError: If the server has no public IPv4 (e.g. an IPv6-only box we can't reach).
    """
    public = server.public_net
    ip = None
    if public is not None:
        if public.ipv4 is not None:
            ip = public.ipv4.ip
        elif public.primary_ipv4 is not None:
            ip = public.primary_ipv4.ip
    if ip is None:
        msg = f"server {server.name!r} has no public IPv4 to ssh/rsync to."
        raise RuntimeError(msg)
    return ip


def fleet_servers(client: Client, prefix: str = NAME_PREFIX) -> list[Server]:
    """Return this fleet's servers (name starts with ``prefix``), sorted for deterministic teardown."""
    matched = [s for s in client.servers.get_all() if (s.name or "").startswith(prefix)]
    return sorted(matched, key=lambda s: s.name or "")


def _ssh_base(identity: str | None) -> list[str]:
    """Return the ssh argv prefix (options + optional identity) shared by ssh and rsync's transport."""
    base = ["ssh", *_SSH_OPTS]
    if identity is not None:
        base += ["-i", identity]
    return base


def _ssh_transport(identity: str | None) -> str:
    """Return the ``-e`` transport string rsync uses to invoke ssh with our options."""
    return " ".join(_ssh_base(identity))


def ssh_command(host: str, remote: str, *, user: str = "root", identity: str | None = None) -> list[str]:
    """Build an ssh argv running ``remote`` on ``host``."""
    return [*_ssh_base(identity), f"{user}@{host}", remote]


def rsync_push(
    local_src: str,
    host: str,
    remote_dest: str,
    *,
    user: str = "root",
    identity: str | None = None,
    excludes: tuple[str, ...] = (),
) -> list[str]:
    """Build an rsync argv pushing ``local_src`` to ``host:remote_dest``."""
    cmd = ["rsync", "-az"]
    for pattern in excludes:
        cmd += ["--exclude", pattern]
    cmd += ["-e", _ssh_transport(identity), local_src, f"{user}@{host}:{remote_dest}"]
    return cmd


def rsync_pull(
    host: str, remote_src: str, local_dest: str, *, user: str = "root", identity: str | None = None
) -> list[str]:
    """Build an rsync argv pulling ``host:remote_src`` back to ``local_dest``."""
    return ["rsync", "-az", "-e", _ssh_transport(identity), f"{user}@{host}:{remote_src}", local_dest]


def _ssh(ip: str, remote: str, args: argparse.Namespace) -> None:
    """Run ``remote`` over ssh on ``ip`` with the run's ssh user/identity; raise on failure."""
    subprocess.run(ssh_command(ip, remote, user=args.ssh_user, identity=args.identity), check=True)


def _push(
    ip: str, local_src: str, remote_dest: str, args: argparse.Namespace, *, excludes: tuple[str, ...] = ()
) -> None:
    """Rsync ``local_src`` up to ``ip:remote_dest``; raise on failure."""
    pushed = rsync_push(local_src, ip, remote_dest, user=args.ssh_user, identity=args.identity, excludes=excludes)
    subprocess.run(pushed, check=True)


def _pull(ip: str, remote_src: str, local_dest: str, args: argparse.Namespace) -> None:
    """Rsync ``ip:remote_src`` back to ``local_dest``, best-effort (a missing dir must not abort teardown)."""
    subprocess.run(rsync_pull(ip, remote_src, local_dest, user=args.ssh_user, identity=args.identity), check=False)


def add_provision_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the provisioning arguments for ``up`` (and the SSH knobs ``down`` reuses)."""
    parser.add_argument("--ssh-key", required=True, help="Name of an SSH key in your hcloud project.")
    parser.add_argument(
        "--type",
        dest="server_type",
        default="cpx62",
        help="Server type (default: cpx62 -- 16 vCPU / 32 GB AMD EPYC Genoa).",
    )
    parser.add_argument("--image", default="ubuntu-24.04", help="OS image (default: ubuntu-24.04).")
    parser.add_argument(
        "--location",
        default="nbg1",
        choices=EU_LOCATIONS,
        help="European datacenter: nbg1 (Nuremberg), fsn1 (Falkenstein), hel1 (Helsinki). Default nbg1.",
    )
    _add_ssh_arguments(parser)


def add_teardown_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the arguments for ``down`` (fetch results, then delete the fleet)."""
    parser.add_argument(
        "--prefix",
        default=NAME_PREFIX,
        help=f"Delete servers whose name starts with this (default: {NAME_PREFIX!r}).",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Local dir fetched results land in."
    )
    parser.add_argument("--no-fetch", action="store_true", help="Skip rsyncing results back before deleting.")
    _add_ssh_arguments(parser)


def _add_ssh_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the SSH user/identity options used by both ``up`` and ``down``."""
    parser.add_argument("--ssh-user", default="root", help="SSH user on the boxes (default: root).")
    parser.add_argument("--identity", default=None, help="Path to the SSH private key for ssh/rsync (optional).")


def wait_for_cloud_init(host: str, *, user: str, identity: str | None) -> None:
    """Block until cloud-init finishes on ``host``, retrying while SSH is not yet reachable.

    Args:
        host: Box IP.
        user: SSH user.
        identity: SSH private-key path, or ``None``.

    Raises:
        RuntimeError: If cloud-init reports an error, or the box never becomes ready in the budget.
    """
    wait = ssh_command(host, "cloud-init status --wait", user=user, identity=identity)
    for _ in range(_BOOT_RETRIES):
        try:
            # A bounded timeout so a wedged `--wait` can't block the whole run forever; a timeout is
            # just another "not ready yet, retry".
            result = subprocess.run(wait, check=False, timeout=_WAIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            time.sleep(_BOOT_DELAY_SECONDS)
            continue
        if result.returncode == 0:
            return
        if result.returncode != _SSH_UNREACHABLE:
            # ssh connected and `cloud-init status --wait` itself reported error/degraded -- the box
            # is broken, not still booting, so fail fast instead of spinning the full retry budget.
            msg = f"cloud-init failed on {host} (status exit {result.returncode})."
            raise RuntimeError(msg)
        time.sleep(_BOOT_DELAY_SECONDS)
    msg = f"{host} did not become reachable within {_BOOT_RETRIES * _BOOT_DELAY_SECONDS}s."
    raise RuntimeError(msg)


def _create_server(
    client: Client, name: str, *, ssh_key: BoundSSHKey, user_data: str, args: argparse.Namespace
) -> Server:
    """Create one server and block until it (and its follow-up actions) are ready."""
    response = client.servers.create(
        name=name,
        server_type=ServerType(name=args.server_type),
        image=Image(name=args.image),
        location=Location(name=args.location),
        ssh_keys=[ssh_key],
        user_data=user_data,
    )
    response.action.wait_until_finished()
    for action in response.next_actions or []:
        action.wait_until_finished()
    return response.server


def _setup_box(server: Server, script_path: Path, args: argparse.Namespace) -> None:
    """Wait for a provisioned box, ship code + data + plan script, sync the env, and start the run."""
    ip = _server_ip(server)
    print(f">>> {server.name} ({ip}): waiting for cloud-init")
    wait_for_cloud_init(ip, user=args.ssh_user, identity=args.identity)

    processed = str(args.processed_dir)
    remote_processed = f"{REMOTE_DIR}/{processed}"
    print(f">>> {server.name} ({ip}): shipping code, {processed}/, and {script_path.name}")
    _ssh(ip, f"mkdir -p {remote_processed}", args)
    _push(ip, f"{PROJECT_ROOT}/", f"{REMOTE_DIR}/", args, excludes=_CODE_EXCLUDES)
    _push(ip, f"{processed}/", f"{remote_processed}/", args)
    _push(ip, str(script_path), f"{REMOTE_DIR}/box.sh", args)
    print(f">>> {server.name} ({ip}): uv sync")
    _ssh(ip, f"cd {REMOTE_DIR} && uv sync", args)
    print(f">>> {server.name} ({ip}): starting run")
    _ssh(ip, f"cd {REMOTE_DIR} && nohup bash box.sh > box.log 2>&1 < /dev/null &", args)


def run_up(args: argparse.Namespace) -> None:
    """Provision the fleet, ship each box its code + data + plan script, and start the runs.

    Args:
        args: Parsed ``dispatch up`` arguments (plan dimensions plus provisioning options).
    """
    boxes = plan_boxes(args)
    paths = write_box_scripts(boxes, args.plan_dir)
    # partition() always returns exactly --boxes lists, padding with empty ones when there is less
    # work than boxes. Provision only the boxes that have cells -- an empty box would be a paid
    # server running an empty script.
    active = [(box_name(i), paths[i]) for i, box in enumerate(boxes) if box]
    if not active:
        print("Nothing to dispatch; all results already exist. Not provisioning.")
        return

    client = _client()
    ssh_key = client.ssh_keys.get_by_name(args.ssh_key)
    if ssh_key is None:
        msg = f"SSH key {args.ssh_key!r} not found in the hcloud project."
        raise SystemExit(msg)
    user_data = CLOUD_INIT.read_text()

    # Create all boxes first so they boot in parallel while we set them up one by one.
    provisioned: list[tuple[Server, Path]] = []
    for name, script_path in active:
        print(f">>> creating {name} ({args.server_type}, {args.location})")
        server = _create_server(client, name, ssh_key=ssh_key, user_data=user_data, args=args)
        provisioned.append((server, script_path))

    for server, script_path in provisioned:
        _setup_box(server, script_path, args)
    print(
        f"Started {len(provisioned)} box(es). Tail progress with: ssh {args.ssh_user}@<ip> tail -f {REMOTE_DIR}/box.log"
    )


def run_down(args: argparse.Namespace) -> None:
    """Fetch results back from the fleet (unless ``--no-fetch``), then delete every box.

    Args:
        args: Parsed ``dispatch down`` arguments.
    """
    client = _client()
    servers = fleet_servers(client, args.prefix)
    if not servers:
        print(f"No servers matching {args.prefix!r}; nothing to tear down.")
        return

    if not args.no_fetch:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        for server in servers:
            ip = _server_ip(server)
            print(f">>> {server.name} ({ip}): fetching results")
            _pull(ip, f"{REMOTE_DIR}/results/", f"{args.results_dir}/", args)

    for server in servers:
        print(f">>> deleting {server.name}")
        client.servers.delete(server)
    print(f"Deleted {len(servers)} box(es).")


# `dispatch plan` only needs the planning args; re-exported so the CLI can wire all three uniformly.
__all__ = ["add_plan_arguments", "add_provision_arguments", "add_teardown_arguments", "run_down", "run_up"]
