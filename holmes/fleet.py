"""Cloud-fleet lifecycle for the baseline fan-out: provision, ship+run, and tear down boxes.

Drives the ``holmes dispatch up`` / ``holmes dispatch down`` lifecycle around the ``dispatch`` plan.
``up`` provisions N Hetzner CPX62 boxes (via the ``hcloud`` CLI), waits for cloud-init to finish on
each, rsyncs the preprocessed datasets and that box's plan script over SSH, and starts the run.
``down`` fetches results back and deletes the fleet.

The command *builders* (``create_command``, ``rsync_push``, ...) are pure so the exact argv is
testable; the orchestration functions are thin ``subprocess`` wrappers, since they touch real
infrastructure and cannot be exercised offline.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from holmes.config import PROJECT_ROOT
from holmes.dispatch import DEFAULT_RESULTS_DIR, add_plan_arguments, plan_boxes, write_box_scripts

if TYPE_CHECKING:
    import argparse

# Hetzner's European datacenters (Nuremberg, Falkenstein, Helsinki). Restricting --location to these
# keeps the whole fleet in a European region, and on one CPU generation, by construction.
EU_LOCATIONS = ("nbg1", "fsn1", "hel1")
CLOUD_INIT = PROJECT_ROOT / "deploy" / "cloud-init.yaml"
REMOTE_DIR = "/opt/holmes"
NAME_PREFIX = "holmes-box-"

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


def render_user_data(repo_url: str, branch: str) -> str:
    """Substitute the repo URL and branch into the cloud-init template.

    Args:
        repo_url: Git remote each box clones.
        branch: Branch to clone.

    Returns:
        str: The cloud-init user-data with placeholders filled.
    """
    # Plain str.replace, not string.Template: cloud-init runcmd lines routinely contain shell ``$``
    # (``$VAR``, ``$(...)``), which Template.substitute would choke on. Only our two literal
    # ``${...}`` markers are touched.
    return CLOUD_INIT.read_text().replace("${REPO_URL}", repo_url).replace("${BRANCH}", branch)


def create_command(name: str, *, server_type: str, image: str, location: str, ssh_key: str) -> list[str]:
    """Build the ``hcloud server create`` argv for one box (user-data piped on stdin via ``-``)."""
    return [
        "hcloud", "server", "create",
        "--name", name,
        "--type", server_type,
        "--image", image,
        "--location", location,
        "--ssh-key", ssh_key,
        "--user-data-from-file", "-",
    ]  # fmt: skip


def delete_command(name: str) -> list[str]:
    """Build the ``hcloud server delete`` argv for one box."""
    return ["hcloud", "server", "delete", name]


def ip_command(name: str) -> list[str]:
    """Build the ``hcloud server ip`` argv for one box."""
    return ["hcloud", "server", "ip", name]


def list_names_command() -> list[str]:
    """Build the ``hcloud server list`` argv that prints one server name per line."""
    return ["hcloud", "server", "list", "--output", "noheader", "--output", "columns=name"]


def fleet_names(all_names: list[str], prefix: str = NAME_PREFIX) -> list[str]:
    """Filter a list of server names down to this fleet's, sorted for deterministic teardown."""
    return sorted(name for name in all_names if name.startswith(prefix))


def _ssh_transport(identity: str | None) -> str:
    """Return the ``-e`` transport string rsync uses to invoke ssh with our options."""
    parts = ["ssh", *_SSH_OPTS]
    if identity is not None:
        parts += ["-i", identity]
    return " ".join(parts)


def ssh_command(host: str, remote: str, *, user: str = "root", identity: str | None = None) -> list[str]:
    """Build an ssh argv running ``remote`` on ``host``."""
    cmd = ["ssh", *_SSH_OPTS]
    if identity is not None:
        cmd += ["-i", identity]
    cmd += [f"{user}@{host}", remote]
    return cmd


def rsync_push(
    local_src: str, host: str, remote_dest: str, *, user: str = "root", identity: str | None = None
) -> list[str]:
    """Build an rsync argv pushing ``local_src`` to ``host:remote_dest``."""
    return ["rsync", "-az", "-e", _ssh_transport(identity), local_src, f"{user}@{host}:{remote_dest}"]


def rsync_pull(
    host: str, remote_src: str, local_dest: str, *, user: str = "root", identity: str | None = None
) -> list[str]:
    """Build an rsync argv pulling ``host:remote_src`` back to ``local_dest``."""
    return ["rsync", "-az", "-e", _ssh_transport(identity), f"{user}@{host}:{remote_src}", local_dest]


def add_provision_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the provisioning arguments shared by ``up`` (and the SSH knobs ``down`` reuses)."""
    parser.add_argument("--repo-url", required=True, help="Git remote each box clones via cloud-init.")
    parser.add_argument("--ssh-key", required=True, help="Name of an SSH key in your hcloud project.")
    parser.add_argument("--branch", default="main", help="Branch to clone (default: main).")
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


def server_ip(name: str) -> str:
    """Resolve a box's public IPv4 via ``hcloud server ip``."""
    result = subprocess.run(ip_command(name), capture_output=True, text=True, check=True)
    return result.stdout.strip()


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


def _setup_box(name: str, script_path: Path, args: argparse.Namespace) -> None:
    """Wait for a provisioned box, ship its data + plan script, and start the run."""
    ip = server_ip(name)
    print(f">>> {name} ({ip}): waiting for cloud-init")
    wait_for_cloud_init(ip, user=args.ssh_user, identity=args.identity)

    processed = str(args.processed_dir)
    remote_processed = f"{REMOTE_DIR}/{processed}"
    print(f">>> {name} ({ip}): shipping {processed}/ and {script_path.name}")
    subprocess.run(
        ssh_command(ip, f"mkdir -p {remote_processed}", user=args.ssh_user, identity=args.identity),
        check=True,
    )
    subprocess.run(
        rsync_push(f"{processed}/", ip, f"{remote_processed}/", user=args.ssh_user, identity=args.identity),
        check=True,
    )
    subprocess.run(
        rsync_push(str(script_path), ip, f"{REMOTE_DIR}/box.sh", user=args.ssh_user, identity=args.identity),
        check=True,
    )
    print(f">>> {name} ({ip}): starting run")
    subprocess.run(
        ssh_command(
            ip,
            f"cd {REMOTE_DIR} && nohup bash box.sh > box.log 2>&1 < /dev/null &",
            user=args.ssh_user,
            identity=args.identity,
        ),
        check=True,
    )


def run_up(args: argparse.Namespace) -> None:
    """Provision the fleet, ship each box its data + plan script, and start the runs.

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

    user_data = render_user_data(args.repo_url, args.branch)
    for name, _ in active:
        print(f">>> creating {name} ({args.server_type}, {args.location})")
        subprocess.run(
            create_command(
                name,
                server_type=args.server_type,
                image=args.image,
                location=args.location,
                ssh_key=args.ssh_key,
            ),
            input=user_data,
            text=True,
            check=True,
        )

    for name, script_path in active:
        _setup_box(name, script_path, args)
    print(f"Started {len(active)} box(es). Tail progress with: ssh {args.ssh_user}@<ip> tail -f {REMOTE_DIR}/box.log")


def run_down(args: argparse.Namespace) -> None:
    """Fetch results back from the fleet (unless ``--no-fetch``), then delete every box.

    Args:
        args: Parsed ``dispatch down`` arguments.
    """
    listing = subprocess.run(list_names_command(), capture_output=True, text=True, check=True)
    names = fleet_names(listing.stdout.split(), args.prefix)
    if not names:
        print(f"No servers matching {args.prefix!r}; nothing to tear down.")
        return

    if not args.no_fetch:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            # Best-effort, end to end: neither an IP lookup that fails nor an empty results dir may
            # abort teardown of the rest -- otherwise one flaky box leaves the whole fleet billing.
            try:
                ip = server_ip(name)
            except subprocess.CalledProcessError:
                print(f">>> {name}: could not resolve IP; skipping result fetch")
                continue
            print(f">>> {name} ({ip}): fetching results")
            subprocess.run(
                rsync_pull(
                    ip, f"{REMOTE_DIR}/results/", f"{args.results_dir}/", user=args.ssh_user, identity=args.identity
                ),
                check=False,
            )

    for name in names:
        print(f">>> deleting {name}")
        subprocess.run(delete_command(name), check=True)
    print(f"Deleted {len(names)} box(es).")


# `dispatch plan` only needs the planning args; re-exported so the CLI can wire all three uniformly.
__all__ = ["add_plan_arguments", "add_provision_arguments", "add_teardown_arguments", "run_down", "run_up"]
