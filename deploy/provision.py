#!/usr/bin/env python3
r"""Provision N identical Hetzner CPX62 boxes in a European region, each prepared to run a HOLMES shard.

Python port of the former provision.sh. Requires the ``hcloud`` CLI, authenticated
(``hcloud context active``). The sibling ``cloud-init.yaml`` installs uv, clones the repo, and runs
``uv sync`` on each box; this script substitutes the repo/branch into it and creates the servers.

Example:
    uv run python deploy/provision.py 8 \\
        --repo-url git@github.com:Data-Simply/holmes.git --ssh-key my-key --location nbg1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from string import Template

# Hetzner's European datacenters: Nuremberg, Falkenstein, Helsinki. Restricting --location to these
# keeps the whole fleet in a European region (and on one CPU generation) by construction.
EU_LOCATIONS = ("nbg1", "fsn1", "hel1")
CLOUD_INIT = Path(__file__).with_name("cloud-init.yaml")


def render_user_data(repo_url: str, branch: str) -> str:
    """Substitute the repo URL and branch into the cloud-init template.

    Args:
        repo_url: Git remote each box clones.
        branch: Branch to clone.

    Returns:
        str: The cloud-init user-data with placeholders filled.
    """
    return Template(CLOUD_INIT.read_text()).substitute(REPO_URL=repo_url, BRANCH=branch)


def create_command(index: int, *, server_type: str, image: str, location: str, ssh_key: str) -> list[str]:
    """Build the ``hcloud server create`` argv for one box (user-data is piped on stdin via ``-``).

    Args:
        index: Box index, used in the server name ``holmes-box-<index>``.
        server_type: Hetzner server type (e.g. ``cpx62``).
        image: OS image (e.g. ``ubuntu-24.04``).
        location: Datacenter location.
        ssh_key: Name of an SSH key in the hcloud project.

    Returns:
        list[str]: The command to run.
    """
    return [
        "hcloud", "server", "create",
        "--name", f"holmes-box-{index}",
        "--type", server_type,
        "--image", image,
        "--location", location,
        "--ssh-key", ssh_key,
        "--user-data-from-file", "-",
    ]  # fmt: skip


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the provisioner's command-line arguments."""
    parser = argparse.ArgumentParser(prog="provision.py", description=__doc__)
    parser.add_argument("boxes", type=int, help="Number of boxes to create (holmes-box-0 .. N-1).")
    parser.add_argument("--repo-url", required=True, help="Git remote each box clones via cloud-init.")
    parser.add_argument("--ssh-key", required=True, help="Name of an SSH key in your hcloud project.")
    parser.add_argument("--branch", default="main", help="Branch to clone (default: main).")
    parser.add_argument(
        "--type",
        dest="server_type",
        default="cpx62",
        help="Server type (default: cpx62 -- 16 vCPU / 32 GB AMD EPYC Genoa, top of the CPX line).",
    )
    parser.add_argument("--image", default="ubuntu-24.04", help="OS image (default: ubuntu-24.04).")
    parser.add_argument(
        "--location",
        default="nbg1",
        choices=EU_LOCATIONS,
        help="European datacenter: nbg1 (Nuremberg), fsn1 (Falkenstein), hel1 (Helsinki). Default nbg1.",
    )
    args = parser.parse_args(argv)
    if args.boxes < 1:
        parser.error(f"boxes must be >= 1, got {args.boxes}.")
    return args


def main(argv: list[str] | None = None) -> None:
    """Create the requested boxes, then print the resulting server list.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).
    """
    args = _parse_args(argv)
    user_data = render_user_data(args.repo_url, args.branch)
    for index in range(args.boxes):
        print(f">>> creating holmes-box-{index} ({args.server_type}, {args.location})")
        command = create_command(
            index,
            server_type=args.server_type,
            image=args.image,
            location=args.location,
            ssh_key=args.ssh_key,
        )
        subprocess.run(command, input=user_data, text=True, check=True)
    print()
    subprocess.run(["hcloud", "server", "list", "--output", "columns=name,status,ipv4,location"], check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
