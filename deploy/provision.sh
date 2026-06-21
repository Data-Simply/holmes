#!/usr/bin/env bash
# Provision N identical Hetzner CPX62 boxes in a European region, each prepared (via cloud-init) to
# run a HOLMES baseline shard. Requires the `hcloud` CLI, authenticated (`hcloud context active`).
#
# Usage:
#   REPO_URL=git@github.com:you/holmes.git BRANCH=main SSH_KEY=my-key ./deploy/provision.sh 8
#
# Env:
#   REPO_URL  (required) git remote cloud-init clones on each box
#   SSH_KEY   (required) name of an SSH key already added to your hcloud project
#   BRANCH    branch to clone (default: main)
#   TYPE      server type (default: cpx62 -- 16 vCPU / 32 GB AMD EPYC Genoa, top of the CPX line)
#   IMAGE     OS image (default: ubuntu-24.04)
#   LOCATION  European datacenter: nbg1 (Nuremberg), fsn1 (Falkenstein), hel1 (Helsinki). Default nbg1.
set -euo pipefail

BOXES=${1:?usage: provision.sh N   (set REPO_URL, SSH_KEY; optionally BRANCH/TYPE/IMAGE/LOCATION)}
: "${REPO_URL:?set REPO_URL to the git remote each box should clone}"
: "${SSH_KEY:?set SSH_KEY to an SSH key name in your hcloud project}"
BRANCH=${BRANCH:-main}
TYPE=${TYPE:-cpx62}
IMAGE=${IMAGE:-ubuntu-24.04}
LOCATION=${LOCATION:-nbg1}

# Enforce a European region structurally -- these are Hetzner's only EU datacenters.
case "$LOCATION" in
  nbg1 | fsn1 | hel1) ;;
  *)
    echo "LOCATION must be a European region (nbg1|fsn1|hel1), got '$LOCATION'." >&2
    exit 1
    ;;
esac

export REPO_URL BRANCH
user_data=$(envsubst '$REPO_URL $BRANCH' < "$(dirname "$0")/cloud-init.yaml")

for i in $(seq 0 $((BOXES - 1))); do
  echo ">>> creating holmes-box-$i ($TYPE, $LOCATION)"
  printf '%s' "$user_data" | hcloud server create \
    --name "holmes-box-$i" \
    --type "$TYPE" \
    --image "$IMAGE" \
    --location "$LOCATION" \
    --ssh-key "$SSH_KEY" \
    --user-data-from-file -
done

echo
hcloud server list --output columns=name,status,ipv4,location
