# Fanning the baselines out across Hetzner CPX62 boxes

`holmes dispatch` plans the `grid`/`random`/`bayes` sweep into independent cells and runs them — on a
fleet of cloud boxes or on your own machine. HOLMES (the agentic loop) is **not** part of this flow;
it stays on the `make holmes` target, which sandboxes each LLM session.

```sh
holmes dispatch plan   # write one runnable script per box (add --run to run here, now)
holmes dispatch up     # provision a fleet, ship data + scripts, start the runs
holmes dispatch down   # fetch results back, then delete the fleet
```

## Target instance: CPX62

| Spec | Value |
|---|---|
| vCPU | 16 (shared) |
| RAM | ~31 GiB (32 GB) |
| Disk | 640 GB SSD |
| CPU | AMD EPYC Genoa (CPX Gen2) |
| Price | ~€38.99/mo (~€0.24/hr) |
| Regions (EU) | nbg1 (Nuremberg), fsn1 (Falkenstein), hel1 (Helsinki) |

It's the top of Hetzner's CPX line. One box runs one baseline cell at a time (a full strategy run is
a multi-GB ALS model, so fits never overlap on a box). A uniform all-CPX62 fleet is one CPU
generation, so a config scores identically on every box — keep the whole campaign on one type.

> **Check first:** measure peak RSS of a `factors=512` fit. If a single model exceeds ~31 GiB,
> no CPX box holds it and you must move to the dedicated CCX line.

## Run locally (no fleet)

Plan a single box and run its cells right here, one at a time — the replacement for the old
`make baselines` target:

```sh
uv run holmes dispatch plan --boxes 1 --run        # everything, serially, on this machine
uv run holmes dispatch plan --boxes 1 --run --strategies grid   # just one strategy
```

Without `--run`, `plan` only writes `plans/box-0.sh` for you to inspect or `bash` yourself. `--run`
honours the same skip-if-exists guard, so it resumes after an interruption.

## Run on a Hetzner fleet

The Hetzner API goes through the `hcloud` Python SDK (a project dependency — no separate CLI to
install). Boxes run **your local working-tree code**: `up` rsyncs the checkout, so there's no repo
URL, branch, or `git push` to coordinate.

Prerequisites:

- **`HCLOUD_TOKEN`** in your environment — a Hetzner Cloud API token (Console → your project →
  Security → API Tokens, Read & Write). The SDK reads it directly; `export HCLOUD_TOKEN=...` (use
  `read -rs HCLOUD_TOKEN` to keep it out of shell history).
- An **SSH key in your hcloud project** (`hcloud` web console, or any Hetzner client) whose name you
  pass as `--ssh-key`, with the matching private key available locally for `--identity`.
- Datasets **preprocessed locally** (the raw data is 30M+ rows — preprocess **once**; `up` ships
  `data/processed/` to each box).

### 1. Provision and start (`up`)

```sh
export HCLOUD_TOKEN=...
uv run holmes dispatch up \
    --boxes 8 \
    --ssh-key my-key \
    --identity ~/.ssh/id_ed25519 \      # private key for ssh/rsync to the boxes
    --location nbg1 \                    # EU only: nbg1 | fsn1 | hel1 (enforced)
    --fit-seeds 0 1 2 --search-seeds 0
```

`up` plans the shards, creates the boxes, waits for cloud-init (which installs uv) on each, rsyncs the
local code + `data/processed/` + that box's plan script, runs `uv sync`, and starts the run in the
background. Tail one with `ssh root@<ip> tail -f /opt/holmes/box.log`.

To eyeball the partition before spending money, run `holmes dispatch plan` with the same flags first.

### 2. Tear down (`down`)

```sh
uv run holmes dispatch down --identity ~/.ssh/id_ed25519
```

`down` rsyncs each box's `results/` back into your local `results/` (skip with `--no-fetch`), then
deletes every `holmes-box-*` server. Results are namespaced per cell, so the fetched files merge
without collision.

## Notes

- Boxes run one cell at a time (the multi-GB-model memory guard); parallelism is purely across boxes.
- Everything is skip-if-exists guarded, so re-running `up` (or a box script) resumes rather than
  redoing finished cells.
- Keep the whole campaign on one instance type so a config scores identically on every box.
