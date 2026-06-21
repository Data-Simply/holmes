# Fanning the baselines out across Hetzner CPX62 boxes

The `holmes dispatch` planner splits the `grid`/`random`/`bayes` sweep into one runnable script per
box; this directory provisions the boxes to run them. HOLMES (the agentic loop) is **not** part of
this flow — it stays on the `make holmes` target, which sandboxes each LLM session.

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

## 1. Provision

```sh
hcloud context create holmes          # one-time: authenticate the CLI
uv run python deploy/provision.py 8 \  # 8 boxes, holmes-box-0 .. holmes-box-7
    --repo-url git@github.com:Data-Simply/holmes.git \
    --branch main \
    --ssh-key my-key \
    --location nbg1                    # EU only: nbg1 | fsn1 | hel1 (enforced)
```

cloud-init installs `uv`, clones the repo, and runs `uv sync` on each box.

## 2. Plan the shards

```sh
uv run holmes dispatch --boxes 8 --fit-seeds 0 1 2 --search-seeds 0
# -> plans/box-0.sh .. plans/box-7.sh
```

## 3. Ship the data + script and run

Each box needs the preprocessed datasets and its own script. Preprocess **once** (the raw data is
30M+ rows) and copy `data/processed/` out — don't re-preprocess per box:

```sh
for i in $(seq 0 7); do
  ip=$(hcloud server ip "holmes-box-$i")
  rsync -a data/processed/        root@"$ip":/opt/holmes/data/processed/
  rsync -a "plans/box-$i.sh"      root@"$ip":/opt/holmes/box.sh
  ssh root@"$ip" 'cd /opt/holmes && nohup bash box.sh > box.log 2>&1 &'
done
```

Results land at `results/<cat>/<strategy>-seed<N>...json`, namespaced per cell so shards never
collide. Pull them back with `rsync` (or write to a shared volume). The scripts are skip-if-exists
guarded, so re-running a box after a crash resumes rather than redoing finished cells.

## Run the baselines serially on one machine

No fleet needed — plan a single box and run its script. This is the replacement for the old
`make baselines` target:

```sh
uv run holmes dispatch --boxes 1     # plans/box-0.sh with every cell
bash plans/box-0.sh                  # runs them one at a time, skip-if-exists guarded
```

## Teardown

```sh
for i in $(seq 0 7); do hcloud server delete "holmes-box-$i"; done
```
