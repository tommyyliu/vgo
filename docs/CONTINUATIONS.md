# Continuing runs and rating checkpoints

Two loops that interact: training runs produce checkpoints, and sampled
tournaments rate them. Neither watches the other, so the ordering below matters.

## Continuing a training run

Re-run the same launcher with a higher `--updates`. The coordinator takes an
flock on `<run>/.pipeline.lock`, keeps progress in `pipeline-state.json`, and
has durable resume points at the update spec, the checkpoint, the ONNX export
and the publication — so it continues rather than restarting, and a finished run
is a no-op.

```bash
VGO_UPDATES=80 ./runs/ddrnet-attn.sh artifacts/ddrnet-fresh-attn
```

`--updates` may be raised, never lowered. Everything outside
`OPERATIONAL_CONFIG_FIELDS` is run identity: change one and the run refuses to
resume. That includes `--samples-per-shard`, `--replay-window`,
`--inference-batch`, `--leaf-batch`, the architecture and every seed.

Stop with `kill` (SIGTERM). `rl_loop` routes it onto the Ctrl-C path, so
children are signalled rather than orphaned and wall time is recorded. A crash
costs at most the shard in flight.

### Runs that predate a flag

A run whose `pipeline-config.json` lacks a field today's code writes cannot
resume: the identity guard sees a new key, and the state's `config_digest`
disagrees. `ddrnet-fresh-attn` hit this with `full_adam` / `muon_learning_rate`.

Fix only when you can establish what the run actually did. For the optimizer,
read the checkpoint: one param group with no `use_muon` is plain Adam, two
groups with `use_muon: true` on one is the hybrid.

```bash
training/.venv/bin/python3 -c "
import torch; c=torch.load('artifacts/<run>/updates/update-000030/candidate.pt',
  map_location='cpu', weights_only=False)
print([{k:v for k,v in g.items() if k!='params'} for g in c['optimizer_state_dict']['param_groups']])"
```

Then write the true value into `pipeline-config.json`, recompute
`config_digest` in `pipeline-state.json` from `canonical_digest(identity_config(config))`,
and back both files up first.

## Rating checkpoints

`scripts/dense-curve.py` plays short round-robins over a sampled pool;
`scripts/build-dense-curve.py` fits one Bradley-Terry rating and renders the
curve. Records append and readers parse by brace depth, so a tournament can be
read while it runs and stopped without losing completed rounds.

```bash
# a tournament over every run, uniform pairing
training/.venv/bin/python3 scripts/dense-curve.py artifacts/<run> [artifacts/<run> ...] \
  --stride 2 --rounds-per-checkpoint 3 --field 8 --pairs 2 \
  --simulations 800 --concurrency 100 --output artifacts/dense-curve-N --seed <n>

# fit and render, pooling every tournament played so far
training/.venv/bin/python3 scripts/build-dense-curve.py \
  --records artifacts/dense-curve/records.jsonl \
  --records artifacts/dense-curve-N/records.jsonl \
  --ratings-json ratings.json --output curve.html
```

`--anchor <run>/<version>` puts zero on a chosen checkpoint; the default is
naive when it is in the field.

### Banded matchmaking

Bradley-Terry information per game is proportional to `p(1-p)`, so a 97-3
pairing is nearly worthless. Feed the previous fit back to match on strength:

```bash
training/.venv/bin/python3 scripts/dense-curve.py artifacts/<run> ... \
  --ratings ratings.json --band 12 --spanning-every 4 ...
```

Measured on an 83-checkpoint field, this raises mean `p(1-p)` from 0.108 to
0.198 — **1.8x the information per game** — while staying one connected
component. Do not drop `--spanning-every`: banding alone turns the comparison
graph into a chain, and comparing its ends then accumulates error through every
link. The spanning rounds are the long baselines.

Pairing on estimated rating does not bias the fit. The MLE stays unbiased as
long as pairing depends on prior estimates rather than on outcomes.

## The pool is fixed at launch

**`dense-curve.py` globs its checkpoints once, when it starts.** Checkpoints
created afterwards are invisible to it — no tournament picks up new updates on
its own. So:

1. finish (or stop) the continuation run,
2. then start a tournament, which globs the pool as it now stands,
3. pool its records with the earlier ones when fitting.

Getting this backwards is why `ddrnet-fresh-muon` updates 45-54 were missing
from two tournaments and the curve appeared to stop at update 42.

Resume is per round index in `<output>/rounds-done.json`, so re-running the same
command continues a tournament. Changing `--rounds-per-checkpoint` or `--seed`
renumbers the schedule — start a new output directory instead and pool the
records.

## Things that bite

**`--samples-per-shard` is inert below about 3,400.** Draining the games already
in flight yields `actors x plies` samples on its own — 64 x ~53. Requesting
1,600 gives a ~5,200-sample shard of which 69% is the tail. Going genuinely
smaller means fewer `--actors`, which costs throughput.

**`--arena-pairs 8` is too small to gate on.** Sixteen games at a 0.55 threshold
needs roughly a +90 Elo gain to pass reliably. Late in a run, where updates
deliver far less, it rejects real progress and occasionally promotes a
regression: `ddrnet-fresh-muon` u53 passed on 9-7 and then lost 51-69 over 120
games to the incumbent it replaced.

**Consecutive checkpoints vary by hundreds of Elo.** Adam swung 373 between
u50 and u51 against a ±59 measurement error. A run's "best checkpoint" is
substantially its luckiest, and a single checkpoint is weak evidence about the
run.

**Rating tournaments run at 800 simulations, generation at 1,600.** Ratings
therefore describe strength at half the search the models trained under. One
180-game sweep across 200/800/3200 found no significant trend for a single pair,
but it has not been checked generally.

**A tournament round holds `field * (field - 1) * pairs` games.** With
`--field 8 --pairs 1` that is 56, so a `--concurrency` above it does nothing.
Raise `--pairs` to fill the machine.
