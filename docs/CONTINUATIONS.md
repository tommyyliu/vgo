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
VGO_UPDATES=80 ./runs/ddrnet-attn.sh artifacts/<run>
```

A relative output path is resolved against **your shell's directory**, so run
these from the project root. The launchers pin it there explicitly, because the
pipeline is exec'd after a `cd` into `training/`: before that guard existed, the
command above sent `mkdir`, `install` and `tee` to `artifacts/<run>` while
`--output` landed in `training/artifacts/<run>`, which is an empty directory and
therefore a cold start. The symptom is a run that begins generating at
`--runtime naive` while writing its log into the run you meant to continue.

`--updates` may be raised, never lowered. Everything outside
`OPERATIONAL_CONFIG_FIELDS` is run identity: change one and the run refuses to
resume. That includes `--samples-per-shard`, `--replay-window`,
`--leaf-batch`, the whole komi group (`--komi-low`, `--komi-high`,
`--dynamic-komi`), the architecture and every seed. Only 22 fields are
operational; `pipeline.py:34` is the list. `--inference-batch` is one of them, a
serving control that may change on resume as long as it does not exceed the
maximum embedded in the current ONNX artifact.

Stop with `kill` (SIGTERM). `rl_loop` routes it onto the Ctrl-C path, so
children are signalled rather than orphaned and wall time is recorded. A crash
costs at most the shard in flight.

### Runs that predate a flag

A missing key blocks a resume only when nothing backfills it. `compatible_config`
(`pipeline.py:216`) fills defaults for fields added without changing historical
semantics — currently `dynamic_komi` and the three `komi_recenter_*` fields — so
runs older than those stay resumable. Anything else absent, like `full_adam` /
`muon_learning_rate`, is an identity mismatch and refuses.

The digest usually needs no attention. `pipeline.py:1083` accepts a stored
`config_digest` that was taken under a known older set of operational fields, so
a run whose digest predates `inference_batch` or `promotion_arena` becoming
operational resumes and quietly re-stamps itself. `ddrnet-fresh-attn` still
carries such a digest today and resumes fine. Recompute one by hand only when
that fallback rejects it.

Fix a missing field only when you can establish what the run actually did. For
the optimizer, read the checkpoint: one param group with no `use_muon` is plain
Adam, two groups with `use_muon: true` on one is the hybrid.

```bash
training/.venv/bin/python3 -c "
import torch; c=torch.load('artifacts/<run>/updates/update-000030/candidate.pt',
  map_location='cpu', weights_only=False)
print([{k:v for k,v in g.items() if k!='params'} for g in c['optimizer_state_dict']['param_groups']])"
```

Then write the true value into `pipeline-config.json`, backing it up first.
`ddrnet-fresh-attn` was patched this way on 2026-08-10; both `.bak-20260810`
files are the pre-patch state.

### The opposite case: a backfilled field you want to change

Backfilling makes an old run resumable at its *historical* behaviour, not at
today's default. `dynamic_komi` backfills to `false`, so any recipe passing
`--dynamic-komi` is an identity change and refuses — deliberately, per the note
at `pipeline.py:220`. Turning the controller on mid-run is exactly what the
guard is for, since the replay window would then mix shards drawn from two komi
policies.

Patching the stored config to match your recipe is the wrong instinct here. The
`full_adam` patch above recorded what the run had *already done*; changing
`dynamic_komi`, `komi_low` or `komi_high` changes what it will do next, and
those are all identity fields.

Seed a new run instead — it keeps the trained weights and the training window
while giving the controller a run it is allowed to steer:

```bash
./runs/ddrnet-attn-komi.sh   # ddrnet-fresh-attn u59 -> a fresh run, komi on
```

That launcher is the worked example. Three things it has to get right, all of
which are easy to miss:

- `--initial-checkpoint` and `--initial-onnx` must be given together, and the
  seeded model enters state as version −1 so *generation* starts from it. On its
  own, `--initial-replay` seeds only the training window.
- Copy the seed shards. Retirement compresses a shard and deletes the
  uncompressed original (`pipeline.py:119`), so `--initial-replay` pointed at
  another run rewrites that run's replay directory as housekeeping.
- Spell the checkpoint path as `"$root/artifacts/..."`. `ancestor_of`
  (`scripts/rate-checkpoints.py:98`) recovers lineage by a regex that only
  understands a literal `root`; through any other variable the child silently
  rates as a cold start.

With replay seeded, the komi controller takes its starting center from the
newest seeded manifest rather than from `--komi-low/--komi-high`, which then
supply only the width. Recentering the configured range is inert in that case.

## Rating checkpoints

`scripts/dense-curve.py` plays short round-robins over a sampled pool;
`scripts/build-dense-curve.py` fits one Bradley-Terry rating and renders the
curve. Records append and readers parse by brace depth, so a tournament can be
read while it runs and stopped without losing completed rounds.

### The two commands

```bash
# 1. rebuild the curve from every tournament ever played
training/.venv/bin/python3 scripts/build-dense-curve.py \
  --ratings-json ratings.json --output curve.html

# 2. play more games (banded off the fit from step 1)
training/.venv/bin/python3 scripts/dense-curve.py artifacts/<run> [artifacts/<run> ...] \
  --stride 2 --rounds-per-checkpoint 3 --field 8 --pairs 2 \
  --ratings ratings.json --band 12 --spanning-every 4 \
  --simulations 800 --concurrency 100 --output artifacts/dense-curve-N --seed <n>
```

Then repeat step 1. That is the whole loop: fit, play, refit.

**Step 1 needs no arguments.** It discovers every `artifacts/*/records.jsonl`
and pools whatever was played at `--simulations` (default 800), which includes
the small hand-run matches, not just the sampled tournaments -- a 120-game
head-to-head is better evidence per pairing than anything the sampling
produces. Records at other budgets are skipped and named, because a model at
1600 simulations is a different player from the same model at 800. Records
naming models outside a run's `updates/` tree -- the cross-training experiment
wrote four such -- are skipped and counted.

`--anchor <run>/<version>` puts zero on a chosen checkpoint; the default is
naive when it is in the field. Anchoring is only a translation: it moves zero,
never the spacing, so no comparison changes. Prefer a reference that wins
*some* games; one that never wins has no finite rating and its distance from
the field grows with the game count instead of converging.

Pass `--records` explicitly only to pool a subset.

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

**Naive is banded like everything else.** Given `--ratings` containing
`naive/-1`, it enters the pool as an ordinary member and the bands pair it with
the update-0-era checkpoints it can still take games from — 1.5x the
information per naive game versus joining whole rounds, and it needs no tuning
as the field improves, because it simply sinks in the ordering. `--naive-rounds`
is then ignored (the run says so); it only applies to uniform matchmaking,
where there is no fit to band on.

Naive is worth keeping in the field because it is the one player whose strength
does not drift between runs. Everything else is rated relative to the rest of
the field; naive is what stops the whole scale from floating.

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

**`--samples-per-shard` is inert below about 3,700.** Draining the games already
in flight yields `actors x plies` samples on its own. Measured from the shard
manifests at `--actors 64`, the floor is steady across shard sizes:

| run | requested | produced (mean ± sd) | floor | tail |
|---|---|---|---|---|
| `ddrnet-fresh-attn` (60 shards) | 1,600 | 5,455 ± 291 | 3,855 | 71% |
| `shard-sweep-10000` (29 shards) | 6,480 | 10,248 ± 354 | 3,768 | 37% |
| `shard-sweep-15000` (40 shards) | 11,480 | 15,016 ± 353 | 3,536 | 24% |

So a 1,600 request is 71% tail. Going genuinely smaller means fewer `--actors`,
which costs throughput. Note the sd: a shard lands within about ±350 of its
mean regardless of size, so two arms differing by less than that are not
differing at all.

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
