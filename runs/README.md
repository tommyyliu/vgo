# Run recipes

One script per experiment. These are tracked; `artifacts/` is not, so this
directory is the only thing that survives a fresh clone — and the only record
of what a run actually was.

```
./scripts/setup.sh          # once per machine
./scripts/smoke.sh          # prove the box, ~90s
./runs/ddrnet-attn.sh       # -> artifacts/ddrnet-attn
```

Each recipe takes an optional output directory, copies itself to
`<run>/launch.sh`, and tees the coordinator's output to `<run>/logs/run.log`.

## Resuming

Re-run the identical command. The coordinator takes an flock lease on
`<run>/.pipeline.lock`, keeps progress in `pipeline-state.json`, and has
durable resume points at the update spec, the checkpoint, the ONNX export and
the publication — so it continues rather than restarting, and a completed run
is a no-op. A crash costs at most the shard being generated.

`--updates` may be raised on resume, never lowered. So sizing a run to a
rental window is free:

```
VGO_UPDATES=1  ./runs/ddrnet-attn.sh      # one update, read the wall time
VGO_UPDATES=40 ./runs/ddrnet-attn.sh      # resumes; the first update is kept
```

## Which settings may be parameterized

Only fields in `OPERATIONAL_CONFIG_FIELDS` (`training/vgo_training/pipeline.py`)
may read from the environment. Everything else is part of the run's identity,
and the coordinator refuses to resume a run whose identity config changed —
so exposing one of those as a variable makes the run unresumable the moment
anyone sets it.

Safe to vary per machine: `VGO_UPDATES`, `VGO_ACTORS`, `VGO_ARENA_ACTORS`,
`VGO_SLOTS`, `VGO_TRAINING_THREADS`.

Not safe, and deliberately literal in every recipe: `--samples-per-shard`,
`--replay-window`, `--inference-batch`, `--leaf-batch`,
`--concurrent-generators`, the board and komi settings, the architecture, the
learning rates, and all seeds.

## Adding one

Copy the nearest recipe, change the seeds, and write down in the header *why*
each non-default value is what it is — the measurement, not the intention. A
recipe whose header says "w64 because w96 memorises, train value MAE 0.018
against 0.247 validation" is worth keeping; one that says "tuned" is not.

Check the optimizer explicitly. `--full-adam` selects Adam; the default is
Muon on the trunk with Adam on the heads. Runs predating that flag have no
`full_adam` key in their `pipeline-config.json` and were Adam, so copying one
of those forward without adding `--full-adam` silently changes the optimizer.

## Sizing a machine

Generation is **CPU-bound**: it runs 87–89% user across 32 logical cores while
the GPU sits at 50–63%. Choose an instance on vCPU count, not GPU class, and
set `VGO_ACTORS` from the core count. Reference throughput is ~6.6 self-play
samples/sec on a 16-core/32-thread 9950X with an RTX 5070 Ti.

`--actors 64`, `--inference-batch 64` and `--inference-slots 2` assume a 16 GB
card. A 40-update run needs roughly 5 GB of disk, almost all of it
`updates/`, which is never pruned.
