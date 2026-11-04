#!/usr/bin/env bash
# The RL loop, retraining in bulk instead of incrementally.
#
# ## Why this exists
#
# The old continuous loop warm-started one epoch on a moving window every 12k
# samples. Twenty-two such updates produced a model that a *from scratch* run on
# the same corpus beat 41-7 -- +307 Elo, CI [+195, +527], 48 games at equal
# simulations. A 26M-parameter model trained the same way scored +154 against
# the same opponent, so the gain is the training regime and not capacity: three
# times the parameters measured slightly worse, on validation and in play.
#
# The incremental chain is what accumulates the damage. This script keeps the
# continuous generator -- which exists to avoid the shard tail, where 30 of 32
# actors idle waiting for the slowest game -- and replaces only what training
# does.
#
# ## The schedule
#
# Retrain from scratch when `turnover` of the window is new, for `epochs`
# passes, and adopt it. There is no gate, deliberately.
#
# A gate earns its cost when a bad model would compound. Warm-starting is where
# that happens: one damaged update poisons every descendant. Training from
# scratch breaks the chain -- a bad model degrades only the games its own round
# produces, and the next retrain starts from the corpus rather than from it. The
# failure is self-limiting.
#
# A gate could not do the job here regardless. The 48-game matches this schedule
# was chosen from returned CIs of [+56, +284] and [+195, +527]; a gate small
# enough to afford every round resolves about +/-120 Elo, which catches a
# catastrophe and nothing finer, for ~23 minutes of a GPU that would otherwise
# be generating.
#
# Progress is measured through a connected graph of matches against sampled
# earlier models and fixed references every `anchor_every` rounds.
#
# The window cancels out of the cost: training duty is `epochs * 0.0125 /
# turnover` regardless of how wide it is, so those two knobs alone set how much
# of the GPU goes to training rather than to the generation that feeds it. At 8
# epochs and 15% turnover that is about two thirds of generation time on paper, on
# figures from a slower recipe, so check the lifecycle logs. Generation never stops
# -- the generator holds the incumbent model throughout -- so this is contention
# for the GPU, not downtime.
#
# Note the window cancels: a narrower one retrains sooner but each retrain is
# proportionally cheaper. Widen it for diversity, not to train less often.
#
# Epoch count is from prior testing: high single digits is safe here. It matters
# because the learner publishes the *final* epoch; `selection_regret` reports
# the difference from the best validation epoch. Adoption is ungated, so epoch
# changes need a separate strength experiment.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="${VGO_OUTPUT:-$root/artifacts/vgo-bulk}"
# Absolute, always. Training runs inside `( cd "$root/training" && ... )`, so a
# relative --games-root resolves against the wrong directory and dies with
# FileNotFoundError -- but only at the first retrain, hours after launch, with
# generation having run correctly the whole time because the loop itself sits in
# the repo root.
mkdir -p "$output"
output="$(cd "$output" && pwd)"
games="$output/games"
models="$output/models"

# 450k since 2026-11-04. The value head memorized a 150k window (train MAE 0.03,
# held-out 0.65), and a window drawn from twice as many generations beat update
# 23's own 34-14. Ownership as bits and the ridge as uint8 brought the window to
# 57 KiB a sample, so 450k trains at a 27 GB peak where 150k once took 18.
window="${VGO_WINDOW_SAMPLES:-450000}"
# Input raster, square. The policy grid stays at 128 either way, so this changes
# what the network sees, not which moves it can propose: boundary cells whose
# centre is illegal still snap to the nearest legal point inside them. 128 searches
# 2-3.6x faster than 256 and scored 30-18 at equal time on the big board.
resolution="${VGO_RESOLUTION:-128}"
# The net's shape. Early in a run generation speed matters more than play, so the
# default is the lighter net from the loss screen: 37k positions/s in the generator
# against 21k. Any net's games train any shape, so a later stage can restart with
# VGO_SHAPE="" VGO_CONTEXT_ATTENTION_BLOCKS=1 VGO_NORMS_PER_BLOCK=2 for the full
# net, at the cost of one retrain. See the Net shapes tab of the dashboard.
shape="${VGO_SHAPE-{\"detail_blocks\":[1,1,1],\"context_blocks\":[2,2],\"tail\":\"conv\"\}}"
context_attention_blocks="${VGO_CONTEXT_ATTENTION_BLOCKS:-0}"
# Board sizes, as space-separated `WEIGHT:UNITS` or `WEIGHT:LOW-HIGH` bands in
# stone diameters across. Komi and the ply cap follow from each game's size.
read -r -a board_mix <<< "${VGO_BOARD_MIX:-50:38 25:18 25:18-38}"
# Fraction of the window replaced before retraining. Training competes with
# generation; the historical duty estimate above was measured on a different
# recipe and should be rechecked from the current lifecycle/performance logs.
# 5% of 450k is the 22.5k samples a retrain waited for at 15% of 150k, so updates
# keep their cadence; 3 epochs keeps the steps per update near the old 8 on 150k.
turnover="${VGO_TURNOVER:-0.05}"
epochs="${VGO_EPOCHS:-3}"
# Value targets: 16 positions per game, spread evenly, at a total value weight
# of 1/16 of an unthinned window. Every ply of a game shares one outcome, so the
# full window memorized ~800 labels; on update 23's window this recipe cut
# held-out value MSE from 0.99 to 0.58, and the policy improved as the value
# weight fell, flat at the no-value bound from 1/8 down. See docs/FINDINGS.md.
value_labels_per_game="${VGO_VALUE_LABELS_PER_GAME:-16}"
value_weight="${VGO_VALUE_WEIGHT:-0.0625}"
updates="${VGO_UPDATES:-200}"
actors="${VGO_ACTORS:-64}"
inference_slots="${VGO_INFERENCE_SLOTS:-3}"
telemetry_seconds="${VGO_TELEMETRY_SECONDS:-30}"
simulations="${VGO_SIMULATIONS:-1000}"
full_search_fraction="${VGO_FULL_SEARCH_FRACTION:-1.0}"
fast_simulations="${VGO_FAST_SIMULATIONS:-200}"
norms_per_block="${VGO_NORMS_PER_BLOCK:-1}"
# Training only, and absent from the exported graph. Unlike the value label, it
# varies within a game, so it can't be memorized by trajectory.
ownership_weight="${VGO_OWNERSHIP_WEIGHT:-1.5}"
# The naive evaluator's value is a heuristic, so deep search on it buys little,
# and with no model the whole generator runs on CPU while the GPU idles. A
# from-scratch run spends its first generation here, so keep it cheap.
naive_simulations="${VGO_NAIVE_SIMULATIONS:-200}"
# Komi, as a fixed number of points on every board -- the same convention as
# Go, where 7 points is roughly right from 9x9 to 19x19.
#
# `komi_centre = coefficient * radius^2`, and a board of radius r holds about
# (1/2r)^2 stones, so `komi * points` is the constant and the coefficient is
# four times it. 1/18 holds ~81 and 1/38 ~361, which is 9x9 and 19x19.
#
# 28.0 is 7 points. The previous 33.696 was 8.42, fit when the models were much
# weaker. Komi is White's compensation -- `analysis.rs` scores
# `black - white - komi` -- and 1/18 currently gives Black 18% of 358 games, so
# it wants less komi, not more. 1/38 sits at 53% and moves a point the wrong way,
# which is the smaller of the two errors.
komi_area_coefficient="${VGO_KOMI_AREA_COEFFICIENT:-28.0}"
# The search tail. Games that look decided play on at a cheaper budget, so search
# moves to positions where it matters and every game still reaches a real result.
# A schedule from recent games says how far behind a side must be, by board and
# stage, and it is rebuilt before each generator. See crates/vgo-selfplay/src/tail.rs.
#
# A stage without enough evidence stays closed, so a new run starts with full search.
tail_target="${VGO_TAIL_TARGET:-0.05}"
tail_games="${VGO_TAIL_GAMES:-1000}"
tail_simulations="${VGO_TAIL_SIMULATIONS:-400}"
tail_window="${VGO_TAIL_WINDOW:-5}"
# Share of tail plies to train on. All of them stay on disk.
tail_keep="${VGO_TAIL_KEEP:-0.25}"
# The random stop, a bootstrap until the value head supports resignation: once a
# side has passed this many times running, each further pass ends the game with
# the probability below, awarded by area. Self-play and rating rounds both use
# it, so ratings measure the game being trained. On the first fresh run's games
# it would have ended 74% of big-board games about 61 plies early, where 72% of
# long streaks ran to the cap. VGO_RANDOM_STOP_AFTER=0 turns it off.
random_stop_after="${VGO_RANDOM_STOP_AFTER:-4}"
random_stop_probability="${VGO_RANDOM_STOP_PROBABILITY:-0.25}"
packed_input="${VGO_PACKED_INPUT:-1}"
root_noise="${VGO_ROOT_NOISE:-0.10}"
# Measurement against a fixed anchor, every N rounds. Both seats always get the
# same simulation count: an unequal budget measures what search is worth, not
# who is stronger.
anchor_every="${VGO_ANCHOR_EVERY:-3}"
# Three pairs against each of three sampled opponents: 18 games an update,
# against 48 before. Any one of them resolves almost nothing; the rating fit
# over the accumulated graph is what gets sharper.
anchor_pairs="${VGO_ANCHOR_PAIRS:-3}"
anchor_samples="${VGO_ANCHOR_SAMPLES:-3}"
# How many recent models the sample is drawn from, alongside the fixed
# references. Wider reaches further back for diversity; narrower keeps every
# match close in strength and therefore informative.
anchor_pool="${VGO_ANCHOR_POOL:-12}"
# Held at 800 while generation moves to 1000. Both seats always match each
# other, which is what makes a match fair; holding the number fixed across runs
# is what makes the *series* comparable, and -44 / -89 / +280 were all measured
# here.
anchor_simulations="${VGO_ANCHOR_SIMULATIONS:-800}"
# A fixed anchor, if set: every round then plays it, and the rating graph keeps
# that model at zero. Unset (the default), the reserved slot goes to the model
# `anchor_lag` updates back instead. A fixed early anchor is swept once the run
# moves on, and a sweep says almost nothing; a model a few updates back is close
# enough to lose to, and still ties each round to the older, connected graph.
anchor_model="${VGO_ANCHOR_MODEL:-}"
anchor_lag="${VGO_ANCHOR_LAG:-6}"
# The board rating rounds are played on, and the ply cap that belongs to it.
# Generation scales its cap by (reference / radius)^2; the arena takes a flat
# number, so it has to be told. 1/38 since 2026-11-04: on 1/18 the first run's
# curve rose to update 12 while 1/38 showed it falling from there, and big
# boards are 73% of self-play plies. 312 is the 1/38 cap; komi is the
# coefficient's centre for this radius (28 * r^2).
anchor_radius="${VGO_ANCHOR_RADIUS:-0.02631578947368421}"
anchor_max_plies="${VGO_ANCHOR_MAX_PLIES:-312}"
anchor_komi="${VGO_ANCHOR_KOMI:-0.019390581717451522}"
maximum_plies="${VGO_MAX_PLIES:-70}"
seed_model="${VGO_SEED_MODEL:-}"
[ -n "$seed_model" ] && seed_model="$(cd "$(dirname "$seed_model")" && pwd)/$(basename "$seed_model")"

python="$root/training/.venv/bin/python"
source "$root/scripts/env/ort.sh"
mkdir -p "$games" "$models"

# Lifecycle timestamps join training contention and generator handoffs to the
# per-process performance.jsonl records. Arguments are data, never Python code.
record_event () {
  "$python" - "$output/events.jsonl" "$@" "loop_pid=$$" <<'PY'
import json, sys, time
event = {"schema": "vgo.loop-event.v1", "unix_seconds": time.time(), "event": sys.argv[2]}
event.update(argument.split("=", 1) for argument in sys.argv[3:])
with open(sys.argv[1], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(event) + "\n")
PY
}

# Samples training would keep, from manifests rather than by loading anything.
count_samples () {
  ( cd "$root/training" && "$python" -m vgo_training.window "$games" --tail-keep "$tail_keep" )
}

# 64 actors keep a full batch waiting, so each slot's deck stays filled. At 32 the
# queue runs dry on a machine with fewer cores: see "Sizing a machine" in RUNNING.md.
#
# `--maximum-batch 32` stays. Three lanes pay for a net this fast at 128, where two
# leave the card idle between batches. On the full net three gain nothing over
# whole games, and they hang at 256 within a minute, so use VGO_INFERENCE_SLOTS=2
# there. At 4 the inference threads livelock, spinning at 100% CPU while every actor
# blocks on a result that never arrives. `nvidia-smi` reports 100% utilization
# either way, so the tell is power draw, which sits near idle (~60 W) instead of
# the 200 W+ of real load.
#
# `--leaf-batch` stays at 4 for a different reason: leaf parallelism trades
# search quality for throughput, measured at -70 Elo for batch 32 in the browser
# client, and halving the simulations already spent the quality budget.
#
# Nothing in the invocation below may be interrupted by a comment. A trailing
# backslash joins the next line, so a `#` on it comments out every remaining
# argument and the binary starts with a silently truncated command line -- no
# `--model` reads as a naive generator, which still runs and still writes games.
# Before each generator, because a schedule describes the games behind it and the
# models that played them. A failure here stops the run rather than hiding: losing
# the tail quietly costs search with nothing else looking wrong.
build_tail_schedule () {
  local path="$1"
  "$python" "$root/training/loop/tail-calibration.py" "$games" --output "$path" \
    --games "$tail_games" --target "$tail_target" --window "$tail_window"
}

start_generator () {
  local label="$1" model="$2" first_game="$3"
  local budget="$simulations"
  [ -z "$model" ] && budget="$naive_simulations"
  # A tail must stay cheaper than a normal move.
  local tail_budget="$tail_simulations"
  [ "$tail_budget" -gt "$budget" ] && tail_budget="$budget"
  local stop_file="$games/$label.stop"
  rm -f "$stop_file"
  local schedule="$output/tail-schedules/$label.json"
  build_tail_schedule "$schedule" || { echo "[loop] could not build $schedule" >&2; return 1; }
  record_event tail_schedule "label=$label" "schedule=$schedule"
  local model_flag=()
  local search_fraction=1.0
  [ -n "$model" ] && model_flag=(--model "$model")
  [ -n "$model" ] && search_fraction="$full_search_fraction"
  setsid nohup "$root/target/release/vgo-generate-continuous" \
    --output-root "$games" --label "$label" \
    --stop-file "$stop_file" --first-game "$first_game" \
    --actors "$actors" --simulations "$budget" \
    --full-search-fraction "$search_fraction" --fast-simulations "$fast_simulations" \
    --telemetry-seconds "$telemetry_seconds" \
    --resolution "$resolution" --policy-resolution 128 --raster-kind compact-radius \
    "${board_mix_flags[@]}" \
    --max-plies "$maximum_plies" --radius 0.05555555555555555 \
    --coarse-pool 16 --widening-coefficient 6.0 --maximum-candidates 321 \
    --komi-area-coefficient "$komi_area_coefficient" \
    --komi-low 0.017 --komi-high 0.137 \
    --tail-schedule "$schedule" --tail-simulations "$tail_budget" --tail-window "$tail_window" \
    --temperature 1.0 --temperature-plies 30 \
    --root-exploration-noise "$root_noise" \
    --random-stop-after "$random_stop_after" --random-stop-probability "$random_stop_probability" \
    --leaf-batch 4 --maximum-batch 32 --delay-ms 1 --inference-slots "$inference_slots" \
    --provider tensorrt --fp16 true \
    --cache-directory "$root/artifacts/onnx-cache" \
    --seed $((70000 + first_game)) \
    "${model_flag[@]}" \
    >> "$output/generate.log" 2>&1 &
  local pid=$!
  record_event generator_started "label=$label" "pid=$pid" "model=$model"
  echo "$pid"
}


board_mix_flags=()
for band in "${board_mix[@]}"; do board_mix_flags+=(--board-mix "$band"); done
shape_flags=()
[ -n "$shape" ] && shape_flags=(--shape "$shape")

# Anything that ends this script ends the generator it started, which would
# otherwise keep filling the disk. `label` is whichever generator is current.
label=""
trap '[ -n "$label" ] && touch "$games/$label.stop"' EXIT

model="$seed_model"
# With no seed and no named anchor, the run's first trained model becomes the
# reference once it exists, so a from-scratch run still builds a rating graph.
# The first model rating rounds may reach back to: a fixed anchor if one is set,
# else the seed, else the first model this process trains.
first_rated="${anchor_model:-$seed_model}"

# Continue the update numbering from whatever is on disk. Restarting from zero
# rewrites `update-000000`, which on a restart is usually the model just passed
# as `--seed-model`: the loop would destroy its own starting point and only
# notice later, when the history it wanted to compare against was gone.
first_update=$(ls "$models"/update-*.pt 2>/dev/null \
  | sed 's/.*update-0*\([0-9]\+\)\.pt/\1/' | sort -n | tail -1)
first_update=$(( ${first_update:-(-1)} + 1 ))
[ "$first_update" -gt 0 ] && echo "[loop] continuing from update $first_update"

# New samples that must land before a retrain. Bash cannot multiply by 0.50.
step=$("$python" -c "print(int($window * $turnover))")
echo "[loop] window $window, retrain every $step new samples, $epochs epochs from scratch"

generation=0
# Game indices never restart: a reused index replays a seed, and `write_game`
# treats an existing directory as done and returns without writing -- so a
# collision discards the finished game in silence rather than failing loudly.
# Derived from disk, not from the update number, because the update number
# repeats whenever a restart resumes the same update.
highest=$(find "$games" -maxdepth 2 -name 'game-*' -type d 2>/dev/null \
  | sed 's/.*game-0*\([0-9]\+\)$/\1/' | sort -n | tail -1)
next_game=$(( 1000000 + first_update * 1000000 ))
[ "${highest:-0}" -ge "$next_game" ] && next_game=$(( highest + 1000 ))
# Tagged with the launch time. Generation numbering restarts at zero every run,
# so two runs against the same games directory used to write into one directory
# -- `gen-000000-seed` held games from two of them. The window selector no
# longer takes recency from these names, but merged directories are still a trap
# for anything that reads them.
run_tag="$(date '+%m%d%H%M')"
label="gen-$(printf '%06d' "$generation")-$run_tag-seed"
generator=$(start_generator "$label" "$model" "$next_game") || exit 1
echo "[loop] generation $generation started (pid $generator, model ${model:-none})"

baseline=$(count_samples)
for ((update = first_update; update < first_update + updates; update++)); do
  target=$(( baseline + step ))
  echo "[loop] update $update: waiting for $target samples (have $baseline)"
  while [ "$(count_samples)" -lt "$target" ]; do
    if ! kill -0 "$generator" 2>/dev/null; then
      echo "[loop] generator exited unexpectedly; see $output/generate.log" >&2
      exit 1
    fi
    sleep 60
  done

  # The naive generator's games stop being useful once there is enough to train
  # on: the first trained model replaces it either way. Left running, it held all
  # 32 threads through the first training, which took 64 minutes against about 9,
  # and wrote 38 GB of games no window used. The stop file drains it instead.
  if [ -z "$model" ] && [ ! -e "$games/$label.stop" ]; then
    touch "$games/$label.stop"
    record_event generator_draining "label=$label" "pid=$generator" "reason=naive_window_full"
    echo "[loop] update $update: naive generator stopped; training has its window"
  fi

  checkpoint="$models/update-$(printf '%06d' "$update").pt"
  echo "[loop] update $update: training from scratch on the most recent $window samples"
  record_event training_started "update=$update" "checkpoint=$checkpoint" "generator_pid=$generator"
  # No `--initial-checkpoint`, and that is the entire point of this script. The
  # incremental chain is what accumulated the damage a from-scratch run on the
  # same corpus beat 41-7.
  #
  # `--precision bfloat16`: the default is float32, which every update of the
  # incremental loop paid for nothing. Measured 11.6 s per 1k samples against
  # 7.5 s per 1k with `--compile` at 3.2x the parameters.
  #
  # Nothing below may be interrupted by a comment: a `#` on a backslash-continued
  # line comments out every remaining argument and the command runs truncated.
  ( cd "$root/training" && "$python" "$root/training/loop/train-once.py" \
      --games-root "$games" --window-samples "$window" \
      --output "$checkpoint" \
      --raster-kind compact-radius --resolution "$resolution" \
      --model-width 64 --blocks 16 --context-attention-blocks "$context_attention_blocks" \
      "${shape_flags[@]}" \
      --attention-heads 8 --norm-groups 8 --norms-per-block "$norms_per_block" --precision bfloat16 \
      --epochs "$epochs" --batch-size 64 --learning-rate 0.0005 \
      --value-weight "$value_weight" --value-labels-per-game "$value_labels_per_game" \
      --ownership-weight "$ownership_weight" --validation-fraction 0.1 \
      --tail-keep "$tail_keep" \
      --schedule wsd --warmup-epochs 0 --compile \
      --seed $((90000 + update)) --report-every 1 \
  ) >> "$output/train.log" 2>&1 || { record_event training_failed "update=$update"; echo "[loop] training failed" >&2; exit 1; }
  record_event training_finished "update=$update" "checkpoint=$checkpoint"

  onnx="${checkpoint%.pt}.onnx"
  packed_flag=()
  [ "$packed_input" = "1" ] && packed_flag=(--packed-input)
  ( cd "$root/training" && "$python" -m vgo_training.export_onnx \
      --checkpoint "$checkpoint" --output "$onnx" --maximum-batch 64 \
      "${packed_flag[@]}" \
  ) >> "$output/train.log" 2>&1 || { record_event export_failed "update=$update"; echo "[loop] export failed" >&2; exit 1; }
  record_event export_finished "update=$update" "model=$onnx"

  # Start the successor before stopping the incumbent, so generation never
  # pauses. `next_game` advances past anything the old process could claim.
  generation=$((generation + 1))
  next_game=$((next_game + 1000000))
  sha=$("$python" -c "
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:8])" "$onnx")
  previous_label="$label"
  previous_generator="$generator"
  label="gen-$(printf '%06d' "$generation")-$run_tag-$sha"
  generator=$(start_generator "$label" "$onnx" "$next_game") || { label="$previous_label"; exit 1; }
  touch "$games/$previous_label.stop"
  record_event generator_draining "label=$previous_label" "pid=$previous_generator"
  model="$onnx"
  [ -z "$anchor_model" ] && [ -z "$first_rated" ] && first_rated="$onnx"
  baseline=$(count_samples)
  echo "[loop] update $update done: generation $generation started (pid $generator), $previous_label draining"

  # Measurement, not a gate. Adoption already happened above.
  #
  # Keyed on the absolute update number, not on rounds since this process
  # started. `first_update` moves with every restart, so a rounds-based cadence
  # resets to zero each time -- across five restarts it fired exactly once, and
  # every measurement in between had to be run by hand.
  if [ -n "$first_rated" ] && [ $(( update % anchor_every )) -eq 0 ]; then
    # A pool, not a fixed opponent. One unchanging anchor stops measuring
    # anything the moment it is outclassed: sl-w64b16 lost 48-0 twice running,
    # which bounds the candidate from below and says nothing else, for 70
    # minutes of GPU a time.
    #
    # Instead sample `anchor_samples` earlier models and play a short match
    # against each. Any single match is noise at this size; what accumulates is
    # a connected graph of pairwise results, and `evaluation/ratings.py` fits every
    # model's rating over the whole history at once. Sampling keeps the graph
    # connected without anyone having to choose a ladder, and re-fitting from
    # scratch means old matches keep informing new ratings.
    # Recent models plus the fixed references, deduplicated.
    #
    # Recent, because a match against something far weaker is a sweep, and a
    # sweep carries almost no information -- the rating prior already assumes a
    # large gap. Neighbours are where the games actually discriminate.
    #
    # The fixed references stay in so the graph keeps a tie to the original
    # scale; without one, ratings drift as a connected component with nothing
    # holding the zero. Deduplicated because `seed_model` is usually also the
    # newest entry in `models/`, and drawing it twice would spend a third of the
    # match on a repeat.
    # A slot is reserved for a fixed reference, explained where it's drawn below.
    # Defined before the pool because the pool filters it out: an empty
    # `$reference` would make `grep -vF ""` match every line and leave nothing.
    if [ -n "$anchor_model" ]; then
      reference="$anchor_model"
    else
      # The model `anchor_lag` updates back, or the run's first model while the
      # run is younger than that. Either way it already has a path to the rest.
      lagged="$models/update-$(printf '%06d' $(( update > anchor_lag ? update - anchor_lag : 0 ))).onnx"
      reference=$([ -f "$lagged" ] && echo "$lagged" || echo "$first_rated")
    fi

    mapfile -t pool < <(
      {
        ls -1 "$models"/update-*.onnx 2>/dev/null | tail -n "$anchor_pool"
        [ -n "$anchor_model" ] && echo "$anchor_model"
      } | grep -v "update-$(printf '%06d' "$update").onnx" \
        | grep -vF "$reference" | awk '!seen[$0]++' \
        | { if [ "$update" -gt "$anchor_lag" ]; then grep -v "/update-000000.onnx$"; else cat; fi; }
    )
    if [ "${#pool[@]}" -gt 0 ]; then
      # Seeded in Python rather than with `shuf --random-source`. A constant
      # stream like `yes $update` has almost no entropy, so shuf returned the
      # same permutation for every update -- three consecutive rating matches
      # would have drawn the identical opponents and the graph would never
      # connect. Seeding a PRNG on the update number is deterministic per update
      # (so a rerun repeats it) and actually varies between them.
      # One fixed reference every round, not left to the draw.
      #
      # Sampling purely from recent models makes them play each other and never
      # the references, which splits the rating graph: the recent cluster has no
      # edge to the anchor, so Bradley-Terry fits it against its own prior and
      # reports numbers on a different scale from the anchored ones, with nothing
      # marking them apart. That happened on the first real round -- update 39
      # came out at +28 while update 33 sat at +676, which reads as a collapse
      # and was an artifact.
      #
      # Reserving one slot for a reference keeps every round connected to the
      # scale by construction, at the cost of one lopsided match in three.
      mapfile -t chosen < <(
        "$python" -c "
import random, sys
pool = [line for line in sys.stdin.read().splitlines() if line]
count = min(int(sys.argv[1]), len(pool))
print('\n'.join(random.Random(int(sys.argv[2])).sample(pool, count)))
" "$((anchor_samples - 1))" "$update" <<< "$(printf '%s\n' "${pool[@]}")"
      )
      [ -n "$reference" ] && chosen=("$reference" "${chosen[@]}")
      opponent_flags=()
      for opponent in "${chosen[@]}"; do opponent_flags+=(--opponent "$opponent"); done
      echo "[loop] update $update: rating match against ${#chosen[@]} sampled models"
      printf '[loop]   %s\n' "${chosen[@]##*/}"
      "$root/target/release/vgo-arena" \
        --candidate "$onnx" "${opponent_flags[@]}" \
        --candidate-raster-kind compact-radius \
        --radius "$anchor_radius" --komi "$anchor_komi" \
        --policy-resolution 128 --resolution "$resolution" \
        --simulations "$anchor_simulations" --pairs "$anchor_pairs" \
        --max-plies "$anchor_max_plies" --threads 8 --maximum-batch 32 \
        --random-stop-after "$random_stop_after" --random-stop-probability "$random_stop_probability" \
        --coarse-pool 16 --widening-coefficient 4.0 --maximum-candidates 321 \
        --seed $((500000 + update)) \
        >> "$output/anchor.jsonl" 2>>"$output/anchor.log"
      # The arena intermittently aborts in teardown after writing its JSON, at 8
      # threads and at 16, so a non-zero exit here is not necessarily a bad
      # measurement. It is recorded rather than acted on.
      status=$?
      [ "$status" -ne 0 ] && echo "[loop] rating arena exited $status (results may still be valid)" >&2
    fi
  fi
done

echo "[loop] finished $updates updates"
