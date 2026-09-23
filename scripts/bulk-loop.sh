#!/usr/bin/env bash
# The RL loop, retraining in bulk instead of incrementally.
#
# ## Why this exists
#
# `continuous-loop.sh` warm-starts one epoch on a moving window every 12k
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
# Progress is measured instead against a *fixed* anchor every `anchor_every`
# rounds. Games against one unchanging opponent accumulate power; pairwise
# gates restart the question every time and never converge.
#
# The window cancels out of the cost: training duty is `epochs * 0.0125 /
# turnover` regardless of how wide it is, so those two knobs alone set how much
# of the GPU goes to training rather than to the generation that feeds it. At 8
# epochs and 50% turnover that is ~20%, a retrain every ~5.3h of generation.
# Generation never stops -- the generator holds the incumbent model throughout --
# so this is contention on a GPU already at 78%, not downtime.
#
# Note the window cancels: a narrower one retrains sooner but each retrain is
# proportionally cheaper. Widen it for diversity, not to train less often.
#
# Epoch count is from prior testing: high single digits is safe here. It matters
# because the learner publishes the *final* epoch and only reports that a better
# one existed (`selection_regret`), so overfitting ships silently. The arena
# gate is what makes an aggressive count safe to run at all.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

# 150k rather than the 200k that produced the +307: that run peaked at 46 GB of
# 60 with nothing else on the box, and here a generator is resident throughout.
window="${VGO_WINDOW_SAMPLES:-150000}"
# Half the window new before retraining. Duty is `epochs * 0.0125 / turnover`
# and the window cancels, so these two knobs alone decide how much of the GPU
# goes to training: 8 epochs at 0.50 is ~20%, at 0.30 it would be 33%. A fresher
# model is worth less than it looks when each round already replaces half the
# window, and training competes with the generation that feeds it.
turnover="${VGO_TURNOVER:-0.15}"
epochs="${VGO_EPOCHS:-8}"
updates="${VGO_UPDATES:-200}"
actors="${VGO_ACTORS:-32}"
simulations="${VGO_SIMULATIONS:-1000}"
# Soft resign. `resign_threshold` at 0.0 is what disables the whole rule, so
# setting it is what turns this on.
#
# Past a soft concession the game keeps playing -- it is a discount on the tail,
# not a skip -- at `resign_soft_simulations`. That default is 2400, which is
# above the normal budget here, so leaving it alone would make conceded games
# *more* expensive than ordinary ones. 400 against 1000 is the discount.
#
# 0.99 rather than the 0.98 tried before: at 0.98 two fifths of recorded samples
# came from the cheap tail, and every ply is recorded, so all of them are
# trained on. A higher bar fires later and keeps that share down.
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
# `resign_threshold` is the fallback, used until calibration says otherwise.
#
# The real value is chosen per generation from measured false positives. Every
# game writes `resign-calibration.jsonl`: for each candidate (threshold, window),
# whether the rule would have conceded, and whether that concession would have
# been wrong. Under soft resignation the game plays on to a real terminal state,
# so every game calibrates rather than only a sampled exemption.
#
# Re-picked before each generator rather than once per run, because calibration
# describes the model that produced it and a model still learning invalidates its
# own history -- one recorded run fired on 15 of 1625 games over fifteen shards
# and 440 of 1686 over the next sixteen. This is what `pipeline.py` did per
# shard; the bulk loop dropped it when it stopped using the pipeline, not
# deliberately.
#
# Set the target to 0 to pin the threshold and switch adaptation off.
resign_threshold="${VGO_RESIGN_THRESHOLD:-0.99}"
resign_target_false_positive="${VGO_RESIGN_TARGET_FP:-0.03}"
# Games pooled when picking. Wider is steadier but reaches back to weaker models.
resign_calibration_games="${VGO_RESIGN_CALIBRATION_GAMES:-400}"
resign_soft_simulations="${VGO_RESIGN_SOFT_SIMULATIONS:-400}"
resign_window="${VGO_RESIGN_WINDOW:-5}"
resign_minimum_ply="${VGO_RESIGN_MINIMUM_PLY:-20}"
# Zero is right under soft resign: the disable fraction exists to sample games
# that ignore the rule so false positives can be counted, and a soft concession
# plays on to a real terminal state, so there is nothing to falsify.
resign_disable_fraction="${VGO_RESIGN_DISABLE_FRACTION:-0.0}"
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
# The anchor itself. Fixed for the life of the run: changing it discards every
# game measured against the old one. Defaults to the seed model.
anchor_model="${VGO_ANCHOR_MODEL:-}"
# The board the anchor is played on, and the ply cap that belongs to it.
# Generation scales its cap by (reference / radius)^2; the arena takes a flat
# number, so it has to be told. 70 is the 1/18 value -- set to 200, two thirds
# of every game was both sides shuffling in a full board, and a pass count read
# off those games said the search was broken when it was the cap.
anchor_radius="${VGO_ANCHOR_RADIUS:-0.05555555555555555}"
anchor_max_plies="${VGO_ANCHOR_MAX_PLIES:-70}"
anchor_komi="${VGO_ANCHOR_KOMI:-0.09}"
seed_model="${VGO_SEED_MODEL:-}"
[ -n "$seed_model" ] && seed_model="$(cd "$(dirname "$seed_model")" && pwd)/$(basename "$seed_model")"

python="$root/training/.venv/bin/python"
source "$root/scripts/env/ort.sh"
mkdir -p "$games" "$models"

# Samples currently on disk, from manifests rather than by loading anything.
count_samples () {
  "$python" - "$games" <<'PY'
import json, sys
from pathlib import Path
total = 0
for generation in sorted(p for p in Path(sys.argv[1]).iterdir() if p.is_dir()):
    for game in sorted(p for p in generation.iterdir() if p.is_dir()):
        manifest = game / "manifest.json"
        if not manifest.is_file() or not (game / "dataset.vgo").is_file():
            continue
        try:
            total += int(json.loads(manifest.read_text())["samples"])
        except (ValueError, KeyError, OSError):
            continue
print(total)
PY
}

# `--inference-slots 2` and `--maximum-batch 32` are the known-good pair. Do not
# raise them without watching a game actually land: at 4 slots the inference
# threads livelock, spinning at 100% CPU while every actor blocks on a result
# that never arrives. `nvidia-smi` reports 100% utilization throughout, because
# a spin-wait is indistinguishable from work by that metric -- the tell is power
# draw, which sits at idle (~52 W) instead of the 200 W+ of real load.
#
# `--leaf-batch` stays at 4 for a different reason: leaf parallelism trades
# search quality for throughput, measured at -70 Elo for batch 32 in the browser
# client, and halving the simulations already spent the quality budget.
#
# Nothing in the invocation below may be interrupted by a comment. A trailing
# backslash joins the next line, so a `#` on it comments out every remaining
# argument and the binary starts with a silently truncated command line -- no
# `--model` reads as a naive generator, which still runs and still writes games.
# Lowest threshold whose measured false-positive rate clears the target, or the
# configured fallback when nothing has enough evidence yet. Self-protecting: with
# no calibration it changes nothing, so it is safe to leave on while the value
# head is still learning and the rule would be worthless.
pick_resign_threshold () {
  local chosen
  if [ "$(echo "$resign_target_false_positive > 0" | bc -l 2>/dev/null)" != "1" ]; then
    echo "$resign_threshold"; return
  fi
  chosen=$("$python" "$root/scripts/resign-calibration.py" "$games" --pick \
    --games "$resign_calibration_games" --window "$resign_window" \
    --target "$resign_target_false_positive" --fallback "$resign_threshold" 2>/dev/null)
  [ -z "$chosen" ] && chosen="$resign_threshold"
  echo "$chosen"
}

start_generator () {
  local label="$1" model="$2" first_game="$3"
  local stop_file="$games/$label.stop"
  rm -f "$stop_file"
  local threshold
  threshold=$(pick_resign_threshold)
  if [ "$threshold" = "$resign_threshold" ]; then
    echo "[resign] $label: no threshold met $(echo "100*$resign_target_false_positive" | bc -l | cut -c1-4)% false positives; holding the $resign_threshold fallback" >&2
  else
    echo "[resign] $label: threshold $threshold chosen from calibration (target $(echo "100*$resign_target_false_positive" | bc -l | cut -c1-4)%)" >&2
  fi
  local model_flag=()
  [ -n "$model" ] && model_flag=(--model "$model")
  setsid nohup "$root/target/release/vgo-generate-continuous" \
    --output-root "$games" --label "$label" \
    --stop-file "$stop_file" --first-game "$first_game" \
    --actors "$actors" --simulations "$simulations" \
    --resolution 256 --policy-resolution 128 --raster-kind compact-radius \
    --board-mix 50:38 --board-mix 25:18 --board-mix 25:18-38 \
    --max-plies 70 --radius 0.05555555555555555 \
    --coarse-pool 16 --widening-coefficient 6.0 --maximum-candidates 321 \
    --komi-area-coefficient "$komi_area_coefficient" \
    --komi-low 0.017 --komi-high 0.137 \
    --resign-threshold "$threshold" \
    --resign-soft-simulations "$resign_soft_simulations" \
    --resign-window "$resign_window" \
    --resign-minimum-ply "$resign_minimum_ply" \
    --resign-disable-fraction "$resign_disable_fraction" \
    --temperature 1.0 --temperature-plies 30 \
    --root-exploration-noise "$root_noise" \
    --leaf-batch 4 --maximum-batch 32 --delay-ms 1 --inference-slots 2 \
    --provider tensorrt --fp16 true \
    --cache-directory "$root/artifacts/onnx-cache" \
    --seed $((70000 + first_game)) \
    "${model_flag[@]}" \
    >> "$output/generate.log" 2>&1 &
  echo $!
}


model="$seed_model"
anchor="${anchor_model:-$seed_model}"

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
generator=$(start_generator "$label" "$model" "$next_game")
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

  checkpoint="$models/update-$(printf '%06d' "$update").pt"
  echo "[loop] update $update: training from scratch on the most recent $window samples"
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
  ( cd "$root/training" && "$python" "$root/scripts/train-once.py" \
      --games-root "$games" --window-samples "$window" \
      --output "$checkpoint" \
      --raster-kind compact-radius \
      --model-width 64 --blocks 16 --context-attention-blocks 1 \
      --attention-heads 8 --norm-groups 8 --precision bfloat16 \
      --epochs "$epochs" --batch-size 64 --learning-rate 0.0005 \
      --value-weight 2.0 --ownership-weight 0.0 --validation-fraction 0.1 \
      --schedule wsd --warmup-epochs 0 --compile \
      --seed $((90000 + update)) --report-every 1 \
  ) >> "$output/train.log" 2>&1 || { echo "[loop] training failed" >&2; exit 1; }

  onnx="${checkpoint%.pt}.onnx"
  packed_flag=()
  [ "$packed_input" = "1" ] && packed_flag=(--packed-input)
  ( cd "$root/training" && "$python" -m vgo_training.export_onnx \
      --checkpoint "$checkpoint" --output "$onnx" --maximum-batch 64 \
      "${packed_flag[@]}" \
  ) >> "$output/train.log" 2>&1 || { echo "[loop] export failed" >&2; exit 1; }

  # Start the successor before stopping the incumbent, so generation never
  # pauses. `next_game` advances past anything the old process could claim.
  generation=$((generation + 1))
  next_game=$((next_game + 1000000))
  sha=$("$python" -c "
import hashlib,sys
print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:8])" "$onnx")
  previous_label="$label"
  label="gen-$(printf '%06d' "$generation")-$run_tag-$sha"
  generator=$(start_generator "$label" "$onnx" "$next_game")
  touch "$games/$previous_label.stop"
  model="$onnx"
  baseline=$(count_samples)
  echo "[loop] update $update done: generation $generation started (pid $generator), $previous_label draining"

  # Measurement, not a gate. Adoption already happened above.
  #
  # Keyed on the absolute update number, not on rounds since this process
  # started. `first_update` moves with every restart, so a rounds-based cadence
  # resets to zero each time -- across five restarts it fired exactly once, and
  # every measurement in between had to be run by hand.
  if [ -n "$anchor" ] && [ $(( update % anchor_every )) -eq 0 ]; then
    # A pool, not a fixed opponent. One unchanging anchor stops measuring
    # anything the moment it is outclassed: sl-w64b16 lost 48-0 twice running,
    # which bounds the candidate from below and says nothing else, for 70
    # minutes of GPU a time.
    #
    # Instead sample `anchor_samples` earlier models and play a short match
    # against each. Any single match is noise at this size; what accumulates is
    # a connected graph of pairwise results, and `scripts/ratings.py` fits every
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
    # One fixed reference every round, not left to the draw.
    #
    # Sampling purely from recent models makes them play each other and never the
    # references, which splits the rating graph: the recent cluster has no edge
    # to the anchor, so Bradley-Terry fits it against its own prior and reports
    # numbers on a different scale from the anchored ones, with nothing marking
    # them apart. That happened on the first real round -- update 39 came out at
    # +28 while update 33 sat at +676, which reads as a collapse and was an
    # artifact of two disconnected components.
    #
    # Reserving a slot keeps every round tied to the scale by construction, at
    # the cost of one lopsided match in three. Defined before the pool because
    # the pool filters it out: an empty `$reference` would make `grep -vF ""`
    # match every line and leave nothing to sample.
    reference="$anchor"
    [ -z "$reference" ] && reference="$seed_model"

    mapfile -t pool < <(
      {
        ls -1 "$models"/update-*.onnx 2>/dev/null | tail -n "$anchor_pool"
        [ -n "$anchor" ] && echo "$anchor"
        [ -n "$seed_model" ] && echo "$seed_model"
      } | grep -v "update-$(printf '%06d' "$update").onnx" \
        | grep -vF "$reference" | awk '!seen[$0]++'
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
        --policy-resolution 128 --resolution 256 \
        --simulations "$anchor_simulations" --pairs "$anchor_pairs" \
        --max-plies "$anchor_max_plies" --threads 8 --maximum-batch 32 \
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

touch "$games/$label.stop"
echo "[loop] finished $updates updates"
