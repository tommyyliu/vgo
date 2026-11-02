"""Dashboard data: speed, signal per compute, strength, lemmas, net shapes and what's next."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

from . import ratings as rt

ARTIFACT_ROOTS = ("artifacts",)  # Walked recursively, so this includes artifacts/old.
# Large trees that hold games and models, not results.
SKIP_DIRS = {"games", "replay", "seed-replay", "onnx-cache", "telemetry", "models", "updates"}
# The current loop, or the archived one until a fresh run exists.
LOOPS = ("artifacts/vgo-bulk", "artifacts/old/main/vgo-continuous")
LOOP = LOOPS[1]
ANCHOR = "sl-w64b16"
MIN_LEDGER_GAMES = 40
SHAPE_SCREEN = "performance/shape-screen.json"
ELO_PER_DOUBLING = 61  # Closed ideas in docs/FINDINGS.md.
VCPU_PER_10K = 5  # "Sizing a machine" in docs/RUNNING.md.
SHORTLIST = {"blocks8", "c_d222_short_n1"}
TELEMETRY_RUNS = {
    "Fixed budget, 1000 sims, 128 input": "artifacts/old/main/experiments-2026-10-14/budget-games/fixed",
    "Randomized budget, 128 input": "artifacts/old/main/experiments-2026-10-14/budget-games/randomized",
}
# Only what the files show. Where a folder doesn't say which model is which, the
# name stays neutral.
EXPERIMENT_NAMES = {
    "control-matched-recipe": "Matched-recipe control vs supervised baseline",
    "depth-ab/h2h": "depth-ab: vs small net, equal sims",
    "depth-ab/rung-deep": "depth-ab: deep net vs small, sims adjusted",
    "depth-ab/rung-wide": "depth-ab: wide net vs small, sims adjusted",
    "depth-ab/time": "depth-ab: vs small net at 2x sims",
    "naive-checks": "Loop model vs naive baseline",
    "r128-s2000": "r128-s2000 run vs r128-w200k",
    "res-ab": "128 vs 256 input",
    "search-scaling": "Simulation budget vs 800 sims",
    "window-ab": "Training window size",
    "deep-vs-komi": "Deep model vs komi model",
}


def records(path: Path) -> list[dict]:
    """Every JSON object in a file, whether pretty-printed or one per line."""
    decoder, text, position, found = json.JSONDecoder(), path.read_text(errors="replace"), 0, []
    while (start := text.find("{", position)) >= 0:
        try:
            value, position = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, dict):
            found.append(value)
    return found


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def model_name(path: str | None) -> str | None:
    return Path(path).name.removesuffix(".onnx") if path else None


def update_number(name: str) -> int | None:
    match = re.fullmatch(r"update-(\d+)", name)
    return int(match.group(1)) if match else None


def percent_change(before: float, after: float) -> float:
    return (after / before - 1.0) * 100.0


def arena_records(root: Path) -> list[tuple[str, dict]]:
    found = []
    for base in ARTIFACT_ROOTS:
        for current, dirs, files in os.walk(root / base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(("gen-", "shard-"))]
            for name in files:
                if not name.endswith((".json", ".jsonl")) or name.endswith((".pt.json", ".onnx.json")):
                    continue
                path = Path(current) / name
                if path.stat().st_size > 3_000_000:
                    continue
                rel = str(path.relative_to(root))
                found += [(rel, r) for r in records(path) if r.get("schema") == "vgo.arena.v1"]
    return found


def loop_matches(root: Path) -> list[dict]:
    path = root / LOOP / "anchor.jsonl"
    if not path.exists():
        return []
    matches = []
    for r in records(path):
        a, b = model_name(r.get("candidate_model")), model_name(r.get("opponent_model"))
        if a and b:
            matches.append({"a": a, "b": b, "a_wins": r["candidate_wins"],
                            "b_wins": r["candidate_losses"], "draws": r.get("draws", 0)})
    return matches


def rating_table(matches: list[dict]) -> list[dict]:
    if not matches:
        return []
    fitted = rt.fit(matches, ANCHOR)
    errors = rt.standard_errors(matches, fitted, ANCHOR)
    joined = rt.connected(matches, ANCHOR)
    table = []
    for name, rating in fitted.items():
        games = sum(m["a_wins"] + m["b_wins"] + m.get("draws", 0)
                    for m in matches if name in (m["a"], m["b"]))
        table.append({
            "model": name, "update": update_number(name), "rating": rating,
            "se": errors.get(name, 0.0), "games": games, "connected": name in joined,
            "bound": name != ANCHOR and rt.only_a_bound(matches, name),
        })
    return table


def loop_updates(root: Path) -> list[dict]:
    """One row per update, with the sample count that triggered it."""
    thresholds: dict[int, int] = {}
    log = root / LOOP / "run.log"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if m := re.search(r"update (\d+): waiting for (\d+) samples", line):
                thresholds[int(m[1])] = int(m[2])
    rows = []
    for path in sorted((root / LOOP / "models").glob("update-*.pt.json")):
        report = read_json(path)
        if not report:
            continue
        number = update_number(path.name.removesuffix(".pt.json"))
        validation = report.get("final_validation") or {}
        rows.append({
            "update": number, "cumulative": thresholds.get(number),
            "finished": path.stat().st_mtime, "window": report.get("samples"),
            "parameters": report.get("parameters"), "epochs": report.get("epochs"),
            "wall": report.get("wall_seconds"), "optimization": report.get("optimization_seconds"),
            "policy_kl": validation.get("policy_kl"), "value_mae": validation.get("value_mae"),
            "target_entropy": validation.get("policy_target_entropy"),
        })
    return rows


def samples_per_hour(updates: list[dict]) -> list[dict]:
    rates = []
    for a, b in zip(updates, updates[1:]):
        hours = (b["finished"] - a["finished"]) / 3600
        if b["update"] == a["update"] + 1 and a["cumulative"] and b["cumulative"] and hours > 0:
            rates.append({"update": b["update"], "rate": (b["cumulative"] - a["cumulative"]) / hours})
    return rates


def telemetry(root: Path) -> list[dict]:
    rows = []
    for label, rel in TELEMETRY_RUNS.items():
        path = root / rel / "performance.jsonl"
        if not path.exists():
            continue
        final = [r for r in records(path) if r.get("kind") == "final"]
        if not final:
            continue
        run, totals = final[-1], final[-1].get("totals") or {}
        samples = totals.get("published_samples") or 0
        plies = totals.get("searched_plies") or 0
        games = totals.get("published_games") or 0
        seconds = run.get("elapsed_seconds") or 0
        rows.append({
            "label": label, "seconds": seconds, "games": games, "samples": samples,
            "games_per_hour": totals.get("published_games_per_hour"),
            "samples_per_hour": totals.get("published_samples_per_hour"),
            "sims_per_ply": totals.get("simulations", 0) / plies if plies else None,
            "samples_per_game": samples / games if games else None,
            "capped": totals.get("capped_games", 0) / games if games else None,
            "discarded": totals.get("discarded_games"),
        })
    return rows


def arena_sims(record: dict) -> int | None:
    return record.get("simulations_per_move") or record.get("simulations")


def ledger(arenas: list[tuple[str, dict]]) -> list[dict]:
    """Group repeated arenas. Repeats aren't pooled: they may share models and seeds."""
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for rel, record in arenas:
        if record.get("games", 0) < MIN_LEDGER_GAMES or rel.startswith(LOOP):
            continue
        folder = Path(rel).parent.name
        stem = re.sub(r"-\d+$", "", Path(rel).stem)
        key = folder
        if folder == "depth-ab":
            key = f"depth-ab/{stem}"
        sims = arena_sims(record)
        opponent = record.get("opponent_simulations_per_move") or sims
        board = next(iter(re.findall(r"r(18|38)", stem)), "")
        groups[f"{key}|{board}|{sims}|{opponent}|{model_name(record.get('opponent_model'))}"].append((rel, record))
    rows = []
    for key, members in groups.items():
        name = key.split("|")[0]
        board = key.split("|")[1]
        scores = sorted(members, key=lambda m: m[1]["candidate_score"])
        rel, typical = scores[len(scores) // 2]
        wins, losses = typical["candidate_wins"], typical["candidate_losses"]
        games = wins + losses + typical.get("draws", 0)
        low, high = rt.wilson(wins + 0.5 * typical.get("draws", 0), games)
        sims = arena_sims(typical)
        opponent = typical.get("opponent_simulations_per_move") or sims
        candidate = model_name(typical.get("candidate_model"))
        rival = model_name(typical.get("opponent_model"))
        versus = f"{candidate} vs {rival}" if candidate and rival and candidate != rival else ""
        rows.append({
            "experiment": EXPERIMENT_NAMES.get(name, name) + (f", 1/{board} board" if board else ""),
            "versus": versus,
            "sims": "" if sims is None else f"{sims}" if sims == opponent else f"{sims} vs {opponent}",
            "sims_order": sims or 0,
            "record": f"{wins}-{losses}", "low": low, "high": high,
            "verdict": "Candidate better" if low > 0.5 else "Candidate worse" if high < 0.5 else "Inconclusive",
            "runs": len(members), "source": rel,
        })
    rows.sort(key=lambda r: (r["experiment"], r["sims_order"]))
    return rows


def speed_section(root: Path, updates: list[dict], arenas: list[tuple[str, dict]]) -> dict:
    blocks = []
    stats = []
    neural = read_json(root / "artifacts/old/main/experiments-2026-10-14/neural-summary.json") or {}
    after = neural.get("neural-p1-after") or {}
    if big := after.get("r38.neural_search_32x1000"):
        stats.append({"label": "Neural search, big board", "value": f"{big['searches_s']:.1f}/s",
                      "sub": "1,000-sim searches, 256 input"})
    if small := after.get("r18.neural_search_32x1000"):
        stats.append({"label": "Neural search, small board", "value": f"{small['searches_s']:.1f}/s",
                      "sub": "1,000-sim searches, 256 input"})
    timed = [u for u in updates if u["wall"] and u["window"]]
    if timed:
        last = timed[-1]
        stats.append({"label": "One model update", "value": f"{last['wall'] / 60:.0f} min",
                      "sub": f"{last['window']:,} samples × {last['epochs']} epochs"})
    rated = [r for _, r in arenas if r.get("wall_seconds") and r.get("model_evaluations")]
    if rated:
        rate = statistics.median(r["model_evaluations"] / r["wall_seconds"] for r in rated)
        stats.append({"label": "Arena evaluations", "value": f"{rate:,.0f}/s",
                      "sub": f"median of {len(rated)} arenas"})
    blocks.append({"type": "stats", "items": stats})

    history = [r for r in records(root / "performance/history.jsonl")
               if r.get("schema") == "vgo.bench.v1"] if (root / "performance/history.jsonl").exists() else []
    if len(history) >= 2:
        first, last = history[0]["results"], history[-1]["results"]
        rows = []
        for name in sorted(set(first) & set(last)):
            before, now = first[name]["us"], last[name]["us"]
            rows.append({"path": name, "before": before, "now": now,
                         "change": percent_change(before, now)})
        rows.sort(key=lambda r: -r["now"])
        blocks.append({
            "type": "table", "title": "Geometry and raster hot paths",
            "caption": f"First and latest of {len(history)} recorded runs, "
                       f"last on {history[-1]['timestamp'][:10]}, same machine. Lower is better.",
            "columns": [
                {"key": "path", "label": "Path", "format": "code"},
                {"key": "before", "label": "First (µs)", "format": "number", "align": "right"},
                {"key": "now", "label": "Latest (µs)", "format": "number", "align": "right"},
                {"key": "change", "label": "Change", "format": "percent", "align": "right"},
            ], "rows": rows[:14],
        })
    points = [{"x": u["update"], "y": u["wall"] / u["window"] * 1000 / u["epochs"]}
              for u in timed if u["epochs"]]
    if points:
        blocks.append({
            "type": "line", "title": "Training cost",
            "caption": "Seconds per 1,000 samples per epoch, for each loop update. Lower is better.",
            "xLabel": "Update", "yLabel": "s / 1k samples / epoch",
            "series": [{"name": "Training cost", "points": points}],
        })
    return {"id": "speed", "title": "Speed", "question": "How fast is everything?", "blocks": blocks}


def signal_section(root: Path, updates: list[dict], runs: list[dict]) -> dict:
    rates = samples_per_hour(updates)
    median_rate = statistics.median(r["rate"] for r in rates) if rates else None
    fixed = runs[0] if runs else None
    total = max((u["cumulative"] or 0 for u in updates), default=0)
    stats = []
    if median_rate:
        stats.append({"label": "Loop generation", "value": f"{median_rate / 1000:.1f}k/h",
                      "sub": "samples, median across updates"})
    if total:
        stats.append({"label": "Samples generated", "value": f"{total / 1e6:.2f}M",
                      "sub": "by the continuous loop"})
    if fixed and fixed["sims_per_ply"]:
        stats.append({"label": "Compute per move", "value": f"{fixed['sims_per_ply']:,.0f}",
                      "sub": "simulations per searched move"})
    if fixed and fixed["samples_per_game"]:
        stats.append({"label": "Samples per game", "value": f"{fixed['samples_per_game']:.0f}",
                      "sub": "each game gives one value outcome"})
    blocks: list[dict] = [{"type": "stats", "items": stats}]
    blocks.append({
        "type": "note", "tone": "warning",
        "text": "Nothing here measures information directly. Samples per hour counts positions, "
                "and a game's positions all share one result, so independent games per hour is the "
                "closer measure of value signal. The closest measure available is strength gained "
                "per sample, in the next section. A direct measure is the first idea listed under "
                "What to try next.",
    })
    if runs:
        blocks.append({
            "type": "table", "title": "Generation recipes",
            "caption": "10-minute telemetry runs that stopped mid-game, so these undercount long "
                       "games. Compare them with each other, not with the loop's rate.",
            "columns": [
                {"key": "label", "label": "Recipe"},
                {"key": "games_per_hour", "label": "Games/h", "format": "number", "align": "right"},
                {"key": "samples_per_hour", "label": "Samples/h", "format": "number", "align": "right"},
                {"key": "sims_per_ply", "label": "Sims/move", "format": "number", "align": "right"},
                {"key": "capped", "label": "Hit ply cap", "format": "share", "align": "right"},
                {"key": "discarded", "label": "Games discarded", "format": "number", "align": "right"},
                {"key": "seconds", "label": "Run (s)", "format": "number", "align": "right"},
            ], "rows": runs,
        })
    if rates:
        blocks.append({
            "type": "line", "title": "Loop generation rate",
            "caption": "Samples per hour between updates. Dips are mostly pauses, not slow generation.",
            "xLabel": "Update", "yLabel": "samples / hour",
            "series": [{"name": "Samples per hour", "points": [{"x": r["update"], "y": r["rate"]} for r in rates]}],
        })
    return {"id": "signal", "title": "Signal per compute",
            "question": "How much information are we generating for the amount of compute?",
            "blocks": blocks}


def model_section(table: list[dict], updates: list[dict], rows: list[dict]) -> dict:
    cumulative = {u["update"]: u["cumulative"] for u in updates}
    curve = sorted(
        (r for r in table if r["update"] is not None and r["connected"] and cumulative.get(r["update"])),
        key=lambda r: r["update"],
    )
    placed = [r for r in curve if not r["bound"]]
    stats = []
    if curve:
        latest = curve[-1]
        stats.append({"label": "Latest rated model", "value": f"update {latest['update']}",
                      "sub": f"{latest['rating']:+.0f} ± {latest['se']:.0f} Elo vs supervised baseline"})
    if placed:
        best = max(placed, key=lambda r: r["rating"] - 2 * r["se"])
        stats.append({"label": "Best-supported model", "value": f"update {best['update']}",
                      "sub": f"{best['rating']:+.0f} ± {best['se']:.0f}, {best['games']} games"})
    bounded = [r for r in curve if r["bound"]]
    if bounded:
        stats.append({"label": "Only a lower bound", "value": str(len(bounded)),
                      "sub": "models that never lost, so they can't be placed"})
    if updates:
        last = updates[-1]
        stats.append({"label": "Model size", "value": f"{last['parameters'] / 1e6:.1f}M",
                      "sub": f"{last['window']:,}-sample window"})
    blocks: list[dict] = [{"type": "stats", "items": stats}]
    if bounded:
        names = ", ".join(str(r["update"]) for r in bounded)
        blocks.append({
            "type": "note", "tone": "warning",
            "text": f"Updates {names} swept every game they played, so their ratings are lower "
                    "bounds. They haven't played the later models, so the apparent drop after them "
                    "may not be real. Re-rating them against update 45 would settle it.",
        })
    if placed:
        series = [{
            "name": "Rating, ±2 SE",
            "points": [{"x": cumulative[r["update"]], "y": r["rating"],
                        "low": r["rating"] - 2 * r["se"], "high": r["rating"] + 2 * r["se"],
                        "label": f"update {r['update']}, {r['games']} games"} for r in placed],
        }]
        blocks.append({
            "type": "line", "title": "Strength against samples generated",
            "caption": "Elo against the supervised baseline. Excludes unplaced models.",
            "xLabel": "Samples generated", "yLabel": "Elo", "xFormat": "millions", "series": series,
        })
    if rows:
        blocks.append({
            "type": "table", "title": "Arena results",
            "caption": f"Every arena of {MIN_LEDGER_GAMES}+ games outside the loop. Where an "
                       "experiment was repeated, this shows the median run: repeats may share "
                       "models, so they aren't pooled. 95% Wilson intervals.",
            "columns": [
                {"key": "experiment", "label": "Experiment"},
                {"key": "versus", "label": "Models", "format": "muted"},
                {"key": "sims", "label": "Sims", "align": "right"},
                {"key": "record", "label": "Record", "align": "right"},
                {"key": "low", "label": "Score, 95%", "format": "interval", "align": "right"},
                {"key": "verdict", "label": "Verdict", "format": "verdict"},
                {"key": "runs", "label": "Runs", "align": "right"},
            ], "rows": rows,
        })
    return {"id": "model", "title": "Model strength",
            "question": "How good a model can we get from the available data?", "blocks": blocks}


def lemma_status(root: Path) -> list[dict]:
    """Each lemma's status, read from the spec."""
    path = root / "spec/LEMMAS.md"
    text = path.read_text() if path.exists() else ""
    rows = []
    headings = list(re.finditer(r"^#{2,3} ([AENO]\d+)\. (.+)$", text, re.M))
    for heading, following in zip(headings, [*headings[1:], None]):
        body = text[heading.end():following.start() if following else len(text)]
        stated = re.search(r"^\*Status: ([^.*]+)", body, re.M)
        status = stated.group(1).strip() if stated else "Proved"
        rows.append({"id": heading.group(1), "lemma": heading.group(2),
                     "status": status.split(" ")[0].rstrip(",:"), "detail": status})
    return rows


def spec_section(root: Path) -> dict:
    rows = lemma_status(root)
    counts = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    blocks = [{"type": "stats", "items": [
        {"label": name, "value": str(counts[name])} for name in ("Proved", "Conditional", "Open")]}]
    blocks.append({"type": "table", "title": "Lemmas",
                   "columns": [{"key": "id", "label": "ID"}, {"key": "lemma", "label": "Lemma"},
                               {"key": "detail", "label": "Status"}],
                   "rows": rows})
    return {"id": "spec", "tab": "research", "title": "Proof status",
            "question": "What do we know, and what is still open?", "blocks": blocks}


def shape_speed(shape: dict, screen: dict) -> dict:
    """What a shape's speed buys, from the generator if it ran there.

    The card ceiling otherwise, which is an upper bound.
    """
    if shape["generator"]:
        rate, baseline = shape["generator"], screen["generator_baseline"]
    else:
        baseline = screen["card_baseline"]
        rate = baseline * (1 + shape["card_gain"] / 100)
    ratio = rate / baseline
    saved = 1 - 1 / ratio
    return {
        "generator_gain": percent_change(screen["generator_baseline"], shape["generator"])
        if shape["generator"] else None,
        "time": 1 / ratio, "elo": ELO_PER_DOUBLING * math.log2(ratio),
        "vcpu": rate / 10_000 * VCPU_PER_10K,
        "worth": "Yes" if saved >= 0.25 else "Some" if saved >= 0.10 else "No",
    }


def shape_deltas(shapes: list[dict]) -> tuple[dict[str, float], float]:
    """Each shape's policy KL against the mean of the controls, and the controls' spread."""
    controls = [s["final"]["policy_kl"] for s in shapes if s["group"] == "Control"]
    mean = statistics.mean(controls)
    return {s["name"]: s["final"]["policy_kl"] - mean for s in shapes}, statistics.stdev(controls)


def shape_sections(root: Path) -> list[dict]:
    screen = read_json(root / SHAPE_SCREEN)
    if not screen:
        return []
    shapes = screen["shapes"]
    named = {s["name"]: s for s in shapes}
    delta, spread = shape_deltas(shapes)
    window = screen["window"]
    limit = {s["name"]: s["points"] for s in screen["cpu_limit"]}
    fast, slow = limit["Fast net, 8 threads of 4 cores"][-1], limit["Current net, 8 threads of 4 cores"][-1]
    fastest = max((s for s in shapes if s["generator"]), key=lambda s: s["generator"])
    early, swift = named["c_d222_short_n1"], named["c_w32_ctx96_bt_a0_n1"]
    early_speed, swift_speed = shape_speed(early, screen), shape_speed(swift, screen)
    tested = [s for s in shapes if s["group"] not in ("Control", "Closed")]

    which = {"id": "shapes-which", "title": "Which net when",
             "question": "Which shape fits each stage of a run?", "blocks": [
        {"type": "stats", "items": [
            {"label": "Current net", "value": f"{screen['generator_baseline'] / 1000:.1f}k/s",
             "sub": "generator, 32 threads, early game"},
            {"label": "Fastest measured", "value": f"{fastest['generator'] / 1000:.1f}k/s",
             "sub": fastest["name"]},
            {"label": "Fast net on 8 vCPU", "value": f"{percent_change(slow['y'], fast['y']):+.0f}%",
             "sub": f"after {fast['x']} s, and still falling"},
            {"label": "Policy KL noise", "value": f"{spread:.3f}", "sub": "spread of three control seeds"},
        ]},
        {"type": "table", "title": "A plan for a run",
         "caption": "Any net's games train any shape, as this screen did, so switching costs one retrain "
                    "from the window, about 8 minutes per 150k samples. Weights do not carry between shapes.",
         "columns": [{"key": "stage", "label": "Stage"}, {"key": "cares", "label": "What matters"},
                     {"key": "use", "label": "Use", "format": "code"}, {"key": "why", "label": "Why"}],
         "rows": [
            {"stage": "Early", "cares": "Games per hour", "use": early["name"],
             "why": f"{early_speed['generator_gain']:+.0f}% in the generator, so the same games take "
                    f"{1 - early_speed['time']:.0%} less time, at {delta[early['name']]:+.3f} KL. "
                    f"{swift['name']} is only {percent_change(early['generator'], swift['generator']):.0f}% faster "
                    f"again for {delta[swift['name']] / delta[early['name']]:.0f} times the KL cost."},
            {"stage": "Middle", "cares": "Both", "use": "blocks8",
             "why": f"{named['blocks8']['card_gain']:+d}% on the card with no policy penalty, and it needs no new "
                    "code. Not yet run in the generator."},
            {"stage": "Late", "cares": "Modeling ability", "use": "control",
             "why": "Eight epochs can't show capacity, so nothing here argues for leaving the current net. "
                    "Bigger nets are closed at this compute, so a late move to one needs new evidence."},
         ]},
        {"type": "note", "text": "These are loss curves from one 8-epoch run each, not strength. A shape that "
                                 "looks free here can still fall behind in a long run, and adoption needs a "
                                 "matched-compute arena."},
    ]}

    what = {"id": "shapes-what", "title": "What each shape is",
            "question": "What does each experiment change?", "blocks": [
        {"type": "table", "title": "Parts of the net",
         "caption": "Input is 128x128, reduced to a 32x32 grid before these parts. Each residual stage is "
                    "about 15% of the arithmetic.",
         "columns": [{"key": "part", "label": "Part"}, {"key": "job", "label": "Job"},
                     {"key": "now", "label": "In the current net"}],
         "rows": [
            {"part": "Detail branch", "job": "Keeps fine placement, at 32x32",
             "now": "Three stages of 4 blocks, written 4/4/4, 64 channels"},
            {"part": "Context branch", "job": "Sees the whole board, at 16x16 and 8x8",
             "now": "Two stages of 4 blocks, written 4/4, 128 and 256 channels"},
            {"part": "Attention", "job": "Lets far parts of the board interact directly",
             "now": "The last block of each context stage"},
            {"part": "Norms", "job": "Keep activations stable. About 22% of inference time",
             "now": "Two GroupNorms in every residual block"},
            {"part": "Tail", "job": "Merges the detail branch before the policy head",
             "now": "One 128-channel residual block. A bottleneck narrows it 128-64-128"},
            {"part": "Policy upsampling", "job": "Turns the 32x32 map into 128x128 policy",
             "now": "One output per cell, resized bilinearly"},
         ]},
        {"type": "table", "title": "The shapes",
         "caption": "Names: c_ a stack, d222 detail blocks, short context 3/3, c22 context 2/2, w32 detail width, "
                    "ctx96 context width, bt bottleneck tail, a0 no attention, n1 one norm, ps PixelShuffle. "
                    "Parameters are the exported net's. Arithmetic is relative to the current net.",
         "columns": [{"key": "name", "label": "Shape", "format": "code"}, {"key": "group", "label": "Kind", "format": "nowrap"},
                     {"key": "change", "label": "What changes"},
                     {"key": "parameters", "label": "Parameters", "format": "compact", "align": "right"},
                     {"key": "arithmetic", "label": "Arithmetic", "format": "share", "align": "right"}],
         "rows": [{k: s[k] for k in ("name", "group", "change", "parameters", "arithmetic")} for s in shapes
                  if s["name"] not in ("control_seed2", "control_seed3")]},
    ]}

    speed_rows = sorted(({**shape_speed(s, screen), "name": s["name"], "card_gain": s["card_gain"]} for s in tested),
                        key=lambda r: r["time"])
    speed = {"id": "shapes-speed", "title": "What is meaningfully faster",
             "question": "Which shapes save real time, and on what machine?", "blocks": [
        {"type": "table", "title": "Speed against the current net",
         "caption": f"Card: the ceiling at 128, batch 32, two lanes, against {screen['card_baseline'] / 1000:.1f}k "
                    f"positions/s. Generator: 75 s early in games on all 32 threads with three lanes, against "
                    f"{screen['generator_baseline'] / 1000:.1f}k, using random weights for the stacks. A dash means "
                    "not run. The last three columns use the generator where it ran and the card otherwise, which "
                    f"is an upper bound. Elo is {ELO_PER_DOUBLING} a doubling, if the shape keeps its strength. "
                    f"vCPU is about {VCPU_PER_10K} per 10k positions/s. Meaningful means at least a quarter of "
                    "the time saved.",
         "columns": [{"key": "name", "label": "Shape", "format": "code"},
                     {"key": "card_gain", "label": "Card", "format": "gain", "align": "right"},
                     {"key": "generator_gain", "label": "Generator", "format": "gain", "align": "right"},
                     {"key": "time", "label": "Same games take", "format": "share", "align": "right"},
                     {"key": "elo", "label": "Elo at equal time", "format": "elo", "align": "right"},
                     {"key": "vcpu", "label": "vCPU per card", "format": "whole", "align": "right"},
                     {"key": "worth", "label": "Meaningful", "align": "right"}],
         "rows": speed_rows},
        {"type": "note", "text": "A faster net only pays where the CPU can feed it. On 8 vCPU the current net is "
                                 "already close to the CPU's limit, so these gains need about 15 to 20 vCPU per card."},
        {"type": "line", "title": "Generator rate as CPU runs short",
         "caption": "c_d222_bt_a0_n1, 63% faster on the card, early in games with two lanes. With fewer CPUs it "
                    "starts fast and falls toward the current net as boards fill. Later boards cost more CPU "
                    "than these two minutes reach.",
         "xLabel": "Seconds", "yLabel": "Positions/s", "series": screen["cpu_limit"]},
    ]}

    def kind(shape: dict) -> str:
        return "closed" if shape["group"] == "Closed" else "pick" if shape["name"] in SHORTLIST else "other"

    plotted = [s for s in shapes if s["name"] not in ("control_seed2", "control_seed3")]
    means = [statistics.mean(c) for c in zip(*(s["kl"] for s in shapes if s["group"] == "Control"))]
    epochs = range(1, len(means) + 1)
    seeds = [delta[s["name"]] for s in shapes if s["group"] == "Control"]
    totals = sorted(s["final"]["loss"] for s in shapes if s["group"] == "Control")
    early_best = sum(min(range(len(s["loss"])), key=s["loss"].__getitem__) < 3 for s in shapes)
    train_mae = statistics.mean(s["final"]["training_value_mae"] for s in shapes)
    held_mae = statistics.mean(s["final"]["value_mae"] for s in shapes)
    learning = {"id": "shapes-learning", "title": "What the speed costs",
                "question": "How well did each shape learn on held-out games?", "blocks": [
        {"type": "scatter", "title": "Speed against policy loss",
         "caption": "Lower right is best: faster, with no loss. The band is the three control seeds.",
         "xLabel": "Card speed gain, %", "yLabel": "Validation policy KL vs control",
         "band": [min(seeds), max(seeds)],
         "legend": {"pick": "Shortlisted", "other": "Tested", "closed": "PixelShuffle, closed"},
         "points": [{"x": s["card_gain"], "y": delta[s["name"]], "label": s["name"], "kind": kind(s)}
                    for s in plotted]},
        {"type": "line", "title": "Validation policy KL by epoch",
         "caption": "Lower is better. PixelShuffle starts behind and stays there, and the stack without it "
                    "tracks the control. The last epochs fall fastest because the learning rate decays.",
         "xLabel": "Epoch", "yLabel": "Policy KL", "decimals": 3, "series": [
            {"name": "Control, mean of 3 seeds", "points": [{"x": e, "y": v} for e, v in zip(epochs, means)]},
            *[{"name": name, "points": [{"x": e, "y": v} for e, v in zip(epochs, named[name]["kl"])]}
              for name in ("c_d222_short_n1", "c_w32_ctx96_bt_a0_n1", "up_ps")]]},
        {"type": "note", "text": f"Total validation loss can't rank these. It is lowest at epoch 1 to 3 in "
                                 f"{early_best} of {len(shapes)} runs, then stays flat or climbs while policy KL keeps "
                                 f"falling, because every net memorizes game outcomes: value MAE averages "
                                 f"{train_mae:.2f} on training games and {held_mae:.2f} on held-out ones. No shortlisted "
                                 f"shape is worse than the controls, which read {totals[0]:.2f} to {totals[-1]:.2f}."},
        {"type": "table", "title": "Final validation",
         "caption": f"Final epoch on {window['validation_samples']:,} held-out positions from about 70 of the "
                    f"{window['games']} games, read from the heads that inference uses. Trained from scratch on the "
                    f"newest {window['samples']:,} samples of {window['run']}. Value columns are too noisy to rank "
                    "shapes: norm1 looks worst alone, yet every stack with one norm reads as well as the controls.",
         "columns": [{"key": "name", "label": "Shape", "format": "code"},
                     {"key": "policy_kl", "label": "Policy KL", "format": "fixed", "digits": 3, "align": "right"},
                     {"key": "delta", "label": "vs control", "format": "delta", "noise": 2 * spread, "align": "right"},
                     {"key": "policy_top1", "label": "Top move", "format": "share", "align": "right"},
                     {"key": "loss", "label": "Total loss", "format": "fixed", "digits": 2, "align": "right"},
                     {"key": "value_sign_accuracy", "label": "Value sign", "format": "share", "align": "right"},
                     {"key": "value_mae", "label": "Value MAE", "format": "fixed", "digits": 2, "align": "right"}],
         "rows": sorted(({"name": s["name"], "delta": delta[s["name"]], **s["final"]} for s in shapes),
                        key=lambda r: r["policy_kl"])},
    ]}
    return [{**section, "tab": "shapes"} for section in (which, what, speed, learning)]


def collect(path: Path) -> dict:
    root = Path(path)
    global LOOP
    LOOP = next((loop for loop in LOOPS if (root / loop).exists()), LOOPS[0])
    updates = loop_updates(root)
    arenas = arena_records(root)
    table = rating_table(loop_matches(root))
    return {
        "title": "vgo",
        "summary": "AlphaZero-style self-play for Voronoi Go.",
        "sections": [
            {**speed_section(root, updates, arenas), "tab": "performance"},
            {**signal_section(root, updates, telemetry(root)), "tab": "research"},
            {**model_section(table, updates, ledger(arenas)), "tab": "research"},
            spec_section(root),
            {"id": "next", "tab": "research", "title": "What to try next",
             "question": "What's most worth testing?", "blocks": [{"type": "ideas"}]},
            *shape_sections(root),
        ],
    }
