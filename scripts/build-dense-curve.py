#!/usr/bin/env python3
"""Render the dense-curve tournament as Elo against cumulative samples.

Reads whatever `dense-curve.py` has written so far -- it appends a record per
completed pairing, so this works mid-run -- fits one Bradley-Terry rating over
the connected part of the match graph, and plots every run on shared axes with
a logarithmic sample axis.

The x axis is cumulative self-play samples rather than update number, because
update number is not comparable across runs with different shard sizes: at
10,000 samples a shard, ten updates cost what nineteen do at 5,139.

    scripts/build-dense-curve.py --output dense.html
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "training"))
sys.path.insert(0, str(_ROOT / "scripts"))
from bt_stats import components, covariance_of, standard_errors  # noqa: E402
from vgo_training.bradley_terry import fit_ratings  # noqa: E402

# Reference categorical order. Line charts are scored on the adjacent pairlist,
# which this order is certified against in both modes; direct labels carry
# identity as well so it never rests on colour.
COLORS = {
    "ddrnet-fresh-attn": ("Adam 5.1k/shard", "--c1"),
    "ddrnet-fresh-muon": ("Muon 5.1k/shard", "--c2"),
    "shard-sweep-10000": ("Muon 10.7k/shard", "--c3"),
    "shard-sweep-5000": ("Muon 5.8k/shard", "--c4"),
    "shard-sweep-15000": ("Muon 15.7k/shard", "--c5"),
    # A new run takes the next free slot rather than being inserted next to the
    # run it continues. Colour follows the entity: shifting the sweeps down a
    # slot to make room would repaint four curves that have not changed, and
    # every earlier copy of this page would disagree with this one.
    # Kept short: this name is also the direct end-label, and the right margin
    # only fits about 19 characters before it runs off the plot.
    "ddrnet-attn-komi": ("Adam 5.1k + komi", "--c6"),
}
_UPDATE = re.compile(r"update-(\d+)")
# Same lineage source as rate-checkpoints.py: --initial-checkpoint is what
# actually determined the parent, so it is read from the launch script rather
# than from pipeline state, which only records where training resumed.
_LINEAGE = re.compile(r'--initial-checkpoint\s+"?\$?\{?root\}?/?([^"\s]+)')


def cumulative_samples(run: str) -> dict[int, int]:
    """Samples generated through each update, from the generation logs."""
    total, out = 0, {}
    for index, path in enumerate(
        sorted((_ROOT / "artifacts" / run / "logs").glob("generate-*.stdout.log"))
    ):
        text = path.read_text(encoding="utf-8", errors="replace")
        start = text.find("{")
        if start >= 0:
            try:
                total += int(json.loads(text[start:])["samples"])
            except (ValueError, KeyError):
                pass
        out[index] = total
    return out


def continuation(run: str) -> tuple[str, int] | None:
    """(parent run, parent update) for a warm-started run, from its launch.sh.

    A continuation's update 0 is not a fresh start: it resumes its parent's
    checkpoint, so on a cumulative-samples axis it belongs where the parent
    stopped, not at zero. Drawn from zero it lands on top of the parent's first
    shard and appears to have reached in one update what the parent needed
    sixty for -- and the join then reads as a collapse of several hundred Elo
    that never happened.

    Only the sample axis is shifted. The rating itself is measured, not
    inherited: if the games have not tied the child to the field, the fit
    leaves it out regardless of what its launch script claims.
    """
    launch = _ROOT / "artifacts" / run / "launch.sh"
    if not launch.exists():
        return None
    match = _LINEAGE.search(launch.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    parts = Path(match.group(1)).parts
    if "updates" not in parts:
        return None
    index = parts.index("updates")
    version = _UPDATE.search(parts[index + 1]) if index + 1 < len(parts) else None
    if version is None or index == 0:
        return None
    return parts[index - 1], int(version.group(1))


def records(text: str) -> list[dict]:
    found, depth, start = [], 0, None
    for position, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(text[start : position + 1])
                except json.JSONDecodeError:
                    continue
                if value.get("schema") == "vgo.arena.v1":
                    found.append(value)
    return found


def identify(path: str) -> tuple[str, int] | None:
    """(run, update) for a checkpoint path, or None if it is not one.

    Ad-hoc tournaments sometimes include models that live outside a run's
    updates/ tree -- the cross-training experiment wrote its four to
    artifacts/crosstrain/models/. Those cannot be placed on a run's sample axis,
    so records mentioning them are skipped rather than aborting the pool.
    """
    if path == "naive":
        return "naive", -1
    parts = Path(path).parts
    match = _UPDATE.search(path)
    if "updates" not in parts or match is None:
        return None
    return parts[parts.index("updates") - 1], int(match.group(1))


def naive_record(matches: list[dict], identifier: int | None) -> dict | None:
    """Naive's wins and losses, so the page can say how much its zero is worth.

    Anchoring on naive only buys an absolute scale if naive actually takes games
    off the field. Against trained checkpoints it does not -- even update-0
    models beat it -- and a winless record has no finite maximum-likelihood
    rating, so the zero comes from the prior rather than from evidence.
    """
    if identifier is None:
        return None
    wins = losses = 0
    for match in matches:
        if match["a"] == identifier:
            wins += match["a_wins"]; losses += match["b_wins"]
        elif match["b"] == identifier:
            wins += match["b_wins"]; losses += match["a_wins"]
    return {"wins": wins, "losses": losses, "games": wins + losses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append",
                        help="repeatable; default is every tournament under "
                             "artifacts/ at the chosen simulation count")
    parser.add_argument("--simulations", type=int, default=800,
                        help="pool only tournaments played at this search "
                             "budget; a model at 1600 simulations is a different "
                             "player from the same model at 800")
    parser.add_argument("--komi", type=float, default=0.034,
                        help="pool only tournaments played at this komi. Like "
                             "the simulation count, a different komi is a "
                             "different game: the balanced value drifts as the "
                             "models improve, and it moved from 0.034 to 0.104 "
                             "on 2026-08-16. Records predating the field are "
                             "0.034, which is the default so the existing scale "
                             "keeps building; pass --komi 0.104 for the new one")
    parser.add_argument("--ratings-json", type=Path, default=None,
                        help="also write {run/version: elo}, which dense-curve.py "
                             "reads to band its matchmaking")
    parser.add_argument("--anchor", default=None,
                        help="run/version to anchor at zero, e.g. "
                             "ddrnet-fresh-attn/9. Prefer a weak checkpoint over "
                             "naive: naive is near-winless here, so it has no "
                             "finite maximum-likelihood rating and the fit will "
                             "slide checkpoints past it that beat it head to head")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).resolve().parent / "dense-curve.template.html")
    arguments = parser.parse_args()

    sources = arguments.records or sorted(
        (_ROOT / "artifacts").glob("*/records.jsonl"))
    found, mixed, mixed_komi = [], [], []
    for source in sources:
        if not source.exists() or not source.stat().st_size:
            continue
        batch = records(source.read_text(encoding="utf-8"))
        if not batch:
            continue
        # Records written before the field existed carry no simulation count.
        # Fall back to the launcher's name only where it is unambiguous; here
        # the one legacy tournament at another budget is the joint one.
        counts = {r.get("simulations") for r in batch}
        if counts == {None}:
            if "joint-tournament" in str(source):
                mixed.append((source.name, "1600, legacy"))
                continue
        elif counts - {None, arguments.simulations}:
            mixed.append((source.parent.name, sorted(c for c in counts if c)))
            batch = [r for r in batch
                     if r.get("simulations") in (None, arguments.simulations)]
        # Same treatment for komi, and for the same reason. A record without the
        # field predates 2026-08-15 and was played at 0.034 by construction, so
        # it matches only when that is what was asked for.
        komis = {r.get("komi") for r in batch}
        wanted = {k for k in komis
                  if k is not None and abs(k - arguments.komi) < 1e-9}
        if komis - wanted - ({None} if abs(arguments.komi - 0.034) < 1e-9 else set()):
            off = sorted({k for k in komis if k is not None} - wanted)
            if None in komis and abs(arguments.komi - 0.034) >= 1e-9:
                off = ["0.034 (unrecorded)"] + [str(k) for k in off]
            mixed_komi.append((source.parent.name, off))
            batch = [
                r for r in batch
                if (r.get("komi") is not None
                    and abs(r["komi"] - arguments.komi) < 1e-9)
                or (r.get("komi") is None and abs(arguments.komi - 0.034) < 1e-9)
            ]
        found += batch
    if mixed:
        print("skipped, different search budget: "
              + ", ".join(f"{n} ({c})" for n, c in mixed))
    if mixed_komi:
        print(f"skipped, not komi {arguments.komi}: "
              + ", ".join(f"{n} ({c})" for n, c in mixed_komi))
    if not found:
        raise SystemExit("no records yet")

    keys, labels = {}, {}
    matches, skipped = [], 0
    for record in found:
        if int(record["completed"]) == 0:
            continue
        pair = []
        for side in ("candidate_model", "opponent_model"):
            placed = identify(record[side])
            if placed is None:
                break
            run, version = placed
            key = f"{run}#{version}"
            keys[key] = (run, version)
            labels[key] = "naive" if run == "naive" else f"{run} v{version}"
            pair.append(key)
        if len(pair) != 2:
            skipped += 1
            continue
        matches.append({"a": pair[0], "b": pair[1],
                        "a_wins": int(record["candidate_wins"]),
                        "b_wins": int(record["candidate_losses"]),
                        "draws": int(record["draws"])})

    # Bradley-Terry needs integer ids and a connected graph; anything the games
    # have not linked yet cannot be placed on the scale and is left out rather
    # than given a number the prior invented.
    index = {key: i for i, key in enumerate(sorted(keys))}
    numeric = [{"a": index[m["a"]], "b": index[m["b"]], "a_wins": m["a_wins"],
                "b_wins": m["b_wins"], "draws": m["draws"]} for m in matches]
    groups = components(numeric)
    main_group = groups[0] if groups else set()
    kept = [m for m in numeric if m["a"] in main_group]
    if not kept:
        raise SystemExit("no connected matches yet")

    naive_key = next((k for k in keys if k.startswith("naive#")), None)
    anchor = None
    if arguments.anchor:
        run, _, version = arguments.anchor.rpartition("/")
        wanted = f"{run}#{int(version)}"
        if wanted in index and index[wanted] in main_group:
            anchor = index[wanted]
        else:
            print(f"warning: anchor {arguments.anchor} is not in the connected "
                  f"field; falling back", file=sys.stderr)
    if anchor is None:
        anchor = index[naive_key] if naive_key and index[naive_key] in main_group \
            else min(main_group)
    ratings = fit_ratings(kept, anchor=anchor, prior_games=0.25)
    covariance, position = covariance_of(kept, ratings, anchor, 0.25)
    errors = standard_errors(covariance, position)

    inverse = {i: k for k, i in index.items()}
    samples = {run: cumulative_samples(run) for run in COLORS}
    # Shift every continuation onto its parent's axis, so the pair reads as one
    # training history rather than two runs that happen to share a chart.
    offsets: dict[str, int] = {}
    lineage: dict[str, dict] = {}
    for run in COLORS:
        parent = continuation(run)
        if parent is None:
            continue
        name, version = parent
        parent_samples = samples.get(name) or cumulative_samples(name)
        if version not in parent_samples:
            print(f"warning: {run} continues {name} v{version}, which has no "
                  f"generation log; drawing it from zero", file=sys.stderr)
            continue
        offsets[run] = parent_samples[version]
        lineage[run] = {"parent": name, "version": version,
                        "offset": parent_samples[version]}
    series: dict[str, list] = {run: [] for run in COLORS}
    for identifier, elo in ratings.items():
        run, version = keys[inverse[identifier]]
        if run not in series:
            continue
        played = sum(m["a_wins"] + m["b_wins"] + m["draws"] for m in kept
                     if identifier in (m["a"], m["b"]))
        own = samples[run].get(version)
        series[run].append({"v": version,
                            "n": 0 if own is None else own + offsets.get(run, 0),
                            "elo": elo, "se": errors.get(identifier, 0.0),
                            "games": played})
    for run in series:
        series[run].sort(key=lambda p: p["v"])

    payload = {
        "series": {run: series[run] for run in COLORS if series[run]},
        "names": {run: COLORS[run][0] for run in COLORS},
        "vars": {run: COLORS[run][1] for run in COLORS},
        "continues": lineage,
        "rated": len(ratings),
        "pool": len(keys),
        "games": sum(m["a_wins"] + m["b_wins"] + m["draws"] for m in kept),
        "pairings": len(kept),
        "stranded": sum(len(g) for g in groups[1:]),
        "anchored_on_naive": naive_key is not None and index.get(naive_key) == anchor,
        "anchor_label": labels.get(inverse[anchor], "?"),
        "naive": naive_record(kept, index.get(naive_key)) if naive_key else None,
    }
    html = arguments.template.read_text(encoding="utf-8").replace(
        "/*DATA*/null/*DATA*/", json.dumps(payload))
    arguments.output.write_text(html, encoding="utf-8")
    if arguments.ratings_json:
        arguments.ratings_json.write_text(json.dumps(
            {f"{keys[inverse[i]][0]}/{keys[inverse[i]][1]}": elo
             for i, elo in ratings.items()}, indent=2) + "\n", encoding="utf-8")
    print(f"-> {arguments.output}  ({payload['rated']} of {payload['pool']} rated, "
          f"{payload['games']} games, {payload['stranded']} not yet connected"
          + (f", {skipped} records skipped: not run checkpoints)" if skipped else ")"))


if __name__ == "__main__":
    main()
