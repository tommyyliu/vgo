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
}
_UPDATE = re.compile(r"update-(\d+)")


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


def identify(path: str) -> tuple[str, int]:
    if path == "naive":
        return "naive", -1
    parts = Path(path).parts
    match = _UPDATE.search(path)
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
                        help="repeatable; several tournaments pool into one fit")
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

    sources = arguments.records or [_ROOT / "artifacts/dense-curve/records.jsonl"]
    found = []
    for source in sources:
        if source.exists():
            found += records(source.read_text(encoding="utf-8"))
    if not found:
        raise SystemExit("no records yet")

    keys, labels = {}, {}
    matches = []
    for record in found:
        if int(record["completed"]) == 0:
            continue
        pair = []
        for side in ("candidate_model", "opponent_model"):
            run, version = identify(record[side])
            key = f"{run}#{version}"
            keys[key] = (run, version)
            labels[key] = "naive" if run == "naive" else f"{run} v{version}"
            pair.append(key)
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
    series: dict[str, list] = {run: [] for run in COLORS}
    for identifier, elo in ratings.items():
        run, version = keys[inverse[identifier]]
        if run not in series:
            continue
        played = sum(m["a_wins"] + m["b_wins"] + m["draws"] for m in kept
                     if identifier in (m["a"], m["b"]))
        series[run].append({"v": version,
                            "n": samples[run].get(version, 0),
                            "elo": elo, "se": errors.get(identifier, 0.0),
                            "games": played})
    for run in series:
        series[run].sort(key=lambda p: p["v"])

    payload = {
        "series": {run: series[run] for run in COLORS if series[run]},
        "names": {run: COLORS[run][0] for run in COLORS},
        "vars": {run: COLORS[run][1] for run in COLORS},
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
          f"{payload['games']} games, {payload['stranded']} not yet connected)")


if __name__ == "__main__":
    main()
