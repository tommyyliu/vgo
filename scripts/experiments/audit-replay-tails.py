#!/usr/bin/env python3
"""Audit existing v8 replay without generating games or rendering rasters.

Run with training/.venv/bin/python. Scores are independently reconstructed by
half-plane clipping of the stored final stones; JSON coordinates are rounded.
Repetition comparisons use exact binary pre-move boards, ignoring the mover.
Sparse replay can only establish repetition between consecutive recorded plies.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from vgo_training.dataset import HEADER_V8, _v4_record_dtype


def signed_score(stones, komi):
    """Independent Voronoi area scorer, with empty board area zero per side."""
    areas = [0.0, 0.0]
    for x, y, color in stones:
        polygon = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
        for ox, oy, _ in stones:
            if (x, y) == (ox, oy):
                continue
            nx, ny = ox-x, oy-y
            limit = (ox*ox+oy*oy-x*x-y*y)/2
            clipped = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                da, db = nx*a[0]+ny*a[1]-limit, nx*b[0]+ny*b[1]-limit
                if da <= 0:
                    clipped.append(a)
                if (da <= 0) != (db <= 0):
                    t = da/(da-db)
                    clipped.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
            polygon = clipped
            if not polygon:
                break
        area = abs(sum(a[0]*b[1]-a[1]*b[0]
                       for a, b in zip(polygon, polygon[1:] + polygon[:1])))/2
        areas[int(color)] += area
    return areas[0]-areas[1]-komi, areas


def stones_of(row):
    return [(float(s['x']), float(s['y']), int(s['color']))
            for s in row['stones'][:int(row['stone_count'])]]


def audit(directory):
    totals = Counter()
    games = []
    inputs = []
    paths = sorted(directory.glob('game-*/games.jsonl'))
    if not paths:
        raise ValueError(f'no completed game records under {directory}')
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        replay = path.with_name('dataset.vgo')
        with replay.open('rb') as file:
            magic, version, samples, channels, height, width, policy, capacity, stones = HEADER_V8.unpack(file.read(HEADER_V8.size))
        if magic != b'VGORPLY1' or version != 8:
            raise ValueError(f'{replay}: requires replay v8')
        dtype = _v4_record_dtype(policy, version, capacity, stones)
        assert replay.stat().st_size == HEADER_V8.size + samples*dtype.itemsize
        rows = np.memmap(replay, dtype=dtype, offset=HEADER_V8.size, mode='r', shape=(samples,))
        inputs.append(dict(path=str(replay), sha256=hashlib.sha256(replay.read_bytes()).hexdigest(),
                           game_records_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for game in records:
            counters = Counter(games=1, rows=game['sample_count'], plies=game['plies'],
                               capped=int(game['reached_ply_cap']), passes=game['passes'],
                               self_captures=game['self_captures'])
            budgets, eligible = game['search_budgets'], game['policy_eligible']
            assert len(budgets) == len(eligible) == game['plies']
            subset = rows[int(game['first_sample']):int(game['first_sample'])+int(game['sample_count'])]
            assert all(int(row['game']) == game['game'] for row in subset)
            assert [int(row['ply']) for row in subset] == [i for i, keep in enumerate(eligible) if keep]
            # Trigger ply itself was searched before the soft concession fired.
            soft = game['soft_resign_ply']
            start = soft+1 if soft is not None else game['plies']
            counters['soft_games'] = int(soft is not None)
            counters['simulations'] = sum(budgets)
            counters['tail_simulations'] = sum(budgets[start:])
            counters['tail_plies'] = len(budgets[start:])
            counters['tail_rows'] = sum(eligible[start:])
            counters['actual_soft_winner_reversals'] = int(soft is not None and
                (game['black_utility'] == 0 or (soft % 2 == 0) == (game['black_utility'] > 0)))
            calibration = path.with_name('resign-calibration.jsonl')
            # Pilot configuration: threshold .99 over five turns per side.
            if len(records) == 1 and calibration.exists():
                for line in calibration.read_text().splitlines():
                    item = json.loads(line)
                    if item['threshold'] == .99 and item['window'] == 5:
                        counters['calibration_fired'] += item['fired']
                        counters['calibration_wrong'] += item['wrong']
            seen = set()
            previous = None
            tail_boards = set()
            tail_pass_mass = []
            for row in subset:
                ply = int(row['ply'])
                board = tuple(stones_of(row))
                tail = ply >= start
                if tail:
                    tail_boards.add(board)
                    counters['tail_repeated_board_rows'] += int(board in seen)
                    if previous is not None and previous[0] >= start and ply == previous[0]+1:
                        counters['tail_adjacent_pairs'] += 1
                        counters['tail_unchanged_board_pairs'] += int(board == previous[1])
                    cells = row['cells'][:int(row['touched'])]
                    visits = cells['visits'].astype(np.float64)
                    if visits.sum():
                        tail_pass_mass.append(float(visits[cells['index'] == policy-1].sum()/visits.sum()))
                    counters['tail_selected_passes'] += int(row['selected_action'] == policy-1)
                assert abs(float(row['value']) - game['black_utility']*(1 if row['to_move'] == 0 else -1)) < 1e-6
                seen.add(board)
                previous = ply, board
            counters['tail_distinct_boards'] = len(tail_boards)
            score, areas = signed_score(game['final_stones'], game['komi'])
            error = max(abs(areas[0]-game['black_area']), abs(areas[1]-game['white_area']),
                        abs(abs(score)-game['margin']))
            counters['score_mismatches'] = int(error > 3e-6)
            counters['winner_mismatches'] = int(abs(score) > 3e-6 and np.sign(score) != game['black_utility'])
            # This measures scoring sensitivity within the existing closing
            # record. It does not infer the outcome of hypothetical continuation.
            closing = [row for row in subset if int(row['ply']) >= game['plies']-8]
            closing_scores = [signed_score(stones_of(row), float(row['komi']))[0] for row in closing]
            flips = sum(int(np.sign(s) != np.sign(score)) for s in closing_scores if abs(s) > 3e-6)
            counters['closing_opposite_winner_rows'] = flips
            counters['closing_rows'] = len(closing)
            counters['games_with_closing_winner_changes'] = int(flips > 0)
            games.append(dict(game=game['game'], units=1/game['radius'], **dict(counters),
                              score_rounding_error=error, final_signed_margin=score,
                              closing_signed_margins=closing_scores,
                              tail_mean_pass_target_mass=float(np.mean(tail_pass_mass)) if tail_pass_mass else None))
            totals.update(counters)
    return dict(directory=str(directory), totals=dict(totals), games=games, inputs=inputs,
                limitations=['Completed pilot games only; timed cutoff and board mixes differ.',
                             'Sparse rows cannot establish unrecorded adjacent transitions.',
                             'Board repetition alone does not establish harmful training targets.',
                             'Final JSON stones/komi are rounded; score tolerance is 3e-6.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [audit(path) for path in args.directories]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2)+'\n')
    for report in reports:
        print(report['directory'], json.dumps(report['totals']), flush=True)


if __name__ == '__main__':
    main()
