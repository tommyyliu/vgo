#!/usr/bin/env python3
"""Bounded local restart experiments. Run one stage at a time, with an idle GPU.

source scripts/env/ort.sh
training/.venv/bin/python scripts/experiments/restart-2026-10-14.py STAGE

All commands, process exit codes and elapsed times go into runs.jsonl. Existing
outputs are deliberately refused: use a new output root for a repeated run.
The baseline benchmark is the same source with legal_set.rs restored to HEAD.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / 'training/.venv/bin/python'


def run(out, label, args, cwd=ROOT):
    stdout, stderr = out / f'{label}.stdout', out / f'{label}.log'
    if stdout.exists() or stderr.exists():
        raise RuntimeError(f'refusing to overwrite {label}')
    input_digests = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for argument in args[1:] if (path := Path(str(argument))).is_file()
        and path.suffix in ('.onnx', '.pt')
    }
    print(f'Starting {label}', time.strftime('%H:%M:%S'), flush=True)
    start = time.time()
    with stdout.open('w') as output, stderr.open('w') as errors:
        result = subprocess.run(list(map(str, args)), cwd=cwd, stdout=output, stderr=errors)
    record = dict(label=label, args=list(map(str, args)), cwd=str(cwd),
                  exit_code=result.returncode, started_unix=start,
                  wall_seconds=time.time()-start)
    executable = Path(args[0])
    if executable.is_file():
        record['executable_sha256'] = hashlib.sha256(executable.read_bytes()).hexdigest()
    record['input_digests'] = input_digests
    with (out / 'runs.jsonl').open('a') as records:
        records.write(json.dumps(record) + '\n')
    print(f'Finished {label}: {record["wall_seconds"]:.1f}s, exit {result.returncode}', flush=True)
    if result.returncode:
        raise RuntimeError(f'{label} failed; see {stderr}')


def benchmark(out):
    binaries = dict(before=Path('/tmp/vgo-production-baseline-target/release/vgo-bench'),
                    after=ROOT / 'target/release/vgo-bench')
    args = ['--positions', ROOT / 'benchmarks/positions.txt', '--seconds', '1',
            '--model', ROOT / 'artifacts/res-ab/r256-w150k.onnx', '--resolution', '256',
            '--actors', '32', '--simulations', '1000', '--maximum-batch', '32',
            '--inference-slots', '2', '--cache-directory', ROOT / 'artifacts/onnx-cache']
    for pair, order in enumerate([('before', 'after'), ('after', 'before')]):
        for variant in order:
            run(out, f'neural-p{pair}-{variant}', [binaries[variant], *args])


def model_benchmark(out):
    for label, model in [('resolution128', ROOT / 'artifacts/res-ab/r128-w150k.onnx'),
                         ('norm1', out / 'norm1.onnx')]:
        run(out, f'{label}-benchmark', [ROOT / 'target/release/vgo-bench',
            '--positions', ROOT / 'benchmarks/positions.txt', '--seconds', '1',
            '--model', model, '--resolution', '128', '--actors', '32',
            '--simulations', '1000', '--maximum-batch', '32', '--inference-slots', '2',
            '--cache-directory', ROOT / 'artifacts/onnx-cache'])


def freeze_corpus(out):
    """Freeze the original ordered resolution corpus, retaining numeric game IDs."""
    report = json.loads((ROOT / 'artifacts/res-ab/r128-w150k.pt.json').read_text())
    generation = out / 'frozen-model-corpus/gen-original'
    generation.mkdir(parents=True, exist_ok=True)
    for source in report['datasets']:
        source = Path(source)
        if not source.is_absolute():
            source = (ROOT / 'training' / source).resolve()
        game = source.parent
        link = generation / game.name
        if not link.exists():
            link.symlink_to(game, target_is_directory=True)
        elif link.resolve() != game.resolve():
            raise RuntimeError(f'conflicting game ID: {link}')


def verify_corpus(out):
    baseline = json.loads((ROOT / 'artifacts/res-ab/r128-w150k.pt.json').read_text())
    variant = json.loads((out / 'norm1.pt.json').read_text())
    for key in ('dataset_digests', 'samples', 'training_samples', 'validation_samples', 'shape'):
        if baseline[key] != variant[key]:
            raise RuntimeError(f'normalization comparison has mismatched {key}')
    (out / 'norm-corpus-parity.json').write_text(json.dumps({
        'ordered_digests_equal': True, 'games': len(variant['dataset_digests']),
        'samples': variant['samples'], 'training_samples': variant['training_samples'],
        'validation_samples': variant['validation_samples']}, indent=2) + '\n')


def train(out, label, games=None, norms=2, initial=None, epochs=8, datasets=None):
    selection = list(datasets) if datasets is not None else ['--games-root', games, '--window-samples', '150000']
    args = [PYTHON, ROOT / 'scripts/train-once.py', *selection, '--output', out / f'{label}.pt',
            '--raster-kind', 'compact-radius', '--resolution', '128',
            '--model-width', '64', '--blocks', '16', '--context-attention-blocks', '1',
            '--attention-heads', '8', '--norm-groups', '8', '--norms-per-block', str(norms),
            '--precision', 'bfloat16', '--epochs', str(epochs), '--batch-size', '64',
            '--learning-rate', '0.0005', '--value-weight', '2', '--ownership-weight', '0',
            '--validation-fraction', '0.1', '--schedule', 'wsd', '--warmup-epochs', '0',
            '--compile', '--seed', '424242', '--report-every', '1']
    if initial:
        args += ['--initial-checkpoint', initial, '--no-restore-optimizer']
    run(out, f'{label}-train', args, ROOT / 'training')
    run(out, f'{label}-export', [PYTHON, '-m', 'vgo_training.export_onnx',
        '--checkpoint', out / f'{label}.pt', '--output', out / f'{label}.onnx',
        '--maximum-batch', '64', '--packed-input'], ROOT / 'training')


def arena(out, label, candidate, opponent, units, timed=False, opponent_resolution=128):
    args = [ROOT / 'target/release/vgo-arena', '--candidate', candidate,
            '--opponent', opponent, '--candidate-raster-kind', 'compact-radius',
            '--resolution', '128', '--opponent-resolution', str(opponent_resolution),
            '--policy-resolution', '128', '--pairs', '24', '--threads', '16',
            '--maximum-batch', '32', '--simulations', '10000' if timed else '200',
            '--coarse-pool', '16', '--maximum-candidates', '321',
            '--widening-coefficient', '6', '--leaf-batch', '4',
            '--radius', str(1/units), '--komi', str(28/(units*units)),
            '--max-plies', str(round(70*(units/18)**2)), '--seed', '910001',
            '--cache-directory', ROOT / 'artifacts/onnx-cache',
            '--sgf-directory', out / f'{label}-sgf']
    if timed:
        args += ['--move-time-ms', '200']
    run(out, label, args)


def generate(out, label, randomized):
    args = [ROOT / 'target/release/vgo-generate-continuous', '--model',
            ROOT / 'artifacts/res-ab/r128-w150k.onnx', '--output-root', out / 'budget-games',
            '--label', label, '--actors', '32', '--resolution', '128',
            '--policy-resolution', '128', '--simulations', '1000',
            '--full-search-fraction', '0.25' if randomized else '1',
            '--fast-simulations', '200', '--coarse-pool', '16',
            '--widening-coefficient', '6', '--maximum-candidates', '321', '--leaf-batch', '4',
            '--maximum-batch', '32', '--inference-slots', '2',
            '--board-mix', '50:38', '--board-mix', '25:18', '--board-mix', '25:18-38',
            '--radius', str(1/18), '--max-plies', '70', '--komi-area-coefficient', '28',
            '--resign-threshold', '0.99', '--resign-soft-simulations', '400',
            '--resign-window', '5', '--resign-minimum-ply', '20',
            '--temperature', '1', '--temperature-plies', '30', '--root-exploration-noise', '0.1',
            '--cache-directory', ROOT / 'artifacts/onnx-cache', '--telemetry-seconds', '30',
            '--run-seconds', '600', '--seed', '810001']
    run_label = f'{label}-generate'
    attempt = 0
    while (out / f'{run_label}.stdout').exists():
        attempt += 1
        run_label = f'{label}-generate-retry-{attempt}'
    run(out, run_label, args)


def report(out):
    summary = {'arenas': {}, 'training': {}, 'generation': {}, 'throughput': {}}
    for label in ('norm1', 'budget-fixed', 'budget-randomized', 'window-fixed', 'window-randomized'):
        path = out / f'{label}.pt.json'
        if path.exists():
            data = json.loads(path.read_text())
            summary['training'][label] = {key: data[key] for key in
                ('samples', 'training_samples', 'validation_samples', 'optimization_seconds',
                 'wall_seconds', 'final_validation', 'parameters')}
    for path in out.glob('*.stdout'):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get('schema') == 'vgo.arena.v1':
            summary['arenas'][path.stem] = data
        elif isinstance(data, dict) and any('neural_search' in key for key in data):
            summary['throughput'][path.stem] = data
    for label in ('fixed', 'randomized'):
        generation = out / 'budget-games' / label
        path = generation / 'performance.jsonl'
        if not path.exists():
            continue
        final = json.loads(path.read_text().splitlines()[-1])
        games = [json.loads(line) for path in generation.glob('game-*/games.jsonl')
                 for line in path.read_text().splitlines()]
        for game in games:
            if len(game['search_budgets']) != game['plies']:
                raise RuntimeError(f'budget metadata missing a ply in {label}')
            if sum(game['policy_eligible']) != game['sample_count']:
                raise RuntimeError(f'eligibility does not reconcile with replay in {label}')
        if len(games) != final['totals']['published_games']:
            raise RuntimeError(f'published game count does not reconcile in {label}')
        groups = {name: {'games': 0, 'samples': 0, 'plies': 0}
                  for name in ('small', 'big', 'intermediate')}
        for game in games:
            units = round(1 / game['radius'])
            group = groups['small' if units == 18 else 'big' if units == 38 else 'intermediate']
            group['games'] += 1
            group['samples'] += game['sample_count']
            group['plies'] += game['plies']
        summary['generation'][label] = {'elapsed_seconds': final['elapsed_seconds'],
            'totals': final['totals'], 'broker': final.get('broker_totals'),
            'boards': final['boards'], 'published_board_groups': groups,
            'completed_game_plies': sum(game['plies'] for game in games),
            'eligible_rows': sum(game['sample_count'] for game in games),
            'soft_tail_eligible_rows': sum(
                sum(game['policy_eligible'][game['soft_resign_ply'] + 1:])
                for game in games if game['soft_resign_ply'] is not None)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Report written to', out / 'summary.json')


def verify_window(out):
    common = json.loads((ROOT / 'artifacts/res-ab/r128-w150k.pt.json').read_text())
    reports = {label: json.loads((out / f'window-{label}.pt.json').read_text())
               for label in ('fixed', 'randomized')}
    for label, data in reports.items():
        if data['dataset_digests'][:len(common['datasets'])] != common['dataset_digests']:
            raise RuntimeError(f'{label} changed the common replay corpus')
    steps = {label: (data['training_samples'] + 63) // 64 * 8
             for label, data in reports.items()}
    if max(steps.values()) - min(steps.values()) > 8:
        raise RuntimeError(f'full-window optimizer budgets differ: {steps}')
    (out / 'window-parity.json').write_text(json.dumps({
        'common_games': len(common['datasets']), 'common_samples': common['samples'],
        'common_ordered_digests_equal': True, 'optimizer_steps': steps,
        'relative_optimizer_step_difference': (max(steps.values()) - min(steps.values())) / min(steps.values()),
        'training_rows': {label: data['training_samples'] for label, data in reports.items()},
        'validation_value_mae': {label: data['final_validation']['value_mae']
                                 for label, data in reports.items()}}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['benchmark', 'model-benchmark', 'norm-train', 'resolution-arena',
        'norm-arena', 'budget-generate', 'budget-train', 'budget-arena',
        'budget-window-train', 'budget-window-verify', 'budget-window-arena', 'report'])
    parser.add_argument('--output-root', type=Path,
                        default=ROOT / 'artifacts/experiments-2026-10-14')
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / 'artifacts/res-ab/r128-w150k.onnx'
    if args.stage == 'benchmark':
        benchmark(out)
    elif args.stage == 'model-benchmark':
        model_benchmark(out)
    elif args.stage == 'norm-train':
        freeze_corpus(out)
        train(out, 'norm1', out / 'frozen-model-corpus', norms=1)
        verify_corpus(out)
    elif args.stage == 'resolution-arena':
        for units in (18, 38):
            arena(out, f'resolution-time-r{units}', baseline,
                  ROOT / 'artifacts/res-ab/r256-w150k.onnx', units, True, 256)
    elif args.stage == 'norm-arena':
        for units in (18, 38):
            for timed in (False, True):
                arena(out, f'norm1-{"time" if timed else "sim"}-r{units}',
                      out / 'norm1.onnx', baseline, units, timed)
    elif args.stage == 'budget-generate':
        # Check precision before interpreting the normalization match as a
        # learning effect. This stage runs after all model arenas, with no
        # concurrent work that could change their advisory time budgets.
        parity = out / 'serving-parity.json'
        verified = parity.exists() and all(
            'TensorrtExecutionProvider' in result['providers'] and result['finite']
            for result in json.loads(parity.read_text())['results'].values())
        if not verified:
            parity_label = 'serving-parity'
            attempt = 0
            while (out / f'{parity_label}.stdout').exists():
                attempt += 1
                parity_label = f'serving-parity-retry-{attempt}'
            run(out, parity_label, [PYTHON, ROOT / 'scripts/experiments/serving-parity.py',
                                      '--output-root', out])
        for randomized in (False, True):
            label = 'randomized' if randomized else 'fixed'
            # One bounded fixed arm was validated after its deliberate stop
            # was incorrectly classified as process failure. Its immutable
            # published corpus is retained, with explicit acceptance evidence.
            if label == 'fixed' and (out / 'fixed-deadline-acceptance.json').exists():
                print('Using validated bounded fixed corpus; see fixed-deadline-acceptance.json', flush=True)
                continue
            generate(out, label, randomized)
    elif args.stage == 'budget-train':
        for label in ('fixed', 'randomized'):
            # Keep window_from_games' expected generation/game hierarchy.
            corpus = out / f'{label}-corpus'
            corpus.mkdir(exist_ok=True)
            link = corpus / label
            if not link.exists():
                link.symlink_to(out / 'budget-games' / label, target_is_directory=True)
            train(out, f'budget-{label}', corpus)
    elif args.stage == 'budget-arena':
        for units in (18, 38):
            for timed in (False, True):
                arena(out, f'budget-{"time" if timed else "sim"}-r{units}',
                      out / 'budget-randomized.onnx', out / 'budget-fixed.onnx', units, timed)
    elif args.stage == 'budget-window-train':
        baseline_report = json.loads((ROOT / 'artifacts/res-ab/r128-w150k.pt.json').read_text())
        common = list(map(Path, baseline_report['datasets']))
        for label in ('fixed', 'randomized'):
            fresh = sorted((out / 'budget-games' / label).glob('game-*/dataset.vgo'))
            train(out, f'window-{label}', datasets=[*common, *fresh])
            trained = json.loads((out / f'window-{label}.pt.json').read_text())
            if trained['dataset_digests'][:len(common)] != baseline_report['dataset_digests']:
                raise RuntimeError(f'{label} changed the common replay corpus')
        verify_window(out)
    elif args.stage == 'budget-window-verify':
        verify_window(out)
    elif args.stage == 'budget-window-arena':
        for units in (18, 38):
            arena(out, f'window-time-r{units}', out / 'window-randomized.onnx',
                  out / 'window-fixed.onnx', units, True)
    elif args.stage == 'report':
        report(out)


if __name__ == '__main__':
    main()
