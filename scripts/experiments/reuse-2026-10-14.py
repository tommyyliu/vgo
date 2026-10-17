#!/usr/bin/env python3
"""Bounded tree reuse arena and inference deadline checks; idle GPU required.

source scripts/env/ort.sh
training/.venv/bin/python scripts/experiments/reuse-2026-10-14.py smoke|arena
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/reuse-2026-10-14'
MODEL = ROOT / 'artifacts/res-ab/r128-w150k.onnx'


def run(label, args, *, environment=None, expect_success=True, expected_exit_code=None, timeout=900):
    OUT.mkdir(parents=True, exist_ok=True)
    stdout, stderr = OUT / f'{label}.stdout', OUT / f'{label}.log'
    if stdout.exists() or stderr.exists():
        raise RuntimeError(f'refusing to overwrite {label}')
    env = dict(os.environ, VGO_INFERENCE_STARTUP_SECONDS='120', VGO_INFERENCE_TIMEOUT_SECONDS='10')
    env.update(environment or {})
    record = dict(label=label, args=list(map(str, args)), started_unix=time.time(),
                  executable_sha256=hashlib.sha256(Path(args[0]).read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                  deadline_environment={key: env[key] for key in env if key.startswith('VGO_INFERENCE_')})
    print('Starting', label, flush=True)
    peak_rss = 0
    with stdout.open('w') as output, stderr.open('w') as errors:
        process = subprocess.Popen(list(map(str, args)), cwd=ROOT, env=env,
                                   stdout=output, stderr=errors, start_new_session=True)
        timed_out = False
        while process.poll() is None:
            try:
                for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        peak_rss = max(peak_rss, int(line.split()[1])*1024)
            except FileNotFoundError:
                pass
            if time.time()-record['started_unix'] > timeout:
                os.killpg(process.pid, signal.SIGKILL)
                timed_out = True
            try:
                process.wait(timeout=.25)
            except subprocess.TimeoutExpired:
                pass
    record.update(exit_code=process.returncode, wall_seconds=time.time()-record['started_unix'],
                  sampled_peak_rss_bytes=peak_rss, supervisor_timed_out=timed_out)
    with (OUT / 'runs.jsonl').open('a') as records:
        records.write(json.dumps(record)+'\n')
    print('Finished', label, json.dumps(record), flush=True)
    if timed_out or ((process.returncode == 0) != expect_success) or (
            expected_exit_code is not None and process.returncode != expected_exit_code):
        raise RuntimeError(f'{label} failed expectations: {stderr}')


def arena():
    for units in [18, 38]:
        run(f'reuse-time-r{units}', [ROOT / 'target/release/vgo-arena',
            '--candidate', MODEL, '--opponent', MODEL, '--tree-reuse',
            '--candidate-raster-kind', 'compact-radius', '--resolution', '128',
            '--policy-resolution', '128', '--pairs', '24', '--threads', '16',
            '--maximum-batch', '32', '--simulations', '10000', '--move-time-ms', '200',
            '--coarse-pool', '16', '--maximum-candidates', '321',
            '--widening-coefficient', '6', '--leaf-batch', '4',
            '--radius', str(1/units), '--komi', str(28/(units*units)),
            '--max-plies', str(round(70*(units/18)**2)), '--seed', '940001',
            '--cache-directory', ROOT / 'artifacts/onnx-cache',
            '--sgf-directory', OUT / f'reuse-time-r{units}-sgf'])


def smoke(slots_list=(2, 4), prefix=''):
    for slots in slots_list:
        run(f'{prefix}inference-{slots}-slots', [ROOT / 'target/release/vgo-generate-continuous',
            '--model', MODEL, '--output-root', OUT / 'smoke-games', '--label', f'{prefix}{slots}-slots',
            '--raster-kind', 'compact-radius', '--resolution', '128', '--policy-resolution', '128',
            '--actors', '8', '--inference-slots', str(slots), '--maximum-batch', '32',
            '--simulations', '16', '--coarse-pool', '16', '--run-seconds', '3',
            '--telemetry-seconds', '1', '--seed', '950001',
            '--cache-directory', ROOT / 'artifacts/onnx-cache'], timeout=180)
    missing_runtime()


def missing_runtime():
    run('missing-runtime-guarded', [ROOT / 'target/release/vgo-arena', '--candidate', MODEL,
        '--resolution', '128', '--policy-resolution', '128', '--candidate-raster-kind', 'compact-radius'],
        environment={'ORT_DYLIB_PATH': str(OUT / 'missing-libonnxruntime.so'),
                     'VGO_INFERENCE_STARTUP_SECONDS': '2'}, expect_success=False,
        expected_exit_code=70, timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'arena', 'missing-runtime', 'final-smoke'])
    parser.add_argument('--output', type=Path, default=OUT, help='Use a fresh directory for repeated runs.')
    args = parser.parse_args()
    OUT = args.output.resolve()
    {'smoke': smoke, 'arena': arena, 'missing-runtime': missing_runtime,
     'final-smoke': lambda: smoke((2,), 'final-')}[args.stage]()
