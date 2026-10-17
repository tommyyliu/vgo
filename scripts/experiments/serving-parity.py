#!/usr/bin/env python3
"""Check trained restart exports against eager PyTorch on actual replay positions.

Run with an idle GPU after sourcing scripts/env/ort.sh. This is an accuracy
probe, not a timing benchmark. TensorRT profiles mirror the native batch-32
serving configuration; CUDA fallback disables the known conv/bias fusion bug.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))

import numpy as np
import onnxruntime as ort
import torch

from vgo_training.dataset import load_dataset
from vgo_training.model import load_model
from vgo_training.packed_input import PackedInputModel
from vgo_training.packed_states import pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root
    torch.set_num_threads(1)
    if 'TensorrtExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('TensorRT provider required for this accuracy probe')
    selected = {18: [], 38: []}
    for game in sorted((out / 'frozen-model-corpus/gen-original').iterdir()):
        record = json.loads((game / 'games.jsonl').read_text().splitlines()[0])
        units = round(1 / record['radius'])
        if units in selected and len(selected[units]) < 4:
            selected[units].append(game)
        if all(len(games) == 4 for games in selected.values()):
            break
    if any(len(games) != 4 for games in selected.values()):
        raise RuntimeError('need four frozen games per board size')
    triples, sources = [], []
    for games in selected.values():
        for game in games:
            dataset = load_dataset(game / 'dataset.vgo', raster_kind='compact-radius', resolution=128)
            packed = dataset.packed_states or pack(dataset.states)
            indices = sorted(set([0, packed.samples // 2, packed.samples - 1]))
            triples.append((packed.bits[indices], packed.continuous[indices], packed.scalars[indices]))
            sources.append({'game': str(game.resolve()), 'rows': indices})
    tensors = tuple(torch.cat([triple[index] for triple in triples]) for index in range(3))
    device_tensors = tuple(tensor.cuda() for tensor in tensors)
    results = {}
    for label, checkpoint in [('norm2', ROOT / 'artifacts/res-ab/r128-w150k.pt'),
                              ('norm1', out / 'norm1.pt')]:
        model, _ = load_model(checkpoint)
        wrapped = PackedInputModel(model.cuda(), binary=(0, 1, 3), continuous=(2,),
                                   scalar=(4, 5, 6), height=128, width=128).cuda().eval()
        with torch.inference_mode():
            reference = tuple(tensor.float().cpu() for tensor in wrapped(*device_tensors))
        path = checkpoint.with_suffix('.onnx')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        cache = ROOT / 'artifacts/onnx-cache' / f'tensorrt-fp16-128x128-batch32-{digest[:16]}'
        cache.mkdir(parents=True, exist_ok=True)
        shape = lambda batch: f'bits:{batch}x3x2048,dense:{batch}x1x128x128,scalars:{batch}x3'
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=options, providers=[
            ('TensorrtExecutionProvider', {
                'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(cache), 'trt_profile_min_shapes': shape(1),
                'trt_profile_opt_shapes': shape(32), 'trt_profile_max_shapes': shape(32)}),
            ('CUDAExecutionProvider', {'fuse_conv_bias': '0'}), 'CPUExecutionProvider'])
        if 'TensorrtExecutionProvider' not in session.get_providers():
            raise RuntimeError('TensorRT registration failed; refusing to call fallback fp16 parity')
        session.disable_fallback()
        feed = {name: tensor.numpy() for name, tensor in
                zip(('bits', 'dense', 'scalars'), tensors)}
        first = session.run(None, feed)
        session.run(None, feed)
        actual = session.run(None, feed)
        repeated_error = max(float(np.max(np.abs(a - b))) for a, b in zip(first, actual))
        actual = tuple(torch.from_numpy(np.asarray(tensor)) for tensor in actual)
        policy_delta = (reference[0].softmax(-1) - actual[0].softmax(-1)).abs().sum(-1) / 2
        result = {'positions': tensors[0].shape[0], 'onnx_sha256': digest,
                  'maximum_logit_error': float((reference[0] - actual[0]).abs().max()),
                  'maximum_value_error': float((reference[1] - actual[1]).abs().max()),
                  'maximum_policy_total_variation': float(policy_delta.max()),
                  'maximum_repeated_output_error': repeated_error,
                  'finite': all(bool(tensor.isfinite().all()) for tensor in actual),
                  'providers': session.get_providers()}
        results[label] = result
        del session, wrapped, model
        torch.cuda.empty_cache()
    (out / 'serving-parity.json').write_text(json.dumps({'sources': sources, 'results': results}, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    if any(not result['finite'] or result['maximum_value_error'] > 0.02
           or result['maximum_policy_total_variation'] > 0.02
           or result['maximum_repeated_output_error'] > 0.02 for result in results.values()):
        raise RuntimeError('serving output differs materially from eager checkpoint')


if __name__ == '__main__':
    main()
