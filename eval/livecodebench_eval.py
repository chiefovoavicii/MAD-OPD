#!/usr/bin/env python3
"""Run LiveCodeBench v6 on a MAD-OPD checkpoint via ``lcb_runner``.

``lcb_runner`` starts its own vLLM server and scores pass@1 on the
code-generation-lite split.

Usage:
    python livecodebench_eval.py --model /path/to/checkpoint --release v6
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local path to the student checkpoint.')
    parser.add_argument('--registry_name', default=None,
                        help='Short name registered in lcb_runner/lm_styles.py; defaults to '
                             'the checkpoint basename.')
    parser.add_argument('--benchmark_root', default='./third_party/LiveCodeBench')
    parser.add_argument('--dataset_path', default=None,
                        help='Optional local copy of the code-generation-lite split '
                             '(falls back to HF download when absent).')
    parser.add_argument('--release', default='v6',
                        help='LiveCodeBench release label, e.g. v5 / v6.')
    parser.add_argument('--scenario', default='codegeneration')
    parser.add_argument('--tp', type=int, default=int(os.environ.get('TP', 1)))
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Samples per problem; pass@1 is averaged over these.')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--max_tokens', type=int, default=16384)
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--eval_workers', type=int, default=8,
                        help='Parallel processes for code execution.')
    parser.add_argument('--eval_timeout', type=int, default=10,
                        help='Per-test timeout (seconds).')
    parser.add_argument('--output_dir', default='./results/livecodebench')
    args = parser.parse_args()

    registry_name = args.registry_name or Path(args.model).name
    root = Path(args.benchmark_root).resolve()
    if not root.exists():
        sys.exit(f'[lcb] benchmark root not found at {root}. '
                 'Clone the benchmark (see eval/README.md).')

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, '-m', 'lcb_runner.runner.main',
        '--tensor_parallel_size', str(args.tp),
        '--model', registry_name,
        '--local_model_path', args.model,
        '--scenario', args.scenario,
        '--release_version', args.release,
        '--n', str(args.n_samples),
        '--temperature', str(args.temperature),
        '--top_p', str(args.top_p),
        '--max_tokens', str(args.max_tokens),
        '--dtype', args.dtype,
        '--evaluate',
        '--num_process_evaluate', str(args.eval_workers),
        '--timeout', str(args.eval_timeout),
        '--trust_remote_code',
        '--enable_prefix_caching',
    ]
    if args.dataset_path:
        cmd.extend(['--local_dataset_path', args.dataset_path])

    env = os.environ.copy()
    env['OUTPUT_DIR'] = str(out)

    print(f'[lcb] $ (cd {root} && {" ".join(cmd)})', flush=True)
    rc = subprocess.run(cmd, cwd=str(root), env=env).returncode
    if rc != 0:
        sys.exit(f'[lcb] lcb_runner exited with code {rc}')

    # lcb_runner writes scores under <root>/output/<registry_name>/...
    # We surface the final score JSON location for convenience.
    score_dir = root / 'output' / registry_name
    print(f'\n[lcb] evaluation complete.  scores in: {score_dir}')


if __name__ == '__main__':
    main()
