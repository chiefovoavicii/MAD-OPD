#!/usr/bin/env python3
"""Run MBPP+ evaluation on a MAD-OPD checkpoint via the EvalPlus CLI.

Generates N samples per task with vLLM, then scores them against the
MBPP+ test suite.  Requires ``pip install evalplus``.

Usage:
    python mbpp_plus_eval.py --model /path/to/checkpoint --n_samples 16
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict | None = None, cwd: str | None = None) -> int:
    print(f'[mbpp+] $ {" ".join(cmd)}', flush=True)
    return subprocess.run(cmd, env=env or os.environ.copy(), cwd=cwd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='HF path or local directory of the student checkpoint.')
    parser.add_argument('--output_dir', default='./results/mbpp_plus', help='Where to put samples + scores.')
    parser.add_argument('--benchmark_root', default='./third_party/evalplus',
                        help='Local clone of evalplus (only used to resolve its package imports).')
    parser.add_argument('--backend', default='vllm', choices=['vllm', 'hf'])
    parser.add_argument('--tp', type=int, default=int(os.environ.get('TP', 1)),
                        help='Tensor-parallel size for vLLM.')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--max_new_tokens', type=int, default=16384)
    parser.add_argument('--n_samples', type=int, default=16,
                        help='Samples per problem; pass@1 is averaged over these.')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--eval_parallel', type=int, default=8,
                        help='CPU parallelism for code-execution evaluation.')
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    samples = out / f'mbpp_n{args.n_samples}_samples.jsonl'
    scores  = out / f'mbpp_n{args.n_samples}_scores.json'

    env = os.environ.copy()
    env['EVALPLUS_ENABLE_THINKING'] = env.get('EVALPLUS_ENABLE_THINKING', 'false')

    # ---------- codegen ----------
    codegen_root = out / '_codegen_tmp'
    if codegen_root.exists():
        shutil.rmtree(codegen_root)

    rc = run([
        sys.executable, '-m', 'evalplus.codegen',
        '--model', args.model,
        '--dataset', 'mbpp',
        '--backend', args.backend,
        '--tp', str(args.tp),
        '--temperature', str(args.temperature),
        '--top_p', str(args.top_p),
        '--max_new_tokens', str(args.max_new_tokens),
        '--n_samples', str(args.n_samples),
        '--dtype', args.dtype,
        '--trust_remote_code',
        '--root', str(codegen_root),
    ], env=env, cwd=args.benchmark_root if Path(args.benchmark_root).exists() else None)
    if rc != 0:
        sys.exit(f'[mbpp+] codegen failed with exit code {rc}')

    # codegen puts the samples under <root>/mbpp/<...>.jsonl
    gen = next(codegen_root.joinpath('mbpp').glob('*.jsonl'), None)
    if gen is None:
        sys.exit('[mbpp+] cannot find generated samples under codegen root')
    shutil.move(str(gen), samples)
    shutil.rmtree(codegen_root, ignore_errors=True)
    print(f'[mbpp+] samples -> {samples}')

    # ---------- evaluate ----------
    rc = run([
        sys.executable, '-m', 'evalplus.evaluate',
        '--dataset', 'mbpp',
        '--samples', str(samples),
        '--parallel', str(args.eval_parallel),
        '--i-just-wanna-run',
        '--output_file', str(scores),
    ], env=env)
    if rc != 0:
        sys.exit(f'[mbpp+] evaluate failed with exit code {rc}')

    # ---------- summary ----------
    if scores.exists():
        data = json.loads(scores.read_text())
        base_pass = data.get('pass_at_k', {}).get('base', {}).get('pass@1', float('nan'))
        plus_pass = data.get('pass_at_k', {}).get('plus', {}).get('pass@1', float('nan'))
        print(f'\n[mbpp+] base pass@1 = {base_pass:.4f}')
        print(f'[mbpp+] plus pass@1 = {plus_pass:.4f}')


if __name__ == '__main__':
    main()
