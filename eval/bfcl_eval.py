#!/usr/bin/env python3
"""Run BFCL v4 on a MAD-OPD checkpoint via the ``bfcl_eval`` CLI.

Launches a local vLLM server for the student, then calls ``bfcl_eval
generate`` and ``bfcl_eval evaluate`` over the configured categories.

Usage:
    python bfcl_eval.py --model /path/to/checkpoint --registry_name student-v1
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


# BFCL-v4 main-table categories (mean over all 8).
DEFAULT_CATEGORIES = [
    'simple', 'multiple', 'parallel', 'parallel_multiple',
    'multi_turn_base', 'multi_turn_miss_func',
    'multi_turn_miss_param', 'multi_turn_long_context',
]


def wait_for_vllm(url: str, timeout: int = 600) -> bool:
    import urllib.request
    import urllib.error
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + '/v1/models', timeout=5) as r:
                if r.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        time.sleep(5)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Student checkpoint path.')
    parser.add_argument('--registry_name', required=True,
                        help='Short name registered in bfcl_eval/_constants/model_config.py.')
    parser.add_argument('--benchmark_root', default='./third_party/gorilla/berkeley-function-call-leaderboard')
    parser.add_argument('--categories', nargs='*', default=DEFAULT_CATEGORIES,
                        help='Which BFCL categories to evaluate.')
    parser.add_argument('--temperature', type=float, default=0.001)
    parser.add_argument('--num_threads', type=int, default=16)

    # vLLM server management (set --skip_server if you already have one running)
    parser.add_argument('--skip_server', action='store_true')
    parser.add_argument('--server_port', type=int, default=8084)
    parser.add_argument('--server_host', default='127.0.0.1')
    parser.add_argument('--tp', type=int, default=int(os.environ.get('TP', 1)))
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=32768)
    args = parser.parse_args()

    bfcl_root = Path(args.benchmark_root).resolve()
    if not bfcl_root.exists():
        sys.exit(f'[bfcl] benchmark root not found: {bfcl_root}')

    # ---------- bring up vLLM server ----------
    server_proc: subprocess.Popen | None = None
    url = f'http://{args.server_host}:{args.server_port}'
    if not args.skip_server:
        vllm_cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', args.model,
            '--served-model-name', args.registry_name,
            '--host', args.server_host,
            '--port', str(args.server_port),
            '--tensor-parallel-size', str(args.tp),
            '--gpu-memory-utilization', str(args.gpu_memory_utilization),
            '--max-model-len', str(args.max_model_len),
            '--trust-remote-code',
        ]
        print(f'[bfcl] launching vLLM: {" ".join(shlex.quote(c) for c in vllm_cmd)}', flush=True)
        server_proc = subprocess.Popen(vllm_cmd, preexec_fn=os.setsid)
        if not wait_for_vllm(url):
            server_proc.send_signal(signal.SIGINT)
            sys.exit('[bfcl] vLLM server failed to come up within timeout')
        print(f'[bfcl] vLLM ready at {url}')

    os.environ.setdefault('OPENAI_BASE_URL', f'{url}/v1')
    os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')

    try:
        # ---------- generate ----------
        rc = subprocess.run([
            sys.executable, '-m', 'bfcl_eval', 'generate',
            '--model', args.registry_name,
            '--test-category', *args.categories,
            '--num-threads', str(args.num_threads),
            '--temperature', str(args.temperature),
            '--skip-server-setup',
            '--include-input-log',
        ], cwd=str(bfcl_root)).returncode
        if rc != 0:
            print(f'[bfcl] generate exited with code {rc} (some categories may be empty, continuing)')

        # ---------- evaluate ----------
        rc = subprocess.run([
            sys.executable, '-m', 'bfcl_eval', 'evaluate',
            '--model', args.registry_name,
            '--test-category', *args.categories,
            '--partial-eval',
        ], cwd=str(bfcl_root)).returncode
        if rc != 0:
            sys.exit(f'[bfcl] evaluate failed with code {rc}')

        # Result CSV path is fixed by BFCL: score/<registry_name>/data_overall.csv
        score_csv = bfcl_root / 'score' / args.registry_name / 'data_overall.csv'
        if score_csv.exists():
            print(f'\n[bfcl] final scores:\n{score_csv.read_text()}')
        else:
            print(f'[bfcl] scores directory: {bfcl_root / "score" / args.registry_name}')

    finally:
        if server_proc is not None:
            print('[bfcl] shutting down vLLM server...')
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)


if __name__ == '__main__':
    main()
