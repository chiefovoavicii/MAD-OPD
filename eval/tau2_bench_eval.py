#!/usr/bin/env python3
"""Run τ²-Bench on a MAD-OPD checkpoint via the ``tau2 run`` CLI.

Starts a vLLM server for the student agent and calls ``tau2 run`` with
the requested user-simulator model.

Usage:
    python tau2_bench_eval.py --model /path/to/checkpoint \
        --agent_model_name student-v1 --user_llm claude-opus-4-6 \
        --domains airline,retail,telecom
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_for_vllm(url: str, timeout: int = 900) -> bool:
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
    parser.add_argument('--model', required=True, help='Agent checkpoint path.')
    parser.add_argument('--agent_model_name', required=True,
                        help='Short name used by LiteLLM / tau2 to address the agent.')
    parser.add_argument('--user_llm', required=True,
                        help='User-simulator LLM, e.g. claude-opus-4-6 via API.')
    parser.add_argument('--domains', default='airline,retail,telecom',
                        help='Comma-separated domains.')
    parser.add_argument('--task_split', default='tau2',
                        help='Task split name; tau2 = full τ² split.')
    parser.add_argument('--num_trials', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--max_concurrency', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Leave off for non-thinking mode (paper default).')

    # vLLM server for the agent
    parser.add_argument('--skip_server', action='store_true')
    parser.add_argument('--agent_port', type=int, default=8002)
    parser.add_argument('--tp', type=int, default=int(os.environ.get('TP', 1)))
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=32768)
    parser.add_argument('--extra', nargs='*', default=[],
                        help='Extra args forwarded verbatim to ``tau2 run``.')
    args = parser.parse_args()

    domains = args.domains

    # ---------- vLLM agent server ----------
    server_proc: subprocess.Popen | None = None
    agent_url = f'http://127.0.0.1:{args.agent_port}'
    if not args.skip_server:
        vllm_cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', args.model,
            '--served-model-name', args.agent_model_name,
            '--host', '0.0.0.0', '--port', str(args.agent_port),
            '--tensor-parallel-size', str(args.tp),
            '--gpu-memory-utilization', str(args.gpu_memory_utilization),
            '--max-model-len', str(args.max_model_len),
            '--trust-remote-code',
        ]
        print(f'[tau2] launching agent vLLM: {" ".join(vllm_cmd)}', flush=True)
        server_proc = subprocess.Popen(vllm_cmd, preexec_fn=os.setsid)
        if not wait_for_vllm(agent_url):
            if server_proc is not None:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
            sys.exit('[tau2] agent vLLM failed to become ready')

    # LiteLLM-style args for tau2: point the agent at our local vLLM.
    agent_llm_args = (
        f'{{"api_base": "{agent_url}/v1", "api_key": "EMPTY", '
        f'"temperature": {args.temperature}, "top_p": {args.top_p}}}'
    )
    user_llm_args = '{}'   # user LLM uses the environment's default (API or other vLLM)

    try:
        cmd = [
            'tau2', 'run',
            '--domain', domains,
            '--agent-llm', args.agent_model_name,
            '--user-llm', args.user_llm,
            '--agent-llm-args', agent_llm_args,
            '--user-llm-args', user_llm_args,
            '--task-split-name', args.task_split,
            '--num-trials', str(args.num_trials),
            '--max-steps', str(args.max_steps),
            '--max-concurrency', str(args.max_concurrency),
            '--seed', str(args.seed),
            '--verbose-logs',
        ]
        if args.enable_thinking:
            cmd.append('--enable-think')
        cmd.extend(args.extra)

        print(f'[tau2] $ {" ".join(cmd)}', flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f'[tau2] tau2 run exited with code {rc} (some tasks may have failed).')

    finally:
        if server_proc is not None:
            print('[tau2] shutting down vLLM...')
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)


if __name__ == '__main__':
    main()
