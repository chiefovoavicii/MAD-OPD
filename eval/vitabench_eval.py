#!/usr/bin/env python3
"""Run VitaBench on a MAD-OPD checkpoint via the ``vita run`` CLI.

Starts a vLLM server for the student agent and calls ``vita run`` with
the requested user-simulator and evaluator models.

Usage:
    python vitabench_eval.py --model /path/to/checkpoint \
        --agent_model_name student-v1 \
        --user_llm claude-opus-4-6 --evaluator_llm claude-opus-4-6 \
        --domains delivery,instore,ota
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
    parser.add_argument('--model', required=True)
    parser.add_argument('--agent_model_name', required=True)
    parser.add_argument('--user_llm', required=True,
                        help='User-simulator LLM identifier (API model name or vLLM-served name).')
    parser.add_argument('--evaluator_llm', required=True,
                        help='Evaluator LLM identifier.')
    parser.add_argument('--domains', default='delivery,instore,ota',
                        help='Comma-separated single-domain list OR a list of domains to join with '
                             '+ for --cross_domain mode.')
    parser.add_argument('--cross_domain', action='store_true',
                        help='Run the multi-domain task split where all domains are merged.')
    parser.add_argument('--num_trials', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--max_concurrency', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--language', default='chinese', choices=['chinese', 'english'])
    parser.add_argument('--enable_thinking', action='store_true')

    # vLLM agent server
    parser.add_argument('--skip_server', action='store_true')
    parser.add_argument('--agent_port', type=int, default=8002)
    parser.add_argument('--tp', type=int, default=int(os.environ.get('TP', 1)))
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_model_len', type=int, default=32768)

    parser.add_argument('--output_dir', default='./results/vitabench')
    parser.add_argument('--extra', nargs='*', default=[],
                        help='Extra args forwarded verbatim to ``vita run``.')
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Domains: vita run expects "+" joined list in cross-domain mode.
    domain_arg = '+'.join(args.domains.split(',')) if args.cross_domain else args.domains

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
        print(f'[vita] launching agent vLLM: {" ".join(vllm_cmd)}', flush=True)
        server_proc = subprocess.Popen(vllm_cmd, preexec_fn=os.setsid)
        if not wait_for_vllm(agent_url):
            if server_proc is not None:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
            sys.exit('[vita] agent vLLM failed to become ready')

    try:
        save_name = f'{args.agent_model_name}_{domain_arg.replace("+", "-")}.json'
        cmd = [
            'vita', 'run',
            '--domain', domain_arg,
            '--agent-llm', args.agent_model_name,
            '--user-llm', args.user_llm,
            '--evaluator-llm', args.evaluator_llm,
            '--num-trials', str(args.num_trials),
            '--max-steps', str(args.max_steps),
            '--max-concurrency', str(args.max_concurrency),
            '--seed', str(args.seed),
            '--language', args.language,
            '--save-to', str(out / save_name),
            '--csv-output', str(out / f'score_table_{domain_arg.replace("+", "-")}.csv'),
        ]
        if args.enable_thinking:
            cmd.append('--enable-think')
        cmd.extend(args.extra)

        # vita run reads the agent endpoint from env (LiteLLM routing).
        os.environ.setdefault('OPENAI_BASE_URL', f'{agent_url}/v1')
        os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')

        print(f'[vita] $ {" ".join(cmd)}', flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f'[vita] vita run exited with code {rc}')
        print(f'[vita] results under {out}')

    finally:
        if server_proc is not None:
            print('[vita] shutting down vLLM...')
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)


if __name__ == '__main__':
    main()
