#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
import argparse
import json
import logging
import os
import subprocess
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [launch_teachers] %(levelname)s %(message)s',
)
logger = logging.getLogger('launch_teachers')


def wait_for_service(url, timeout_seconds=600, interval=5, label='service'):
    logger.info(f'Waiting for {label} at {url} (timeout={timeout_seconds}s)...')
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                logger.info(f'✓ {label} is ready!')
                return True
        except Exception:
            pass
        time.sleep(interval)
    logger.error(f'✗ Timeout waiting for {label} at {url}')
    return False


def start_vllm_server(model_path, host, port, gpu_id, gpu_memory_utilization,
                      max_model_len, log_file):
    cmd = (
        f'{sys.executable} -m vllm.entrypoints.openai.api_server '
        f'--model "{model_path}" '
        f'--host {host} '
        f'--port {port} '
        f'--gpu-memory-utilization {gpu_memory_utilization} '
        f'--max-model-len {max_model_len} '
        f'--dtype bfloat16 '
        f'--trust-remote-code '
        # f'--disable-log-requests '
        f'>> "{log_file}" 2>&1'
    )
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTHONUNBUFFERED'] = '1'

    logger.info(f'Starting vLLM: model={os.path.basename(model_path)}, '
                f'GPU={gpu_id}, port={port}')
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    
    logger.info(f'  PID={process.pid}, log={log_file}')
    return process


def start_sidecar_server(model_path, host, port, gpu_id,
                         gpu_memory_fraction, log_file,
                         enable_thinking=False, max_length=40960):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sidecar_script = os.path.join(script_dir, 'force_decode_server.py')

    cmd = (
        f'{sys.executable} "{sidecar_script}" '
        f'--model "{model_path}" '
        f'--host {host} '
        f'--port {port} '
        f'--device cuda:0 '
        f'--gpu-memory-fraction {gpu_memory_fraction} '
        f'--max-length {max_length} '
        f'--enable-thinking {str(enable_thinking).lower()} '
        f'>> "{log_file}" 2>&1'
    )
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['PYTHONUNBUFFERED'] = '1'

    logger.info(f'Starting Sidecar: model={os.path.basename(model_path)}, '
                f'GPU={gpu_id}, port={port}')
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    
    logger.info(f'  PID={process.pid}, log={log_file}')
    return process


def tail_log(log_file, lines=20):
    if not os.path.exists(log_file):
        return
    try:
        with open(log_file, 'r') as fh:
            all_lines = fh.readlines()
            for line in all_lines[-lines:]:
                print(line, end='', file=sys.stderr)
    except Exception:
        pass


def detect_total_gpus():
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True, text=True, timeout=10,
        )
        count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
        return max(count, 1)
    except Exception:
        return 8


def launch_external_mode(args):
    logger.info('External Teachers mode: using pre-deployed services')

    teacher_1_url = args.vllm_teacher_1_external_url
    teacher_2_url = args.vllm_teacher_2_external_url or ''
    sidecar_1_url = args.sidecar_1_external_url or ''
    sidecar_2_url = args.sidecar_2_external_url or ''

    if not teacher_1_url:
        logger.error('VLLM_TEACHER_1_EXTERNAL_URL must be set in external mode')
        sys.exit(1)

    if not wait_for_service(f'{teacher_1_url}/models', timeout_seconds=30,
                            interval=3, label='External Teacher-1'):
        logger.error(f'External Teacher-1 at {teacher_1_url} is not reachable')
        sys.exit(1)

    if teacher_2_url:
        wait_for_service(f'{teacher_2_url}/models', timeout_seconds=30,
                         interval=3, label='External Teacher-2')

    if args.use_sidecar:
        if not sidecar_1_url:
            logger.error('SIDECAR_1_EXTERNAL_URL must be set when USE_SIDECAR=true in external mode')
            sys.exit(1)
        wait_for_service(f'{sidecar_1_url}/health', timeout_seconds=30,
                         interval=3, label='External Sidecar-1')
        if sidecar_2_url:
            wait_for_service(f'{sidecar_2_url}/health', timeout_seconds=30,
                             interval=3, label='External Sidecar-2')

    total_gpus = detect_total_gpus()
    all_gpu_ids = ','.join(str(i) for i in range(total_gpus))

    return {
        'teacher_1_url': teacher_1_url,
        'teacher_2_url': teacher_2_url,
        'sidecar_1_url': sidecar_1_url,
        'sidecar_2_url': sidecar_2_url,
        'cuda_visible_devices': all_gpu_ids,
        'train_nproc_per_node': total_gpus,
        'use_sidecar': args.use_sidecar,
        'pids': [],
    }


def launch_local_mode(args):
    pids = []
    teacher_1_url = ''
    teacher_2_url = ''
    sidecar_1_url = ''
    sidecar_2_url = ''
    use_sidecar = args.use_sidecar

    is_multi_node = args.nnodes >= 2
    has_teacher_2 = bool(args.teacher_model_2) and args.teacher_model_2 != 'None'

    os.makedirs(args.log_dir, exist_ok=True)

    if is_multi_node:
        # =================================================================
        local_host = '127.0.0.1'
        logger.info(f'[Worker {args.node_rank}] Multi-node self-contained mode: '
                    f'each node launches its own vLLM + Sidecar on GPU 0-1')

        log_prefix = f'node{args.node_rank}_'

        vllm_gpu_util = args.vllm_gpu_memory_utilization
        sidecar_mem = args.sidecar_gpu_memory_fraction
        if use_sidecar:
            vllm_gpu_util = args.vllm_sidecar_coexist_gpu_util
            sidecar_mem = args.sidecar_coexist_gpu_memory_fraction
            logger.info(f'[Worker {args.node_rank}] vLLM + Sidecar coexist on same GPU: '
                        f'vLLM gpu_util={vllm_gpu_util}, Sidecar mem_frac={sidecar_mem}')

        logger.info(f'[Worker {args.node_rank}] Starting vLLM Teacher-1 on GPU {args.vllm_teacher_1_gpu}...')
        proc1 = start_vllm_server(
            model_path=args.teacher_model_1,
            host=local_host,
            port=args.vllm_teacher_1_port,
            gpu_id=args.vllm_teacher_1_gpu,
            gpu_memory_utilization=vllm_gpu_util,
            max_model_len=args.vllm_teacher_1_max_model_len,
            log_file=os.path.join(args.log_dir, f'{log_prefix}vllm_teacher_1.log'),
        )
        pids.append(proc1.pid)

        if has_teacher_2:
            logger.info(f'[Worker {args.node_rank}] Starting vLLM Teacher-2 on GPU {args.vllm_teacher_2_gpu}...')
            proc2 = start_vllm_server(
                model_path=args.teacher_model_2,
                host=local_host,
                port=args.vllm_teacher_2_port,
                gpu_id=args.vllm_teacher_2_gpu,
                gpu_memory_utilization=vllm_gpu_util,
                max_model_len=args.vllm_teacher_2_max_model_len,
                log_file=os.path.join(args.log_dir, f'{log_prefix}vllm_teacher_2.log'),
            )
            pids.append(proc2.pid)

        vllm_1_health = f'http://127.0.0.1:{args.vllm_teacher_1_port}/v1/models'
        if not wait_for_service(vllm_1_health, timeout_seconds=300, label='vLLM Teacher-1'):
            tail_log(os.path.join(args.log_dir, f'{log_prefix}vllm_teacher_1.log'))
            sys.exit(1)

        if has_teacher_2:
            vllm_2_health = f'http://127.0.0.1:{args.vllm_teacher_2_port}/v1/models'
            if not wait_for_service(vllm_2_health, timeout_seconds=300, label='vLLM Teacher-2'):
                tail_log(os.path.join(args.log_dir, f'{log_prefix}vllm_teacher_2.log'))
                sys.exit(1)

        if use_sidecar:
            logger.info(f'[Worker {args.node_rank}] Starting Force Decode sidecar servers (co-located on GPU 0-1)...')

            proc_s1 = start_sidecar_server(
                model_path=args.teacher_model_1,
                host=local_host,
                port=args.sidecar_1_port,
                gpu_id=args.sidecar_1_gpu,
                gpu_memory_fraction=sidecar_mem,
                log_file=os.path.join(args.log_dir, f'{log_prefix}sidecar_1.log'),
                enable_thinking=args.sidecar_enable_thinking,
                max_length=args.sidecar_max_length,
            )
            pids.append(proc_s1.pid)

            if has_teacher_2:
                proc_s2 = start_sidecar_server(
                    model_path=args.teacher_model_2,
                    host=local_host,
                    port=args.sidecar_2_port,
                    gpu_id=args.sidecar_2_gpu,
                    gpu_memory_fraction=sidecar_mem,
                    log_file=os.path.join(args.log_dir, f'{log_prefix}sidecar_2.log'),
                    enable_thinking=args.sidecar_enable_thinking,
                    max_length=args.sidecar_max_length,
                )
                pids.append(proc_s2.pid)

            s1_health = f'http://127.0.0.1:{args.sidecar_1_port}/health'
            if not wait_for_service(s1_health, timeout_seconds=900, label='Sidecar-1'):
                tail_log(os.path.join(args.log_dir, f'{log_prefix}sidecar_1.log'), lines=50)
                logger.warning('Continuing without sidecar (fallback to sparse prompt_logprobs)')
                use_sidecar = False

            if has_teacher_2 and use_sidecar:
                s2_health = f'http://127.0.0.1:{args.sidecar_2_port}/health'
                if not wait_for_service(s2_health, timeout_seconds=900, label='Sidecar-2'):
                    tail_log(os.path.join(args.log_dir, f'{log_prefix}sidecar_2.log'), lines=50)
                    logger.warning('Continuing without sidecar')
                    use_sidecar = False

        teacher_1_url = f'http://{local_host}:{args.vllm_teacher_1_port}/v1'
        teacher_2_url = f'http://{local_host}:{args.vllm_teacher_2_port}/v1' if has_teacher_2 else ''

        if use_sidecar:
            sidecar_1_url = f'http://{local_host}:{args.sidecar_1_port}'
            sidecar_2_url = f'http://{local_host}:{args.sidecar_2_port}' if has_teacher_2 else ''

        logger.info(f'[Worker {args.node_rank}] All services launched locally:')
        logger.info(f'  Teacher-1: {teacher_1_url}')
        logger.info(f'  Teacher-2: {teacher_2_url or "N/A"}')
        logger.info(f'  Sidecar-1: {sidecar_1_url or "N/A"}')
        logger.info(f'  Sidecar-2: {sidecar_2_url or "N/A"}')

    else:
        # =================================================================
        if args.node_rank == 0:
            logger.info('[Single-node] Starting vLLM teacher servers...')

            vllm_gpu_util = args.vllm_gpu_memory_utilization
            sidecar_mem = args.sidecar_gpu_memory_fraction
            if use_sidecar:
                vllm_gpu_util = args.vllm_sidecar_coexist_gpu_util
                sidecar_mem = args.sidecar_coexist_gpu_memory_fraction
                logger.info(f'Single-node + Sidecar coexist: '
                            f'vLLM={vllm_gpu_util}, Sidecar={sidecar_mem}')

            proc1 = start_vllm_server(
                model_path=args.teacher_model_1,
                host=args.vllm_teacher_1_host,
                port=args.vllm_teacher_1_port,
                gpu_id=args.vllm_teacher_1_gpu,
                gpu_memory_utilization=vllm_gpu_util,
                max_model_len=args.vllm_teacher_1_max_model_len,
                log_file=os.path.join(args.log_dir, 'vllm_teacher_1.log'),
            )
            pids.append(proc1.pid)

            if has_teacher_2:
                proc2 = start_vllm_server(
                    model_path=args.teacher_model_2,
                    host=args.vllm_teacher_2_host,
                    port=args.vllm_teacher_2_port,
                    gpu_id=args.vllm_teacher_2_gpu,
                    gpu_memory_utilization=vllm_gpu_util,
                    max_model_len=args.vllm_teacher_2_max_model_len,
                    log_file=os.path.join(args.log_dir, 'vllm_teacher_2.log'),
                )
                pids.append(proc2.pid)

            vllm_1_health = f'http://{args.vllm_teacher_1_host}:{args.vllm_teacher_1_port}/v1/models'
            if not wait_for_service(vllm_1_health, timeout_seconds=300, label='vLLM Teacher-1'):
                tail_log(os.path.join(args.log_dir, 'vllm_teacher_1.log'))
                if has_teacher_2:
                    tail_log(os.path.join(args.log_dir, 'vllm_teacher_2.log'))
                sys.exit(1)

            if has_teacher_2:
                vllm_2_health = f'http://{args.vllm_teacher_2_host}:{args.vllm_teacher_2_port}/v1/models'
                if not wait_for_service(vllm_2_health, timeout_seconds=300, label='vLLM Teacher-2'):
                    tail_log(os.path.join(args.log_dir, 'vllm_teacher_2.log'))
                    sys.exit(1)

            teacher_1_url = f'http://{args.vllm_teacher_1_host}:{args.vllm_teacher_1_port}/v1'
            teacher_2_url = (f'http://{args.vllm_teacher_2_host}:{args.vllm_teacher_2_port}/v1'
                             if has_teacher_2 else '')

            if use_sidecar:
                logger.info('[Single-node] Starting Force Decode sidecar servers (co-located)...')
                sidecar_host = '127.0.0.1'

                proc_s1 = start_sidecar_server(
                    model_path=args.teacher_model_1,
                    host=sidecar_host,
                    port=args.sidecar_1_port,
                    gpu_id=args.vllm_teacher_1_gpu,
                    gpu_memory_fraction=sidecar_mem,
                    log_file=os.path.join(args.log_dir, 'sidecar_1.log'),
                    enable_thinking=args.sidecar_enable_thinking,
                    max_length=args.sidecar_max_length,
                )
                pids.append(proc_s1.pid)

                if has_teacher_2:
                    proc_s2 = start_sidecar_server(
                        model_path=args.teacher_model_2,
                        host=sidecar_host,
                        port=args.sidecar_2_port,
                        gpu_id=args.vllm_teacher_2_gpu,
                        gpu_memory_fraction=sidecar_mem,
                        log_file=os.path.join(args.log_dir, 'sidecar_2.log'),
                        enable_thinking=args.sidecar_enable_thinking,
                        max_length=args.sidecar_max_length,
                    )
                    pids.append(proc_s2.pid)

                s1_health = f'http://{sidecar_host}:{args.sidecar_1_port}/health'
                if not wait_for_service(s1_health, timeout_seconds=300, label='Sidecar-1'):
                    tail_log(os.path.join(args.log_dir, 'sidecar_1.log'))
                    logger.warning('Continuing without sidecar (fallback to sparse prompt_logprobs)')
                    use_sidecar = False

                if has_teacher_2 and use_sidecar:
                    s2_health = f'http://{sidecar_host}:{args.sidecar_2_port}/health'
                    if not wait_for_service(s2_health, timeout_seconds=300, label='Sidecar-2'):
                        tail_log(os.path.join(args.log_dir, 'sidecar_2.log'))
                        logger.warning('Continuing without sidecar')
                        use_sidecar = False

                if use_sidecar:
                    sidecar_1_url = f'http://{sidecar_host}:{args.sidecar_1_port}'
                    sidecar_2_url = (f'http://{sidecar_host}:{args.sidecar_2_port}'
                                     if has_teacher_2 else '')

    total_gpus = detect_total_gpus()
    reserved_gpus = 2
    train_gpus = total_gpus - reserved_gpus
    train_gpu_ids = ','.join(str(i) for i in range(reserved_gpus, total_gpus))

    if is_multi_node:
        logger.info(f'[Worker {args.node_rank}] GPU 0-1 for vLLM+Sidecar, '
                    f'GPU {reserved_gpus}-{total_gpus-1} for training')
    elif args.node_rank == 0:
        logger.info(f'[Single-node] GPU 0-1 for vLLM+Sidecar, GPU {reserved_gpus}-{total_gpus-1} for training')

    logger.info(f'Training on GPUs: {train_gpu_ids} ({train_gpus} GPUs per node)')

    return {
        'teacher_1_url': teacher_1_url,
        'teacher_2_url': teacher_2_url,
        'sidecar_1_url': sidecar_1_url,
        'sidecar_2_url': sidecar_2_url,
        'cuda_visible_devices': train_gpu_ids,
        'train_nproc_per_node': train_gpus,
        'use_sidecar': use_sidecar,
        'pids': pids,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Launch vLLM Teacher & Sidecar services for distillation training',
    )

    parser.add_argument('--external-teachers', type=str, default='false',
                        help='Use pre-deployed external vLLM/Sidecar services')

    parser.add_argument('--vllm-teacher-1-external-url', type=str, default='')
    parser.add_argument('--vllm-teacher-2-external-url', type=str, default='')
    parser.add_argument('--sidecar-1-external-url', type=str, default='')
    parser.add_argument('--sidecar-2-external-url', type=str, default='')

    parser.add_argument('--teacher-model-1', type=str, required=True)
    parser.add_argument('--teacher-model-2', type=str, default='')

    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--node-rank', type=int, default=0)
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--sidecar-remote-host', type=str, default='')
    parser.add_argument('--worker-addrs', type=str, default='')

    parser.add_argument('--vllm-teacher-1-host', type=str, default='127.0.0.1')
    parser.add_argument('--vllm-teacher-1-port', type=int, default=8001)
    parser.add_argument('--vllm-teacher-1-gpu', type=int, default=0)
    parser.add_argument('--vllm-teacher-1-max-model-len', type=int, default=32768)
    parser.add_argument('--vllm-teacher-2-host', type=str, default='127.0.0.1')
    parser.add_argument('--vllm-teacher-2-port', type=int, default=8002)
    parser.add_argument('--vllm-teacher-2-gpu', type=int, default=1)
    parser.add_argument('--vllm-teacher-2-max-model-len', type=int, default=32768)
    parser.add_argument('--vllm-gpu-memory-utilization', type=float, default=0.75)
    parser.add_argument('--vllm-sidecar-coexist-gpu-util', type=float, default=0.4,
                        help='vLLM GPU util when co-located with Sidecar (single-node)')

    parser.add_argument('--use-sidecar', type=str, default='true')
    parser.add_argument('--sidecar-1-port', type=int, default=8011)
    parser.add_argument('--sidecar-2-port', type=int, default=8012)
    parser.add_argument('--sidecar-1-gpu', type=int, default=0)
    parser.add_argument('--sidecar-2-gpu', type=int, default=1)
    parser.add_argument('--sidecar-gpu-memory-fraction', type=float, default=0.75)
    parser.add_argument('--sidecar-coexist-gpu-memory-fraction', type=float, default=0.4,
                        help='Sidecar GPU mem fraction when co-located with vLLM (single-node)')
    parser.add_argument('--sidecar-enable-thinking', type=str, default='false',
                        help='Enable thinking mode in Sidecar chat template (for Qwen3). '
                             'Should match the student model enable_thinking training config.')
    parser.add_argument('--sidecar-max-length', type=int, default=40960,
                        help='Maximum sequence length the force-decode sidecar tokenizes: '
                             'must cover prompt + full debate transcript + student response '
                             '(default 40960 covers both ToolACE and code configurations).')

    parser.add_argument('--log-dir', type=str, required=True)

    args = parser.parse_args()

    args.external_teachers = args.external_teachers.lower() in ('true', '1', 'yes')
    args.use_sidecar = args.use_sidecar.lower() in ('true', '1', 'yes')
    args.sidecar_enable_thinking = args.sidecar_enable_thinking.lower() in ('true', '1', 'yes')

    return args


def main():
    args = parse_args()

    if args.external_teachers:
        result = launch_external_mode(args)
    else:
        result = launch_local_mode(args)

    output_json = json.dumps(result, ensure_ascii=False)
    print(output_json)


if __name__ == '__main__':
    main()
