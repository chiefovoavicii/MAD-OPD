# Copyright (c) ModelScope Contributors. All rights reserved.
"""
vLLM Teacher Manager for Online Distillation (HTTP Server Mode)

This module provides a VLLMTeacherManager class that communicates with
independently deployed vLLM servers via OpenAI-compatible HTTP API.

Key features:
1. Teacher-1 (14B) and Teacher-2 (8B) run as independent vLLM servers
2. Each teacher uses a dedicated GPU (e.g., 1x H100 each)
3. Communication via standard OpenAI-compatible HTTP API
4. Force Decoding via prompt_logprobs API
5. Completely decoupled from training torch.distributed process groups
6. No TP coordination issues, no new_group() deadlocks

Architecture:
- Teacher-1 (14B): vLLM server on GPU-A, e.g., http://localhost:8001
- Teacher-2 (8B):  vLLM server on GPU-B, e.g., http://localhost:8002
- Training process sends HTTP requests to both servers
- Only rank 0 sends requests; results are broadcast to all ranks
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from swift.utils import get_logger

logger = get_logger()

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

try:
    import requests as _http_requests
    _requests_available = True
except ImportError:
    _requests_available = False

try:
    from safetensors.torch import load as safetensors_load
    _safetensors_available = True
except ImportError:
    _safetensors_available = False


class VLLMTeacherManager:
    """
    Manages dual vLLM teacher servers for online distillation.

    Communicates with independently deployed vLLM servers via HTTP.
    Only rank 0 sends requests; results are broadcast to all ranks
    via torch.distributed.
    """

    def __init__(
        self,
        teacher_1_url: str,
        teacher_2_url: Optional[str] = None,
        tokenizer=None,
        max_model_len: int = 16384,
        max_model_len_1: Optional[int] = None,
        max_model_len_2: Optional[int] = None,
        teacher_1_model_name: Optional[str] = None,
        teacher_2_model_name: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        sidecar_1_url: Optional[str] = None,
        sidecar_2_url: Optional[str] = None,
        sidecar_max_length: int = 40960,
        sidecar_top_k: Optional[int] = None,
    ):
        """
        Initialize vLLM teacher manager with HTTP server endpoints.

        Args:
            teacher_1_url: Base URL of Teacher-1 vLLM server (e.g., 'http://localhost:8001/v1')
            teacher_2_url: Base URL of Teacher-2 vLLM server (e.g., 'http://localhost:8002/v1')
            tokenizer: Shared tokenizer instance
            max_model_len: Default maximum sequence length (fallback for both teachers)
            max_model_len_1: Maximum sequence length for Teacher-1 (overrides max_model_len)
            max_model_len_2: Maximum sequence length for Teacher-2 (overrides max_model_len)
            teacher_1_model_name: Model name for Teacher-1 (auto-detected if None)
            teacher_2_model_name: Model name for Teacher-2 (auto-detected if None)
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            sidecar_1_url: Base URL of Teacher-1 Force Decode sidecar (e.g., 'http://localhost:8011')
            sidecar_2_url: Base URL of Teacher-2 Force Decode sidecar (e.g., 'http://localhost:8012')
            sidecar_max_length: Maximum sequence length for sidecar force decode requests
        """
        if not _openai_available:
            raise ImportError(
                'openai package is required for vLLM teacher server mode. '
                'Install with: pip install openai'
            )

        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        self.max_model_len_1 = max_model_len_1 or max_model_len
        self.max_model_len_2 = max_model_len_2 or max_model_len
        self.sidecar_max_length = sidecar_max_length
        self.sidecar_top_k = sidecar_top_k
        self.timeout = timeout
        self.max_retries = max_retries

        # Force Decode sidecar URLs (for full-vocabulary logits)
        self.sidecar_1_url = sidecar_1_url
        self.sidecar_2_url = sidecar_2_url
        self.use_sidecar = (sidecar_1_url is not None) and _requests_available and _safetensors_available

        if self.use_sidecar and not _safetensors_available:
            logger.warning(
                '[VLLMTeacherManager] safetensors not available, falling back to prompt_logprobs. '
                'Install with: pip install safetensors'
            )
            self.use_sidecar = False
        # Rank info & node-local process group (multi-node support)
        # In multi-node mode, each node has its own vLLM/Sidecar on GPU 0-1.
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', self.world_size))
        self.node_rank = self.global_rank // self.local_world_size if self.local_world_size > 0 else 0
        self.num_nodes = self.world_size // self.local_world_size if self.local_world_size > 0 else 1

        # Each node's local_rank=0 is the "main rank" that sends HTTP requests
        self.is_main_rank = (self.local_rank == 0)

        # Create node-local process group for intra-node broadcast/allgather
        self.node_group = None
        self.node_src_rank = 0  # global rank of local_rank=0 within this node
        if dist.is_initialized() and self.num_nodes > 1:
            for node_idx in range(self.num_nodes):
                node_start = node_idx * self.local_world_size
                node_ranks = list(range(node_start, node_start + self.local_world_size))
                group = dist.new_group(ranks=node_ranks)
                if self.global_rank in node_ranks:
                    self.node_group = group
                    self.node_src_rank = node_start
            logger.info(
                f'[VLLMTeacherManager] Multi-node mode: node_rank={self.node_rank}, '
                f'node_src_rank={self.node_src_rank}, local_rank={self.local_rank}, '
                f'is_main_rank={self.is_main_rank}'
            )
        else:
            # Single-node: node_group=None → _node_broadcast falls back to global broadcast(src=0)
            self.node_src_rank = 0

        # Initialize OpenAI clients (only on rank 0 to avoid redundant connections)
        self.teacher_1_client = None
        self.teacher_2_client = None
        self.teacher_1_model = teacher_1_model_name
        self.teacher_2_model = teacher_2_model_name

        if self.is_main_rank:
            self.teacher_1_client = OpenAI(
                api_key='EMPTY',
                base_url=teacher_1_url,
                timeout=timeout,
                max_retries=max_retries,
            )
            if self.teacher_1_model is None:
                self.teacher_1_model = self._detect_model_name(self.teacher_1_client, 'teacher_1')

            if teacher_2_url is not None:
                self.teacher_2_client = OpenAI(
                    api_key='EMPTY',
                    base_url=teacher_2_url,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                if self.teacher_2_model is None:
                    self.teacher_2_model = self._detect_model_name(self.teacher_2_client, 'teacher_2')

        # Broadcast model names within node
        if dist.is_initialized():
            model_names = [self.teacher_1_model, self.teacher_2_model] if self.is_main_rank else [None, None]
            self._node_broadcast(model_names)
            self.teacher_1_model, self.teacher_2_model = model_names

        logger.info(
            f'[VLLMTeacherManager] rank={self.global_rank} | '
            f'teacher_1={teacher_1_url} (model={self.teacher_1_model}, max_len={self.max_model_len_1}) | '
            f'teacher_2={teacher_2_url} (model={self.teacher_2_model}, max_len={self.max_model_len_2}) | '
            f'sidecar_1={sidecar_1_url} | sidecar_2={sidecar_2_url} | '
            f'use_sidecar={self.use_sidecar} | mode=http_server | '
            f'num_nodes={self.num_nodes} | node_rank={self.node_rank}'
        )

    def _detect_model_name(self, client: 'OpenAI', name: str) -> str:
        """Auto-detect model name from vLLM server."""
        try:
            models = client.models.list()
            model_name = models.data[0].id
            logger.info(f'[VLLMTeacherManager] {name} model detected: {model_name}')
            return model_name
        except Exception as exc:
            logger.warning(f'[VLLMTeacherManager] Failed to detect {name} model: {exc}')
            return name

    def _wait_for_server(self, client: 'OpenAI', name: str, max_wait: float = 600.0):
        """Wait for vLLM server to be ready."""
        if client is None:
            return
        start = time.time()
        while time.time() - start < max_wait:
            try:
                client.models.list()
                logger.info(f'[VLLMTeacherManager] {name} server is ready')
                return
            except Exception:
                time.sleep(5)
                logger.info(f'[VLLMTeacherManager] Waiting for {name} server...')
        raise TimeoutError(f'{name} server not ready after {max_wait}s')

    def _node_broadcast(self, data_list: list):
        """
        Broadcast data from this node's local_rank=0 to all ranks in the same node.
        Single-node: falls back to global broadcast with src=0.
        """
        if self.node_group is not None:
            dist.broadcast_object_list(data_list, src=self.node_src_rank, group=self.node_group)
        else:
            dist.broadcast_object_list(data_list, src=0)

    def _node_allgather(self, local_obj):
        """
        Gather objects from all ranks within the same node.
        Single-node: falls back to global all_gather.
        """
        if self.node_group is not None:
            gathered = [None] * self.local_world_size
            dist.all_gather_object(gathered, local_obj, group=self.node_group)
            return gathered
        else:
            gathered = [None] * self.world_size
            dist.all_gather_object(gathered, local_obj)
            return gathered
    # Generation via OpenAI API
    def _generate_via_server(
        self,
        client: 'OpenAI',
        model_name: str,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
    ) -> List[str]:
        """
        Generate text via vLLM server's OpenAI-compatible API.
        Uses ThreadPoolExecutor to send all requests concurrently,
        leveraging vLLM's continuous batching for maximum throughput.
        Only called on rank 0.
        """
        def _single_generate(prompt: str) -> str:
            try:
                extra = {'repetition_penalty': 1.1}
                if top_k > 0:
                    extra['top_k'] = top_k
                response = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body=extra,
                )
                result = response.choices[0].text.strip()
                logger.debug(
                    f'[vLLM Generate] model={model_name} | '
                    f'prompt: {prompt[-500:]} | output: {result[-500:]}'
                )
                return result
            except Exception as exc:
                logger.warning(f'[VLLMTeacherManager] Generation failed: {exc}')
                return ''

        if len(prompts) <= 1:
            return [_single_generate(p) for p in prompts]

        with ThreadPoolExecutor(max_workers=min(len(prompts), 16)) as executor:
            futures = [executor.submit(_single_generate, p) for p in prompts]
            results = [f.result() for f in futures]
        return results

    def generate(
        self,
        engine_name: str,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
    ) -> List[str]:
        """
        Generate text using a teacher server.
        Only rank 0 sends requests; results are broadcast to all ranks.
        """
        if self.is_main_rank:
            client = self.teacher_1_client if engine_name == 'teacher_1' else self.teacher_2_client
            model_name = self.teacher_1_model if engine_name == 'teacher_1' else self.teacher_2_model
            if client is None:
                results = [''] * len(prompts)
            else:
                results = self._generate_via_server(
                    client, model_name, prompts, max_tokens, temperature, top_p, top_k
                )
        else:
            results = None

        # Broadcast results to all ranks
        if dist.is_initialized():
            data = [results] if self.is_main_rank else [None]
            self._node_broadcast(data)
            results = data[0]

        return results
    # Force Decoding via prompt_logprobs
    def _force_decode_via_server(
        self,
        client: 'OpenAI',
        model_name: str,
        context_texts: List[str],
        student_output_texts: List[str],
        device: str = 'cuda',
    ) -> List[Optional[torch.Tensor]]:
        """
        Force Decoding: get teacher logits at student output positions.

        Uses vLLM's prompt_logprobs feature via completions API.
        All requests are sent concurrently via ThreadPoolExecutor to leverage
        vLLM's continuous batching for maximum throughput.

        Only called on rank 0.
        """
        vocab_size = len(self.tokenizer)

        def _single_force_decode(context_text: str, student_text: str) -> Optional[torch.Tensor]:
            try:
                # Tokenize context and student separately, then concatenate at token level
                # to avoid BPE merge effects that cause token boundary misalignment
                context_ids = self.tokenizer(context_text, add_special_tokens=False)['input_ids']
                student_ids = self.tokenizer(student_text, add_special_tokens=False)['input_ids']
                context_length = len(context_ids)

                # Concatenate at token level and pass token_ids directly to vLLM
                full_ids = context_ids + student_ids

                # Request with prompt_logprobs
                response = client.completions.create(
                    model=model_name,
                    prompt=full_ids,
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=1,
                    extra_body={'prompt_logprobs': 1},
                )

                # Extract prompt logprobs from response
                prompt_logprobs = None
                if hasattr(response.choices[0], 'prompt_logprobs'):
                    prompt_logprobs = response.choices[0].prompt_logprobs

                if prompt_logprobs is None:
                    logger.warning(
                        '[VLLMTeacherManager] prompt_logprobs not available in response, '
                        'force decode returning None'
                    )
                    return None

                # Build sparse logits tensor from prompt_logprobs
                full_length = len(prompt_logprobs)
                student_start = context_length
                student_end = full_length

                if student_start >= student_end:
                    return None

                # Create on CPU to avoid GPU OOM on rank 0
                student_logits = torch.zeros(
                    1, student_end - student_start, vocab_size,
                    dtype=torch.bfloat16, device='cpu'
                )

                for pos_idx, abs_pos in enumerate(range(student_start, student_end)):
                    if abs_pos < len(prompt_logprobs) and prompt_logprobs[abs_pos] is not None:
                        logprob_dict = prompt_logprobs[abs_pos]
                        for token_str, logprob_value in logprob_dict.items():
                            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                            if token_ids and token_ids[0] < vocab_size:
                                if isinstance(logprob_value, (int, float)):
                                    student_logits[0, pos_idx, token_ids[0]] = logprob_value
                                elif hasattr(logprob_value, 'logprob'):
                                    student_logits[0, pos_idx, token_ids[0]] = logprob_value.logprob

                return student_logits

            except Exception as exc:
                logger.warning(f'[VLLMTeacherManager] Force decode failed: {exc}')
                return None

        num_samples = len(context_texts)
        if num_samples <= 1:
            return [_single_force_decode(c, s) for c, s in zip(context_texts, student_output_texts)]

        with ThreadPoolExecutor(max_workers=min(num_samples, 16)) as executor:
            futures = [
                executor.submit(_single_force_decode, ctx, stu)
                for ctx, stu in zip(context_texts, student_output_texts)
            ]
            results = [f.result() for f in futures]
        return results
    # Force Decoding via Sidecar (full-vocabulary logits)
    def _force_decode_via_sidecar(
        self,
        sidecar_url: str,
        context_texts: List[str],
        student_output_texts: List[str],
        device: str = 'cuda',
        student_output_token_ids: Optional[List[List[int]]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """
        Force Decoding via sidecar server: get full-vocabulary teacher logits.

        The sidecar server runs the same teacher model and performs a standard
        HuggingFace forward pass, returning raw logits as safetensors binary.

        The sidecar internally:
        1. Applies chat template to context_text as user message
        2. Appends student_output_text after assistant generation prompt
        3. Runs model.forward() to get full logits
        4. Extracts logits at student output positions
        5. Returns [1, response_len, vocab_size] tensor in bfloat16

        When student_output_token_ids is provided, the sidecar uses exact token IDs
        instead of re-tokenizing text, ensuring perfect alignment with student tokens.

        Only called on rank 0.

        Args:
            sidecar_url: Base URL of the sidecar server (e.g., 'http://localhost:8011')
            context_texts: List of context strings (prompt + optional debate history)
            student_output_texts: List of student output strings
            device: Target device for returned tensors
            student_output_token_ids: Optional list of token ID lists for precise alignment

        Returns:
            List of logits tensors [1, response_len, vocab_size] or None per sample
        """
        endpoint = f'{sidecar_url}/force_decode'

        request_body = {
            'context_texts': context_texts,
            'student_output_texts': student_output_texts,
            'max_length': self.sidecar_max_length,
        }
        if student_output_token_ids is not None:
            request_body['student_output_token_ids'] = student_output_token_ids
        if self.sidecar_top_k is not None and self.sidecar_top_k > 0:
            request_body['top_k'] = self.sidecar_top_k

        logger.debug(
            f'[Sidecar Request] endpoint={endpoint}, num_samples={len(context_texts)}, '
            f'context[0]: {context_texts[0][-500:]} | student[0]: {student_output_texts[0][-500:]}'
        )

        try:
            response = _http_requests.post(
                endpoint,
                json=request_body,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                f'[VLLMTeacherManager] Sidecar force decode request failed: {exc}. '
                f'Falling back to prompt_logprobs API.'
            )
            return None  # Signal caller to fall back

        # Deserialize safetensors binary response
        try:
            binary_data = response.content
            tensors = safetensors_load(binary_data)
        except Exception as exc:
            logger.warning(
                f'[VLLMTeacherManager] Failed to deserialize sidecar response: {exc}. '
                f'Falling back to prompt_logprobs API.'
            )
            return None

        # Reconstruct results list.
        # Keep tensors on CPU to avoid GPU OOM on rank 0; broadcast_object_list
        # serializes CPU tensors without allocating GPU memory, and
        # scatter_results_to_ranks moves only this rank's sample to GPU later.
        is_top_k_mode = '__top_k__' in tensors
        results = []
        for idx in range(len(context_texts)):
            if is_top_k_mode:
                # top_k sparse format: values [1, seq_len, k] + indices [1, seq_len, k]
                val_key = f'logits_values_{idx}'
                idx_key = f'logits_indices_{idx}'
                if val_key in tensors and idx_key in tensors:
                    values = tensors[val_key].cpu()
                    indices = tensors[idx_key].cpu()
                    if values.shape[1] == 0:
                        results.append(None)
                    else:
                        # Store as dict; JSD loss will handle sparse format
                        results.append({'values': values, 'indices': indices})
                else:
                    results.append(None)
            else:
                # Full vocab format: [1, seq_len, vocab_size]
                key = f'logits_{idx}'
                if key in tensors:
                    tensor = tensors[key]
                    # Empty tensor (shape [1, 0, 1]) indicates failure for that sample
                    if tensor.shape[1] == 0:
                        results.append(None)
                    else:
                        results.append(tensor.cpu())
                else:
                    results.append(None)

        num_valid = sum(1 for r in results if r is not None)
        logger.info(
            f'[VLLMTeacherManager] Sidecar force decode: {num_valid}/{len(results)} samples '
            f'with full-vocab logits (binary size={len(binary_data)} bytes)'
        )

        return results

    def _wait_for_sidecar(self, sidecar_url: str, name: str, max_wait: float = 600.0):
        """Wait for sidecar server to be ready."""
        start = time.time()
        while time.time() - start < max_wait:
            try:
                resp = _http_requests.get(f'{sidecar_url}/health', timeout=5)
                if resp.status_code == 200:
                    logger.info(f'[VLLMTeacherManager] {name} sidecar is ready')
                    return
            except Exception:
                pass
            time.sleep(5)
            logger.info(f'[VLLMTeacherManager] Waiting for {name} sidecar at {sidecar_url}...')
        raise TimeoutError(f'{name} sidecar not ready after {max_wait}s')

    def force_decode_batch(
        self,
        engine_name: str,
        context_texts: List[str],
        student_output_texts: List[str],
        device: str = 'cuda',
    ) -> List[Optional[torch.Tensor]]:
        """
        Batch force decoding via teacher server.
        Only rank 0 sends requests; results are broadcast to all ranks.

        Priority:
        1. If sidecar is available, use it for full-vocabulary logits (same quality as DeepSpeed)
        2. Otherwise, fall back to vLLM prompt_logprobs API (sparse, degraded quality)
        """
        if self.is_main_rank:
            sidecar_url = self.sidecar_1_url if engine_name == 'teacher_1' else self.sidecar_2_url
            client = self.teacher_1_client if engine_name == 'teacher_1' else self.teacher_2_client
            model_name = self.teacher_1_model if engine_name == 'teacher_1' else self.teacher_2_model

            results = None

            # Priority 1: Try sidecar for full-vocabulary logits
            if self.use_sidecar and sidecar_url is not None:
                results = self._force_decode_via_sidecar(
                    sidecar_url, context_texts, student_output_texts, device
                )
                if results is not None:
                    logger.info(
                        f'[VLLMTeacherManager] Using sidecar full-vocab logits for {engine_name}'
                    )

            # If sidecar is configured but failed, do NOT fall back to vLLM prompt_logprobs API.
            # Fallback would cause vLLM to compute log_softmax over full vocabulary for long
            # sequences, leading to GPU OOM and teacher vLLM crash.
            if results is None:
                if self.use_sidecar and sidecar_url is not None:
                    logger.warning(
                        f'[VLLMTeacherManager] Sidecar failed for {engine_name}, '
                        f'returning [None]*N instead of falling back to prompt_logprobs '
                        f'(which would risk teacher vLLM OOM on long sequences)'
                    )
                    results = [None] * len(context_texts)
                elif client is None:
                    results = [None] * len(context_texts)
                else:
                    logger.info(
                        f'[VLLMTeacherManager] Using prompt_logprobs fallback for {engine_name} '
                        f'(sparse logits, degraded distillation quality)'
                    )
                    results = self._force_decode_via_server(
                        client, model_name, context_texts, student_output_texts, device
                    )
        else:
            results = None

        # Broadcast results to all ranks
        if dist.is_initialized():
            data = [results] if self.is_main_rank else [None]
            self._node_broadcast(data)
            results = data[0]

        return results
    # Parallel generation: two teachers concurrently
    def _generate_on_rank0(
        self,
        engine_name: str,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
    ) -> List[str]:
        """
        Generate text on rank 0 only, WITHOUT broadcast.
        Used as a building block for parallel generation.
        """
        client = self.teacher_1_client if engine_name == 'teacher_1' else self.teacher_2_client
        model_name = self.teacher_1_model if engine_name == 'teacher_1' else self.teacher_2_model
        if client is None:
            return [''] * len(prompts)
        return self._generate_via_server(
            client, model_name, prompts, max_tokens, temperature, top_p, top_k
        )

    def generate_parallel(
        self,
        engine_1_name: str,
        prompts_1: List[str],
        max_tokens_1: int,
        engine_2_name: str,
        prompts_2: List[str],
        max_tokens_2: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate text from two teachers concurrently using ThreadPoolExecutor.
        Only rank 0 sends requests; results are broadcast to all ranks.

        Returns:
            (results_1, results_2): generation results from both teachers
        """
        if self.is_main_rank:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_1 = executor.submit(
                    self._generate_on_rank0,
                    engine_1_name, prompts_1, max_tokens_1, temperature, top_p, top_k,
                )
                future_2 = executor.submit(
                    self._generate_on_rank0,
                    engine_2_name, prompts_2, max_tokens_2, temperature, top_p, top_k,
                )
                results_1 = future_1.result()
                results_2 = future_2.result()
        else:
            results_1 = None
            results_2 = None

        # Broadcast both results in a single call
        if dist.is_initialized():
            data = [results_1, results_2] if self.is_main_rank else [None, None]
            self._node_broadcast(data)
            results_1, results_2 = data

        return results_1, results_2
    # Parallel Force Decoding: two teachers concurrently
    def _force_decode_on_rank0(
        self,
        engine_name: str,
        context_texts: List[str],
        student_output_texts: List[str],
        device: str = 'cuda',
        student_output_token_ids: Optional[List[List[int]]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """
        Force decode on rank 0 only, WITHOUT broadcast.
        Used as a building block for parallel force decoding.
        """
        sidecar_url = self.sidecar_1_url if engine_name == 'teacher_1' else self.sidecar_2_url
        client = self.teacher_1_client if engine_name == 'teacher_1' else self.teacher_2_client
        model_name = self.teacher_1_model if engine_name == 'teacher_1' else self.teacher_2_model

        results = None

        # Priority 1: Try sidecar for full-vocabulary logits
        if self.use_sidecar and sidecar_url is not None:
            results = self._force_decode_via_sidecar(
                sidecar_url, context_texts, student_output_texts, device,
                student_output_token_ids=student_output_token_ids,
            )
            if results is not None:
                logger.info(
                    f'[VLLMTeacherManager] Using sidecar full-vocab logits for {engine_name}'
                )

        # If sidecar is configured but failed, do NOT fall back to vLLM prompt_logprobs API.
        # Fallback would cause vLLM to compute log_softmax over full vocabulary for long
        # sequences, leading to GPU OOM and teacher vLLM crash.
        if results is None:
            if self.use_sidecar and sidecar_url is not None:
                logger.warning(
                    f'[VLLMTeacherManager] Sidecar failed for {engine_name}, '
                    f'returning [None]*N instead of falling back to prompt_logprobs '
                    f'(which would risk teacher vLLM OOM on long sequences)'
                )
                results = [None] * len(context_texts)
            elif client is None:
                results = [None] * len(context_texts)
            else:
                logger.info(
                    f'[VLLMTeacherManager] Using prompt_logprobs fallback for {engine_name} '
                    f'(sparse logits, degraded distillation quality)'
                )
                results = self._force_decode_via_server(
                    client, model_name, context_texts, student_output_texts, device
                )

        return results

    def force_decode_parallel(
        self,
        context_texts_1: List[str],
        student_output_texts_1: List[str],
        context_texts_2: List[str],
        student_output_texts_2: List[str],
        device: str = 'cuda',
        student_output_token_ids: Optional[List[List[int]]] = None,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """
        Force decode from two teachers concurrently using ThreadPoolExecutor.
        Only rank 0 sends requests; results are broadcast to all ranks.

        Args:
            student_output_token_ids: Optional list of token ID lists for precise alignment.
                Both teachers receive the same student token IDs.

        Returns:
            (teacher_1_logits, teacher_2_logits)
        """
        if self.is_main_rank:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_1 = executor.submit(
                    self._force_decode_on_rank0,
                    'teacher_1', context_texts_1, student_output_texts_1, device,
                    student_output_token_ids,
                )
                future_2 = executor.submit(
                    self._force_decode_on_rank0,
                    'teacher_2', context_texts_2, student_output_texts_2, device,
                    student_output_token_ids,
                )
                results_1 = future_1.result()
                results_2 = future_2.result()
        else:
            results_1 = None
            results_2 = None

        # Broadcast both results in a single call
        if dist.is_initialized():
            data = [results_1, results_2] if self.is_main_rank else [None, None]
            self._node_broadcast(data)
            results_1, results_2 = data

        return results_1, results_2

    # MAD Debate + Force Decode (all-in-one)
    @torch.no_grad()
    def batch_blind_mad_debate_and_force_decode(
        self,
        all_prompts: List[str],
        all_responses: List[str],
        all_response_token_ids: Optional[List[List[int]]] = None,
        rounds: int = 2,
        t1_max_tokens: int = 4096,
        t2_max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        device: str = 'cuda',
        toolace_mode: bool = False,
        code_mode: bool = False,
        dynamic_weights: bool = False,
    ) -> Optional[Dict]:
        """Blind parallel debate + batched force decoding (Paper §4).

        Round 1 is independent: each teacher answers the prompt without seeing
        the student reply or the other teacher. Rounds 2..R are cross-revision:
        every teacher is shown the full debate transcript so far and revises.
        Neither the student response nor any reference answer is injected into
        the debate prompt.

        Weighting:
        - ``dynamic_weights=False`` fixes the combination weights at 0.5/0.5.
        - ``dynamic_weights=True`` asks each teacher for a ``<<CONFIDENCE=N>>``
          tag in its last-round reply and applies Paper Eq. 7 softmax
          reweighting over the two final-round confidences.

        Pipeline:
          1. Round 1   — independent answers (prompt only).
          2. Rounds 2..R — cross-revision seeing the full debate history.
          3. Build the final debate transcript H.
          4. Force decoding — both teachers score the student's on-policy reply
             conditioned on ``prompt + H + role header``; full-vocab logits are
             returned at the student-response positions.

        Args:
            all_prompts: gathered prompts from every rank.
            all_responses: gathered student responses (force-decoding only).
            rounds: number of debate rounds R (≥1).
            t1_max_tokens / t2_max_tokens: generation budget per round.
            temperature: sampling temperature for debate generation.
            device: target device for the returned logits.

        Returns:
            A dict broadcast to every rank:
            {
                'teacher_a_logits': List[Tensor],
                'teacher_b_logits': List[Tensor],
                'weights':          List[Tuple[float, float]],
                'debate_histories': List[str],
            }
        """
        num_samples = len(all_prompts)
        all_debate_parts = [[] for _ in range(num_samples)]

        # Require each teacher to emit a <<CONFIDENCE=N>> tag when dynamic
        # weighting is active. Using this special token avoids false positives
        # from natural-text phrases that may appear in code or reasoning.
        confidence_suffix = (
            '\n\nIMPORTANT: At the very end of your response, output your confidence '
            'using EXACTLY this format: <<CONFIDENCE=N>> where N is an integer from 0 to 100. '
            'Do not include any other text after this tag.'
        ) if dynamic_weights else ''

        # ---------- Round 1: both teachers answer independently from the prompt only ----------
        if toolace_mode:
            round1_prompts = [
                f'You are an expert in AI tool-use and function calling. '
                f'Please analyze the following user request and provide the best tool-use response.\n\n'
                f'User request: {p}\n\nYour response:{confidence_suffix}'
                for p in all_prompts
            ]
        elif code_mode:
            round1_prompts = [
                f'You are an expert programmer specializing in competitive programming and code golf. '
                f'Please solve the following programming challenge. '
                f'Focus on algorithm correctness, edge case handling, and code efficiency.\n\n'
                f'Problem:\n{p}\n\nYour code solution:{confidence_suffix}'
                for p in all_prompts
            ]
        else:
            round1_prompts = [
                f'You are a knowledgeable expert. Please solve the following problem step by step.\n\n'
                f'Question: {p}\n\nYour solution:{confidence_suffix}'
                for p in all_prompts
            ]

        # Round 1: Two teachers generate concurrently (parallel)
        teacher_1_answers, teacher_2_answers = self.generate_parallel(
            'teacher_1', round1_prompts, t1_max_tokens,
            'teacher_2', round1_prompts, t2_max_tokens,
            temperature, top_p, top_k,
        )

        for i in range(num_samples):
            all_debate_parts[i].append(f'[Expert A Round 1]: {teacher_1_answers[i][:]}')
            all_debate_parts[i].append(f'[Expert B Round 1]: {teacher_2_answers[i][:]}')

        logger.info(f'[BlindDebate] Round 1 complete (parallel): Teacher-1 and Teacher-2 independently answered {num_samples} questions')

        # ---------- Rounds 2..R: cross-revision given full debate history ----------
        # Keep the last-round outputs of both teachers for confidence extraction
        # under dynamic_weights.
        last_round_a_outputs = teacher_1_answers
        last_round_b_outputs = teacher_2_answers

        for round_idx in range(2, rounds + 1):
            debate_history_strs = ['\n'.join(parts) for parts in all_debate_parts]

            if toolace_mode:
                round_a_prompts = [
                    f'{p}\n\n'
                    f'The following is the debate history between two experts on this request:\n\n'
                    f'{dh}\n\n'
                    f'You are Expert A. You don\'t have to agree with the other expert\'s response. '
                    f'Please continue the debate about the best tool-use strategy. '
                    f'Having seen the debate history, please reconsider and provide your '
                    f'improved response focusing on tool selection accuracy, parameter correctness, '
                    f'and call sequence optimization.{confidence_suffix}'
                    for p, dh in zip(all_prompts, debate_history_strs)
                ]
                round_b_prompts = [
                    f'{p}\n\n'
                    f'The following is the debate history between two experts on this request:\n\n'
                    f'{dh}\n\n'
                    f'You are Expert B. You don\'t have to agree with the other expert\'s response. '
                    f'Please continue the debate about the best tool-use strategy. '
                    f'Having seen the debate history, please reconsider and provide your '
                    f'improved response focusing on tool selection accuracy, parameter correctness, '
                    f'and call sequence optimization.{confidence_suffix}'
                    for p, dh in zip(all_prompts, debate_history_strs)
                ]
            elif code_mode:
                round_a_prompts = [
                    f'{p}\n\n'
                    f'The following is the debate history between two expert programmers on this challenge:\n\n'
                    f'{dh}\n\n'
                    f'You are Expert A. You don\'t have to agree with the other expert\'s solution. '
                    f'Please continue the debate about the best approach. '
                    f'Having seen the debate history, please reconsider and provide your '
                    f'improved solution focusing on algorithm correctness, edge case handling, '
                    f'time/space complexity, and code conciseness.{confidence_suffix}'
                    for p, dh in zip(all_prompts, debate_history_strs)
                ]
                round_b_prompts = [
                    f'{p}\n\n'
                    f'The following is the debate history between two expert programmers on this challenge:\n\n'
                    f'{dh}\n\n'
                    f'You are Expert B. You don\'t have to agree with the other expert\'s solution. '
                    f'Please continue the debate about the best approach. '
                    f'Having seen the debate history, please reconsider and provide your '
                    f'improved solution focusing on algorithm correctness, edge case handling, '
                    f'time/space complexity, and code conciseness.{confidence_suffix}'
                    for p, dh in zip(all_prompts, debate_history_strs)
                ]
            else:
                round_a_prompts = [
                    f'{p}\n\n'
                    f'The following is the debate history between two experts on this problem:\n\n'
                    f'{dh}\n\n'
                    f'You are Expert A, you don\'t have to agree with the other expert\'s solution.'
                    f'please continue the debate about this problem.'
                    f'Having seen the debate history, please reconsider and provide your '
                    f'response with improved reasoning.{confidence_suffix}'
                    for p, dh in zip(all_prompts, debate_history_strs)
                ]
                round_b_prompts = [
                    f'{p}\n\n'
                    f'The following is the debate history between two experts on this problem:\n\n'
                    f'{dh}\n\n'
                    f'You are Expert B, you don\'t have to agree with the other expert\'s solution.'
                    f'please continue the debate about this problem.'
                    f'Having seen the debate history, please reconsider and provide your '
                    f'response with improved reasoning.{confidence_suffix}'
                    for p, dh in zip(all_prompts, debate_history_strs)
                ]

            # Two teachers revise concurrently (parallel)
            round_a_outputs, round_b_outputs = self.generate_parallel(
                'teacher_1', round_a_prompts, t1_max_tokens,
                'teacher_2', round_b_prompts, t2_max_tokens,
                temperature, top_p, top_k,
            )

            for i in range(num_samples):
                all_debate_parts[i].append(f'[Expert A Round {round_idx}]: {round_a_outputs[i][:]}')
                all_debate_parts[i].append(f'[Expert B Round {round_idx}]: {round_b_outputs[i][:]}')

            # Diagnostic: detect empty teacher replies per round.
            r_empty_a = sum(1 for r in round_a_outputs if not r.strip())
            r_empty_b = sum(1 for r in round_b_outputs if not r.strip())
            if num_samples > 0:
                r_lens_a = [len(r) for r in round_a_outputs[:3]]
                r_lens_b = [len(r) for r in round_b_outputs[:3]]
                r_prompt_lens = [len(p) for p in round_a_prompts[:3]]
                logger.info(
                    f'[BlindDebate] Round {round_idx} complete | empty: A={r_empty_a}/{num_samples}, B={r_empty_b}/{num_samples} | '
                    f'sample lens A={r_lens_a}, B={r_lens_b} | prompt lens={r_prompt_lens}'
                )
                for tag, revised_list in [('A', round_a_outputs), ('B', round_b_outputs)]:
                    for idx, r in enumerate(revised_list[:3]):
                        if r.strip():
                            logger.info(f'[BlindDebate-R{round_idx}] Teacher-{tag} sample[{idx}] tail(200): ...{r[-200:]}')
                            break
                    else:
                        logger.warning(f'[BlindDebate-R{round_idx}] Teacher-{tag} ALL first 3 samples are EMPTY')

            last_round_a_outputs = round_a_outputs
            last_round_b_outputs = round_b_outputs

        # ---------- Build the final debate transcript H ----------
        all_debate_histories = [
            '[Debate Start]\n' + '\n'.join(parts) + '\n[Debate End]'
            for parts in all_debate_parts
        ]
        # Dynamic weights (Paper Eq. 7):
        #     w_k = softmax(c_k / 100 / tau_conf), tau_conf = 1.0.
        # When any confidence is missing, fall back to uniform weights.
        from .mad_opd_core import _softmax_confidence_weights
        if dynamic_weights:
            last_round = rounds
            all_weights = []
            for i in range(num_samples):
                score_a = self._extract_last_confidence(last_round_a_outputs[i])
                score_b = self._extract_last_confidence(last_round_b_outputs[i])
                if score_a >= 0 and score_b >= 0:
                    all_weights.append(_softmax_confidence_weights(score_a, score_b))
                else:
                    all_weights.append((0.5, 0.5))
            if num_samples > 0:
                sample_weights = all_weights[0]
                logger.info(f'[BlindDebate] Dynamic weights enabled (from Round {last_round}), sample[0]: alpha={sample_weights[0]:.3f}, beta={sample_weights[1]:.3f}')
                if sample_weights == (0.5, 0.5):
                    # Diagnostic: log the tail of the last-round outputs when
                    # confidence tags were missing.
                    for tag, outputs in [('A', last_round_a_outputs), ('B', last_round_b_outputs)]:
                        for idx, r in enumerate(outputs[:3]):
                            if r.strip():
                                logger.warning(f'[BlindDebate-DIAG] Teacher-{tag} Round {last_round} sample[{idx}] tail(300): ...{r[-300:]}')
                                break
                    conf_a = self._extract_last_confidence(last_round_a_outputs[0])
                    conf_b = self._extract_last_confidence(last_round_b_outputs[0])
                    logger.warning(f'[BlindDebate-DIAG] Per-output confidence: A={conf_a}, B={conf_b}')
        else:
            all_weights = [(0.5, 0.5)] * num_samples

        # ---------- Force decoding phase (parallel across both teachers) ----------
        # Force-decode context = prompt + identity + debate history + trailing cue.
        # Expert A and Expert B share the debate history but keep distinct identities.
        fd_contexts_a = []
        fd_contexts_b = []
        for i, (prompt, debate_history) in enumerate(zip(all_prompts, all_debate_histories)):
            if toolace_mode:
                context_a = (f'{prompt}\n\n'
                             f'You are Expert A. The following is the debate between you and other experts:\n\n'
                             f'{debate_history}\n\n'
                             f'Now based on the debate, provide the best tool-use response:')
                context_b = (f'{prompt}\n\n'
                             f'You are Expert B. The following is the debate between you and other experts:\n\n'
                             f'{debate_history}\n\n'
                             f'Now based on the debate, provide the best tool-use response:')
            elif code_mode:
                context_a = (f'{prompt}\n\n'
                             f'You are Expert A. The following is the debate between you and other experts:\n\n'
                             f'{debate_history}\n\n'
                             f'Now based on the debate results, please write the best code solution to this programming challenge:')
                context_b = (f'{prompt}\n\n'
                             f'You are Expert B. The following is the debate between you and other experts:\n\n'
                             f'{debate_history}\n\n'
                             f'Now based on the debate results, please write the best code solution to this programming challenge:')
            else:
                context_a = (f'{prompt}\n\n'
                             f'You are Expert A. The following is the debate between you and other experts:\n\n'
                             f'{debate_history}\n\n'
                             f'Now based on the debate results, please solve this problem again using your own method and process:')
                context_b = (f'{prompt}\n\n'
                             f'You are Expert B. The following is the debate between you and other experts:\n\n'
                             f'{debate_history}\n\n'
                             f'Now based on the debate results, please solve this problem again using your own method and process:')
            fd_contexts_a.append(context_a)
            fd_contexts_b.append(context_b)

        # Two teachers force decode concurrently
        all_t1_logits, all_t2_logits = self.force_decode_parallel(
            fd_contexts_a, all_responses,
            fd_contexts_b, all_responses,
            device,
            student_output_token_ids=all_response_token_ids,
        )

        return {
            'teacher_a_logits': all_t1_logits,
            'teacher_b_logits': all_t2_logits,
            'weights': all_weights,
            'debate_histories': all_debate_histories,
        }

    @staticmethod
    def _extract_section(debate_history: str, start_tag: str, end_tag: str) -> str:
        """Extract text between the *last* occurrence of start_tag and end_tag."""
        start_idx = debate_history.rfind(start_tag)
        if start_idx < 0:
            return ''
        end_idx = debate_history.find(end_tag, start_idx + len(start_tag))
        if end_idx < 0:
            return debate_history[start_idx + len(start_tag):]
        return debate_history[start_idx + len(start_tag):end_idx]

    @staticmethod
    def _extract_last_confidence(text: str) -> float:
        """Extract the last confidence score from a single teacher's text section.

        Only matches the special tag <<CONFIDENCE=N>> to avoid false positives
        from code/reasoning content. Returns -1 if no tag found.
        """
        scores = re.findall(r'<<CONFIDENCE=(\d+)>>', text)
        if scores:
            return float(scores[-1])
        return -1

    def _parse_weights(self, debate_history: str) -> Tuple[float, float]:
        """Parse confidence weights from debate history.

        Extracts <<CONFIDENCE=N>> tags from Expert A and Expert B sections.
        If either side is missing the tag, falls back to equal weights (0.5, 0.5).
        """
        # Extract each teacher's section independently
        section_a = self._extract_section(
            debate_history, '[Expert A Revised Answer]:', '[Expert B Revised Answer]:'
        )
        section_b = self._extract_section(
            debate_history, '[Expert B Revised Answer]:', '[Debate End]'
        )

        score_a = self._extract_last_confidence(section_a)
        score_b = self._extract_last_confidence(section_b)

        # Fallback: also try non-blind debate format "[Expert A Evaluation]" / "[Expert B Response]"
        if score_a < 0 or score_b < 0:
            section_a_alt = self._extract_section(
                debate_history, '[Expert A Evaluation]:', '[Expert B Response]:'
            )
            section_b_alt = self._extract_section(
                debate_history, '[Expert B Response]:', '[Debate End]'
            )
            if score_a < 0:
                score_a = self._extract_last_confidence(section_a_alt)
            if score_b < 0:
                score_b = self._extract_last_confidence(section_b_alt)

        if score_a >= 0 and score_b >= 0:
            from .mad_opd_core import _softmax_confidence_weights
            return _softmax_confidence_weights(score_a, score_b)

        # Missing <<CONFIDENCE=N>> tag from either side → uniform fallback.
        return (0.5, 0.5)
    # Scatter results to per-rank (simplified: results already on all ranks)
    def scatter_results_to_ranks(
        self,
        full_results: Optional[Dict],
        device: str = 'cuda',
    ) -> Dict:
        """Split the full debate result dict into this rank's sample slice.

        The blind-parallel debate + force-decode batch has already broadcast
        the complete result to every rank; here we just index by rank.
        """
        if full_results is None:
            return None

        my_rank = self.global_rank

        # When the gathered batch has fewer samples than world_size, every
        # extra rank falls back to sample 0 (single-sample batch fan-out).
        num_samples = len(full_results['teacher_a_logits'])
        sample_idx = my_rank if my_rank < num_samples else 0

        teacher_a_logits = full_results['teacher_a_logits'][sample_idx]
        teacher_b_logits = full_results['teacher_b_logits'][sample_idx]
        alpha, beta = full_results['weights'][sample_idx]
        debate_history = full_results['debate_histories'][sample_idx]

        # Move only the current rank's sample to GPU (logits are on CPU after broadcast)
        if teacher_a_logits is not None:
            teacher_a_logits = teacher_a_logits.to(device)
        if teacher_b_logits is not None:
            teacher_b_logits = teacher_b_logits.to(device)

        # Release all samples' CPU tensor references to free memory
        for i in range(num_samples):
            full_results['teacher_a_logits'][i] = None
            full_results['teacher_b_logits'][i] = None

        return {
            'teacher_a_logits': teacher_a_logits,
            'teacher_b_logits': teacher_b_logits,
            'alpha': alpha,
            'beta': beta,
            'debate_history': debate_history,
        }
    # Backward compatibility aliases
    def allgather_texts(self, local_texts: List[str]) -> List[List[str]]:
        """AllGather: collect text lists from every rank onto all ranks."""
        if not dist.is_initialized():
            return [local_texts]
        return self._node_allgather(local_texts)

    def broadcast_generation_results(
        self,
        results: List[str],
        source_ranks: List[int],
    ) -> List[str]:
        """Broadcast generation results from source rank to all ranks."""
        if not dist.is_initialized():
            return results
        if self.is_main_rank:
            broadcast_data = [results]
        else:
            broadcast_data = [None]
        self._node_broadcast(broadcast_data)
        return broadcast_data[0]

    def debate_loop(
        self,
        prompt: str,
        student_output: str,
        rounds: int = 2,
        temperature: float = 0.7,
    ) -> Tuple[str, float, float]:
        """Execute multi-agent debate (single sample convenience wrapper)."""
        results = self.batch_blind_mad_debate_and_force_decode(
            all_prompts=[prompt],
            all_responses=[student_output],
            rounds=rounds,
            temperature=temperature,
        )
        if results is not None:
            alpha, beta = results['weights'][0]
            debate_history = results['debate_histories'][0]
            return debate_history, alpha, beta
        return '', 0.5, 0.5

    def cleanup(self):
        """Clean up HTTP clients."""
        self.teacher_1_client = None
        self.teacher_2_client = None
        logger.info(f'[VLLMTeacherManager] Cleaned up on rank={self.global_rank}')