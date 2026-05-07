#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Force Decode Sidecar Server

A lightweight FastAPI service that runs alongside vLLM teacher servers,
providing full-vocabulary logits for knowledge distillation via Force Decoding.

Why a sidecar instead of using vLLM's prompt_logprobs API?
- vLLM's OpenAI-compatible API returns logprobs as JSON dicts
- Full vocab (~150K tokens) × seq_len positions = millions of JSON entries
- JSON serialization/deserialization is extremely slow for this volume
- This sidecar uses safetensors binary format for efficient tensor transfer

Architecture:
- Loads the same model as the corresponding vLLM server
- Shares the same GPU (vLLM server uses ~90% GPU mem, sidecar uses remaining)
- Actually, sidecar loads model separately with controlled GPU memory
- Returns raw logits (not logprobs) as safetensors binary, matching DeepSpeed mode

Endpoints:
- POST /force_decode: Force decode a batch of (context, student_output) pairs
- GET /health: Health check
"""
import argparse
import logging
import os
import sys
import time
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('force_decode_server')

app = FastAPI(title='Force Decode Sidecar Server')

# Global model reference (initialized on startup)
_model = None
_tokenizer = None
_device = None
_enable_thinking = False


class ForceDecodeRequest(BaseModel):
    """Request body for force decode endpoint."""
    context_texts: List[str]
    student_output_texts: List[str]
    student_output_token_ids: Optional[List[List[int]]] = None
    max_length: int = 40960
    top_k: Optional[int] = None  # If set, only return top-k logits (indices + values) to save bandwidth


class ForceDecodeResponse(BaseModel):
    """Metadata response (actual tensor data is in binary body)."""
    num_samples: int
    shapes: List[List[int]]
    dtype: str


def _load_model(model_path: str, device: str, gpu_memory_fraction: float = 0.45):
    """
    Load model using standard HuggingFace transformers for Force Decoding.

    We use HF model (not vLLM LLM class) because:
    1. We need raw logits tensor on GPU, not logprobs
    2. HF model.forward() returns logits directly as torch.Tensor
    3. No need for vLLM's sampling/scheduling overhead for pure forward pass
    4. Can precisely control GPU memory via max_memory parameter
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f'Loading model from {model_path} on {device}...')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Control GPU memory allocation
    gpu_id = int(device.split(':')[-1]) if ':' in device else 0
    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    max_mem_bytes = int(total_mem * gpu_memory_fraction)
    max_memory = {gpu_id: max_mem_bytes, 'cpu': '16GiB'}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={'': device},
        trust_remote_code=True,
        max_memory=max_memory,
    )
    model.eval()

    logger.info(f'Model loaded successfully on {device}')
    return model, tokenizer


def _apply_chat_template(tokenizer, user_content: str) -> str:
    """Apply chat template to format prompt for teacher model.

    Uses the global _enable_thinking flag (set via --enable-thinking CLI arg)
    to control whether Qwen3's thinking mode is activated in the chat template.
    This should match the student model's enable_thinking training config so that
    Force Decode logits align with the student's expected token format.
    """
    import inspect

    if not hasattr(tokenizer, 'apply_chat_template'):
        return user_content

    messages = [{'role': 'user', 'content': user_content}]
    kwargs = {'tokenize': False, 'add_generation_prompt': True}

    template_params = inspect.signature(tokenizer.apply_chat_template).parameters
    if 'enable_thinking' in template_params:
        kwargs['enable_thinking'] = _enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return user_content


@torch.no_grad()
def _force_decode_single(
    model,
    tokenizer,
    context_text: str,
    student_output_text: str,
    device: str,
    max_length: int = 40960,
    student_output_token_ids: Optional[List[int]] = None,
    top_k: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Force decode: return the teacher's full logits at the student response positions.

    The teacher sees ``context_text`` wrapped as a user message via the chat
    template, followed by the student's answer after the assistant generation
    prompt. A single forward pass is run, and we slice the logits at the
    positions covering the student response (Ys) only.

    When ``student_output_token_ids`` is provided, those exact token IDs are
    used instead of re-tokenizing the text, guaranteeing alignment with the
    student's tokenization.

    Returns:
        logits tensor of shape [1, response_len, vocab_size] in bfloat16,
        or None on failure.
    """
    # Build the context prefix via the tokenizer's chat template.
    prefix = _apply_chat_template(tokenizer, context_text)

    # Tokenize prefix
    prefix_ids = tokenizer(prefix, return_tensors='pt', add_special_tokens=False)['input_ids']
    prefix_length = prefix_ids.shape[1]

    # Use exact token IDs if provided, otherwise tokenize text
    if student_output_token_ids is not None:
        student_ids = torch.tensor([student_output_token_ids], dtype=prefix_ids.dtype)
    else:
        student_ids = tokenizer(student_output_text, return_tensors='pt', add_special_tokens=False)['input_ids']

    full_input_ids = torch.cat([prefix_ids, student_ids], dim=1)
    # Truncate to max_length if needed
    full_input_ids = full_input_ids[:, :max_length].to(device)
    attention_mask = torch.ones_like(full_input_ids)
    full_length = full_input_ids.shape[1]

    if prefix_length >= full_length:
        return None

    # Forward pass to get full logits
    outputs = model(input_ids=full_input_ids, attention_mask=attention_mask)
    full_logits = outputs.logits  # [1, full_length, vocab_size]

    # Extract logits at student output positions
    # position i predicts token i+1 (autoregressive), so we take [prefix-1 : full-1]
    ys_logits = full_logits[:, prefix_length - 1:full_length - 1, :]

    # top_k truncation: only keep top-k logits per position to drastically reduce
    # transfer size (e.g., 152K vocab → 20 tokens = 7600x smaller)
    if top_k is not None and top_k > 0 and top_k < ys_logits.shape[-1]:
        top_k_values, top_k_indices = torch.topk(ys_logits, k=top_k, dim=-1)
        result_values = top_k_values.to(torch.bfloat16).cpu()
        result_indices = top_k_indices.cpu()  # int64
        del top_k_values, top_k_indices, ys_logits, full_logits, outputs, full_input_ids, attention_mask
        # Return as dict-like: caller will handle the two tensors
        return {'values': result_values, 'indices': result_indices}

    # Move to CPU immediately and release GPU tensors to prevent OOM
    # when processing multiple samples sequentially in a batch request
    result = ys_logits.to(torch.bfloat16).cpu()
    del ys_logits, full_logits, outputs, full_input_ids, attention_mask
    return result


def _serialize_logits_list(logits_list: List, top_k: Optional[int] = None) -> bytes:
    """
    Serialize a list of logits tensors to safetensors binary format.

    Format:
    - Header (JSON): metadata about each tensor (shape, dtype, offset)
    - Body: concatenated raw tensor bytes

    When top_k is set, each sample is stored as two tensors:
    - logits_values_{idx}: [1, seq_len, top_k] bf16
    - logits_indices_{idx}: [1, seq_len, top_k] int64 (token indices)

    We use safetensors because:
    1. Zero-copy deserialization on the client side
    2. Much faster than JSON for large tensors
    3. Standard format supported by HuggingFace ecosystem
    """
    from safetensors.torch import save

    tensors = {}
    for idx, logits in enumerate(logits_list):
        if logits is not None and isinstance(logits, dict):
            # top_k mode: store values and indices separately
            tensors[f'logits_values_{idx}'] = logits['values'].cpu()
            tensors[f'logits_indices_{idx}'] = logits['indices'].cpu()
        elif logits is not None:
            # Full vocab mode
            tensors[f'logits_{idx}'] = logits.cpu()
        else:
            # Placeholder: empty tensor to indicate None
            tensors[f'logits_{idx}'] = torch.zeros(1, 0, 1, dtype=torch.bfloat16)

    # Store top_k flag as a scalar tensor so the client knows which format to expect
    if top_k is not None and top_k > 0:
        tensors['__top_k__'] = torch.tensor([top_k], dtype=torch.int64)

    # safetensors.torch.save returns bytes directly
    return save(tensors)


@app.post('/force_decode')
async def force_decode(request: ForceDecodeRequest):
    """
    Force decode endpoint: returns full-vocabulary logits as safetensors binary.

    Request body:
        context_texts: List of context strings (prompt + debate history)
        student_output_texts: List of student output strings
        max_length: Maximum sequence length for tokenization

    Response:
        Binary safetensors data containing logits tensors.
        Each tensor is named 'logits_0', 'logits_1', etc.
        Shape: [1, response_len, vocab_size] per sample.
        Empty tensor (shape [1, 0, 1]) indicates failure for that sample.
    """
    global _model, _tokenizer, _device

    if _model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    if len(request.context_texts) != len(request.student_output_texts):
        raise HTTPException(
            status_code=400,
            detail='context_texts and student_output_texts must have the same length'
        )

    # Check if token IDs are provided and have matching length
    has_token_ids = (request.student_output_token_ids is not None
                     and len(request.student_output_token_ids) == len(request.context_texts))

    start_time = time.time()
    logits_list = []

    for idx, (context_text, student_text) in enumerate(
        zip(request.context_texts, request.student_output_texts)
    ):
        logger.debug(
            f'[sample {idx}] context: {context_text[:]} | student_output: {student_text[:]}'
        )
        try:
            sample_token_ids = (request.student_output_token_ids[idx]
                                if has_token_ids else None)
            logits = _force_decode_single(
                _model, _tokenizer, context_text, student_text,
                _device, request.max_length,
                student_output_token_ids=sample_token_ids,
                top_k=request.top_k,
            )
            logits_list.append(logits)
        except Exception as exc:
            logger.warning(f'Force decode failed for sample: {exc}')
            logits_list.append(None)

    # Release GPU cache once after the entire batch to prevent OOM accumulation
    torch.cuda.empty_cache()

    # Serialize to safetensors binary
    binary_data = _serialize_logits_list(logits_list, top_k=request.top_k)

    elapsed = time.time() - start_time
    logger.info(
        f'Force decode completed: {len(logits_list)} samples, '
        f'{elapsed:.2f}s, {len(binary_data)} bytes'
    )

    return Response(
        content=binary_data,
        media_type='application/octet-stream',
        headers={
            'X-Num-Samples': str(len(logits_list)),
            'X-Elapsed-Seconds': f'{elapsed:.3f}',
        },
    )


@app.get('/health')
async def health():
    """Health check endpoint."""
    return {
        'status': 'ok',
        'model_loaded': _model is not None,
        'device': str(_device) if _device else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Force Decode Sidecar Server')
    parser.add_argument('--model', type=str, required=True, help='Path to teacher model')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8011, help='Server port')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.45,
                        help='Fraction of GPU memory to use (default 0.45)')
    parser.add_argument('--max-length', type=int, default=40960,
                        help='Default max sequence length (must cover prompt + full '
                             'debate transcript + student response; 40960 covers both '
                             'ToolACE and code configurations).')
    parser.add_argument('--enable-thinking', type=str, default='false',
                        help='Enable thinking mode in chat template (for Qwen3 models). '
                             'Should match the student model enable_thinking training config.')
    args = parser.parse_args()

    global _model, _tokenizer, _device, _enable_thinking
    _device = args.device
    _enable_thinking = args.enable_thinking.lower() in ('true', '1', 'yes')
    logger.info(f'enable_thinking={_enable_thinking}')
    _model, _tokenizer = _load_model(args.model, args.device, args.gpu_memory_fraction)
    logger.info(f'Starting Force Decode Sidecar on {args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
