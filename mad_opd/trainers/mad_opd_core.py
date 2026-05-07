# Copyright 2026 The MAD-OPD authors. Licensed under the Apache License, Version 2.0.
"""
MAD-OPD: Multi-Agent Debate driven On-Policy Distillation — core module.

Algorithmic pieces used by the MAD-OPD trainer:

* ``multi_teacher_jsd_loss`` /
  ``_chunked_jsd``                  Memory-efficient, top-k truncated JSD used
                                    for agentic OPD (Paper Eq. 8 with D=JSD,
                                    bounded by Lemma 1).
* ``generalized_jsd_loss_single``   Single-teacher JSD primitive (also the
                                    backend for OPD baselines).
* ``compute_mad_distillation_loss`` Assembles Paper Eq. 8 with confidence-
                                    based weights w_k.

Reverse-KL for code generation (Paper Proposition 2) is enabled by setting
``divergence='reverse_kl'`` on the loss primitives; JSD is the default.
"""
import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from swift.utils import get_logger

logger = get_logger()


def _softmax_confidence_weights(score_a: float, score_b: float,
                                tau_conf: float = 1.0) -> Tuple[float, float]:
    """Paper Eq. 7: w_k = softmax(c_k / 100 / tau_conf) with tau_conf = 1.0."""
    logits = [score_a / 100.0 / tau_conf, score_b / 100.0 / tau_conf]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return exps[0] / z, exps[1] / z



def _chunked_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    beta: float,
    chunk_size: int,
    num_valid: int,
) -> torch.Tensor:
    """Core chunked JSD computation on already-aligned, flattened logits [N, dim]."""
    total_loss = torch.zeros((), dtype=torch.float32, device=student_logits.device)

    if beta != 0 and beta != 1:
        beta_t = torch.tensor(beta, dtype=torch.float32, device=student_logits.device)
        log_beta = torch.log(beta_t)
        log_1_minus_beta = torch.log1p(-beta_t)
    else:
        beta_t = log_beta = log_1_minus_beta = None

    for start_idx in range(0, num_valid, chunk_size):
        end_idx = min(start_idx + chunk_size, num_valid)
        s_chunk = student_logits[start_idx:end_idx]
        t_chunk = teacher_logits[start_idx:end_idx]

        s_log_probs = F.log_softmax(s_chunk, dim=-1)
        t_log_probs = F.log_softmax(t_chunk, dim=-1)
        del s_chunk, t_chunk

        if beta == 0:
            jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
        elif beta == 1:
            jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
            kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
            del mixture_log_probs
            jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
            del kl_teacher, kl_student

        total_loss = total_loss + jsd_chunk.sum()
        del jsd_chunk, s_log_probs, t_log_probs

    result = (total_loss / num_valid).clamp(min=0.0)
    return result


def generalized_jsd_loss_single(    student_logits: torch.Tensor,
    teacher_logits,
    labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    chunk_size: int = 256,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute generalized Jensen-Shannon Divergence (JSD) loss for single teacher.

    Mathematical formula:
    JSD_beta(P || Q) = beta * KL(M || Q) + (1-beta) * KL(M || P)
    where M = beta * Q + (1-beta) * P is the mixture distribution

    Special cases:
    - beta = 0: Degenerates to KL(P || Q), forward KL
    - beta = 1: Degenerates to KL(Q || P), reverse KL
    - beta = 0.5: Standard symmetric JSD

    Args:
        student_logits: Student model logits, shape [batch, seq_len, vocab_size]
        teacher_logits: Teacher model logits, shape [batch, seq_len, vocab_size],
            or a dict {'values': [batch, seq_len, k], 'indices': [batch, seq_len, k]}
            when sidecar returns pre-truncated top-k logits.
        labels: Labels for masking invalid positions (-100 means ignore)
        beta: Beta parameter for JSD
        temperature: Temperature for softmax scaling
        chunk_size: Chunk size for memory-efficient computation
        top_k: If set, restricts JSD to only the top-k tokens of the teacher distribution.
               Both distributions are renormalized over these k tokens before computing JSD.
               This reduces memory from [seq_len, vocab_size] to [seq_len, top_k] per chunk.
               Default: None (full vocabulary).
               Ignored when teacher_logits is already in sparse top-k format.

    Returns:
        torch.Tensor: Average JSD loss
    """
    # Handle sparse top-k format from sidecar: {'values': Tensor, 'indices': Tensor}
    # In this mode, teacher only sent top-k logits. We gather the corresponding
    # student logits using the same indices, then compute JSD on the k-dim slice.
    # This avoids ever materializing full-vocab tensors on the training GPU.
    if isinstance(teacher_logits, dict) and 'values' in teacher_logits:
        teacher_values = teacher_logits['values']  # [batch, seq_len, k]
        teacher_indices = teacher_logits['indices']  # [batch, seq_len, k]

        # Align sequence length
        s_seq = student_logits.shape[-2]
        t_seq = teacher_values.shape[-2]
        if s_seq != t_seq:
            min_seq = min(s_seq, t_seq)
            student_logits = student_logits[..., :min_seq, :]
            teacher_values = teacher_values[..., :min_seq, :]
            teacher_indices = teacher_indices[..., :min_seq, :]

        # Ensure indices are on the same device as student_logits
        teacher_values = teacher_values.to(student_logits.device)
        teacher_indices = teacher_indices.to(student_logits.device)

        if labels is not None:
            mask = labels != -100
            student_logits_flat = student_logits[mask]
            teacher_values = teacher_values.view(-1, teacher_values.size(-1))[mask.view(-1)]
            teacher_indices = teacher_indices.view(-1, teacher_indices.size(-1))[mask.view(-1)]
        else:
            student_logits_flat = student_logits.view(-1, student_logits.size(-1))
            teacher_values = teacher_values.view(-1, teacher_values.size(-1))
            teacher_indices = teacher_indices.view(-1, teacher_indices.size(-1))

        num_valid = student_logits_flat.size(0)
        if num_valid == 0:
            return student_logits.new_zeros(())

        # Gather student logits at teacher's top-k positions
        student_gathered = torch.gather(student_logits_flat, dim=-1, index=teacher_indices)
        # Now both are [num_valid, k] — proceed with chunked JSD
        return _chunked_jsd(student_gathered, teacher_values, beta, chunk_size, num_valid)

    original_dtype = student_logits.dtype

    # Align sequence length (external Sidecar may return one fewer token than student logits)
    if student_logits.dim() >= 2 and teacher_logits.dim() >= 2:
        s_seq = student_logits.shape[-2]
        t_seq = teacher_logits.shape[-2]
        if s_seq != t_seq:
            min_seq = min(s_seq, t_seq)
            student_logits = student_logits[..., :min_seq, :]
            teacher_logits = teacher_logits[..., :min_seq, :]
        # Align vocab dim (different models may have different vocab sizes)
        s_vocab = student_logits.shape[-1]
        t_vocab = teacher_logits.shape[-1]
        if s_vocab != t_vocab:
            min_vocab = min(s_vocab, t_vocab)
            student_logits = student_logits[..., :min_vocab]
            teacher_logits = teacher_logits[..., :min_vocab]

    if labels is not None:
        mask = labels != -100
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]
        num_valid = mask.sum()
    else:
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        num_valid = student_logits.size(0)

    if num_valid == 0:
        return student_logits.new_zeros(()).to(original_dtype)

    # Top-k truncation: select top-k tokens from teacher distribution to reduce memory
    # Keep original dtype (bf16) during truncation to avoid materialising a full-vocab fp32 tensor
    if top_k is not None and top_k > 0 and top_k < student_logits.size(-1):
        with torch.no_grad():
            _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)
        # Subsequent log_softmax will renormalise over the truncated top-k dimension

    num_valid_int = num_valid if isinstance(num_valid, int) else num_valid.item()
    return _chunked_jsd(student_logits, teacher_logits, beta, chunk_size, num_valid_int)


def multi_teacher_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits_list: List[torch.Tensor],
    teacher_weights: Optional[List[float]] = None,
    labels: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    temperature: float = 1.0,
    chunk_size: int = 512,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute weighted JSD loss from multiple teacher models.

    Core formula:
    total_loss = sum(weight_i * JSD(student || teacher_i))

    When beta=1, JSD degenerates to reverse KL:
    loss_i = KL(teacher_i || student)

    Args:
        student_logits: Student model logits
        teacher_logits_list: List of teacher model logits
        teacher_weights: Weight for each teacher, defaults to equal weights
        labels: Labels for masking invalid positions
        beta: JSD beta parameter (beta=1 for reverse KL)
        temperature: Temperature parameter
        chunk_size: Chunk size for memory efficiency
        top_k: If set, restricts JSD to only the top-k tokens of the teacher distribution.
               Both distributions are renormalized over these k tokens before computing JSD.
               This reduces memory from [seq_len, vocab_size] to [seq_len, top_k] per chunk.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - total_loss: Weighted average JSD loss
            - individual_losses: Dictionary of individual teacher losses
    """
    num_teachers = len(teacher_logits_list)

    if teacher_weights is None:
        teacher_weights = [1.0 / num_teachers] * num_teachers
    else:
        weight_sum = sum(teacher_weights)
        teacher_weights = [w / weight_sum for w in teacher_weights]

    assert len(teacher_weights) == num_teachers, \
        f'Number of weights ({len(teacher_weights)}) must equal number of teachers ({num_teachers})'

    individual_losses = {}
    total_loss = student_logits.new_zeros(())

    for i, (teacher_logits, weight) in enumerate(zip(teacher_logits_list, teacher_weights)):
        loss_i = generalized_jsd_loss_single(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=beta,
            temperature=temperature,
            chunk_size=chunk_size,
            top_k=top_k,
        )

        individual_losses[f'teacher_{i}_loss'] = loss_i.detach().clone()
        total_loss = total_loss + weight * loss_i

    return total_loss, individual_losses




def compute_mad_distillation_loss(
    student_logits: torch.Tensor,
    teacher_a_logits: torch.Tensor,
    teacher_b_logits: torch.Tensor,
    alpha: float,
    beta: float,
    labels: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    jsd_beta: float = 1.0,
    response_mask: Optional[torch.Tensor] = None,
    jsd_top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute MAD distillation loss with dynamic weights using JSD.

    Core formula:
    loss = alpha * JSD(student || teacher_A) + beta * JSD(student || teacher_B)

    where JSD uses generalized_jsd_loss_single with configurable jsd_beta:
    - jsd_beta=1.0: Degenerates to reverse KL(teacher || student), consistent with
                     default/multi_teacher when --beta 1 (default)
    - jsd_beta=0.5: Standard symmetric JSD (bounded by ln(2) ≈ 0.693)
    - jsd_beta=0.0: Degenerates to forward KL(student || teacher)

    Key points:
    - Teacher logits are computed with debate history H as context
    - This makes teacher probability distributions influenced by debate analysis
    - Weights alpha and beta are dynamically allocated based on debate results
    - Each teacher's JSD is computed independently, then weighted by alpha/beta
    - student_logits must retain autograd computation graph (requires_grad=True)
    - Uses contiguous slice (not boolean indexing) to preserve grad_fn

    Args:
        student_logits: Student model logits WITH grad, shape [batch, seq_len, vocab_size]
        teacher_a_logits: Teacher A logits (with debate context), detached
        teacher_b_logits: Teacher B logits (with debate context), detached
        alpha: Dynamic weight for Teacher A (from debate)
        beta: Dynamic weight for Teacher B (from debate)
        labels: Optional labels for masking invalid positions (-100 means ignore).
                When provided, uses contiguous slice to extract valid response logits
                (preserving autograd graph), NOT boolean indexing.
        temperature: Temperature for softmax scaling
        jsd_beta: Beta parameter for JSD divergence (default 0.5 for symmetric JSD)

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]:
            - loss: Dynamically weighted JSD loss (with grad)
            - metrics: Dictionary of metrics including per-teacher losses
    """
    # Extract valid response logits using contiguous slice (preserves grad_fn)
    if labels is not None:
        valid_mask = labels != -100
        valid_indices = valid_mask.nonzero(as_tuple=False)
        if valid_indices.numel() == 0:
            zero_loss = (student_logits * 0).sum()
            return zero_loss, {'mad_loss': 0.0, 'alpha': alpha, 'beta': beta,
                               'teacher_a_jsd': 0.0, 'teacher_b_jsd': 0.0}
        first_valid = valid_indices[:, 1].min().item()
        last_valid = valid_indices[:, 1].max().item() + 1
        student_logits = student_logits[:, first_valid:last_valid, :]
    # Handle None teacher logits (Force Decode OOM or failure)
    if teacher_a_logits is None and teacher_b_logits is None:
        zero_loss = (student_logits * 0).sum()
        return zero_loss, {'mad_loss': 0.0, 'alpha': alpha, 'beta': beta,
                           'teacher_a_jsd': 0.0, 'teacher_b_jsd': 0.0,
                           'teacher_a_coverage': 0.0, 'teacher_b_coverage': 0.0}

    if teacher_a_logits is None:
        # Only Teacher B available, give all weight to B
        alpha = 0.0
        beta = 1.0
    else:
        teacher_a_logits = teacher_a_logits.detach()

    if teacher_b_logits is None:
        # Only Teacher A available, give all weight to A
        alpha = 1.0
        beta = 0.0
    else:
        teacher_b_logits = teacher_b_logits.detach()

    # Declare variables first, then run validation checks
    response_len = student_logits.shape[1]
    teacher_a_len = teacher_a_logits.shape[1] if teacher_a_logits is not None else 0
    teacher_b_len = teacher_b_logits.shape[1] if teacher_b_logits is not None else 0

    # Length-alignment diagnostics
    if teacher_a_logits is not None and abs(response_len - teacher_a_len) > 5:
        logger.warning(
            f'[MAD] Token alignment gap: student_len={response_len}, teacher_a_len={teacher_a_len}, '
            f'diff={abs(response_len - teacher_a_len)}. This may indicate tokenization mismatch.'
        )
    if teacher_b_logits is not None and abs(response_len - teacher_b_len) > 5:
        logger.warning(
            f'[MAD] Token alignment gap: student_len={response_len}, teacher_b_len={teacher_b_len}, '
            f'diff={abs(response_len - teacher_b_len)}. This may indicate tokenization mismatch.'
        )

    

    # Compute JSD for each teacher independently with per-teacher alignment.
    # Each teacher's logits are aligned with student logits separately,
    # so a shorter teacher_a does NOT truncate the range used for teacher_b.
    # This prevents the "shortest-teacher-wins" bug where the global min_seq_len
    # caused only a small prefix of student tokens to receive gradient signal,
    # incentivizing the student to produce shorter and shorter outputs.

    # ---- Teacher A ----
    effective_len_a = min(response_len, teacher_a_len) if teacher_a_logits is not None else 0
    if effective_len_a > 0:
        student_logits_a = student_logits[:, :effective_len_a, :]
        teacher_a_aligned = teacher_a_logits[:, :effective_len_a, :]
        # Align vocab size
        min_vocab_a = min(student_logits_a.shape[-1], teacher_a_aligned.shape[-1])
        student_logits_a = student_logits_a[..., :min_vocab_a]
        teacher_a_aligned = teacher_a_aligned[..., :min_vocab_a]

        loss_a = generalized_jsd_loss_single(
            student_logits=student_logits_a,
            teacher_logits=teacher_a_aligned,
            labels=None,
            beta=jsd_beta,
            temperature=temperature,
            top_k=jsd_top_k,
        )
        del student_logits_a, teacher_a_aligned
    else:
        loss_a = (student_logits * 0).sum()
    if teacher_a_logits is not None:
        del teacher_a_logits
    torch.cuda.empty_cache()

    # ---- Teacher B ----
    effective_len_b = min(response_len, teacher_b_len) if teacher_b_logits is not None else 0
    if effective_len_b > 0:
        student_logits_b = student_logits[:, :effective_len_b, :]
        teacher_b_aligned = teacher_b_logits[:, :effective_len_b, :]
        # Align vocab size
        min_vocab_b = min(student_logits_b.shape[-1], teacher_b_aligned.shape[-1])
        student_logits_b = student_logits_b[..., :min_vocab_b]
        teacher_b_aligned = teacher_b_aligned[..., :min_vocab_b]

        loss_b = generalized_jsd_loss_single(
            student_logits=student_logits_b,
            teacher_logits=teacher_b_aligned,
            labels=None,
            beta=jsd_beta,
            temperature=temperature,
            top_k=jsd_top_k,
        )
        del student_logits_b, teacher_b_aligned
    else:
        loss_b = (student_logits * 0).sum()
    if teacher_b_logits is not None:
        del teacher_b_logits
    torch.cuda.empty_cache()

    # If both teachers have zero effective length, return zero loss
    if effective_len_a == 0 and effective_len_b == 0:
        zero_loss = (student_logits * 0).sum()
        return zero_loss, {'mad_loss': 0.0, 'alpha': alpha, 'beta': beta,
                           'teacher_a_jsd': 0.0, 'teacher_b_jsd': 0.0,
                           'teacher_a_coverage': 0.0, 'teacher_b_coverage': 0.0}
    # Dynamic weighted combination
    weight_sum = alpha + beta
    normalized_alpha = alpha / weight_sum if weight_sum > 0 else 0.5
    normalized_beta = beta / weight_sum if weight_sum > 0 else 0.5

    # loss_a and loss_b are in float32 (generalized_jsd_loss_single keeps float32).
    # Compute weighted combination in float32, then cast back to original dtype.
    loss = (normalized_alpha * loss_a + normalized_beta * loss_b)

    # Debug: log raw loss values for first few steps to verify no negative values
    loss_a_val = loss_a.item()
    loss_b_val = loss_b.item()
    if loss_a_val < 0 or loss_b_val < 0:
        logger.error(
            f'[MAD-DEBUG] Negative JSD after clamp! loss_a={loss_a_val:.8f}, loss_b={loss_b_val:.8f}, '
            f'eff_len_a={effective_len_a}, eff_len_b={effective_len_b}, response_len={response_len}'
        )

    # Coverage: fraction of student response tokens covered by each teacher
    coverage_a = effective_len_a / response_len if response_len > 0 else 0.0
    coverage_b = effective_len_b / response_len if response_len > 0 else 0.0

    metrics = {
        'mad_loss': loss.item(),
        'alpha': normalized_alpha,
        'beta': normalized_beta,
        'teacher_a_jsd': loss_a.item(),
        'teacher_b_jsd': loss_b.item(),
        'teacher_a_coverage': coverage_a,
        'teacher_b_coverage': coverage_b,
    }

    return loss, metrics

