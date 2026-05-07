"""MAD-OPD-specific extensions of ms-swift's argument dataclasses.

Both dataclasses are defined at **module scope** (not inside a helper function)
so they remain picklable — ms-swift serialises the training-args object into
each DDP worker's checkpoint, and local classes cannot be pickled.

Installed by :mod:`mad_opd._bootstrap`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

# Defer importing swift dataclasses until after ``swift`` is importable.  The
# helper functions below materialise the subclasses at bootstrap time.

_MADOPDArguments = None
_MADOPDGKDConfig = None


def _define_mad_opd_arguments():
    from swift.arguments import RLHFArguments

    @dataclass
    class MADOPDArguments(RLHFArguments):
        # 'opd'     = single-teacher On-Policy Distillation (Paper §5.1 OPD row)
        # 'mt_opd'  = multi-teacher OPD without debate, fixed weights
        # 'mad_opd' = Multi-Agent-Debate OPD — paper main contribution
        # 'mt_seqkd'= off-policy multi-teacher SeqKD (dataset pre-merged 7:3)
        gkd_algorithm: Literal['opd', 'mt_opd', 'mad_opd', 'mt_seqkd'] = 'opd'

        # --- MAD-OPD hyper-parameters (Paper §4.1–§4.3) ---
        mad_debate_rounds: int = field(
            default=2, metadata={'help': 'Number of debate rounds R (Paper Eq. 5).'})
        mad_dynamic_weights: bool = field(
            default=True, metadata={'help': 'Confidence-based dynamic weighting (Paper §4.2).'})
        teacher_weights: Optional[List[float]] = field(
            default=None,
            metadata={'help': 'Prior weights [w_1, w_2] before dynamic reweighting.'})
        mad_max_debate_tokens: int = 512
        mad_debate_temperature: float = 0.7
        mad_debate_top_p: float = 0.9
        mad_debate_top_k: int = 50
        mad_t1_max_tokens: int = 4096
        mad_t2_max_tokens: int = 4096
        jsd_top_k: Optional[int] = None


        # --- Code-mode prompts (Paper §3.3, Remark 1 — code) ---
        code_mode: bool = False

        # --- vLLM teacher service orchestration ---
        use_vllm_teachers: bool = False
        vllm_teacher_tp_size: int = 8
        vllm_teacher_gpu_memory_utilization: float = 0.25
        vllm_teacher_max_model_len: int = 32768
        vllm_teacher_1_max_model_len: Optional[int] = None
        vllm_teacher_2_max_model_len: Optional[int] = None
        vllm_teacher_1_url: Optional[str] = None
        vllm_teacher_2_url: Optional[str] = None
        vllm_sidecar_1_url: Optional[str] = None
        vllm_sidecar_2_url: Optional[str] = None
        vllm_sidecar_max_length: int = 40960
        vllm_teacher_timeout: int = 300

        teacher_review_interval: int = 0

        # --- ToolACE agentic distillation ---
        toolace_mode: bool = False
        toolace_onpolicy_mode: str = 'teacher_forcing'
        toolace_max_turns: int = 10
        toolace_mask_tool_response: bool = True

        teacher_model_2: Optional[str] = None
        teacher_model_2_type: Optional[str] = None
        teacher_model_2_revision: Optional[str] = None
        max_logits_bytes: int = 1 << 30

    return MADOPDArguments


def _define_mad_opd_gkd_config():
    from swift.rlhf_trainers.arguments import GKDConfig

    @dataclass
    class MADOPDGKDConfig(GKDConfig):
        gkd_algorithm: str = 'opd'
        mad_debate_rounds: int = 2
        mad_dynamic_weights: bool = True
        teacher_weights: Optional[List[float]] = None
        mad_max_debate_tokens: int = 512
        mad_debate_temperature: float = 0.7
        mad_debate_top_p: float = 0.9
        mad_debate_top_k: int = 50
        mad_t1_max_tokens: int = 4096
        mad_t2_max_tokens: int = 4096
        jsd_top_k: Optional[int] = None
        code_mode: bool = False
        use_vllm_teachers: bool = False
        vllm_teacher_1_url: Optional[str] = None
        vllm_teacher_2_url: Optional[str] = None
        vllm_sidecar_1_url: Optional[str] = None
        vllm_sidecar_2_url: Optional[str] = None
        vllm_sidecar_max_length: int = 40960
        vllm_teacher_timeout: int = 300
        vllm_teacher_max_model_len: int = 32768
        vllm_teacher_1_max_model_len: Optional[int] = None
        vllm_teacher_2_max_model_len: Optional[int] = None
        toolace_mode: bool = False
        toolace_onpolicy_mode: str = 'teacher_forcing'
        toolace_max_turns: int = 10
        toolace_mask_tool_response: bool = True
        teacher_review_interval: int = 0
        max_logits_bytes: int = 1 << 30

    return MADOPDGKDConfig


def build_mad_opd_args_class():
    """Materialise (once) and return the module-level ``MADOPDArguments`` class."""
    global _MADOPDArguments
    if _MADOPDArguments is None:
        _MADOPDArguments = _define_mad_opd_arguments()
        # Expose at module scope so pickle can locate the class.
        globals()['MADOPDArguments'] = _MADOPDArguments
        _MADOPDArguments.__module__ = __name__
        _MADOPDArguments.__qualname__ = 'MADOPDArguments'
    return _MADOPDArguments


def build_mad_opd_gkd_config():
    """Materialise (once) and return the module-level ``MADOPDGKDConfig`` class."""
    global _MADOPDGKDConfig
    if _MADOPDGKDConfig is None:
        _MADOPDGKDConfig = _define_mad_opd_gkd_config()
        globals()['MADOPDGKDConfig'] = _MADOPDGKDConfig
        _MADOPDGKDConfig.__module__ = __name__
        _MADOPDGKDConfig.__qualname__ = 'MADOPDGKDConfig'
    return _MADOPDGKDConfig
