"""MAD-OPD trainer + vLLM teacher manager exports."""
from .mad_opd_core import (
    compute_mad_distillation_loss,
    generalized_jsd_loss_single,
    multi_teacher_jsd_loss,
)
from .mad_opd_trainer import MADOPDTrainer, GKDTrainer  # noqa: F401 — alias
from .vllm_teacher_manager import VLLMTeacherManager

__all__ = [
    'MADOPDTrainer',
    'VLLMTeacherManager',
    'compute_mad_distillation_loss',
    'generalized_jsd_loss_single',
    'multi_teacher_jsd_loss',
]
