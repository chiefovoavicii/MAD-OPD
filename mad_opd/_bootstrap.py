"""Runtime patches that splice MAD-OPD into a stock ``ms-swift`` install.

We perform three patches, all idempotent:

1. Extend ``swift.arguments.RLHFArguments.gkd_algorithm`` so it accepts the
   value ``'mad_opd'`` (the canonical algorithm name used in the paper).
2. Replace ``swift.rlhf_trainers.GKDTrainer`` with
   ``mad_opd.trainers.MADOPDTrainer`` so ``ms-swift``'s RLHF pipeline picks
   up our trainer when ``--rlhf_type gkd`` is requested.
3. Teach the RLHF pipeline to load a second teacher model when
   ``gkd_algorithm == 'mad_opd'`` (already the case in the copy of the
   pipeline we ship — nothing to patch there; we simply validate).

All patches are applied lazily the first time this module is imported.
"""
import logging
import os
from typing import Any

_PATCHED = False


def apply_patches() -> None:
    """Apply MAD-OPD runtime patches.  Safe to call multiple times.

    The trainer patch imports heavy ms-swift submodules (which pull in vllm,
    deepspeed, ...) — that's fine at training time but slows down ``import
    mad_opd`` in lightweight contexts (docs, CI smoke-imports, interactive
    shells).  To keep import time cheap we only install the trainer patch
    when ``MAD_OPD_AUTO_PATCH`` is truthy (the default for any process
    launched through ``mad_opd.cli.train``); the args patch is cheap and
    runs unconditionally.
    """
    global _PATCHED
    if _PATCHED:
        return
    _patch_rlhf_args()
    if os.environ.get('MAD_OPD_AUTO_PATCH', '1') not in ('0', 'false', 'False'):
        _patch_gkd_trainer()
    _PATCHED = True
    logging.getLogger('mad_opd').info('MAD-OPD patches installed on ms-swift.')


def _patch_rlhf_args() -> None:
    """Install ``MADOPDArguments`` as the ``args_class`` for ms-swift RLHF and
    extend ``GKDConfig`` with the same MAD-OPD fields so they propagate to the
    trainer's ``training_args``.
    """
    import swift.arguments as arguments_mod
    import swift.arguments.rlhf_args as rlhf_args_mod
    import swift.pipelines.train.rlhf as rlhf_pipeline_mod
    import swift.rlhf_trainers.arguments as trainer_args_mod
    import swift.trainers.trainer_factory as trainer_factory_mod

    from mad_opd._mad_opd_args import build_mad_opd_args_class
    MADOPDArguments = build_mad_opd_args_class()

    rlhf_args_mod.RLHFArguments = MADOPDArguments
    arguments_mod.RLHFArguments = MADOPDArguments
    rlhf_pipeline_mod.RLHFArguments = MADOPDArguments
    rlhf_pipeline_mod.SwiftRLHF.args_class = MADOPDArguments

    # Extend GKDConfig (the inner training-arg dataclass) so MAD-OPD fields
    # reach the trainer via ``kwargs['args']``.  We rebind the class object
    # in every import site that ``TrainerFactory.get_cls`` may consult.
    import swift.rlhf_trainers as rlhf_trainers_pkg
    from mad_opd._mad_opd_args import build_mad_opd_gkd_config
    MADOPDGKDConfig = build_mad_opd_gkd_config()
    trainer_args_mod.GKDConfig = MADOPDGKDConfig
    rlhf_trainers_pkg.GKDConfig = MADOPDGKDConfig


def _patch_gkd_trainer() -> None:
    """Install MADOPDTrainer as the implementation of ``swift.rlhf_trainers.GKDTrainer``.

    ms-swift's RLHF pipeline imports the trainer class lazily, so we both
    (a) rebind the attribute on the module, and (b) evict the cached import
    from ``sys.modules`` so subsequent imports see our subclass.
    """
    import sys
    import swift.rlhf_trainers as trainers_mod
    import swift.rlhf_trainers.gkd_trainer as stock_gkd

    from mad_opd.trainers.mad_opd_trainer import MADOPDTrainer

    # Patch the public attribute that the pipeline looks up.
    trainers_mod.GKDTrainer = MADOPDTrainer
    stock_gkd.GKDTrainer = MADOPDTrainer

    # Also rebind the class inside ms-swift's module namespace so reflections
    # (e.g. ``isinstance`` checks inside ``get_trainer``) resolve the subclass.
    sys.modules['swift.rlhf_trainers.gkd_trainer'].GKDTrainer = MADOPDTrainer
