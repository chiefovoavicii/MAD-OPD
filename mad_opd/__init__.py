"""MAD-OPD: Multi-Agent Debate driven On-Policy Distillation.

Importing this package installs a small set of runtime patches on the
already-installed ``ms-swift`` so that ``--rlhf_type gkd --gkd_algorithm
mad_opd`` dispatches to the MAD-OPD trainer shipped here.

No changes to ``swift``'s source files are made — only attribute-level
monkey patches applied to already-imported objects.  Importing this module
more than once is a no-op.
"""
from ._bootstrap import apply_patches

apply_patches()

__version__ = '0.1.0'
__all__ = ['apply_patches']
