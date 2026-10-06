"""Verbatim copies of the user's pc-fsvi-continual-learning modules (gp, predictive_coding, variational,
continual_learning) + a utils shim. continual_learning.py also needs uncertainty.py and metrics.py (not uploaded),
so it is NOT imported here; it is kept for the E5 stage."""
from .gp import ExactGP as RepoExactGP
from .predictive_coding import closed_form_optimum, pc_infer
from .variational import FSVI

__all__ = ["FSVI", "pc_infer", "closed_form_optimum", "RepoExactGP"]
