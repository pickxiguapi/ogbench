"""Explicit IMPALA-small LeWM variant; not the paper-reproduction model."""

from lewm_jax.impala.loss import lewm_loss, sigreg_loss
from lewm_jax.impala.model import ImpalaLeWM

__all__ = ['ImpalaLeWM', 'lewm_loss', 'sigreg_loss']
