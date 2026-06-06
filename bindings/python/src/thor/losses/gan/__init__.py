"""Generative adversarial network losses."""

from __future__ import annotations

from ..._thor.losses.gan import (
    HingeGANDiscriminatorLoss,
    HingeGANGeneratorLoss,
    LSGANDiscriminatorLoss,
    LSGANGeneratorLoss,
    WassersteinGANCriticGradientPenaltyLoss,
    WassersteinGANCriticLoss,
    WassersteinGANGeneratorLoss,
)

__all__ = [
    "HingeGANDiscriminatorLoss",
    "HingeGANGeneratorLoss",
    "WassersteinGANCriticLoss",
    "WassersteinGANGeneratorLoss",
    "WassersteinGANCriticGradientPenaltyLoss",
    "LSGANDiscriminatorLoss",
    "LSGANGeneratorLoss",
]
