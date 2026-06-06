from __future__ import annotations

from .._thor import losses as _native_losses
from .._thor.losses import *  # noqa: F401,F403

from . import classification as classification
from . import detection as detection
from . import distribution as distribution
from . import gan as gan
from . import metric_learning as metric_learning
from . import ranking as ranking
from . import segmentation as segmentation

__all__ = [name for name in dir(_native_losses) if not name.startswith("_")]
for _domain in (
    "classification",
    "detection",
    "distribution",
    "gan",
    "metric_learning",
    "ranking",
    "segmentation",
):
    if _domain not in __all__:
        __all__.append(_domain)
