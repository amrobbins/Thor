from __future__ import annotations

from .._thor.training import *  # noqa: F401,F403
from ._kfold import make_k_fold_run_specs
from ._kfold import training_runs_from_k_fold_split

try:
    __all__  # type: ignore[name-defined]
except NameError:
    __all__ = [name for name in globals() if not name.startswith("_")]
else:
    __all__ = list(__all__)  # type: ignore[name-defined]

for _name in ("make_k_fold_run_specs", "training_runs_from_k_fold_split"):
    if _name not in __all__:
        __all__.append(_name)


def __dir__() -> list[str]:
    return sorted(__all__)
