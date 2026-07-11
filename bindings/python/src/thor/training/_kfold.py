from __future__ import annotations

from collections.abc import Callable, Sequence
import inspect
from typing import Any

from ..data import StratifiedHoldoutKFoldManifest
from ..data import StratifiedKFoldManifest


def _coerce_folds(split: StratifiedKFoldManifest | StratifiedHoldoutKFoldManifest) -> tuple[Any, ...]:
    if isinstance(split, StratifiedHoldoutKFoldManifest):
        return tuple(split.folds.folds)
    if isinstance(split, StratifiedKFoldManifest):
        return tuple(split.folds)
    raise TypeError(
        "split must be a thor.data.StratifiedKFoldManifest or "
        "thor.data.StratifiedHoldoutKFoldManifest"
    )


def _holdout_metadata(split: StratifiedKFoldManifest | StratifiedHoldoutKFoldManifest) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
    if isinstance(split, StratifiedHoldoutKFoldManifest):
        return tuple(split.test_keys), tuple(split.test_groups)
    return None, None


def _format_run_name(template: str, *, fold: Any, ensemble_group: str) -> str:
    try:
        return template.format(fold_index=fold.fold_index, fold=fold, ensemble_group=ensemble_group)
    except AttributeError as exc:  # pragma: no cover - guarded by split type checks, kept for clearer errors.
        raise TypeError("fold objects must expose fold_index") from exc


def _resolve_ensemble_weight(ensemble_weight: float | Callable[..., float], *, fold: Any, run_name: str, ensemble_group: str) -> float:
    if callable(ensemble_weight):
        return float(_call_user_factory(ensemble_weight, fold=fold, run_name=run_name, ensemble_group=ensemble_group))
    return float(ensemble_weight)


def _call_user_factory(factory: Callable[..., Any], **available_kwargs: Any) -> Any:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        try:
            return factory(**available_kwargs)
        except TypeError:
            return factory(available_kwargs["fold"])

    parameters = signature.parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_kwargs:
        return factory(**available_kwargs)

    keyword_kwargs = {
        name: value
        for name, value in available_kwargs.items()
        if name in parameters
        and parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if "fold" in keyword_kwargs:
        return factory(**keyword_kwargs)

    positional_params = [
        param
        for param in parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if positional_params:
        return factory(available_kwargs["fold"], **keyword_kwargs)

    raise TypeError(
        "make_trainer must accept a fold argument, either as a keyword named 'fold' "
        "or as a positional argument"
    )


def make_k_fold_run_specs(
    split: StratifiedKFoldManifest | StratifiedHoldoutKFoldManifest,
    *,
    make_trainer: Callable[..., Any],
    ensemble_group: str,
    run_name_template: str = "fold_{fold_index}",
    ensemble_weight: float | Callable[..., float] = 1.0,
) -> list[tuple[str, Any, str, float]]:
    """Build ``TrainingRuns`` run specs from a stratified k-fold split manifest.

    The helper is intentionally small: it owns the repetitive fold-to-run-spec
    plumbing, while caller code still owns how a fold becomes a ``Trainer`` and
    TrainingData recipe. ``make_trainer`` is called once per fold. It may accept any of these
    keyword arguments when useful: ``fold``, ``run_name``, ``ensemble_group``,
    ``test_keys``, and ``test_groups``. For simple callables, a single
    positional ``fold`` argument is also supported.

    ``split`` may be a plain ``StratifiedKFoldManifest`` or a
    ``StratifiedHoldoutKFoldManifest``. For holdout manifests, the shared
    ``test_keys`` and ``test_groups`` are passed to ``make_trainer`` when its
    signature accepts them.
    """

    if not isinstance(ensemble_group, str) or not ensemble_group:
        raise ValueError("ensemble_group must be a non-empty string")
    if not isinstance(run_name_template, str) or not run_name_template:
        raise ValueError("run_name_template must be a non-empty string")

    folds = _coerce_folds(split)
    test_keys, test_groups = _holdout_metadata(split)
    run_specs: list[tuple[str, Any, str, float]] = []
    seen_run_names: set[str] = set()

    for fold in folds:
        run_name = _format_run_name(run_name_template, fold=fold, ensemble_group=ensemble_group)
        if run_name in seen_run_names:
            raise ValueError(f"run_name_template produced duplicate run name {run_name!r}")
        seen_run_names.add(run_name)

        trainer = _call_user_factory(
            make_trainer,
            fold=fold,
            run_name=run_name,
            ensemble_group=ensemble_group,
            test_keys=test_keys,
            test_groups=test_groups,
        )
        weight = _resolve_ensemble_weight(ensemble_weight, fold=fold, run_name=run_name, ensemble_group=ensemble_group)
        if weight <= 0.0:
            raise ValueError("ensemble_weight must resolve to a positive value")
        run_specs.append((run_name, trainer, ensemble_group, weight))

    return run_specs


def training_runs_from_k_fold_split(
    split: StratifiedKFoldManifest | StratifiedHoldoutKFoldManifest,
    *,
    make_trainer: Callable[..., Any],
    ensemble_group: str,
    run_name_template: str = "fold_{fold_index}",
    ensemble_weight: float | Callable[..., float] = 1.0,
    **training_runs_kwargs: Any,
) -> Any:
    """Create ``TrainingRuns`` from a stratified k-fold split manifest.

    This is a convenience wrapper around ``make_k_fold_run_specs`` plus the
    native ``TrainingRuns`` constructor. All extra keyword arguments are passed
    through to ``TrainingRuns`` unchanged, e.g. ``max_parallel_runs`` or
    ``min_successful_models``.
    """

    from . import TrainingRuns  # Imported lazily so this module can be imported during package initialization.

    return TrainingRuns(
        make_k_fold_run_specs(
            split,
            make_trainer=make_trainer,
            ensemble_group=ensemble_group,
            run_name_template=run_name_template,
            ensemble_weight=ensemble_weight,
        ),
        **training_runs_kwargs,
    )


__all__ = ["make_k_fold_run_specs", "training_runs_from_k_fold_split"]
