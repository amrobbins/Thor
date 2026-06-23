from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


class StratificationMode(str, Enum):
    """How split units are assigned to strata before sampling."""

    CATEGORICAL = "categorical"
    QUANTILE = "quantile"
    EXPLICIT_BUCKETS = "explicit_buckets"


@dataclass(frozen=True)
class StratifiedSplit:
    """One train/validation split over keys and, when supplied, groups."""

    train_keys: tuple[Any, ...]
    validate_keys: tuple[Any, ...]
    train_groups: tuple[Any, ...]
    validate_groups: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_keys": list(self.train_keys),
            "validate_keys": list(self.validate_keys),
            "train_groups": list(self.train_groups),
            "validate_groups": list(self.validate_groups),
        }


@dataclass(frozen=True)
class StratifiedFold(StratifiedSplit):
    """One fold in a stratified k-fold manifest."""

    fold_index: int

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        out["fold_index"] = self.fold_index
        return out


@dataclass(frozen=True)
class StratifiedKFoldManifest:
    """Reproducible stratified k-fold split manifest."""

    k: int
    seed: int | None
    mode: str
    num_bins: int | None
    stratify_edges: tuple[float, ...]
    folds: tuple[StratifiedFold, ...]

    def __iter__(self) -> Iterator[StratifiedFold]:
        return iter(self.folds)

    def __len__(self) -> int:
        return len(self.folds)

    def __getitem__(self, index: int) -> StratifiedFold:
        return self.folds[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "splitter": "stratified_group_k_fold",
            "k": self.k,
            "seed": self.seed,
            "stratify_mode": self.mode,
            "num_bins": self.num_bins,
            "stratify_edges": list(self.stratify_edges),
            "folds": [fold.to_dict() for fold in self.folds],
        }


@dataclass(frozen=True)
class StratifiedHoldoutKFoldManifest:
    """A stratified holdout test split plus k-fold train/validation splits."""

    test_keys: tuple[Any, ...]
    test_groups: tuple[Any, ...]
    folds: StratifiedKFoldManifest

    def to_dict(self) -> dict[str, Any]:
        out = self.folds.to_dict()
        out["splitter"] = "stratified_group_holdout_plus_k_fold"
        out["test_keys"] = list(self.test_keys)
        out["test_groups"] = list(self.test_groups)
        return out


@dataclass(frozen=True)
class StratifiedTrainValidationTestSplit:
    """One stratified train/validation/test split over keys and groups."""

    train_keys: tuple[Any, ...]
    validate_keys: tuple[Any, ...]
    test_keys: tuple[Any, ...]
    train_groups: tuple[Any, ...]
    validate_groups: tuple[Any, ...]
    test_groups: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "splitter": "stratified_group_train_validation_test_split",
            "train_keys": list(self.train_keys),
            "validate_keys": list(self.validate_keys),
            "test_keys": list(self.test_keys),
            "train_groups": list(self.train_groups),
            "validate_groups": list(self.validate_groups),
            "test_groups": list(self.test_groups),
        }


@dataclass(frozen=True)
class NumpyDictSplit:
    """Named NumPy arrays partitioned into train/validation[/test] dictionaries."""

    train: dict[str, np.ndarray]
    validate: dict[str, np.ndarray]
    test: dict[str, np.ndarray] | None = None


def make_numpy_dict_splits(
    tensors: Mapping[str, Any],
    *,
    split: StratifiedSplit | StratifiedTrainValidationTestSplit,
    keys: Sequence[Any] | None = None,
    groups: Sequence[Any] | None = None,
) -> NumpyDictSplit:
    """Partition named array-like tensors according to a stratified split.

    This helper is intended for demand-style in-memory loaders where each named
    input/label/weight tensor shares a leading example dimension. Pass
    ``groups`` when many rows belong to one split unit, e.g. product/date rows
    grouped by product id. If ``groups`` is omitted, ``keys`` are matched
    against the split's key fields.
    """

    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("tensors must be a non-empty mapping of tensor name to array-like values")
    if keys is None and groups is None:
        raise ValueError("exactly one of keys or groups must be provided")
    if keys is not None and groups is not None:
        raise ValueError("exactly one of keys or groups must be provided")

    arrays: dict[str, np.ndarray] = {name: np.asarray(value) for name, value in tensors.items()}
    first_name, first_array = next(iter(arrays.items()))
    if first_array.ndim == 0:
        raise ValueError(f"tensor {first_name!r} must have a leading example dimension")
    num_examples = int(first_array.shape[0])
    for name, array in arrays.items():
        if array.ndim == 0:
            raise ValueError(f"tensor {name!r} must have a leading example dimension")
        if int(array.shape[0]) != num_examples:
            raise ValueError(
                "all tensors must have the same leading example dimension; "
                f"tensor {name!r} has {array.shape[0]} examples but {first_name!r} has {num_examples}"
            )

    selector_values = tuple(groups if groups is not None else keys)
    if len(selector_values) != num_examples:
        selector_name = "groups" if groups is not None else "keys"
        raise ValueError(f"{selector_name} length must match the tensor leading dimension")

    train_values = set(split.train_groups if groups is not None else split.train_keys)
    validate_values = set(split.validate_groups if groups is not None else split.validate_keys)
    test_values = None
    if isinstance(split, StratifiedTrainValidationTestSplit):
        test_values = set(split.test_groups if groups is not None else split.test_keys)

    train_indices = _indices_for_values(selector_values, train_values)
    validate_indices = _indices_for_values(selector_values, validate_values)
    test_indices = _indices_for_values(selector_values, test_values) if test_values is not None else None

    return NumpyDictSplit(
        train={name: np.ascontiguousarray(array[train_indices]) for name, array in arrays.items()},
        validate={name: np.ascontiguousarray(array[validate_indices]) for name, array in arrays.items()},
        test=(
            {name: np.ascontiguousarray(array[test_indices]) for name, array in arrays.items()}
            if test_indices is not None
            else None
        ),
    )


def _indices_for_values(values: Sequence[Any], selected: set[Any] | None) -> np.ndarray:
    if selected is None:
        return np.asarray([], dtype=np.int64)
    return np.asarray([index for index, value in enumerate(values) if value in selected], dtype=np.int64)


@dataclass(frozen=True)
class _SplitUnit:
    group: Any
    keys: tuple[Any, ...]
    stratum: Any
    order: int


class StratifiedSplitter:
    """Group-aware stratified splitter for train/validate/test manifests.

    ``keys`` are the values returned in split manifests. If ``groups`` is
    omitted, each key is its own split unit. If ``groups`` is supplied, all keys
    with the same group are assigned to the same split to avoid leakage.

    For demand forecasting, use product ids as both ``keys`` and ``groups`` (or
    omit ``groups`` when keys are already unique products) and pass product
    demand magnitude as ``stratify_values`` with ``mode="quantile"``.
    """

    def __init__(
        self,
        keys: Sequence[Any],
        stratify_values: Sequence[Any] | None = None,
        *,
        groups: Sequence[Any] | None = None,
        mode: str | StratificationMode = StratificationMode.QUANTILE,
        num_bins: int | None = None,
        bucket_labels: Sequence[Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.keys = tuple(keys)
        if len(self.keys) == 0:
            raise ValueError("keys must contain at least one key")

        self.mode = StratificationMode(mode)
        self.seed = seed
        self.num_bins = num_bins

        if bucket_labels is not None and len(bucket_labels) != len(self.keys):
            raise ValueError("bucket_labels length must match keys length")

        if stratify_values is None:
            if bucket_labels is None:
                raise ValueError("stratify_values is required unless bucket_labels is supplied")
            raw_stratify_values: tuple[Any, ...] = tuple(bucket_labels)
        else:
            if len(stratify_values) != len(self.keys):
                raise ValueError("stratify_values length must match keys length")
            raw_stratify_values = tuple(stratify_values)

        if groups is None:
            self.groups = self.keys
        else:
            if len(groups) != len(self.keys):
                raise ValueError("groups length must match keys length")
            self.groups = tuple(groups)

        if self.mode == StratificationMode.EXPLICIT_BUCKETS:
            if bucket_labels is None:
                raw_stratify_values = tuple(raw_stratify_values)
            else:
                raw_stratify_values = tuple(bucket_labels)

        if self.mode == StratificationMode.QUANTILE:
            self._units, self._stratify_edges = self._build_quantile_units(raw_stratify_values)
        else:
            self._units = self._build_categorical_units(raw_stratify_values)
            self._stratify_edges = tuple()

        self._units_by_group = {unit.group: unit for unit in self._units}

    @property
    def stratify_edges(self) -> tuple[float, ...]:
        return self._stratify_edges

    @property
    def num_split_units(self) -> int:
        return len(self._units)

    def train_validation_split(
        self,
        *,
        validation_fraction: float | None = None,
        validation_size: int | None = None,
    ) -> StratifiedSplit:
        """Create one stratified train/validation split.

        Exactly one of ``validation_fraction`` or ``validation_size`` must be
        provided. Sizes are measured in split units, not raw key rows, so group
        leakage is prevented when ``groups`` is supplied.
        """

        validate_unit_count = self._resolve_sample_size(
            fraction=validation_fraction,
            size=validation_size,
            total=len(self._units),
            fraction_name="validation_fraction",
            size_name="validation_size",
        )
        validate_groups = set(self._sample_stratified_units(validate_unit_count))
        return self._split_from_validate_groups(validate_groups)

    def train_validation_test_split(
        self,
        *,
        validation_fraction: float | None = None,
        validation_size: int | None = None,
        test_fraction: float | None = None,
        test_size: int | None = None,
    ) -> StratifiedTrainValidationTestSplit:
        """Create one stratified train/validation/test split.

        Exactly one validation size selector and exactly one test size selector
        must be provided. Fractions and sizes are measured against the original
        split-unit population, not against the post-test remainder. That makes
        ``validation_fraction=0.2, test_fraction=0.1`` mean roughly 70/20/10
        train/validation/test over products or groups.
        """

        total_units = len(self._units)
        test_unit_count = self._resolve_sample_size(
            fraction=test_fraction,
            size=test_size,
            total=total_units,
            fraction_name="test_fraction",
            size_name="test_size",
        )
        validate_unit_count = self._resolve_sample_size(
            fraction=validation_fraction,
            size=validation_size,
            total=total_units,
            fraction_name="validation_fraction",
            size_name="validation_size",
        )

        if test_unit_count + validate_unit_count >= total_units:
            raise ValueError(
                "validation and test split sizes must leave at least one train split unit; "
                f"got validation={validate_unit_count}, test={test_unit_count}, total={total_units}"
            )

        test_groups = set(self._sample_stratified_units(test_unit_count))
        train_validate_units = [unit for unit in self._units if unit.group not in test_groups]
        train_validate_splitter = self._splitter_from_units(train_validate_units)
        train_validate_split = train_validate_splitter.train_validation_split(validation_size=validate_unit_count)

        return StratifiedTrainValidationTestSplit(
            train_keys=train_validate_split.train_keys,
            validate_keys=train_validate_split.validate_keys,
            test_keys=self._keys_for_groups(test_groups),
            train_groups=train_validate_split.train_groups,
            validate_groups=train_validate_split.validate_groups,
            test_groups=self._ordered_groups(test_groups),
        )

    def k_fold(self, k: int) -> StratifiedKFoldManifest:
        """Create a reproducible group-aware stratified k-fold manifest."""

        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 2:
            raise ValueError("k must be at least 2")
        if k > len(self._units):
            raise ValueError(f"k={k} is larger than the number of split units ({len(self._units)})")

        rng = np.random.default_rng(self.seed)
        units_by_stratum = self._units_by_stratum()
        folds: list[list[_SplitUnit]] = [[] for _ in range(k)]
        fold_sizes = [0] * k

        for stratum in sorted(units_by_stratum, key=self._stable_sort_key):
            stratum_units = list(units_by_stratum[stratum])
            rng.shuffle(stratum_units)
            for unit in stratum_units:
                fold_index = min(range(k), key=lambda i: (fold_sizes[i], i))
                folds[fold_index].append(unit)
                fold_sizes[fold_index] += 1

        all_groups = {unit.group for unit in self._units}
        fold_specs = []
        for fold_index, validate_units in enumerate(folds):
            validate_groups = {unit.group for unit in validate_units}
            train_groups = all_groups - validate_groups
            fold_specs.append(self._fold_from_groups(fold_index, train_groups, validate_groups))

        return StratifiedKFoldManifest(
            k=k,
            seed=self.seed,
            mode=self.mode.value,
            num_bins=self.num_bins if self.mode == StratificationMode.QUANTILE else None,
            stratify_edges=self._stratify_edges,
            folds=tuple(fold_specs),
        )

    def k_buckets(self, k: int) -> list[list[Any]]:
        """Return only the validation-key buckets from ``k_fold``.

        This mirrors demand-forecasting helper code that wants a list of
        product-key buckets and builds folds externally.
        """

        return [list(fold.validate_keys) for fold in self.k_fold(k).folds]

    def holdout_plus_k_fold(
        self,
        *,
        test_fraction: float | None = None,
        test_size: int | None = None,
        k: int,
    ) -> StratifiedHoldoutKFoldManifest:
        """Create a stratified test holdout, then k-folds over the remainder."""

        holdout = self.train_validation_split(validation_fraction=test_fraction, validation_size=test_size)
        test_group_set = set(holdout.validate_groups)
        train_units = [unit for unit in self._units if unit.group not in test_group_set]

        nested = self._splitter_from_units(train_units)
        folds = nested.k_fold(k)
        return StratifiedHoldoutKFoldManifest(
            test_keys=holdout.validate_keys,
            test_groups=holdout.validate_groups,
            folds=folds,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "splitter": "stratified_splitter",
            "seed": self.seed,
            "stratify_mode": self.mode.value,
            "num_bins": self.num_bins if self.mode == StratificationMode.QUANTILE else None,
            "stratify_edges": list(self._stratify_edges),
            "num_split_units": len(self._units),
        }

    def _splitter_from_units(self, units: Sequence[_SplitUnit]) -> StratifiedSplitter:
        keys = []
        strata = []
        groups = []
        for unit in units:
            keys.extend(unit.keys)
            strata.extend([unit.stratum] * len(unit.keys))
            groups.extend([unit.group] * len(unit.keys))

        return StratifiedSplitter(
            keys,
            strata,
            groups=groups,
            mode=StratificationMode.EXPLICIT_BUCKETS,
            seed=self.seed,
        )

    def _resolve_sample_size(
        self,
        *,
        fraction: float | None,
        size: int | None,
        total: int,
        fraction_name: str,
        size_name: str,
    ) -> int:
        if (fraction is None) == (size is None):
            raise ValueError(f"exactly one of {fraction_name} or {size_name} must be provided")
        if size is not None:
            if not isinstance(size, int):
                raise TypeError(f"{size_name} must be an integer")
            if size <= 0 or size >= total:
                raise ValueError(f"{size_name} must be between 1 and {total - 1}")
            return size

        assert fraction is not None
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction >= 1.0:
            raise ValueError(f"{fraction_name} must be greater than 0 and less than 1")
        resolved = int(math.ceil(total * fraction))
        return min(max(resolved, 1), total - 1)

    def _sample_stratified_units(self, sample_size: int) -> list[Any]:
        rng = np.random.default_rng(self.seed)
        units_by_stratum = self._units_by_stratum()
        target_by_stratum = self._allocate_stratified_counts(units_by_stratum, sample_size)

        sampled_groups = []
        for stratum in sorted(units_by_stratum, key=self._stable_sort_key):
            units = list(units_by_stratum[stratum])
            rng.shuffle(units)
            target = target_by_stratum[stratum]
            sampled_groups.extend(unit.group for unit in units[:target])
        return sampled_groups

    def _allocate_stratified_counts(self, units_by_stratum: Mapping[Any, Sequence[_SplitUnit]], sample_size: int) -> dict[Any, int]:
        total = sum(len(units) for units in units_by_stratum.values())
        quotas = []
        for stratum, units in units_by_stratum.items():
            exact = sample_size * len(units) / total
            base = int(math.floor(exact))
            quotas.append([stratum, base, exact - base, len(units)])

        allocated = sum(int(quota[1]) for quota in quotas)
        remaining = sample_size - allocated
        quotas.sort(key=lambda item: (-float(item[2]), self._stable_sort_key(item[0])))
        while remaining > 0:
            progressed = False
            for quota in quotas:
                if remaining == 0:
                    break
                if quota[1] < quota[3]:
                    quota[1] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise RuntimeError("unable to allocate stratified sample counts")

        return {quota[0]: int(quota[1]) for quota in quotas}

    def _split_from_validate_groups(self, validate_groups: set[Any]) -> StratifiedSplit:
        all_groups = {unit.group for unit in self._units}
        return self._split_from_groups(all_groups - validate_groups, validate_groups)

    def _fold_from_groups(self, fold_index: int, train_groups: set[Any], validate_groups: set[Any]) -> StratifiedFold:
        split = self._split_from_groups(train_groups, validate_groups)
        return StratifiedFold(
            fold_index=fold_index,
            train_keys=split.train_keys,
            validate_keys=split.validate_keys,
            train_groups=split.train_groups,
            validate_groups=split.validate_groups,
        )

    def _split_from_groups(self, train_groups: set[Any], validate_groups: set[Any]) -> StratifiedSplit:
        train_group_tuple = self._ordered_groups(train_groups)
        validate_group_tuple = self._ordered_groups(validate_groups)
        return StratifiedSplit(
            train_keys=self._keys_for_groups(train_groups),
            validate_keys=self._keys_for_groups(validate_groups),
            train_groups=train_group_tuple,
            validate_groups=validate_group_tuple,
        )

    def _keys_for_groups(self, groups: set[Any]) -> tuple[Any, ...]:
        out = []
        for key, group in zip(self.keys, self.groups, strict=True):
            if group in groups:
                out.append(key)
        return tuple(out)

    def _ordered_groups(self, groups: set[Any]) -> tuple[Any, ...]:
        return tuple(unit.group for unit in sorted(self._units, key=lambda unit: unit.order) if unit.group in groups)

    def _units_by_stratum(self) -> dict[Any, list[_SplitUnit]]:
        out: dict[Any, list[_SplitUnit]] = defaultdict(list)
        for unit in self._units:
            out[unit.stratum].append(unit)
        return out

    def _build_quantile_units(self, stratify_values: Sequence[Any]) -> tuple[tuple[_SplitUnit, ...], tuple[float, ...]]:
        grouped_values: dict[Any, list[float]] = defaultdict(list)
        grouped_keys: dict[Any, list[Any]] = defaultdict(list)
        group_order: dict[Any, int] = {}

        for index, (key, group, value) in enumerate(zip(self.keys, self.groups, stratify_values, strict=True)):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("quantile stratify_values must be numeric") from exc
            if not math.isfinite(numeric_value):
                raise ValueError("quantile stratify_values must be finite")
            grouped_values[group].append(numeric_value)
            grouped_keys[group].append(key)
            group_order.setdefault(group, index)

        group_magnitudes = {group: float(np.mean(values)) for group, values in grouped_values.items()}
        values = np.asarray(list(group_magnitudes.values()), dtype=np.float64)
        target_bins = self.num_bins if self.num_bins is not None else self._default_num_bins(len(values))
        if target_bins < 1:
            raise ValueError("num_bins must be at least 1")
        target_bins = min(target_bins, len(values))

        edges = tuple(float(edge) for edge in np.unique(np.quantile(values, np.linspace(0.0, 1.0, target_bins + 1)[1:-1])))

        units = []
        for group, keys in grouped_keys.items():
            value = group_magnitudes[group]
            stratum = int(np.searchsorted(edges, value, side="left"))
            units.append(_SplitUnit(group=group, keys=tuple(keys), stratum=stratum, order=group_order[group]))
        units.sort(key=lambda unit: unit.order)
        return tuple(units), edges

    def _build_categorical_units(self, stratify_values: Sequence[Any]) -> tuple[_SplitUnit, ...]:
        grouped_values: dict[Any, list[Any]] = defaultdict(list)
        grouped_keys: dict[Any, list[Any]] = defaultdict(list)
        group_order: dict[Any, int] = {}

        for index, (key, group, value) in enumerate(zip(self.keys, self.groups, stratify_values, strict=True)):
            grouped_values[group].append(value)
            grouped_keys[group].append(key)
            group_order.setdefault(group, index)

        units = []
        for group, values in grouped_values.items():
            unique_values = set(values)
            if len(unique_values) != 1:
                raise ValueError(
                    f"group {group!r} has multiple stratification labels; use quantile mode for numeric aggregation"
                )
            units.append(_SplitUnit(group=group, keys=tuple(grouped_keys[group]), stratum=values[0], order=group_order[group]))
        units.sort(key=lambda unit: unit.order)
        return tuple(units)

    @staticmethod
    def _default_num_bins(num_units: int) -> int:
        if num_units >= 50:
            return 10
        if num_units >= 20:
            return 5
        return max(1, min(3, num_units))

    @staticmethod
    def _stable_sort_key(value: Any) -> tuple[str, str]:
        return (type(value).__name__, repr(value))


__all__ = [
    "StratificationMode",
    "StratifiedFold",
    "NumpyDictSplit",
    "StratifiedHoldoutKFoldManifest",
    "StratifiedKFoldManifest",
    "StratifiedSplit",
    "StratifiedTrainValidationTestSplit",
    "StratifiedSplitter",
    "make_numpy_dict_splits",
]
