from __future__ import annotations

import pytest
import numpy as np

import thor


def _as_set_list(folds):
    return [set(fold.validate_keys) for fold in folds]


def test_quantile_k_fold_balances_long_tailed_product_buckets():
    keys = [f"product_{i}" for i in range(20)]
    magnitudes = list(range(20))

    splitter = thor.data.StratifiedSplitter(
        keys,
        magnitudes,
        mode=thor.data.StratificationMode.QUANTILE,
        num_bins=4,
        seed=123,
    )

    manifest = splitter.k_fold(5)

    assert len(manifest) == 5
    assert manifest.k == 5
    assert manifest.seed == 123
    assert manifest.mode == "quantile"
    assert len(manifest.stratify_edges) == 3

    validate_sets = _as_set_list(manifest)
    assert set().union(*validate_sets) == set(keys)
    assert sum(len(s) for s in validate_sets) == len(keys)
    assert all(len(fold.validate_keys) == 4 for fold in manifest)
    assert all(len(fold.train_keys) == 16 for fold in manifest)

    # With 20 sorted magnitudes, 4 quantile bins contain five products each.
    # The k-fold splitter should place one product from each bin in every fold.
    for fold in manifest:
        magnitudes_in_fold = sorted(int(key.removeprefix("product_")) for key in fold.validate_keys)
        assert sum(0 <= value <= 4 for value in magnitudes_in_fold) == 1
        assert sum(5 <= value <= 9 for value in magnitudes_in_fold) == 1
        assert sum(10 <= value <= 14 for value in magnitudes_in_fold) == 1
        assert sum(15 <= value <= 19 for value in magnitudes_in_fold) == 1


def test_train_validation_split_uses_exact_stratified_unit_count_and_is_reproducible():
    keys = [f"product_{i}" for i in range(30)]
    magnitudes = [float(i * i) for i in range(30)]

    first = thor.data.StratifiedSplitter(keys, magnitudes, mode="quantile", num_bins=5, seed=77)
    second = thor.data.StratifiedSplitter(keys, magnitudes, mode="quantile", num_bins=5, seed=77)

    first_split = first.train_validation_split(validation_fraction=0.2)
    second_split = second.train_validation_split(validation_fraction=0.2)

    assert first_split.validate_keys == second_split.validate_keys
    assert len(first_split.validate_keys) == 6
    assert len(first_split.train_keys) == 24
    assert set(first_split.train_keys).isdisjoint(first_split.validate_keys)
    assert set(first_split.train_keys) | set(first_split.validate_keys) == set(keys)


def test_grouped_quantile_split_never_leaks_group_across_folds():
    keys = []
    groups = []
    magnitudes = []
    for product_index in range(12):
        for row_index in range(3):
            keys.append(f"product_{product_index}:row_{row_index}")
            groups.append(f"product_{product_index}")
            magnitudes.append(float(product_index))

    splitter = thor.data.StratifiedSplitter(
        keys,
        magnitudes,
        groups=groups,
        mode="quantile",
        num_bins=3,
        seed=5,
    )

    manifest = splitter.k_fold(4)

    assert splitter.num_split_units == 12
    seen_validate_groups = set()
    for fold in manifest:
        validate_groups = set(fold.validate_groups)
        train_groups = set(fold.train_groups)
        assert validate_groups.isdisjoint(train_groups)
        assert len(fold.validate_keys) == len(validate_groups) * 3
        for key in fold.validate_keys:
            group = key.split(":", 1)[0]
            assert group in validate_groups
        seen_validate_groups.update(validate_groups)

    assert seen_validate_groups == {f"product_{i}" for i in range(12)}


def test_categorical_k_fold_preserves_class_proportions_when_divisible():
    keys = [f"a_{i}" for i in range(9)] + [f"b_{i}" for i in range(6)] + [f"c_{i}" for i in range(3)]
    labels = ["a"] * 9 + ["b"] * 6 + ["c"] * 3

    splitter = thor.data.StratifiedSplitter(keys, labels, mode="categorical", seed=101)
    manifest = splitter.k_fold(3)

    for fold in manifest:
        label_counts = {"a": 0, "b": 0, "c": 0}
        for key in fold.validate_keys:
            label_counts[key[0]] += 1
        assert label_counts == {"a": 3, "b": 2, "c": 1}


def test_k_buckets_returns_validation_key_buckets_for_legacy_demand_style_usage():
    keys = [f"product_{i}" for i in range(10)]
    magnitudes = list(range(10))

    splitter = thor.data.StratifiedSplitter(keys, magnitudes, mode="quantile", num_bins=2, seed=8)
    buckets = splitter.k_buckets(5)

    assert len(buckets) == 5
    assert all(len(bucket) == 2 for bucket in buckets)
    assert set().union(*(set(bucket) for bucket in buckets)) == set(keys)


def test_holdout_plus_k_fold_reserves_stratified_test_groups():
    keys = [f"product_{i}" for i in range(15)]
    magnitudes = list(range(15))

    splitter = thor.data.StratifiedSplitter(keys, magnitudes, mode="quantile", num_bins=3, seed=22)
    manifest = splitter.holdout_plus_k_fold(test_size=3, k=4)

    test_keys = set(manifest.test_keys)
    assert len(test_keys) == 3
    for fold in manifest.folds:
        assert test_keys.isdisjoint(fold.train_keys)
        assert test_keys.isdisjoint(fold.validate_keys)

    folded_keys = set()
    for fold in manifest.folds:
        folded_keys.update(fold.validate_keys)
    assert folded_keys | test_keys == set(keys)


def test_train_validation_test_split_reserves_three_disjoint_stratified_sets():
    keys = [f"product_{i}" for i in range(30)]
    magnitudes = list(range(30))

    splitter = thor.data.StratifiedSplitter(keys, magnitudes, mode="quantile", num_bins=5, seed=42)
    split = splitter.train_validation_test_split(validation_fraction=0.2, test_fraction=0.1)

    assert isinstance(split, thor.data.StratifiedTrainValidationTestSplit)
    assert len(split.train_keys) == 21
    assert len(split.validate_keys) == 6
    assert len(split.test_keys) == 3

    train_keys = set(split.train_keys)
    validate_keys = set(split.validate_keys)
    test_keys = set(split.test_keys)

    assert train_keys.isdisjoint(validate_keys)
    assert train_keys.isdisjoint(test_keys)
    assert validate_keys.isdisjoint(test_keys)
    assert train_keys | validate_keys | test_keys == set(keys)

    manifest = split.to_dict()
    assert manifest["splitter"] == "stratified_group_train_validation_test_split"
    assert set(manifest["train_keys"]) == train_keys
    assert set(manifest["validate_keys"]) == validate_keys
    assert set(manifest["test_keys"]) == test_keys


def test_grouped_train_validation_test_split_never_leaks_groups():
    keys = []
    groups = []
    magnitudes = []
    for product_index in range(20):
        for row_index in range(2):
            keys.append(f"product_{product_index}:row_{row_index}")
            groups.append(f"product_{product_index}")
            magnitudes.append(float(product_index))

    splitter = thor.data.StratifiedSplitter(
        keys,
        magnitudes,
        groups=groups,
        mode="quantile",
        num_bins=4,
        seed=11,
    )

    split = splitter.train_validation_test_split(validation_size=4, test_size=5)

    assert len(split.train_groups) == 11
    assert len(split.validate_groups) == 4
    assert len(split.test_groups) == 5
    assert len(split.train_keys) == 22
    assert len(split.validate_keys) == 8
    assert len(split.test_keys) == 10

    train_groups = set(split.train_groups)
    validate_groups = set(split.validate_groups)
    test_groups = set(split.test_groups)

    assert train_groups.isdisjoint(validate_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validate_groups.isdisjoint(test_groups)
    assert train_groups | validate_groups | test_groups == {f"product_{i}" for i in range(20)}

    for key in split.train_keys:
        assert key.split(":", 1)[0] in train_groups
    for key in split.validate_keys:
        assert key.split(":", 1)[0] in validate_groups
    for key in split.test_keys:
        assert key.split(":", 1)[0] in test_groups


def test_train_validation_test_split_rejects_sizes_that_leave_no_training_units():
    splitter = thor.data.StratifiedSplitter(["a", "b", "c"], [0.0, 1.0, 2.0], seed=4)

    with pytest.raises(ValueError, match="leave at least one train split unit"):
        splitter.train_validation_test_split(validation_size=1, test_size=2)


def test_splitter_manifest_is_serializable_and_contains_edges():
    splitter = thor.data.StratifiedSplitter(["p0", "p1", "p2", "p3"], [0.0, 1.0, 2.0, 3.0], num_bins=2, seed=9)

    manifest = splitter.k_fold(2).to_dict()

    assert manifest["splitter"] == "stratified_group_k_fold"
    assert manifest["seed"] == 9
    assert manifest["stratify_mode"] == "quantile"
    assert manifest["num_bins"] == 2
    assert manifest["stratify_edges"] == [1.5]
    assert [fold["fold_index"] for fold in manifest["folds"]] == [0, 1]


def test_splitter_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="keys must contain"):
        thor.data.StratifiedSplitter([], [])

    with pytest.raises(ValueError, match="stratify_values length"):
        thor.data.StratifiedSplitter(["a", "b"], [1.0])

    with pytest.raises(ValueError, match="groups length"):
        thor.data.StratifiedSplitter(["a", "b"], [1.0, 2.0], groups=["g"])

    with pytest.raises(ValueError, match="finite"):
        thor.data.StratifiedSplitter(["a", "b"], [1.0, float("nan")], mode="quantile")

    with pytest.raises(ValueError, match="multiple stratification labels"):
        thor.data.StratifiedSplitter(
            ["a0", "a1"],
            ["cold", "hot"],
            groups=["a", "a"],
            mode="categorical",
        )

    splitter = thor.data.StratifiedSplitter(["a", "b"], [0.0, 1.0])
    with pytest.raises(ValueError, match="at least 2"):
        splitter.k_fold(1)
    with pytest.raises(ValueError, match="larger than"):
        splitter.k_fold(3)
    with pytest.raises(ValueError, match="exactly one"):
        splitter.train_validation_split()
    with pytest.raises(ValueError, match="greater than 0"):
        splitter.train_validation_split(validation_fraction=0.0)


def test_make_numpy_dict_splits_partitions_demand_style_arrays_by_group_with_explicit_test_split():
    product_ids = [f"product_{i}" for i in range(8)]
    row_groups = []
    row_keys = []
    for product_id in product_ids:
        for date_idx in range(3):
            row_groups.append(product_id)
            row_keys.append(f"{product_id}:date_{date_idx}")

    splitter = thor.data.StratifiedSplitter(
        row_keys,
        [float(product_id.rsplit("_", 1)[1]) for product_id in row_groups],
        groups=row_groups,
        mode="quantile",
        num_bins=4,
        seed=7,
    )
    split = splitter.train_validation_test_split(validation_size=2, test_size=2)

    tensors = {
        "trend_inputs": np.arange(len(row_keys) * 2, dtype=np.float32).reshape(len(row_keys), 2),
        "seasonality_inputs": np.arange(len(row_keys) * 3, dtype=np.float32).reshape(len(row_keys), 3),
        "monotone_increasing_inputs": np.arange(len(row_keys), dtype=np.float32).reshape(len(row_keys), 1),
        "forecast": np.arange(len(row_keys), dtype=np.float32).reshape(len(row_keys), 1),
        "example_weights": np.ones((len(row_keys), 1), dtype=np.float32),
    }

    partitions = thor.data.make_numpy_dict_splits(tensors, groups=row_groups, split=split)

    assert isinstance(partitions, thor.data.NumpyDictSplit)
    assert partitions.test is not None
    assert set(partitions.train) == set(tensors)
    assert set(partitions.validate) == set(tensors)
    assert set(partitions.test) == set(tensors)
    assert partitions.train["trend_inputs"].shape[0] == len(split.train_groups) * 3
    assert partitions.validate["trend_inputs"].shape[0] == len(split.validate_groups) * 3
    assert partitions.test["trend_inputs"].shape[0] == len(split.test_groups) * 3
    assert partitions.train["seasonality_inputs"].shape[1:] == (3,)
    assert partitions.train["monotone_increasing_inputs"].shape[1:] == (1,)

    validate_groups = set(split.validate_groups)
    expected_validate_indices = [idx for idx, group in enumerate(row_groups) if group in validate_groups]
    assert np.array_equal(partitions.validate["forecast"], tensors["forecast"][expected_validate_indices])


def test_make_numpy_dict_splits_partitions_by_key_for_plain_train_validation_split():
    keys = [f"row_{i}" for i in range(6)]
    splitter = thor.data.StratifiedSplitter(keys, [0, 0, 1, 1, 2, 2], mode="categorical", seed=2)
    split = splitter.train_validation_split(validation_size=2)
    tensors = {"x": np.arange(12, dtype=np.float32).reshape(6, 2), "y": np.arange(6, dtype=np.float32).reshape(6, 1)}

    partitions = thor.data.make_numpy_dict_splits(tensors, keys=keys, split=split)

    assert partitions.test is None
    assert partitions.train["x"].shape[0] == len(split.train_keys)
    assert partitions.validate["x"].shape[0] == len(split.validate_keys)
    expected_train_indices = [idx for idx, key in enumerate(keys) if key in set(split.train_keys)]
    assert np.array_equal(partitions.train["y"], tensors["y"][expected_train_indices])


def test_make_numpy_dict_splits_rejects_ambiguous_or_mismatched_selectors():
    splitter = thor.data.StratifiedSplitter(["a", "b", "c"], [0.0, 1.0, 2.0], seed=1)
    split = splitter.train_validation_split(validation_size=1)
    tensors = {"x": np.zeros((3, 2), dtype=np.float32)}

    with pytest.raises(ValueError, match="exactly one of keys or groups"):
        thor.data.make_numpy_dict_splits(tensors, split=split)
    with pytest.raises(ValueError, match="exactly one of keys or groups"):
        thor.data.make_numpy_dict_splits(tensors, keys=["a", "b", "c"], groups=["a", "b", "c"], split=split)
    with pytest.raises(ValueError, match="keys length must match"):
        thor.data.make_numpy_dict_splits(tensors, keys=["a", "b"], split=split)
    with pytest.raises(ValueError, match="same leading example dimension"):
        thor.data.make_numpy_dict_splits(
            {"x": np.zeros((3, 2), dtype=np.float32), "y": np.zeros((2, 1), dtype=np.float32)},
            keys=["a", "b", "c"],
            split=split,
        )
