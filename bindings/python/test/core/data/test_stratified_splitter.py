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


def test_numpy_dataset_manifest_partitions_demand_style_rows_by_group_with_explicit_test_split():
    product_ids = [f"product_{i}" for i in range(8)]
    row_groups = [product_id for product_id in product_ids for _ in range(3)]
    tensors = {
        "trend_inputs": np.arange(len(row_groups) * 2, dtype=np.float32).reshape(len(row_groups), 2),
        "forecast": np.arange(len(row_groups), dtype=np.float32).reshape(len(row_groups), 1),
    }
    dataset = thor.data.NumpyDataset({name: np.ascontiguousarray(value) for name, value in tensors.items()})
    manifest = thor.data.StratifiedSplitter.train_validation_test_manifest(
        dataset=dataset,
        strata=[float(product_id.rsplit("_", 1)[1]) for product_id in row_groups],
        groups=row_groups,
        validation_size=2,
        test_size=2,
        mode="quantile",
        num_bins=4,
        seed=7,
    )

    train_groups = {row_groups[index] for index in manifest.train.indices}
    validate_groups = {row_groups[index] for index in manifest.validate.indices}
    test_groups = {row_groups[index] for index in manifest.test.indices}
    assert train_groups.isdisjoint(validate_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validate_groups.isdisjoint(test_groups)
    assert len(manifest.train) + len(manifest.validate) + len(manifest.test) == len(row_groups)
    assert manifest.dataset_id == dataset.id
    assert not tensors["forecast"].flags.writeable


def test_numpy_dataset_manifest_uses_canonical_row_ids_for_plain_train_validation_split():
    tensors = {
        "x": np.ascontiguousarray(np.arange(12, dtype=np.float32).reshape(6, 2)),
        "y": np.ascontiguousarray(np.arange(6, dtype=np.float32).reshape(6, 1)),
    }
    dataset = thor.data.NumpyDataset(tensors)
    manifest = thor.data.StratifiedSplitter.train_validation_manifest(
        dataset=dataset,
        strata=[0, 0, 1, 1, 2, 2],
        validation_size=2,
        mode="categorical",
        seed=2,
    )

    assert len(manifest.train) == 4
    assert len(manifest.validate) == 2
    assert manifest.test_aliases_validate
    assert set(manifest.train.indices).isdisjoint(manifest.validate.indices)
    assert set(manifest.train.indices) | set(manifest.validate.indices) == set(range(6))


def test_numpy_dataset_and_manifest_reject_mismatched_table_or_selector_lengths():
    with pytest.raises(ValueError, match="same leading dimension"):
        thor.data.NumpyDataset(
            {
                "x": np.zeros((3, 2), dtype=np.float32),
                "y": np.zeros((2, 1), dtype=np.float32),
            }
        )

    dataset = thor.data.NumpyDataset({"x": np.zeros((3, 2), dtype=np.float32)})
    with pytest.raises(ValueError, match="strata length"):
        thor.data.StratifiedSplitter.train_validation_manifest(
            dataset=dataset, strata=[0.0, 1.0], validation_size=1
        )
    with pytest.raises(ValueError, match="groups length"):
        thor.data.StratifiedSplitter.train_validation_manifest(
            dataset=dataset, strata=[0.0, 1.0, 2.0], groups=["a", "b"], validation_size=1
        )


def test_materialized_numpy_split_helpers_are_removed():
    assert not hasattr(thor.data, "NumpyDictSplit")
    assert not hasattr(thor.data, "NumpyDictSplitIndices")
    assert not hasattr(thor.data, "make_numpy_dict_split_indices")
    assert not hasattr(thor.data, "make_numpy_dict_splits_DEPRECATED")
