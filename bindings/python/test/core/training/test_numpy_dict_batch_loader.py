from __future__ import annotations

import numpy as np
import pytest

import thor


def test_numpy_float32_dict_batch_loader_accepts_named_tensors_and_scalar_weights():
    train = {
        "trend_inputs": np.arange(24, dtype=np.float32).reshape(6, 4),
        "seasonality_inputs": np.arange(18, dtype=np.float32).reshape(6, 3),
        "monotone_increasing_inputs": np.arange(12, dtype=np.float32).reshape(6, 2),
        "sales": np.arange(30, dtype=np.float32).reshape(6, 5),
        "example_weights": np.ones(6, dtype=np.float32),
    }
    validate = {
        "trend_inputs": np.zeros((4, 4), dtype=np.float32),
        "seasonality_inputs": np.zeros((4, 3), dtype=np.float32),
        "monotone_increasing_inputs": np.zeros((4, 2), dtype=np.float32),
        "sales": np.zeros((4, 5), dtype=np.float32),
        "example_weights": np.ones(4, dtype=np.float32),
    }

    loader = thor.training.NumpyFloat32DictBatchLoader(
        train=train,
        validate=validate,
        batch_size=4,
        dataset_name="keyed_demand",
        randomize_train=True,
    )

    assert loader.get_dataset_name() == "keyed_demand"
    assert loader.get_batch_size() == 4
    assert loader.get_num_train_examples() == 6
    assert loader.get_num_validate_examples() == 4
    assert loader.get_num_train_batches() == 2
    assert loader.get_num_validate_batches() == 1
    assert set(loader.get_tensor_names()) == set(train)
    assert loader.get_tensor_shapes() == {
        "trend_inputs": [4],
        "seasonality_inputs": [3],
        "monotone_increasing_inputs": [2],
        "sales": [5],
        "example_weights": [1],
    }


def test_numpy_float32_dict_batch_loader_converts_array_like_values_to_float32():
    loader = thor.training.NumpyFloat32DictBatchLoader(
        train={"x": [[1.0, 2.0], [3.0, 4.0]], "y": [[1.0], [0.0]]},
        validate={"x": np.zeros((1, 2), dtype=np.float64), "y": np.zeros((1, 1), dtype=np.float64)},
        batch_size=2,
    )

    assert loader.get_tensor_shapes() == {"x": [2], "y": [1]}


def test_numpy_float32_dict_batch_loader_rejects_mismatched_split_names():
    with pytest.raises(ValueError, match="same tensor names"):
        thor.training.NumpyFloat32DictBatchLoader(
            train={"x": np.zeros((2, 3), dtype=np.float32), "y": np.zeros((2, 1), dtype=np.float32)},
            validate={"x": np.zeros((2, 3), dtype=np.float32)},
            batch_size=2,
        )


def test_numpy_float32_dict_batch_loader_rejects_mismatched_leading_dimension():
    with pytest.raises(ValueError, match="same leading dimension"):
        thor.training.NumpyFloat32DictBatchLoader(
            train={"x": np.zeros((2, 3), dtype=np.float32), "y": np.zeros((3, 1), dtype=np.float32)},
            validate={"x": np.zeros((2, 3), dtype=np.float32), "y": np.zeros((2, 1), dtype=np.float32)},
            batch_size=2,
        )


def test_numpy_float32_dict_batch_loader_rejects_mismatched_non_batch_shape():
    with pytest.raises(ValueError, match="matching non-batch shapes"):
        thor.training.NumpyFloat32DictBatchLoader(
            train={"x": np.zeros((2, 3), dtype=np.float32)},
            validate={"x": np.zeros((2, 4), dtype=np.float32)},
            batch_size=2,
        )


def test_numpy_float32_dict_batch_loader_accepts_explicit_test_split_and_reports_counts():
    train = {
        "trend_inputs": np.zeros((7, 4), dtype=np.float32),
        "sales": np.zeros((7, 2), dtype=np.float32),
    }
    validate = {
        "trend_inputs": np.zeros((3, 4), dtype=np.float32),
        "sales": np.zeros((3, 2), dtype=np.float32),
    }
    test = {
        "trend_inputs": np.zeros((5, 4), dtype=np.float32),
        "sales": np.zeros((5, 2), dtype=np.float32),
    }

    loader = thor.training.NumpyFloat32DictBatchLoader(
        train=train,
        validate=validate,
        test=test,
        batch_size=4,
        randomize_train=False,
        batch_queue_depth=3,
    )

    assert loader.has_explicit_test_split()
    assert not loader.get_randomize_train()
    assert loader.get_random_seed() is None
    assert loader.get_batch_queue_depth() == 3
    assert loader.get_num_train_examples() == 7
    assert loader.get_num_validate_examples() == 3
    assert loader.get_num_test_examples() == 5
    assert loader.get_num_train_batches() == 2
    assert loader.get_num_validate_batches() == 1
    assert loader.get_num_test_batches() == 2


def test_numpy_float32_dict_batch_loader_defaults_test_to_validate_for_backcompat():
    train = {"x": np.zeros((4, 2), dtype=np.float32)}
    validate = {"x": np.zeros((3, 2), dtype=np.float32)}

    loader = thor.training.NumpyFloat32DictBatchLoader(
        train=train,
        validate=validate,
        batch_size=2,
        randomize_train=False,
    )

    assert not loader.has_explicit_test_split()
    assert loader.get_num_validate_examples() == 3
    assert loader.get_num_test_examples() == 3
    assert loader.get_num_validate_batches() == 2
    assert loader.get_num_test_batches() == 2


def test_numpy_float32_dict_batch_loader_records_seed_and_requires_randomization():
    loader = thor.training.NumpyFloat32DictBatchLoader(
        train={"x": np.zeros((4, 2), dtype=np.float32)},
        validate={"x": np.zeros((2, 2), dtype=np.float32)},
        batch_size=2,
        randomize_train=True,
        random_seed=12345,
    )

    assert loader.get_randomize_train()
    assert loader.get_random_seed() == 12345

    with pytest.raises(ValueError, match="random_seed requires randomize_train=True"):
        thor.training.NumpyFloat32DictBatchLoader(
            train={"x": np.zeros((4, 2), dtype=np.float32)},
            validate={"x": np.zeros((2, 2), dtype=np.float32)},
            batch_size=2,
            randomize_train=False,
            random_seed=12345,
        )


def test_numpy_float32_dict_batch_loader_rejects_mismatched_test_split():
    with pytest.raises(ValueError, match="train and test dicts must have the same tensor names"):
        thor.training.NumpyFloat32DictBatchLoader(
            train={"x": np.zeros((2, 3), dtype=np.float32), "y": np.zeros((2, 1), dtype=np.float32)},
            validate={"x": np.zeros((2, 3), dtype=np.float32), "y": np.zeros((2, 1), dtype=np.float32)},
            test={"x": np.zeros((2, 3), dtype=np.float32)},
            batch_size=2,
        )

    with pytest.raises(ValueError, match="train and test tensor 'x' must have matching non-batch shapes"):
        thor.training.NumpyFloat32DictBatchLoader(
            train={"x": np.zeros((2, 3), dtype=np.float32)},
            validate={"x": np.zeros((2, 3), dtype=np.float32)},
            test={"x": np.zeros((2, 4), dtype=np.float32)},
            batch_size=2,
        )


def test_numpy_float32_dict_batch_loader_rejects_non_dict_test_split():
    with pytest.raises(TypeError, match="test must be a dict or None"):
        thor.training.NumpyFloat32DictBatchLoader(
            train={"x": np.zeros((2, 3), dtype=np.float32)},
            validate={"x": np.zeros((2, 3), dtype=np.float32)},
            test=[("x", np.zeros((2, 3), dtype=np.float32))],
            batch_size=2,
        )


def test_numpy_float32_dict_batch_loader_composes_with_stratified_demand_split_helpers():
    product_ids = [f"product_{i}" for i in range(6)]
    row_groups = []
    for product_id in product_ids:
        row_groups.extend([product_id, product_id])
    num_rows = len(row_groups)

    tensors = {
        "trend_inputs": np.arange(num_rows, dtype=np.float32).reshape(num_rows, 1),
        "seasonality_inputs": np.arange(num_rows * 2, dtype=np.float32).reshape(num_rows, 2),
        "monotone_increasing_inputs": np.arange(num_rows, dtype=np.float32).reshape(num_rows, 1),
        "forecast_labels": np.arange(num_rows * 4, dtype=np.float32).reshape(num_rows, 4),
        "example_weights": np.ones((num_rows, 1), dtype=np.float32),
    }

    split_manifest = thor.data.StratifiedSplitter(
        product_ids,
        [float(index) for index in range(len(product_ids))],
        mode="quantile",
        num_bins=3,
        seed=11,
    ).holdout_plus_k_fold(test_size=2, k=2)
    fold = split_manifest.folds[0]
    fold_with_holdout = thor.data.StratifiedTrainValidationTestSplit(
        train_keys=fold.train_keys,
        validate_keys=fold.validate_keys,
        test_keys=split_manifest.test_keys,
        train_groups=fold.train_groups,
        validate_groups=fold.validate_groups,
        test_groups=split_manifest.test_groups,
    )

    split = thor.data.make_numpy_dict_splits(tensors, split=fold_with_holdout, groups=row_groups)
    loader = thor.training.NumpyFloat32DictBatchLoader(
        train=split.train,
        validate=split.validate,
        test=split.test,
        batch_size=2,
        randomize_train=False,
        dataset_name="demand_kfold_smoke",
    )

    assert set(loader.get_tensor_names()) == set(tensors)
    assert loader.has_explicit_test_split()
    assert loader.get_num_train_examples() == 4
    assert loader.get_num_validate_examples() == 4
    assert loader.get_num_test_examples() == 4
    assert loader.get_tensor_shapes() == {
        "trend_inputs": [1],
        "seasonality_inputs": [2],
        "monotone_increasing_inputs": [1],
        "forecast_labels": [4],
        "example_weights": [1],
    }
