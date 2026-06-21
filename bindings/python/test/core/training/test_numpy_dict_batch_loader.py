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
