import gc
import weakref

import numpy as np
import pytest
import thor


_STORABLE_THOR_DTYPES = [
    thor.DataType.bool,
    thor.DataType.int8,
    thor.DataType.uint8,
    thor.DataType.int16,
    thor.DataType.uint16,
    thor.DataType.int32,
    thor.DataType.uint32,
    thor.DataType.int64,
    thor.DataType.uint64,
    thor.DataType.fp8_e4m3,
    thor.DataType.fp8_e5m2,
    thor.DataType.fp16,
    thor.DataType.bf16,
    thor.DataType.fp32,
    thor.DataType.fp64,
]


def _dataset():
    features = np.ascontiguousarray(np.arange(24, dtype=np.float32).reshape(8, 3))
    labels = np.ascontiguousarray(np.arange(8, dtype=np.float32).reshape(8, 1))
    weights = np.ascontiguousarray(np.linspace(0.5, 1.5, 8, dtype=np.float32))
    return thor.data.NumpyDataset(
        {
            "features": features,
            "labels": labels,
            "weights": weights,
        }
    ), features, labels, weights


def test_numpy_dataset_owns_one_immutable_named_tensor_table():
    dataset, features, labels, weights = _dataset()

    assert dataset.num_examples == 8
    assert dataset.schema.names == ["features", "labels", "weights"]
    assert dataset.field("features").shape == [3]
    assert dataset.field("labels").shape == [1]
    assert dataset.field("weights").shape == [1]
    assert dataset.field("features").dtype == thor.DataType.fp32
    assert not features.flags.writeable
    assert not labels.flags.writeable
    assert not weights.flags.writeable


@pytest.mark.parametrize("data_type", _STORABLE_THOR_DTYPES)
def test_numpy_dataset_accepts_every_storable_thor_dtype_without_conversion(data_type):
    numpy_dtype = thor.physical.numpy_dtypes.from_thor(data_type)
    values = np.arange(12, dtype=np.float32).astype(numpy_dtype).reshape(4, 3)
    values = np.ascontiguousarray(values)
    original_pointer = values.__array_interface__["data"][0]
    dataset = thor.data.NumpyDataset({"values": values})

    assert dataset.field("values").dtype == data_type
    assert dataset.field("values").shape == [3]
    assert values.__array_interface__["data"][0] == original_pointer
    assert not values.flags.writeable


def test_numpy_dataset_requires_exact_contiguous_supported_arrays():
    with pytest.raises(TypeError, match="numpy.ndarray"):
        thor.data.NumpyDataset({"features": [[1.0], [2.0]]})

    non_contiguous = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2]
    with pytest.raises(TypeError, match="C-contiguous"):
        thor.data.NumpyDataset({"features": non_contiguous})

    with pytest.raises(TypeError, match="canonical NumPy/ml_dtypes representation"):
        thor.data.NumpyDataset({"features": np.ones((4, 2), dtype=np.complex64)})

    with pytest.raises(ValueError, match="same leading dimension"):
        thor.data.NumpyDataset(
            {
                "features": np.ones((4, 2), dtype=np.float32),
                "labels": np.ones((3, 1), dtype=np.float32),
            }
        )


def test_numpy_training_data_opens_independent_sessions_and_uses_manifest_membership():
    dataset, *_ = _dataset()
    manifest = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=[0, 1, 2, 3, 4],
        validate_indices=[5, 6],
        test_indices=[7],
    )
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=manifest,
        batching=thor.data.BatchPolicy(batch_size=2, randomize_train=True, random_seed=17),
        dataset_name="numpy_named_examples",
        device_storage="off",
    )

    first = data.open_session(max_in_flight_batches=3)
    second = data.open_session(max_in_flight_batches=3)
    assert first is not second
    assert first.get_num_train_examples() == 5
    assert first.get_num_validate_examples() == 2
    assert first.get_num_test_examples() == 1
    assert first.get_num_train_batches() == 3
    assert first.get_num_validate_batches() == 1
    assert second.get_num_train_batches() == 3


def test_numpy_dataset_can_have_unused_fields_and_bind_only_model_subset():
    dataset, *_ = _dataset()
    network = thor.Network("numpy_subset_contract")
    features = thor.layers.NetworkInput(network, "features", [3], thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "features_out", features.get_feature_output(), thor.DataType.fp32)

    bindings = thor.training.DatasetInputBindings.by_exact_name(network=network, dataset=dataset)
    assert len(bindings) == 1

    manifest = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=[0, 1, 2, 3],
        validate_indices=[4, 5],
    )
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=manifest,
        batching=thor.data.BatchPolicy(batch_size=2, randomize_train=False),
        device_storage="off",
    )
    trainer = thor.training.Trainer(network=network, data=data, input_bindings=bindings)
    assert trainer is not None


def test_numpy_dataset_supports_weak_references():
    dataset, *_ = _dataset()
    dataset_ref = weakref.ref(dataset)
    assert dataset_ref() is dataset
    del dataset
    gc.collect()
    assert dataset_ref() is None


def test_removed_numpy_loader_and_materialized_split_surface_is_absent():
    for name in (
        "NumpyFloat32BatchLoader",
        "NumpyFloat16BatchLoader",
        "NumpyFloat32DictBatchLoader",
        "IndexedNumpyFloat32DictBatchLoader",
    ):
        assert not hasattr(thor.training, name)
    for name in (
        "NumpyDictSplit",
        "NumpyDictSplitIndices",
        "make_numpy_dict_split_indices",
        "make_numpy_dict_splits_DEPRECATED",
    ):
        assert not hasattr(thor.data, name)
