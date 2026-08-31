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


@pytest.mark.parametrize("offset_dtype", [np.uint32, np.uint64])
def test_numpy_dataset_accepts_canonical_ragged_batches(offset_dtype):
    features = np.ascontiguousarray(np.arange(15, dtype=np.float32).reshape(5, 3))
    label_values = np.ascontiguousarray(np.asarray([10, 20, 21, 40, 50, 51], dtype=np.int32))
    label_offsets = np.ascontiguousarray(np.asarray([0, 1, 3, 3, 4, 6], dtype=offset_dtype))
    vector_values = np.ascontiguousarray(
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    )
    vector_offsets = np.ascontiguousarray(np.asarray([0, 1, 1, 2, 3, 3], dtype=offset_dtype))

    dataset = thor.data.NumpyDataset(
        {"features": features},
        ragged_tensors={
            "labels": thor.data.RaggedBatch(label_values, label_offsets),
            "vectors": thor.data.RaggedBatch(vector_values, vector_offsets),
        },
    )

    assert dataset.num_examples == 5
    assert dataset.schema.names == ["features", "labels", "vectors"]
    assert dataset.field("labels").kind == thor.data.DatasetFieldKind.RAGGED
    assert dataset.field("labels").shape == []
    assert dataset.field("labels").dtype == thor.DataType.int32
    assert dataset.field("vectors").kind == thor.data.DatasetFieldKind.RAGGED
    assert dataset.field("vectors").shape == [2]
    assert dataset.field("vectors").dtype == thor.DataType.fp32
    assert not features.flags.writeable
    assert not label_values.flags.writeable
    assert not label_offsets.flags.writeable
    assert not vector_values.flags.writeable
    assert not vector_offsets.flags.writeable


def test_numpy_dataset_supports_ragged_only_and_all_empty_rows():
    values = np.ascontiguousarray(np.empty((0, 2), dtype=np.float32))
    offsets = np.ascontiguousarray(np.zeros((5,), dtype=np.uint64))
    dataset = thor.data.NumpyDataset(
        ragged_tensors={"tokens": thor.data.RaggedBatch(values, offsets)}
    )

    assert dataset.num_examples == 4
    assert dataset.schema.names == ["tokens"]
    assert dataset.field("tokens").kind == thor.data.DatasetFieldKind.RAGGED
    assert dataset.field("tokens").shape == [2]


def test_numpy_dataset_rejects_malformed_ragged_batches():
    values = np.ascontiguousarray(np.asarray([1, 2, 3], dtype=np.int32))

    with pytest.raises(TypeError, match="thor.data.RaggedBatch"):
        thor.data.NumpyDataset(ragged_tensors={"labels": (values, np.asarray([0, 3], dtype=np.uint32))})

    with pytest.raises(TypeError, match="numpy.uint32 or numpy.uint64"):
        thor.data.NumpyDataset(
            ragged_tensors={
                "labels": thor.data.RaggedBatch(values.copy(), np.asarray([0, 3], dtype=np.int32))
            }
        )

    with pytest.raises(ValueError, match=r"offsets\[0\] must be zero"):
        thor.data.NumpyDataset(
            ragged_tensors={
                "labels": thor.data.RaggedBatch(values.copy(), np.asarray([1, 3], dtype=np.uint32))
            }
        )

    with pytest.raises(ValueError, match="nondecreasing"):
        thor.data.NumpyDataset(
            ragged_tensors={
                "labels": thor.data.RaggedBatch(values.copy(), np.asarray([0, 2, 1, 3], dtype=np.uint32))
            }
        )

    with pytest.raises(ValueError, match=r"offsets\[-1\] must equal values.shape\[0\]"):
        thor.data.NumpyDataset(
            ragged_tensors={
                "labels": thor.data.RaggedBatch(values.copy(), np.asarray([0, 2], dtype=np.uint32))
            }
        )

    with pytest.raises(ValueError, match="same example count"):
        thor.data.NumpyDataset(
            {"features": np.ones((3, 1), dtype=np.float32)},
            ragged_tensors={
                "labels": thor.data.RaggedBatch(values.copy(), np.asarray([0, 1, 3], dtype=np.uint32))
            },
        )

    with pytest.raises(ValueError, match="duplicate field name"):
        thor.data.NumpyDataset(
            {"labels": np.ones((1, 1), dtype=np.float32)},
            ragged_tensors={
                "labels": thor.data.RaggedBatch(values.copy(), np.asarray([0, 3], dtype=np.uint32))
            },
        )


@pytest.mark.cuda
@pytest.mark.parametrize("device_storage", ["off", "strict"])
@pytest.mark.parametrize("source_offsets_dtype", [np.uint32, np.uint64])
def test_numpy_dataset_ragged_batches_train_through_canonical_ctc_with_exact_partial_tail(
    device_storage, source_offsets_dtype
):
    num_examples = 5
    time_steps = 4
    input_features = 2
    num_classes = 3
    batch_size = 3

    features = np.ascontiguousarray(
        np.linspace(-0.5, 0.5, num_examples * time_steps * input_features, dtype=np.float32).reshape(
            num_examples, time_steps, input_features
        )
    )
    input_lengths_values = np.ascontiguousarray(
        np.full((num_examples, 1), time_steps, dtype=np.int32)
    )
    label_values = np.ascontiguousarray(np.asarray([1, 1, 2, 2, 1, 2], dtype=np.int32))
    # Per-example targets: [1], [1,2], [], [2], [1,2].
    label_offsets = np.ascontiguousarray(
        np.asarray([0, 1, 3, 3, 4, 6], dtype=source_offsets_dtype)
    )
    dataset = thor.data.NumpyDataset(
        {"features": features, "input_lengths": input_lengths_values},
        ragged_tensors={"labels": thor.data.RaggedBatch(label_values, label_offsets)},
    )
    manifest = thor.data.DatasetSplitManifest(
        dataset=dataset,
        train_indices=[4, 0, 2],
        validate_indices=[1, 3],
    )
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=manifest,
        batching=thor.data.BatchPolicy(batch_size=batch_size, randomize_train=False),
        device_storage=device_storage,
    )

    network = thor.Network(
        f"numpy_ragged_ctc_exact_partial_tail_{device_storage}_{np.dtype(source_offsets_dtype).name}"
    )
    feature_input = thor.layers.NetworkInput(
        network, "features", [time_steps, input_features], thor.DataType.fp32
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        "labels",
        thor.DataType.int32,
        [],
        max_total_values=5,
        max_values_per_row=2,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    input_lengths = thor.layers.NetworkInput(
        network, "input_lengths", [1], thor.DataType.int32
    )
    logits = thor.layers.FullyConnected(
        network,
        feature_input.get_feature_output(),
        num_classes,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_initializer=thor.initializers.UniformRandom(-0.05, 0.05),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    )
    loss = thor.losses.CtcLoss(
        network,
        logits.get_feature_output(),
        labels,
        input_lengths.get_feature_output(),
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)

    bindings = thor.training.DatasetInputBindings.by_exact_name(network=network, dataset=dataset)
    trainer = thor.training.Trainer(
        network=network,
        data=data,
        input_bindings=bindings,
        optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        stats_color="never",
    )
    result = trainer.fit(1)
    assert result.status == "completed"
    report = result.final_training_stats.device_dataset_storage
    if device_storage == "strict":
        assert report.used
        assert report.reason == ""
        expected_resident_bytes = (
            features.nbytes
            + input_lengths_values.nbytes
            + label_values.nbytes
            + num_examples * 2 * np.dtype(np.uint64).itemsize
        )
        assert report.resident_bytes == expected_resident_bytes
        assert report.required_bytes >= report.resident_bytes
    else:
        assert not report.used


def _numpy_ragged_conv_training_fixture(*, max_values_per_row: int):
    batch_size = 3
    channels = 2
    output_channels = 3
    row_lengths = [5, 2, 4, 1]
    offsets = np.ascontiguousarray(
        np.asarray([0, 5, 7, 11, 12], dtype=np.uint32)
    )
    values = np.ascontiguousarray(
        np.linspace(-0.75, 0.75, offsets[-1] * channels, dtype=np.float32).reshape(
            int(offsets[-1]), channels
        )
    )
    labels = np.ascontiguousarray(
        np.linspace(-0.2, 0.2, len(row_lengths) * output_channels, dtype=np.float32).reshape(
            len(row_lengths), output_channels
        )
    )
    dataset = thor.data.NumpyDataset(
        {"labels": labels},
        ragged_tensors={"history": thor.data.RaggedBatch(values, offsets)},
    )
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=thor.data.DatasetSplitManifest(
            dataset=dataset,
            train_indices=[0, 1, 2],
            validate_indices=[],
        ),
        batching=thor.data.BatchPolicy(batch_size=batch_size, randomize_train=False),
        # N1 specifically covers the host-backed NumpyBatchSession adapter.
        device_storage="off",
    )

    network = thor.Network(f"numpy_ragged_conv_max_row_{max_values_per_row}")
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.fp32,
        [channels],
        max_total_values=12,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=max_values_per_row,
    )
    labels_input = thor.layers.NetworkInput(
        network, "labels", [output_channels], thor.DataType.fp32
    )
    convolved = thor.layers.Convolution1d(
        network,
        history,
        num_output_channels=output_channels,
        filter_width=3,
        padding="causal",
        dilation=2,
        activation=None,
        weights_initializer=thor.initializers.UniformRandom(-0.05, 0.05),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    ).get_feature_output()
    pooled = thor.layers.SegmentedReduction(
        network, convolved, thor.layers.SegmentedReduction.Type.mean
    ).get_feature_output()
    loss = thor.losses.MSE(
        network,
        pooled,
        labels_input.get_feature_output(),
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)

    trainer = thor.training.Trainer(
        network=network,
        data=data,
        input_bindings=thor.training.DatasetInputBindings.by_exact_name(
            network=network, dataset=dataset
        ),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
        debug_synchronous=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        stats_color="never",
    )
    return trainer


@pytest.mark.cuda
def test_numpy_dataset_preserves_ragged_capacity_and_runtime_extent_for_conv1d_training():
    trainer = _numpy_ragged_conv_training_fixture(max_values_per_row=5)

    result = trainer.fit(1)

    assert result.status == "completed"


@pytest.mark.cuda
def test_numpy_dataset_rejects_selected_row_exceeding_requested_max_values_per_row():
    trainer = _numpy_ragged_conv_training_fixture(max_values_per_row=4)

    with pytest.raises(RuntimeError, match=r"exceeding maxValuesPerRow=4"):
        trainer.fit(1)


@pytest.mark.cuda
def test_numpy_dataset_device_residency_supports_all_empty_ragged_storage():
    num_examples = 3
    time_steps = 3
    num_classes = 3
    batch_size = 2
    logits_values = np.ascontiguousarray(
        np.linspace(-0.2, 0.2, num_examples * time_steps * num_classes, dtype=np.float32).reshape(
            num_examples, time_steps, num_classes
        )
    )
    input_lengths_values = np.ascontiguousarray(
        np.full((num_examples, 1), time_steps, dtype=np.int32)
    )
    label_values = np.ascontiguousarray(np.empty((0,), dtype=np.int32))
    label_offsets = np.ascontiguousarray(np.zeros((num_examples + 1,), dtype=np.uint64))
    dataset = thor.data.NumpyDataset(
        {"logits": logits_values, "input_lengths": input_lengths_values},
        ragged_tensors={"labels": thor.data.RaggedBatch(label_values, label_offsets)},
    )
    data = thor.data.TrainingData(
        dataset=dataset,
        splits=thor.data.DatasetSplitManifest(
            dataset=dataset,
            train_indices=[0, 1, 2],
            validate_indices=[],
        ),
        batching=thor.data.BatchPolicy(batch_size=batch_size, randomize_train=False),
        device_storage="strict",
    )

    network = thor.Network("numpy_ragged_device_resident_all_empty")
    logits = thor.layers.NetworkInput(
        network, "logits", [time_steps, num_classes], thor.DataType.fp32
    )
    labels = thor.layers.RaggedNetworkInput(
        network,
        "labels",
        thor.DataType.int32,
        [],
        max_total_values=1,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint64,
    )
    input_lengths = thor.layers.NetworkInput(
        network, "input_lengths", [1], thor.DataType.int32
    )
    # Keep this as a real training-path residency test.  CTC itself has no trainable
    # parameters, so project the resident input through a tiny trainable layer rather
    # than relying on Trainer.fit() to accept a parameter-free graph.
    projected_logits = thor.layers.FullyConnected(
        network,
        logits.get_feature_output(),
        num_classes,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_initializer=thor.initializers.UniformRandom(-0.05, 0.05),
        biases_initializer=thor.initializers.UniformRandom(0.0, 0.0),
    )
    loss = thor.losses.CtcLoss(
        network,
        projected_logits.get_feature_output(),
        labels,
        input_lengths.get_feature_output(),
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)
    trainer = thor.training.Trainer(
        network=network,
        data=data,
        input_bindings=thor.training.DatasetInputBindings.by_exact_name(
            network=network, dataset=dataset
        ),
        optimizer=thor.optimizers.Sgd(initial_learning_rate=0.01, momentum=0.0),
        stats_interval_s=0.0,
        max_in_flight_batches=2,
        stats_color="never",
    )
    result = trainer.fit(1)
    assert result.status == "completed"
    report = result.final_training_stats.device_dataset_storage
    assert report.used
    assert report.reason == ""
    assert report.resident_bytes == (
        logits_values.nbytes
        + input_lengths_values.nbytes
        + num_examples * 2 * np.dtype(np.uint64).itemsize
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
