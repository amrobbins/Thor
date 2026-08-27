import pytest
import thor


def _net():
    return thor.Network("test_net_concatenate")


def _tensor(dims, dtype=thor.DataType.fp32):
    return thor.Tensor(list(dims), dtype)


def test_concatenate_constructs_valid_axis0():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([4, 3], thor.DataType.fp32)  # axis0 can differ
    layer = thor.layers.Concatenate(n, [t1, t2], 0)
    assert layer is not None
    assert isinstance(layer, thor.layers.Concatenate)


def test_concatenate_constructs_valid_axis1():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([2, 5], thor.DataType.fp32)  # axis1 can differ
    layer = thor.layers.Concatenate(n, [t1, t2], 1)
    assert isinstance(layer, thor.layers.Concatenate)


def test_concatenate_rejects_empty_list():
    n = _net()
    with pytest.raises(ValueError, match=r"feature_inputs must be a non-empty"):
        thor.layers.Concatenate(n, [], 0)


def test_concatenate_rejects_axis_out_of_range():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"concatenation_axis .* out of range"):
        thor.layers.Concatenate(n, [t1, t2], 2)  # rank=2, valid axes are 0/1


def test_concatenate_rejects_rank_mismatch():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([2, 3, 4], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"same number of dimensions"):
        thor.layers.Concatenate(n, [t1, t2], 0)


def test_concatenate_rejects_dtype_mismatch():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([2, 3], thor.DataType.fp16)

    with pytest.raises(ValueError, match=r"same data type"):
        thor.layers.Concatenate(n, [t1, t2], 0)


def test_concatenate_rejects_dim_mismatch_non_axis():
    n = _net()
    # axis=0 => dim1 must match, but we make it mismatch
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([4, 5], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"dimensions must match except along concatenation_axis"):
        thor.layers.Concatenate(n, [t1, t2], 0)


def _cpu_tensor_from_numpy(values, dtype):
    import numpy as np

    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _physical_ragged(values, offsets, *, max_values_per_row):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor_from_numpy(values, thor.DataType.fp32),
        _cpu_tensor_from_numpy(offsets, thor.DataType.uint32),
        max_values_per_row=max_values_per_row,
    )


def _ragged_reassemble_network(name):
    network = thor.Network(name)
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.fp32,
        [5],
        max_total_values=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=4,
    )
    first = thor.layers.Slice(network, history, axis=0, start=0, length=2).get_feature_output()
    second = thor.layers.Slice(network, history, axis=0, start=2, length=3).get_feature_output()
    concatenate = thor.layers.Concatenate(network, [first, second], 0)
    joined = concatenate.get_feature_output()
    thor.layers.RaggedNetworkOutput(network, "joined", joined)
    return network, history, concatenate, joined


def test_concatenate_ragged_preserves_shared_partition_and_serializes():
    import json

    network, history, concatenate, joined = _ragged_reassemble_network("ragged-concatenate-shape")

    assert concatenate.use_ragged
    assert isinstance(joined, thor.RaggedTensor)
    assert joined.get_trailing_dimensions() == [5]
    assert joined.offsets == history.offsets
    assert joined.max_values_per_row == history.max_values_per_row == 4

    architecture = json.loads(network.get_architecture_json())
    layer = next(item for item in architecture["layers"] if item["layer_type"] == "concatenate")
    assert layer["use_ragged"] is True
    # RaggedTensor architecture JSON uses the canonical nested Tensor schema
    # for values.  The first values dimension is packed capacity; the
    # remaining dimensions are the logical trailing dimensions.
    assert [item["values"]["dimensions"] for item in layer["ragged_inputs"]] == [[9, 2], [9, 3]]
    assert layer["ragged_output"]["values"]["dimensions"] == [9, 5]
    assert all(item["batch_size"] == 3 for item in layer["ragged_inputs"])
    assert all(item["max_total_values"] == 9 for item in layer["ragged_inputs"])
    assert all(item["max_values_per_row"] == 4 for item in layer["ragged_inputs"])
    assert layer["ragged_output"]["batch_size"] == 3
    assert layer["ragged_output"]["max_total_values"] == 9
    assert layer["ragged_output"]["max_values_per_row"] == 4
    assert all(
        item["offsets"]["id"] == layer["ragged_output"]["offsets"]["id"]
        for item in layer["ragged_inputs"]
    )


def test_concatenate_ragged_rejects_distinct_row_partitions():
    network = thor.Network("ragged-concatenate-distinct-partitions")
    left = thor.layers.RaggedNetworkInput(
        network,
        "left",
        thor.DataType.fp32,
        [2],
        max_total_values=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=4,
    )
    right = thor.layers.RaggedNetworkInput(
        network,
        "right",
        thor.DataType.fp32,
        [3],
        max_total_values=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=4,
    )
    with pytest.raises(RuntimeError, match=r"share the exact same offsets tensor"):
        thor.layers.Concatenate(network, [left, right], 0)


@pytest.mark.cuda
def test_concatenate_ragged_forward_and_save_load_touch_only_active_prefix(tmp_path):
    import numpy as np

    name = "ragged-concatenate-forward-save-load"
    network, _, _, _ = _ragged_reassemble_network(name)
    offsets = np.asarray([0, 3, 3, 7], dtype=np.uint32)
    values = np.arange(9 * 5, dtype=np.float32).reshape(9, 5)
    # Inactive packed capacity is deliberately poisoned. Concatenate is a
    # consumer of its ragged inputs and must not read or canonicalize this tail.
    values[7:] = np.asarray(
        [[np.nan, np.inf, -np.inf, np.nan, np.inf], [-np.inf, np.nan, np.inf, -np.inf, np.nan]],
        dtype=np.float32,
    )
    physical = _physical_ragged(values, offsets, max_values_per_row=4)

    placed = network.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    result = placed.infer({"history": physical})["joined"]
    np.testing.assert_array_equal(result.offsets.numpy(), offsets)
    np.testing.assert_array_equal(np.asarray(result.values.numpy())[:7], values[:7])

    save_dir = tmp_path / "ragged_concatenate"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    loaded_result = loaded_placed.infer({"history": physical})["joined"]
    np.testing.assert_array_equal(loaded_result.offsets.numpy(), offsets)
    np.testing.assert_array_equal(np.asarray(loaded_result.values.numpy())[:7], values[:7])


@pytest.mark.cuda
def test_concatenate_ragged_backward_preserves_batch_size_for_upstream_fused_optimizer():
    import numpy as np

    batch_size = 2
    max_total_values = 6
    max_values_per_row = 3
    feature_width = 4

    network = thor.Network("ragged-concatenate-training-batch-cardinality")
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.fp32,
        [feature_width],
        max_total_values=max_total_values,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=max_values_per_row,
    )
    normalized = thor.layers.RMSNorm(
        network,
        history,
        epsilon=1.0e-5,
        parameter_data_type=thor.DataType.fp32,
    ).get_feature_output()
    left = thor.layers.Slice(network, normalized, axis=0, start=0, length=2).get_feature_output()
    right = thor.layers.Slice(network, normalized, axis=0, start=2, length=2).get_feature_output()
    joined = thor.layers.Concatenate(network, [left, right], 0).get_feature_output()
    pooled = thor.layers.SegmentedReduction(
        network,
        joined,
        thor.layers.SegmentedReduction.Type.mean,
    ).get_feature_output()
    labels = thor.layers.NetworkInput(
        network,
        "labels",
        [feature_width],
        thor.DataType.fp32,
    ).get_feature_output()
    loss = thor.losses.MSE(network, pooled, labels, thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)

    offsets = np.asarray([0, 2, 5], dtype=np.uint32)
    values = np.ascontiguousarray(
        np.arange(int(offsets[-1]) * feature_width, dtype=np.float32).reshape(int(offsets[-1]), feature_width) / 17.0
    )
    expected = np.zeros((batch_size, feature_width), dtype=np.float32)
    dataset = thor.data.NumpyDataset(
        {"labels": expected},
        ragged_tensors={"history": thor.data.RaggedBatch(values, offsets)},
    )
    training_data = thor.data.TrainingData(
        dataset=dataset,
        splits=thor.data.DatasetSplitManifest(
            dataset=dataset,
            train_indices=list(range(batch_size)),
            validate_indices=[],
        ),
        batching=thor.data.BatchPolicy(batch_size=batch_size, randomize_train=False),
        device_storage="off",
    )
    trainer = thor.training.Trainer(
        network=network,
        data=training_data,
        input_bindings=thor.training.DatasetInputBindings.by_exact_name(network=network, dataset=dataset),
        optimizer=thor.optimizers.AdamW(
            alpha=0.01,
            beta1=0.9,
            beta2=0.99,
            epsilon=1.0e-8,
            weight_decay=0.0,
        ),
        debug_synchronous=True,
        stats_interval_s=0.0,
        max_in_flight_batches=1,
        scalar_tensors_to_report=[],
        stats_color="never",
    )

    result = trainer.fit(1, max_training_batches_per_epoch=1)
    assert result.status == "completed"


def test_concatenate_ragged_rejects_duplicate_values_port():
    network = thor.Network("ragged-concatenate-duplicate-values")
    value = thor.layers.RaggedNetworkInput(
        network,
        "value",
        thor.DataType.fp32,
        [2],
        max_total_values=9,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=4,
    )
    with pytest.raises(RuntimeError, match=r"values inputs must be distinct tensors"):
        thor.layers.Concatenate(network, [value, value], 0)
