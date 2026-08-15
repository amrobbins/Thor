import pytest
import thor


def _net():
    return thor.Network("test_net_dropout")


def _tensor_1d(size: int = 1, dtype=thor.DataType.fp32):
    # API tensor: dims + dtype
    return thor.Tensor([size], dtype)


def test_dropout_constructs_and_reports_drop_proportion():
    n = _net()
    x = _tensor_1d(1, thor.DataType.fp32)

    layer = thor.layers.DropOut(n, x, 0.25)
    assert layer is not None
    assert isinstance(layer, thor.layers.DropOut)

    # Should return exactly what we passed (float compare)
    assert layer.get_drop_proportion() == pytest.approx(0.25)


def test_dropout_allows_endpoints():
    n = _net()
    x = _tensor_1d()

    l0 = thor.layers.DropOut(n, x, 0.0)
    assert l0.get_drop_proportion() == pytest.approx(0.0)

    l1 = thor.layers.DropOut(n, x, 1.0)
    assert l1.get_drop_proportion() == pytest.approx(1.0)


def test_dropout_rejects_out_of_range_values():
    n = _net()
    x = _tensor_1d()

    with pytest.raises(ValueError, match=r"0 <= drop_proportion <= 1"):
        thor.layers.DropOut(n, x, -0.01)

    with pytest.raises(ValueError, match=r"0 <= drop_proportion <= 1"):
        thor.layers.DropOut(n, x, 1.01)


def test_dropout_rejects_wrong_types_and_arity():
    n = _net()
    x = _tensor_1d()

    with pytest.raises(TypeError):
        thor.layers.DropOut()  # missing args

    with pytest.raises(TypeError):
        thor.layers.DropOut(n, x)  # missing drop_proportion

    with pytest.raises(TypeError):
        thor.layers.DropOut(n, x, 0.5, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.DropOut("not a network", x, 0.5)

    with pytest.raises(TypeError):
        thor.layers.DropOut(n, "not a tensor", 0.5)

    with pytest.raises(TypeError):
        thor.layers.DropOut(n, x, "0.5")


def test_dropout_training_control_is_transient_and_network_wide():
    n = _net()
    x = _tensor_1d(4, thor.DataType.fp32)
    layer = thor.layers.DropOut(n, x, 0.25)

    assert layer.is_training_dropout_enabled() is True
    assert n.get_num_training_dropout_controllable_layers() == 1
    assert n.is_training_dropout_enabled() is True

    n.set_training_dropout_enabled(False)
    assert layer.is_training_dropout_enabled() is False
    assert n.is_training_dropout_enabled() is False
    assert "training_dropout_enabled" not in n.get_architecture_json()

    layer.set_training_dropout_enabled(True)
    assert layer.is_training_dropout_enabled() is True
    assert n.is_training_dropout_enabled() is True


def _cpu_tensor(values, dtype):
    import numpy as np

    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def test_dropout_accepts_ragged_tensor_and_preserves_partition():
    n = thor.Network("test_ragged_dropout_build")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [3],
        max_total_values=8,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )
    layer = thor.layers.DropOut(n, x, 0.25)
    y = layer.get_feature_output()

    assert layer.get_use_ragged() is True
    assert isinstance(y, thor.RaggedTensor)
    assert y.values.get_dimensions() == [8, 3]
    assert y.offsets == x.offsets


@pytest.mark.cuda
def test_ragged_dropout_inference_is_identity_on_active_prefix_and_survives_save_load(tmp_path):
    import numpy as np

    name = "test_ragged_dropout_inference_save_load"
    n = thor.Network(name)
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )
    layer = thor.layers.DropOut(n, x, 0.5)
    thor.layers.RaggedNetworkOutput(n, "output", layer.get_feature_output())

    values_np = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9999.0, 9999.0], [-9999.0, -9999.0]],
        dtype=np.float32,
    )
    offsets_np = np.array([0, 1, 4], dtype=np.uint32)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values_np, thor.DataType.fp32),
        _cpu_tensor(offsets_np, thor.DataType.uint32),
    )

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    result = placed.infer({"tokens": physical})["output"]
    expected = values_np.copy()
    expected[4:] = 0.0
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    np.testing.assert_allclose(result.values.numpy(), expected, rtol=0, atol=0)

    save_dir = tmp_path / "model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    reloaded = loaded_placed.infer({"tokens": physical})["output"]
    assert np.array_equal(reloaded.offsets.numpy(), offsets_np)
    np.testing.assert_allclose(reloaded.values.numpy(), expected, rtol=0, atol=0)
