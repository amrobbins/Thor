import pytest
import thor


def _net():
    return thor.Network("test_net_type_converter")


def test_type_converter_constructs_with_network_input_tensor():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp32)
    x = ni.get_feature_output()

    tc = thor.layers.TypeConverter(n, x, thor.DataType.fp16)
    assert tc is not None
    assert isinstance(tc, thor.layers.TypeConverter)

    y = tc.get_feature_output()
    assert isinstance(y, thor.Tensor)
    assert y.get_data_type() == thor.DataType.fp16
    assert y.get_dimensions() == x.get_dimensions()


def test_type_converter_rejects_wrong_types_and_arity():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp32)
    x = ni.get_feature_output()

    with pytest.raises(TypeError):
        thor.layers.TypeConverter()  # missing args

    with pytest.raises(TypeError):
        thor.layers.TypeConverter(n)  # missing feature_input + new_data_type

    with pytest.raises(TypeError):
        thor.layers.TypeConverter(n, x)  # missing new_data_type

    with pytest.raises(TypeError):
        thor.layers.TypeConverter(n, x, thor.DataType.fp16, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.TypeConverter("not a network", x, thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.TypeConverter(n, "not a tensor", thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.TypeConverter(n, x, "fp16")  # new_data_type must be enum


def _cpu_tensor(array, dtype):
    import numpy as np

    array = np.asarray(array, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(array.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = array
    return tensor


def test_type_converter_constructs_with_ragged_input_and_preserves_partition():
    n = _net()
    x = thor.layers.RaggedNetworkInput(
        n,
        "ragged_input",
        thor.DataType.fp32,
        [3],
        max_total_values=8,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )

    tc = thor.layers.TypeConverter(n, x, thor.DataType.bf16)
    assert tc.get_use_ragged()
    y = tc.get_feature_output()
    assert isinstance(y, thor.RaggedTensor)
    assert y.values.get_data_type() == thor.DataType.bf16
    assert y.values.get_dimensions() == [8, 3]
    assert y.offsets == x.offsets


@pytest.mark.cuda
def test_ragged_type_converter_executes_only_active_prefix_and_preserves_offsets():
    import numpy as np

    batch_size = 2
    n = thor.Network("test_ragged_type_converter_infer")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    tc = thor.layers.TypeConverter(n, x, thor.DataType.fp16)
    y = tc.get_feature_output()
    thor.layers.RaggedNetworkOutput(n, "output", y)

    placed = n.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    values_np = np.array(
        [
            [1.25, -2.5],
            [3.75, 4.5],
            [-6.0, 7.25],
            [99.0, 99.0],
            [99.0, 99.0],
            [99.0, 99.0],
        ],
        dtype=np.float32,
    )
    offsets_np = np.array([0, 1, 3], dtype=np.uint32)
    physical = thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values_np, thor.DataType.fp32),
        _cpu_tensor(offsets_np, thor.DataType.uint32),
    )

    result = placed.infer({"tokens": physical})["output"]
    assert isinstance(result, thor.physical.PhysicalRaggedTensor)
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    expected = np.zeros((6, 2), dtype=np.float16)
    expected[:3] = values_np[:3].astype(np.float16)
    assert np.array_equal(result.values.numpy(), expected)


@pytest.mark.cuda
def test_ragged_type_converter_save_load_preserves_expression_execution(tmp_path):
    import numpy as np

    batch_size = 2
    network_name = "test_ragged_type_converter_save_load"
    n = thor.Network(network_name)
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp16,
        [2],
        max_total_values=5,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
    )
    y = thor.layers.TypeConverter(n, x, thor.DataType.fp32).get_feature_output()
    thor.layers.RaggedNetworkOutput(n, "output", y)

    save_dir = tmp_path / "ragged_type_converter"
    n.save(str(save_dir), overwrite=False)

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    placed = loaded.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)

    values_np = np.array([[1.5, -2.0], [3.0, 0.25], [99.0, 99.0], [99.0, 99.0], [99.0, 99.0]], dtype=np.float16)
    offsets_np = np.array([0, 1, 2], dtype=np.uint32)
    result = placed.infer(
        {
            "tokens": thor.physical.PhysicalRaggedTensor(
                _cpu_tensor(values_np, thor.DataType.fp16),
                _cpu_tensor(offsets_np, thor.DataType.uint32),
            )
        }
    )["output"]

    expected = np.zeros((5, 2), dtype=np.float32)
    expected[:2] = values_np[:2].astype(np.float32)
    assert np.array_equal(result.offsets.numpy(), offsets_np)
    assert np.array_equal(result.values.numpy(), expected)
