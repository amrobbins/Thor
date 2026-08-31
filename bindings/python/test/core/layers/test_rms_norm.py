import json

import numpy as np
import pytest
import thor


def _net():
    return thor.Network("test_net_rms_norm")


def _input_tensor(n: thor.Network, dims, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", dims, dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def _swish_epilogue():
    return thor.activations.Swish().to_expression(thor.layers.RMSNorm.epilogue_input())


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _rms_norm_reference(x: np.ndarray, normalized_shape, epsilon: float) -> np.ndarray:
    rank = len(normalized_shape)
    axes = tuple(range(x.ndim - rank, x.ndim))
    x32 = x.astype(np.float32)
    square_mean = np.mean(x32 * x32, axis=axes, keepdims=True)
    return x32 / np.sqrt(square_mean + np.float32(epsilon))


def _rms_norm_reference_for_dtype(values: np.ndarray, normalized_shape, epsilon: float, dtype: thor.DataType) -> np.ndarray:
    # Match the values actually provided to Thor: _cpu_tensor stores the input
    # in the requested dtype before device execution.  For fp16/bf16 this can
    # slightly change the normalized values before cuDNN's fp32 accumulation.
    quantized_values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype))
    return _rms_norm_reference(quantized_values, normalized_shape, epsilon).astype(thor.physical.numpy_dtypes.from_thor(dtype))


def _run_rms_norm_network(values: np.ndarray, feature_dims, dtype: thor.DataType, *, normalized_shape=None, epsilon=1e-5) -> np.ndarray:
    dtype_name = str(dtype).split(".")[-1]
    n = thor.Network(f"test_net_rms_norm_numerical_{dtype_name}_{len(feature_dims)}d")
    x = _input_tensor(n, feature_dims, dtype)
    kwargs = {"epsilon": epsilon}
    if normalized_shape is not None:
        kwargs["normalized_shape"] = normalized_shape
    rn = thor.layers.RMSNorm(n, x, **kwargs)
    thor.layers.NetworkOutput(n, "output", rn.get_feature_output(), dtype)

    placed = n.place(
        values.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer({"input": _cpu_tensor(values, dtype)})
    assert set(outputs.keys()) == {"output"}
    return np.array(outputs["output"].numpy(), copy=True)


def test_rms_norm_constructs_default_last_dim_and_output_preserves_shape_dtype():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    rn = thor.layers.RMSNorm(n, x)

    assert isinstance(rn, thor.layers.RMSNorm)
    assert rn.get_normalized_shape() == [16]
    assert rn.get_epsilon() == pytest.approx(1e-5)
    assert rn.get_parameter_data_type() == thor.DataType.fp32

    y = rn.get_feature_output()
    assert isinstance(y, thor.Tensor)
    assert y.get_dimensions() == x.get_dimensions()
    assert y.get_data_type() == x.get_data_type()


def test_rms_norm_constructs_explicit_trailing_shape_and_serializes_weights_only():
    n = _net()
    x = _input_tensor(n, [4, 8, 16], thor.DataType.bf16)

    rn = thor.layers.RMSNorm(n, x, normalized_shape=[8, 16], epsilon=1e-4)
    assert rn.get_normalized_shape() == [8, 16]
    assert rn.get_epsilon() == pytest.approx(1e-4)

    arch = _only_layer_architecture(n, "rms_norm")
    assert arch["normalized_shape"] == [8, 16]
    assert arch["epsilon"] == pytest.approx(1e-4)
    assert "weights" in arch["parameters"]
    assert "biases" not in arch["parameters"]
    assert arch["epilogue"] is None
    assert "fused_activation" not in arch


def test_rms_norm_rejects_bad_normalized_shape():
    n = _net()
    x = _input_tensor(n, [4, 8, 16], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.RMSNorm(n, x, normalized_shape=[16, 8])

    with pytest.raises((RuntimeError, ValueError), match="normalizedShape"):
        thor.layers.RMSNorm(n, x, normalized_shape=[])


def test_rms_norm_rejects_bad_epsilon():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.RMSNorm(n, x, epsilon=0.0)

    with pytest.raises(ValueError, match="epsilon must be > 0"):
        thor.layers.RMSNorm(n, x, epsilon=-1e-5)


def test_rms_norm_rejects_unsupported_dtypes():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises((RuntimeError, ValueError), match="fp32"):
        thor.layers.RMSNorm(n, x, parameter_data_type=thor.DataType.fp16)

    n2 = thor.Network("test_net_rms_norm_int")
    int_x = _input_tensor(n2, [8, 16], thor.DataType.int32)
    with pytest.raises((RuntimeError, ValueError), match="dtype"):
        thor.layers.RMSNorm(n2, int_x)


def test_rms_norm_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.RMSNorm()

    with pytest.raises(TypeError):
        thor.layers.RMSNorm(n)

    with pytest.raises(TypeError):
        thor.layers.RMSNorm("not a network", x)

    with pytest.raises(TypeError):
        thor.layers.RMSNorm(n, "not a tensor")

    with pytest.raises(TypeError):
        thor.layers.RMSNorm(n, x, epsilon="1e-5")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype,atol,rtol", [
    (thor.DataType.fp16, 1.5e-3, 1.5e-3),
    (thor.DataType.fp32, 2.5e-5, 2.5e-5),
])
def test_rms_norm_forward_matches_numpy_default_last_dim(dtype, atol, rtol):
    values = np.array(
        [
            [[-2.0, -1.0, 0.0, 1.0], [1.5, 2.0, 3.0, 4.0], [-3.0, 0.5, 2.5, 5.0]],
            [[0.25, -0.75, 1.25, 2.25], [4.0, 1.0, -2.0, -5.0], [3.5, 3.0, 2.5, 2.0]],
        ],
        dtype=np.float32,
    )

    actual = _run_rms_norm_network(values, [3, 4], dtype)
    expected = _rms_norm_reference_for_dtype(values, [4], 1e-5, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=atol, rtol=rtol)


@pytest.mark.cuda
def test_rms_norm_forward_matches_numpy_explicit_trailing_shape():
    dtype = thor.DataType.fp16
    epsilon = 1e-4
    values = (np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4) - 11.5) / 3.0

    actual = _run_rms_norm_network(values, [2, 3, 4], dtype, normalized_shape=[3, 4], epsilon=epsilon)
    expected = _rms_norm_reference_for_dtype(values, [3, 4], epsilon, dtype)

    np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=1.5e-3, rtol=1.5e-3)


def test_rms_norm_accepts_swish_epilogue_and_serializes_expression():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.bf16)

    rn = thor.layers.RMSNorm(n, x, epilogue=_swish_epilogue())

    assert rn.get_parameter_data_type() == thor.DataType.fp32
    arch = _only_layer_architecture(n, "rms_norm")
    assert arch["epilogue"] is not None
    assert "fused_activation" not in arch


def test_rms_norm_accepts_bf16_weights_for_swish_epilogue_fusion_candidate():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.bf16)

    rn = thor.layers.RMSNorm(n, x, parameter_data_type=thor.DataType.bf16, epilogue=_swish_epilogue())
    assert rn.get_parameter_data_type() == thor.DataType.bf16

    n2 = thor.Network("test_net_rms_norm_bf16_without_swish")
    x2 = _input_tensor(n2, [8, 16], thor.DataType.bf16)
    with pytest.raises((RuntimeError, ValueError), match="Swish epilogue"):
        thor.layers.RMSNorm(n2, x2, parameter_data_type=thor.DataType.bf16)

    n3 = thor.Network("test_net_rms_norm_bf16_weights_bad_input")
    x3 = _input_tensor(n3, [8, 16], thor.DataType.fp16)
    with pytest.raises((RuntimeError, ValueError), match="bf16 feature inputs"):
        thor.layers.RMSNorm(n3, x3, parameter_data_type=thor.DataType.bf16, epilogue=_swish_epilogue())


@pytest.mark.cuda
def test_rms_norm_bf16_swish_fusion_candidate_places_via_custom_layer():
    n = thor.Network("test_net_rms_norm_bf16_swish_places")
    x = _input_tensor(n, [3, 4], thor.DataType.bf16)
    rn = thor.layers.RMSNorm(n, x, parameter_data_type=thor.DataType.bf16, epilogue=_swish_epilogue())
    thor.layers.NetworkOutput(n, "output", rn.get_feature_output(), thor.DataType.bf16)

    placed = n.place(
        2,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    values = np.array(
        [
            [[-2.0, -1.0, 0.0, 1.0], [1.5, 2.0, 3.0, 4.0], [-3.0, 0.5, 2.5, 5.0]],
            [[0.25, -0.75, 1.25, 2.25], [4.0, 1.0, -2.0, -5.0], [3.5, 3.0, 2.5, 2.0]],
        ],
        dtype=np.float32,
    )
    outputs = placed.infer({"input": _cpu_tensor(values, thor.DataType.bf16)})
    actual = np.array(outputs["output"].numpy(), copy=True).astype(np.float32)
    normalized = _rms_norm_reference_for_dtype(values, [4], 1.0e-5, thor.DataType.bf16).astype(np.float32)
    expected = normalized / (1.0 + np.exp(-normalized))
    np.testing.assert_allclose(actual, expected, atol=2.0e-2, rtol=2.0e-2)


def test_rms_norm_rejects_bad_epilogue_type():
    n = _net()
    x = _input_tensor(n, [8, 16], thor.DataType.fp16)

    with pytest.raises(TypeError, match="epilogue"):
        thor.layers.RMSNorm(n, x, epilogue="swish")


def _physical_ragged(
    values: np.ndarray,
    offsets: np.ndarray,
    dtype: thor.DataType = thor.DataType.fp32,
    *,
    offsets_dtype: thor.DataType = thor.DataType.uint32,
    max_values_per_row=None,
):
    values_tensor = _cpu_tensor(values, dtype)
    offsets_tensor = _cpu_tensor(offsets, offsets_dtype)
    if max_values_per_row is None:
        return thor.physical.PhysicalRaggedTensor(values_tensor, offsets_tensor)
    return thor.physical.PhysicalRaggedTensor(
        values_tensor,
        offsets_tensor,
        max_values_per_row=max_values_per_row,
    )


def test_rms_norm_ragged_epilogue_auxiliary_requires_exact_partition():
    n = thor.Network("test_net_rms_norm_ragged_epilogue_partition_guard")
    x = thor.layers.RaggedNetworkInput(
        n, "tokens", thor.DataType.fp32, [4], max_total_values=8, batch_size=3, max_values_per_row=5
    )
    other = thor.layers.RaggedNetworkInput(
        n, "other", thor.DataType.fp32, [4], max_total_values=8, batch_size=3, max_values_per_row=5
    )
    normalized = thor.layers.RMSNorm.epilogue_input(
        output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32
    )
    auxiliary = thor.layers.RMSNorm.epilogue_aux_input(
        "aux", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32
    )

    with pytest.raises(ValueError, match="row partition|RaggedTensor|ragged"):
        thor.layers.RMSNorm(
            n,
            x,
            epilogue=normalized + auxiliary,
            epilogue_inputs={"aux": other},
        )

    with pytest.raises(ValueError, match="RaggedTensor|ragged"):
        thor.layers.RMSNorm(
            n,
            x,
            epilogue=normalized + auxiliary,
            epilogue_inputs={"aux": other.values},
        )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "offsets_dtype,np_offsets_dtype",
    [
        (thor.DataType.uint32, np.uint32),
        (thor.DataType.uint64, np.uint64),
    ],
)
def test_rms_norm_ragged_epilogue_auxiliary_is_active_prefix_aware_and_round_trips(
    tmp_path, offsets_dtype, np_offsets_dtype
):
    batch_size = 3
    capacity = 8
    max_values_per_row = 5
    features = 4
    name = f"test_net_rms_norm_ragged_epilogue_auxiliary_{np_offsets_dtype.__name__}"
    n = thor.Network(name)
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [features],
        max_total_values=capacity,
        batch_size=batch_size,
        max_values_per_row=max_values_per_row,
        offsets_data_type=offsets_dtype,
    )
    aux = thor.activations.Relu().add_to_network(n, x)
    normalized = thor.layers.RMSNorm.epilogue_input(
        output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32
    )
    auxiliary = thor.layers.RMSNorm.epilogue_aux_input(
        "aux", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32
    )
    rms_norm = thor.layers.RMSNorm(
        n,
        x,
        epilogue=normalized * 0.0 + auxiliary,
        epilogue_inputs={"aux": aux},
    )
    y = rms_norm.get_feature_output()
    assert isinstance(y, thor.RaggedTensor)
    assert y.offsets == x.offsets
    thor.layers.RaggedNetworkOutput(n, "output", y)

    arch = _only_layer_architecture(n, "rms_norm")
    assert arch["version"] == "1.1.0"
    assert arch["epilogue_inputs"][0]["name"] == "aux"
    assert arch["epilogue_inputs"][0]["ragged_tensor"]["offsets"]["id"] == arch["ragged_inputs"][0]["offsets"]["id"]

    def run(placed, offsets_values, active_values):
        values = np.full((capacity, features), np.nan, dtype=np.float32)
        values[: len(active_values)] = np.asarray(active_values, dtype=np.float32)
        result = placed.infer(
            {
                "tokens": _physical_ragged(
                    values,
                    np.asarray(offsets_values, dtype=np_offsets_dtype),
                    offsets_dtype=offsets_dtype,
                    max_values_per_row=max_values_per_row,
                )
            }
        )["output"]
        return result, np.array(result.values.numpy(), copy=True)

    placed = n.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    short_offsets = [0, 2, 2, 5]
    short_values = np.asarray(
        [
            [-2.0, 1.0, -0.5, 3.0],
            [4.0, -1.0, 2.0, -3.0],
            [-4.0, 5.0, 6.0, -7.0],
            [8.0, -9.0, 10.0, -11.0],
            [12.0, 13.0, -14.0, 15.0],
        ],
        dtype=np.float32,
    )
    short_result, short_output = run(placed, short_offsets, short_values)
    np.testing.assert_array_equal(short_result.offsets.numpy(), np.asarray(short_offsets, dtype=np_offsets_dtype))
    np.testing.assert_allclose(short_output[:5], np.maximum(short_values, 0.0), rtol=0.0, atol=0.0)

    long_offsets = [0, 1, 5, 8]
    long_values = (np.arange(capacity * features, dtype=np.float32).reshape(capacity, features) - 9.0) / 3.0
    long_result, long_output = run(placed, long_offsets, long_values)
    np.testing.assert_array_equal(long_result.offsets.numpy(), np.asarray(long_offsets, dtype=np_offsets_dtype))
    np.testing.assert_allclose(long_output[:capacity], np.maximum(long_values, 0.0), rtol=0.0, atol=0.0)

    empty_offsets = [0, 0, 0, 0]
    empty_result, _ = run(placed, empty_offsets, np.empty((0, features), dtype=np.float32))
    np.testing.assert_array_equal(empty_result.offsets.numpy(), np.asarray(empty_offsets, dtype=np_offsets_dtype))

    short_result_2, short_output_2 = run(placed, short_offsets, short_values)
    np.testing.assert_array_equal(short_result_2.offsets.numpy(), np.asarray(short_offsets, dtype=np_offsets_dtype))
    np.testing.assert_allclose(short_output_2[:5], np.maximum(short_values, 0.0), rtol=0.0, atol=0.0)

    save_dir = tmp_path / f"ragged_rms_norm_epilogue_aux_{np_offsets_dtype.__name__}"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))
    loaded_arch = _only_layer_architecture(loaded, "rms_norm")
    assert loaded_arch["epilogue_inputs"][0]["ragged_tensor"]["offsets"]["id"] == loaded_arch["ragged_inputs"][0]["offsets"]["id"]
    loaded_placed = loaded.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    loaded_offsets = [0, 3, 3, 7]
    loaded_values = (np.arange(7 * features, dtype=np.float32).reshape(7, features) - 12.0) / 5.0
    loaded_result, loaded_output = run(loaded_placed, loaded_offsets, loaded_values)
    np.testing.assert_array_equal(loaded_result.offsets.numpy(), np.asarray(loaded_offsets, dtype=np_offsets_dtype))
    np.testing.assert_allclose(loaded_output[:7], np.maximum(loaded_values, 0.0), rtol=0.0, atol=0.0)


def test_rms_norm_accepts_ragged_tensor_and_preserves_partition():
    n = thor.Network("test_ragged_rms_norm_build")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )

    rn = thor.layers.RMSNorm(n, x)
    y = rn.get_feature_output()

    assert rn.get_use_ragged() is True
    assert isinstance(y, thor.RaggedTensor)
    assert y.values.get_dimensions() == [66, 4]
    assert y.offsets == x.offsets
    assert rn.get_normalized_shape() == [4]

    arch = _only_layer_architecture(n, "rms_norm")
    assert arch["use_ragged"] is True
    assert arch["ragged_inputs"][0]["offsets"]["id"] == arch["ragged_outputs"][0]["offsets"]["id"]


def test_rms_norm_rejects_ragged_normalized_shape_that_includes_packed_row_dimension():
    n = thor.Network("test_ragged_rms_norm_bad_shape")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )

    with pytest.raises((RuntimeError, ValueError), match="packed ragged row dimension"):
        thor.layers.RMSNorm(n, x, normalized_shape=[66, 4])


@pytest.mark.cuda
@pytest.mark.parametrize("active_rows", [7, 9, 31, 33, 66])
def test_ragged_rms_norm_matches_dense_prefix_across_capacity_buckets(active_rows):
    epsilon = 1e-5
    n = thor.Network(f"test_ragged_rms_norm_bucket_{active_rows}")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=2,
        offsets_data_type=thor.DataType.uint32,
    )
    rn = thor.layers.RMSNorm(n, x, epsilon=epsilon)
    thor.layers.RaggedNetworkOutput(n, "output", rn.get_feature_output())

    values = np.full((66, 4), np.float32(9999.0), dtype=np.float32)
    rows = np.arange(active_rows, dtype=np.float32)[:, None]
    cols = np.arange(4, dtype=np.float32)[None, :]
    values[:active_rows] = ((rows % 7.0) - 3.0) * np.float32(0.3) + (cols + 1.0) * np.float32(0.125)
    offsets = np.array([0, active_rows // 2, active_rows], dtype=np.uint32)

    placed = n.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    result = placed.infer({"tokens": _physical_ragged(values, offsets)})["output"]
    actual = np.array(result.values.numpy(), copy=True)
    expected = _rms_norm_reference(values[:active_rows], [4], epsilon)

    assert np.array_equal(result.offsets.numpy(), offsets)
    np.testing.assert_allclose(actual[:active_rows], expected, rtol=2.5e-5, atol=2.5e-5)


@pytest.mark.cuda
def test_dense_rms_norm_save_load_preserves_execution(tmp_path):
    epsilon = 1e-4
    name = "test_dense_rms_norm_save_load"
    n = thor.Network(name)
    x = _input_tensor(n, [3, 4], thor.DataType.fp32)
    rn = thor.layers.RMSNorm(n, x, normalized_shape=[4], epsilon=epsilon)
    thor.layers.NetworkOutput(n, "output", rn.get_feature_output(), thor.DataType.fp32)

    save_dir = tmp_path / "model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))

    values = (np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) - 7.0) / 3.0
    placed = loaded.place(2, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    actual = np.array(placed.infer({"input": _cpu_tensor(values, thor.DataType.fp32)})["output"].numpy(), copy=True)
    expected = _rms_norm_reference(values, [4], epsilon)

    np.testing.assert_allclose(actual, expected, rtol=2.5e-5, atol=2.5e-5)


@pytest.mark.cuda
def test_ragged_rms_norm_save_load_preserves_execution(tmp_path):
    epsilon = 1e-4
    name = "test_ragged_rms_norm_save_load"
    n = thor.Network(name)
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=16,
        batch_size=3,
        offsets_data_type=thor.DataType.uint32,
    )
    rn = thor.layers.RMSNorm(n, x, epsilon=epsilon)
    thor.layers.RaggedNetworkOutput(n, "output", rn.get_feature_output())

    save_dir = tmp_path / "model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))

    values = np.full((16, 4), np.float32(-7777.0), dtype=np.float32)
    active_rows = 11
    values[:active_rows] = (np.arange(active_rows * 4, dtype=np.float32).reshape(active_rows, 4) - 10.0) / 5.0
    offsets = np.array([0, 3, 3, active_rows], dtype=np.uint32)
    physical = _physical_ragged(values, offsets)

    placed = loaded.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    result = placed.infer({"tokens": physical})["output"]
    actual = np.array(result.values.numpy(), copy=True)
    expected = _rms_norm_reference(values[:active_rows], [4], epsilon)

    assert np.array_equal(result.offsets.numpy(), offsets)
    np.testing.assert_allclose(actual[:active_rows], expected, rtol=2.5e-5, atol=2.5e-5)
