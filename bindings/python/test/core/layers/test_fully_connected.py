import json

import numpy as np
import pytest

import thor
from thor.physical import numpy_dtypes


def _net():
    return thor.Network("test_net_fully_connected")


def _input_tensor(n: thor.Network, in_features: int, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", [in_features], dtype)
    return ni.get_feature_output()


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = [layer for layer in json.loads(n.get_architecture_json())["layers"] if layer["layer_type"] == layer_type]
    assert len(layers) == 1
    return layers[0]


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _physical_ragged(
    values: np.ndarray,
    offsets: np.ndarray,
    *,
    offsets_dtype: thor.DataType,
    max_values_per_row: int,
):
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, thor.DataType.fp32),
        _cpu_tensor(offsets, offsets_dtype),
        max_values_per_row=max_values_per_row,
    )


def test_fully_connected_constructs_defaults_and_output_shape_dtype():
    n = _net()
    x = _input_tensor(n, 32, thor.DataType.fp16)

    fc = thor.layers.FullyConnected(n, x, 64, True)
    assert fc is not None
    assert isinstance(fc, thor.layers.FullyConnected)

    y = fc.get_feature_output()
    assert y is not None
    assert isinstance(y, thor.Tensor)

    # API expectation: output is 1D [num_output_features]
    assert y.get_dimensions() == [64]
    # Typically FC preserves dtype of feature_input at API level; if your builder forces fp32, change this.
    assert y.get_data_type() == x.get_data_type()

    fc_arch = _only_layer_architecture(n, "fully_connected")
    assert fc_arch["activation"]["layer_type"] == "gelu"


def test_fully_connected_fp32_defaults_to_fp32_compute_and_accepts_explicit_tf32():
    default_network = _net()
    default_input = _input_tensor(default_network, 32, thor.DataType.fp32)
    default_fc = thor.layers.FullyConnected(default_network, default_input, 64, False, activation=None)

    assert default_fc.get_weights_data_type() == thor.DataType.fp32
    assert default_fc.get_compute_data_type() == thor.DataType.fp32
    assert default_fc.get_output_data_type() == thor.DataType.fp32
    assert _only_layer_architecture(default_network, "fully_connected")["compute_data_type"] == "fp32"

    tf32_network = thor.Network("test_net_fully_connected_tf32")
    tf32_input = _input_tensor(tf32_network, 32, thor.DataType.fp32)
    tf32_fc = thor.layers.FullyConnected(
        tf32_network,
        tf32_input,
        64,
        False,
        activation=None,
        compute_data_type=thor.DataType.tf32,
    )

    assert tf32_fc.get_compute_data_type() == thor.DataType.tf32
    assert _only_layer_architecture(tf32_network, "fully_connected")["compute_data_type"] == "tf32"


def test_fully_connected_constructs_no_activation_when_none():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    fc = thor.layers.FullyConnected(
        n,
        x,
        8,
        True,
        activation=None,
    )
    assert isinstance(fc, thor.layers.FullyConnected)
    y = fc.get_feature_output()
    assert y.get_dimensions() == [8]

    fc_arch = _only_layer_architecture(n, "fully_connected")
    assert fc_arch["activation"] is None




def test_fully_connected_can_preserve_prefix_dimensions_for_tokenwise_projection():
    n = thor.Network("test_net_fully_connected_tokenwise")
    x_in = thor.layers.NetworkInput(n, "tokens", [5, 16], thor.DataType.fp16)

    fc = thor.layers.FullyConnected(
        n,
        x_in.get_feature_output(),
        8,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
    )

    y = fc.get_feature_output()
    assert y.get_dimensions() == [5, 8]
    assert y.get_data_type() == thor.DataType.fp16

    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["preserve_input_prefix_dimensions"] is True
    assert arch["outputs"][0]["dimensions"] == [5, 8]


def test_fully_connected_default_flattens_prefix_dimensions():
    n = thor.Network("test_net_fully_connected_flatten_prefix")
    x_in = thor.layers.NetworkInput(n, "tokens", [5, 16], thor.DataType.fp16)

    fc = thor.layers.FullyConnected(
        n,
        x_in.get_feature_output(),
        8,
        True,
        activation=None,
    )

    assert fc.get_feature_output().get_dimensions() == [8]
    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["preserve_input_prefix_dimensions"] is False

def test_fully_connected_constructs_with_activation_and_initializers():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    act = thor.activations.Elu(1.0) if hasattr(thor.activations, "Elu") else None
    winit = thor.initializers.Glorot(thor.initializers.Glorot.Mode.UNIFORM)
    binit = thor.initializers.Glorot(thor.initializers.Glorot.Mode.NORMAL)

    fc = thor.layers.FullyConnected(
        n,
        x,
        10,
        True,
        activation=act,
        weights_initializer=winit,
        biases_initializer=binit,
    )
    assert isinstance(fc, thor.layers.FullyConnected)
    y = fc.get_feature_output()
    assert y.get_dimensions() == [10]
    assert y.get_data_type() == x.get_data_type()


def test_fully_connected_rejects_num_output_features_zero():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    # Requires the binding-side check (recommended). If you didn't add it,
    # this might assert/crash, so keep this test only after adding the check.
    with pytest.raises(ValueError, match=r"num_output_features must be > 0"):
        thor.layers.FullyConnected(n, x, 0, True)


def test_fully_connected_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.FullyConnected()  # missing args

    with pytest.raises(TypeError):
        thor.layers.FullyConnected(n, x)  # missing num_output_features

    with pytest.raises(TypeError):
        thor.layers.FullyConnected(n, x, 8, True, 123)  # activation wrong type

    with pytest.raises(TypeError):
        thor.layers.FullyConnected("not a network", x, 8, True)

    with pytest.raises(TypeError):
        thor.layers.FullyConnected(n, "not a tensor", 8, True)

    with pytest.raises(TypeError):
        thor.layers.FullyConnected(n, x, 8, "True")


def test_fully_connected_accepts_epilogue_expression_and_serializes_it():
    n = thor.Network("test_net_fully_connected_epilogue")
    x = _input_tensor(n, 16, thor.DataType.fp16)

    epilogue_input = thor.layers.FullyConnected.epilogue_input(
        output_dtype=thor.DataType.fp32,
        compute_dtype=thor.DataType.fp32,
    )
    epilogue = epilogue_input * 2.0 + 1.0

    fc = thor.layers.FullyConnected(
        n,
        x,
        8,
        True,
        activation=None,
        epilogue=epilogue,
    )

    assert fc.get_feature_output().get_dimensions() == [8]
    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["activation"] is None
    assert arch["epilogue"] is not None
    assert arch["epilogue"]["expected_input_names"] == ["__fully_connected_epilogue_input"]
    assert arch["epilogue"]["expected_output_names"] == ["__fully_connected_epilogue_output"]


def test_fully_connected_rejects_wrong_epilogue_type():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp16)

    with pytest.raises(TypeError, match=r"argument 'epilogue'.*thor\.physical\.Expression or None"):
        thor.layers.FullyConnected(n, x, 8, True, epilogue=123)


def test_fully_connected_serializes_weight_constraints():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp32)

    fc = thor.layers.FullyConnected(
        n,
        x,
        8,
        True,
        activation=None,
        weights_constraints=thor.constraints.NonNegative(),
    )

    assert isinstance(fc, thor.layers.FullyConnected)
    arch = _only_layer_architecture(n, "fully_connected")
    weight_constraints = arch["parameters"]["weights"].get("constraints", [])
    assert len(weight_constraints) == 1
    assert weight_constraints[0]["constraint_type"] == "non_negative"


def test_fully_connected_rejects_invalid_weight_constraint():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.fp32)

    with pytest.raises(TypeError, match="weights_constraints"):
        thor.layers.FullyConnected(
            n,
            x,
            8,
            True,
            activation=None,
            weights_constraints=123,
        )


def test_fully_connected_accepts_bf16_storage_compute_with_fp32_output():
    n = _net()
    x = _input_tensor(n, 16, thor.DataType.bf16)

    fc = thor.layers.FullyConnected(
        n,
        x,
        8,
        True,
        activation=None,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.bf16,
        output_data_type=thor.DataType.fp32,
    )

    assert fc.get_feature_output().get_dimensions() == [8]
    assert fc.get_feature_output().get_data_type() == thor.DataType.fp32
    assert fc.get_weights_data_type() == thor.DataType.bf16
    assert fc.get_compute_data_type() == thor.DataType.bf16
    assert fc.get_output_data_type() == thor.DataType.fp32

    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["weights_data_type"] == "bf16"
    assert arch["compute_data_type"] == "bf16"
    assert arch["output_data_type"] == "fp32"
    assert arch["parameters"]["weights"]["dtype"] == "bf16"
    assert arch["parameters"]["biases"]["dtype"] == "fp32"


def test_fully_connected_accepts_ragged_input_and_preserves_partition():
    n = thor.Network("test_net_fully_connected_ragged")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=3,
        offsets_data_type=thor.DataType.uint64,
    )

    fc = thor.layers.FullyConnected(
        n,
        x,
        3,
        True,
        activation=None,
    )

    y = fc.get_feature_output()
    assert fc.get_use_ragged()
    assert isinstance(y, thor.RaggedTensor)
    assert y.values.get_dimensions() == [66, 3]
    assert y.offsets == x.offsets

    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["use_ragged"] is True
    assert arch["preserve_input_prefix_dimensions"] is True
    assert arch["ragged_inputs"][0]["max_total_values"] == 66
    assert arch["ragged_outputs"][0]["max_total_values"] == 66


def test_fully_connected_ragged_uses_regular_default_activation():
    n = thor.Network("test_net_fully_connected_ragged_activation")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=3,
    )

    fc = thor.layers.FullyConnected(n, x, 3, True)
    y = fc.get_feature_output()
    assert isinstance(y, thor.RaggedTensor)

    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["use_ragged"] is True
    assert arch["activation"] is not None


def test_fully_connected_ragged_rejects_disabling_prefix_preservation():
    n = thor.Network("test_net_fully_connected_ragged_prefix")
    x = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [4],
        max_total_values=66,
        batch_size=3,
    )

    with pytest.raises(ValueError, match="prefix|Ragged|ragged"):
        thor.layers.FullyConnected(
            n,
            x,
            3,
            True,
            activation=None,
            preserve_prefix_dimensions=False,
        )


def test_fully_connected_ragged_epilogue_auxiliary_requires_exact_partition():
    n = thor.Network("test_net_fully_connected_ragged_epilogue_partition_guard")
    x = thor.layers.RaggedNetworkInput(
        n, "tokens", thor.DataType.fp32, [4], max_total_values=8, batch_size=3, max_values_per_row=5
    )
    other = thor.layers.RaggedNetworkInput(
        n, "other", thor.DataType.fp32, [4], max_total_values=8, batch_size=3, max_values_per_row=5
    )
    projected = thor.layers.FullyConnected.epilogue_input(
        output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32
    )
    auxiliary = thor.layers.FullyConnected.epilogue_aux_input(
        "aux", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32
    )

    with pytest.raises(ValueError, match="row partition|RaggedTensor|ragged"):
        thor.layers.FullyConnected(
            n,
            x,
            4,
            False,
            activation=None,
            epilogue=projected + auxiliary,
            epilogue_inputs={"aux": other},
        )

    with pytest.raises(ValueError, match="RaggedTensor|ragged"):
        thor.layers.FullyConnected(
            n,
            x,
            4,
            False,
            activation=None,
            epilogue=projected + auxiliary,
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
def test_fully_connected_ragged_epilogue_auxiliary_is_active_prefix_aware_and_round_trips(
    tmp_path, offsets_dtype, np_offsets_dtype
):
    batch_size = 3
    capacity = 8
    max_values_per_row = 5
    features = 4
    n = thor.Network(f"test_net_fully_connected_ragged_epilogue_auxiliary_{np_offsets_dtype.__name__}")
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
    projected = thor.layers.FullyConnected.epilogue_input(
        output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32
    )
    auxiliary = thor.layers.FullyConnected.epilogue_aux_input(
        "aux", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32
    )
    fc = thor.layers.FullyConnected(
        n,
        x,
        features,
        False,
        activation=None,
        epilogue=projected * 0.0 + auxiliary,
        epilogue_inputs={"aux": aux},
    )
    y = fc.get_feature_output()
    assert isinstance(y, thor.RaggedTensor)
    assert y.offsets == x.offsets
    thor.layers.RaggedNetworkOutput(n, "output", y)

    arch = _only_layer_architecture(n, "fully_connected")
    assert arch["version"] == "1.2.0"
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
        [[-2.0, 1.0, -0.5, 3.0], [4.0, -1.0, 2.0, -3.0], [-4.0, 5.0, 6.0, -7.0],
         [8.0, -9.0, 10.0, -11.0], [12.0, 13.0, -14.0, 15.0]],
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

    save_dir = tmp_path / f"ragged_fc_epilogue_aux_{np_offsets_dtype.__name__}"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network(f"test_net_fully_connected_ragged_epilogue_auxiliary_{np_offsets_dtype.__name__}")
    loaded.load(str(save_dir))
    loaded_arch = _only_layer_architecture(loaded, "fully_connected")
    assert loaded_arch["epilogue_inputs"][0]["ragged_tensor"]["offsets"]["id"] == loaded_arch["ragged_inputs"][0]["offsets"]["id"]
    loaded_placed = loaded.place(batch_size, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    loaded_offsets = [0, 3, 3, 7]
    loaded_values = (np.arange(7 * features, dtype=np.float32).reshape(7, features) - 12.0) / 5.0
    loaded_result, loaded_output = run(loaded_placed, loaded_offsets, loaded_values)
    np.testing.assert_array_equal(loaded_result.offsets.numpy(), np.asarray(loaded_offsets, dtype=np_offsets_dtype))
    np.testing.assert_allclose(loaded_output[:7], np.maximum(loaded_values, 0.0), rtol=0.0, atol=0.0)
