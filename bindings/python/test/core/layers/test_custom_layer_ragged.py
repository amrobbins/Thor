import numpy as np
import pytest

import thor


BATCH_SIZE = 3
CAPACITY = 8
FEATURES = 4
OFFSETS = np.asarray([0, 3, 3, 5], dtype=np.uint32)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _physical_ragged(values: np.ndarray) -> thor.physical.PhysicalRaggedTensor:
    return thor.physical.PhysicalRaggedTensor(
        _cpu_tensor(values, thor.DataType.fp32),
        _cpu_tensor(OFFSETS, thor.DataType.uint32),
    )


def _ragged_input(network: thor.Network, name: str = "history") -> thor.RaggedTensor:
    return thor.layers.RaggedNetworkInput(
        network,
        name,
        thor.DataType.fp32,
        [FEATURES],
        max_total_values=CAPACITY,
        batch_size=BATCH_SIZE,
        offsets_data_type=thor.DataType.uint32,
    )


def test_python_custom_layer_ragged_single_input_preserves_partition_and_logical_specs():
    network = thor.Network("python-custom-layer-ragged-single")
    history = _ragged_input(network)
    seen_specs: list[tuple[list[int], thor.DataType]] = []

    def output_specs(context: thor.layers.CustomLayerSpecContext):
        spec = context.input_spec("feature_input")
        seen_specs.append((spec.shape, spec.dtype))
        return {"feature_output": thor.layers.TensorSpec([FEATURES], thor.DataType.fp32)}

    def build(context: thor.layers.CustomLayerBuildContext):
        x = context.input("feature_input")
        return {"feature_output": -thor.physical.Expression.expm1(-x)}

    layer = thor.layers.CustomLayer(
        network=network,
        inputs=history,
        build=build,
        output_specs=output_specs,
    )

    assert layer.use_ragged is True
    assert seen_specs == [([FEATURES], thor.DataType.fp32)]
    assert isinstance(layer["feature_output"], thor.RaggedTensor)
    assert layer["feature_output"].trailing_dimensions == [FEATURES]
    assert layer["feature_output"].values.get_dimensions() == [CAPACITY, FEATURES]
    assert layer["feature_output"].offsets == history.offsets
    assert isinstance(layer.outputs["feature_output"], thor.RaggedTensor)
    assert layer.get_input_interface()["feature_input"] == history
    assert layer.get_output_interface({"feature_input": history})["feature_output"] == layer["feature_output"]


def test_python_custom_layer_rejects_mixed_dense_and_ragged_named_inputs():
    network = thor.Network("python-custom-layer-ragged-mixed-reject")
    dense = thor.layers.NetworkInput(network, "dense", [FEATURES], thor.DataType.fp32).get_feature_output()
    ragged = _ragged_input(network, "ragged")

    with pytest.raises(TypeError, match="may not mix"):
        thor.layers.CustomLayer(
            network=network,
            inputs={"dense": dense, "ragged": ragged},
            output_names=["feature_output"],
            build=lambda context: {"feature_output": context.input("dense")},
        )


@pytest.mark.cuda
def test_python_custom_layer_ragged_multi_input_multi_output_and_parameter_place():
    network = thor.Network("python-custom-layer-ragged-mimo-parameter")
    history = _ragged_input(network)
    activated = thor.activations.Relu().add_to_network(network, history)
    scale = thor.parameters.ParameterSpecification(
        name="scale",
        shape=[FEATURES],
        dtype=thor.DataType.fp32,
        trainable=True,
    )

    def output_specs(context: thor.layers.CustomLayerSpecContext):
        assert context.input_spec("lhs").shape == [FEATURES]
        assert context.input_spec("rhs").shape == [FEATURES]
        return {
            "wide": thor.layers.TensorSpec([FEATURES], thor.DataType.fp32),
            "narrow": thor.layers.TensorSpec([2], thor.DataType.fp32),
        }

    def build(context: thor.layers.CustomLayerBuildContext):
        lhs = context.input("lhs")
        rhs = context.input("rhs")
        scale_expr = context.param("scale")
        shared = (lhs + rhs) * scale_expr
        rows, width = context.input_tensor("lhs").get_dimensions()
        assert width == FEATURES
        return {
            "wide": shared,
            "narrow": shared.strided_view([rows, 2], [width, 1], 1),
        }

    layer = thor.layers.CustomLayer(
        network=network,
        inputs={"lhs": history, "rhs": activated},
        output_names=["wide", "narrow"],
        parameters=[scale],
        build=build,
        output_specs=output_specs,
    )

    wide = layer["wide"]
    narrow = layer["narrow"]
    assert isinstance(wide, thor.RaggedTensor)
    assert isinstance(narrow, thor.RaggedTensor)
    assert wide.offsets == history.offsets
    assert narrow.offsets == history.offsets
    assert wide.trailing_dimensions == [FEATURES]
    assert narrow.trailing_dimensions == [2]
    assert [parameter.name for parameter in layer.get_parameters()] == ["scale"]

    thor.layers.RaggedNetworkOutput(network, "wide", wide)
    thor.layers.RaggedNetworkOutput(network, "narrow", narrow)
    placed = network.place(BATCH_SIZE, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    bound = layer.get_bound_parameter(placed, "scale")
    assert bound is not None
    assert bound.name == "scale"


@pytest.mark.cuda
def test_python_custom_layer_ragged_executes_active_prefix_and_round_trips(tmp_path):
    network_name = "python-custom-layer-ragged-round-trip"
    network = thor.Network(network_name)
    history = _ragged_input(network)

    def build(context: thor.layers.CustomLayerBuildContext):
        x = context.input("feature_input")
        return {"feature_output": -thor.physical.Expression.expm1(-x)}

    layer = thor.layers.CustomLayer(network=network, inputs=history, build=build)
    output = layer["feature_output"]
    assert isinstance(output, thor.RaggedTensor)
    thor.layers.RaggedNetworkOutput(network, "output", output)

    active = np.asarray(
        [
            [-1.0, 0.5, 2.0, -0.25],
            [0.25, -0.5, 1.5, 3.0],
            [-2.0, -1.0, 0.25, 0.75],
            [1.0, 2.0, -3.0, -4.0],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    values = np.full((CAPACITY, FEATURES), 12345.0, dtype=np.float32)
    values[: active.shape[0]] = active
    expected = -np.expm1(-active)

    placed = network.place(BATCH_SIZE, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    result = placed.infer({"history": _physical_ragged(values)})["output"]
    assert isinstance(result, thor.physical.PhysicalRaggedTensor)
    np.testing.assert_array_equal(result.offsets.numpy(), OFFSETS)
    np.testing.assert_allclose(result.values.numpy()[: active.shape[0]], expected, rtol=1e-5, atol=1e-5)

    save_dir = tmp_path / "ragged_custom_layer"
    placed.save(str(save_dir), overwrite=False, save_optimizer_state=False)
    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    loaded_placed = loaded.place(BATCH_SIZE, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    loaded_result = loaded_placed.infer({"history": _physical_ragged(values)})["output"]
    np.testing.assert_array_equal(loaded_result.offsets.numpy(), OFFSETS)
    np.testing.assert_allclose(
        loaded_result.values.numpy()[: active.shape[0]],
        result.values.numpy()[: active.shape[0]],
        rtol=1e-5,
        atol=1e-5,
    )
