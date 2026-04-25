import numpy as np
import pytest

import thor
from thor.physical import Expression as ex


class Weights(thor.Parameter):

    def __init__(self, out_features):
        super().__init__(
            name="weights",
            shape=[1, out_features],
            dtype=thor.DataType.fp32,
            trainable=True,
        )
        self.out_features = out_features
        self.storage_ctx_input_names = None
        self.storage_shape_seen = None

    def create_storage(self, ctx):
        self.storage_ctx_input_names = ctx.input_names()

        x = ctx.get_input("feature_input")
        dims = x.get_descriptor().get_dimensions()
        batch = dims[0]
        in_features = x.get_descriptor().get_total_num_elements() // batch

        self.storage_shape_seen = [in_features, self.out_features]

        return self.createStorage(
            x,
            shape=[in_features, self.out_features],
            dtype=x.get_descriptor().get_data_type(),
        )


class FusedLinear(thor.layers.CustomLayer):

    def __init__(self, network: thor.Network, x: thor.Tensor, num_output_features: int, has_bias: bool):
        self.out_features = num_output_features
        self.has_bias = has_bias
        super().__init__(
            network=network,
            inputs={"feature_input": x},          # API tensor
            outputs={
                "feature_output": thor.Tensor([num_output_features], x.get_data_type())
            },
        )

    def parameters(self) -> list[thor.Parameter]:
        params: list[thor.Parameter] = [Weights(self.out_features)]
        if self.has_bias:
            params.append(
                thor.Parameter(
                    name="biases",
                    shape=[self.out_features],
                    dtype=thor.DataType.fp32,
                    trainable=True,
                ))
        return params

    def build(
        self,
        ctx: thor.layers.CustomLayerBuildContext,
    ) -> dict[str, thor.physical.Expression]:
        x = ctx.input("feature_input")
        w = ctx.param("weights")

        # Ensure the allocated output tensor is the right size for the expression that will fill it.
        if ctx.has_output("feature_output"):
            feature_output: thor.physical.PhysicalTensor = ctx.output_tensors["feature_output"]
            assert len(feature_output.get_dimensions()) == 2
            assert feature_output.get_dimensions()[1] == w.get_dimensions()[1]

        # Define the expression whose result will be placed in the output tensor.
        feature_output = x @ w
        if self.has_bias:
            b = ctx.param("biases")
            feature_output = feature_output + b

        # Bind the output tensor as the memory that the expression will fill.
        return {
            "feature_output": feature_output,
        }


# FIXME: This is an antipattern:
class DictParameterLayer(thor.layers.CustomLayer):

    def __init__(self, network, x, out_features):
        self.out_features = out_features
        super().__init__(
            network=network,
            inputs={
                "feature_input": x,
            },
            outputs={
                "feature_output": thor.Tensor([out_features], x.get_data_type()),
            },
        )

    def parameters(self) -> dict[str, thor.Parameter]:
        # FIXME: dict of parameters should not be supported. We require a list, the parameter owns its name
        #        can't have the duplication.
        return {
            "weights":
                Weights(self.out_features),
            "biases":
                thor.Parameter(
                    name="biases",
                    shape=[self.out_features],
                    dtype=thor.DataType.fp32,
                    trainable=True,
                ),
        }

    def build(
        self,
        ctx: thor.layers.CustomLayerBuildContext,
    ) -> dict[str, thor.physical.Expression]:
        x = ctx.input("feature_input")
        w = ctx.param("weights")
        b = ctx.param("biases")
        return {
            "feature_output": x @ w + b,
        }


class CommonSubexpressionLayer(thor.layers.CustomLayer):

    def __init__(self, network, x, out_features):
        self.out_features = out_features
        self.build_ctx_seen = False

        super().__init__(
            network=network,
            inputs={
                "feature_input": x,
            },
            outputs={
                "feature_output": thor.Tensor([out_features], x.get_data_type()),
                "aux_output": thor.Tensor([out_features], x.get_data_type()),
            },
        )

    def parameters(self) -> list[thor.Parameter]:
        return [
            Weights(self.out_features),
            thor.Parameter(
                name="biases",
                shape=[self.out_features],
                dtype=thor.DataType.fp32,
                trainable=True,
            ),
        ]

    def build(
        self,
        ctx: thor.layers.CustomLayerBuildContext,
    ) -> dict[str, thor.physical.Expression]:
        self.build_ctx_seen = True

        x = ctx.input("feature_input")
        w = ctx.param("weights")
        b = ctx.param("biases")

        # Shared trunk.
        projected = x @ w
        shifted = projected + b

        # Reuse the same common subexpression in multiple outputs.
        feature_output = shifted * shifted
        aux_output = shifted + shifted

        return {
            "feature_output": feature_output,
            "aux_output": aux_output,
        }


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, shape)
    return thor.physical.PhysicalTensor(placement, descriptor)


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> thor.physical.PhysicalTensor:
    placement = thor.physical.Placement(thor.physical.DeviceType.gpu, gpu_num)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, shape)
    return thor.physical.PhysicalTensor(placement, descriptor)


def _copy_numpy_to_gpu(
    values: np.ndarray,
    stream: thor.physical.Stream,
    dtype: thor.DataType,
    gpu_num: int = 0,
) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")

    cpu = _cpu_tensor(list(values.shape), dtype)
    cpu.numpy()[...] = values

    gpu = _gpu_tensor(list(values.shape), dtype, gpu_num=gpu_num)
    gpu.copy_from_async(cpu, stream)
    return gpu


def _copy_gpu_to_numpy(
    tensor: thor.physical.PhysicalTensor,
    dtype: thor.DataType,
    stream: thor.physical.Stream,
) -> np.ndarray:
    cpu = _cpu_tensor(list(tensor.get_descriptor().get_dimensions()), dtype)
    cpu.copy_from_async(tensor, stream)
    stream.synchronize()
    return np.array(cpu.numpy(), copy=True)


def test_python_custom_layer_builds_logical_output_interface_without_bias():
    network = thor.Network("custom-layer-smoke-no-bias")
    x = thor.Tensor([5], thor.DataType.fp16)

    layer = FusedLinear(network, x, 3, has_bias=False)

    y = layer["feature_output"]
    assert y.get_dimensions() == [3]
    assert y.get_data_type() == thor.DataType.fp16

    assert layer.outputs["feature_output"].get_dimensions() == [3]
    assert layer.outputs["feature_output"].get_data_type() == thor.DataType.fp16

    assert layer.get_input_names() == ["feature_input"]
    assert layer.get_output_names() == ["feature_output"]

    assert [parameter.name for parameter in layer.get_parameters()] == ["weights"]


def test_python_custom_layer_builds_logical_output_interface_with_bias():
    network = thor.Network("custom-layer-smoke-with-bias")
    x = thor.Tensor([7], thor.DataType.fp32)

    layer = FusedLinear(network, x, 4, has_bias=True)

    y = layer.get_output("feature_output")
    assert y.get_dimensions() == [4]
    assert y.get_data_type() == thor.DataType.fp32

    assert layer.get_input_names() == ["feature_input"]
    assert layer.get_output_names() == ["feature_output"]

    assert [parameter.name for parameter in layer.get_parameters()] == ["weights", "biases"]


def test_python_custom_layer_accepts_dict_returned_from_parameters():
    network = thor.Network("custom-layer-dict-parameters")
    x = thor.Tensor([6], thor.DataType.fp16)

    layer = DictParameterLayer(network, x, 2)

    assert layer["feature_output"].get_dimensions() == [2]
    assert layer["feature_output"].get_data_type() == thor.DataType.fp16
    assert [parameter.name for parameter in layer.get_parameters()] == ["weights", "biases"]


def test_python_custom_layer_supports_multiple_named_outputs():
    network = thor.Network("custom-layer-multi-output")
    x = thor.Tensor([5], thor.DataType.fp16)

    layer = CommonSubexpressionLayer(network, x, 3)

    feature_output = layer["feature_output"]
    aux_output = layer["aux_output"]

    assert feature_output.get_dimensions() == [3]
    assert feature_output.get_data_type() == thor.DataType.fp16

    assert aux_output.get_dimensions() == [3]
    assert aux_output.get_data_type() == thor.DataType.fp16

    assert layer.get_input_names() == ["feature_input"]
    assert layer.get_output_names() == ["feature_output", "aux_output"]
    assert set(layer.outputs.keys()) == {"feature_output", "aux_output"}


def test_python_custom_layer_rejects_missing_output_name():
    network = thor.Network("custom-layer-missing-output")
    x = thor.Tensor([5], thor.DataType.fp16)

    layer = FusedLinear(network, x, 3, has_bias=False)

    with pytest.raises(RuntimeError, match="missing"):
        layer["missing"]
