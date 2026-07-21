import gc
import json
import weakref

import numpy as np
import pytest

import thor


class FusedLinear(thor.layers.CustomLayer):

    def __init__(self, network: thor.Network, x: thor.Tensor, num_units: int, has_bias: bool):
        self.num_units: int = num_units
        self.has_bias: bool = has_bias

        super().__init__(
            network=network,
            inputs=x,
        )

    def parameters(self) -> list[thor.parameters.ParameterSpecification]:
        num_units = self.num_units

        def create_weights_storage_from_context(
                context: thor.parameters.ParameterSpecification.StorageContext) -> thor.physical.PhysicalTensor:
            input_tensor = context.get_feature_input()
            batch_size = input_tensor.get_descriptor().get_dimensions()[0]
            num_input_features = input_tensor.get_descriptor().get_total_num_elements() // batch_size
            return thor.parameters.ParameterSpecification.allocate_storage(
                input_tensor,
                shape=[num_input_features, num_units],
                dtype=input_tensor.get_descriptor().get_data_type(),
            )

        weights = thor.parameters.ParameterSpecification(
            name="weights",
            create_storage_from_context=create_weights_storage_from_context,
            trainable=True,
        )

        params: list[thor.parameters.ParameterSpecification] = [weights]
        if self.has_bias:
            biases = thor.parameters.ParameterSpecification(
                name="biases",
                shape=[num_units],
                dtype=thor.DataType.fp32,
                trainable=True,
                training_initially_enabled=True,
            )
            params.append(biases)
        return params

    def build(
        self,
        context: thor.layers.CustomLayerBuildContext,
    ) -> dict[str, thor.physical.Expression]:
        # Physical tensors carry information only known after placement:
        # batch-inflated shapes, placement, and parameter storage shapes.
        x_tensor = context.input_tensor("feature_input")
        w_tensor = context.parameter_tensor("weights")

        assert len(x_tensor.get_dimensions()) == 2
        assert len(w_tensor.get_dimensions()) == 2
        assert x_tensor.get_dimensions()[1] == w_tensor.get_dimensions()[0]

        # The first shape-inference prepare may not have preallocated outputs yet,
        # but compile-time prepares should expose the concrete output tensor.
        if context.has_output("feature_output"):
            feature_output_tensor: thor.physical.PhysicalTensor = context.output_tensor("feature_output")
            assert len(feature_output_tensor.get_dimensions()) == 2
            assert feature_output_tensor.get_dimensions()[1] == w_tensor.get_dimensions()[1]

        x = context.input("feature_input")
        w = context.param("weights")
        feature_output = x @ w
        if self.has_bias:
            b = context.param("biases")
            feature_output = feature_output + b

        return {
            "feature_output": feature_output,
        }


class MultiInputMultiOutputAffine(thor.layers.CustomLayer):
    """Small MIMO layer used as a Python CustomLayer contract test.

    It takes two feature inputs, allocates two vector parameters from the
    placement-time StorageContext, and returns two outputs that share a common
    subexpression. The outputs intentionally exercise normal same-shape output
    tensors and vector-broadcast parameters.
    """

    def __init__(self, network: thor.Network, lhs: thor.Tensor, rhs: thor.Tensor):
        assert lhs.get_dimensions() == rhs.get_dimensions()
        assert lhs.get_data_type() == rhs.get_data_type()

        self.storage_context_input_names: list[list[str]] = []
        self.storage_context_input_dims: list[dict[str, list[int]]] = []
        self.build_contexts: list[dict[str, object]] = []

        super().__init__(
            network=network,
            inputs={
                "lhs": lhs,
                "rhs": rhs,
            },
            output_names=["sum_output", "affine_output"],
                    )

    def parameters(self) -> list[thor.parameters.ParameterSpecification]:
        storage_context_input_names = self.storage_context_input_names
        storage_context_input_dims = self.storage_context_input_dims

        def allocate_vector(
                context: thor.parameters.ParameterSpecification.StorageContext, name: str) -> thor.physical.PhysicalTensor:
            assert context.has_input("lhs") is True
            assert context.has_input("rhs") is True
            assert context.input_names() == ["lhs", "rhs"]

            lhs = context.get_input("lhs")
            rhs = context.get_input("rhs")
            lhs_dims = lhs.get_descriptor().get_dimensions()
            rhs_dims = rhs.get_descriptor().get_dimensions()
            assert len(lhs_dims) == 2
            assert lhs_dims == rhs_dims

            with pytest.raises(RuntimeError, match="There is not exactly 1 input available"):
                context.get_feature_input()

            storage_context_input_names.append(context.input_names())
            storage_context_input_dims.append({
                "lhs": lhs_dims,
                "rhs": rhs_dims,
            })

            return thor.parameters.ParameterSpecification.allocate_storage(
                lhs,
                shape=[lhs_dims[1]],
                dtype=lhs.get_descriptor().get_data_type(),
            )

        return [
            thor.parameters.ParameterSpecification(
                name="scale",
                create_storage_from_context=lambda ctx: allocate_vector(ctx, "scale"),
                trainable=True,
            ),
            thor.parameters.ParameterSpecification(
                name="bias",
                create_storage_from_context=lambda ctx: allocate_vector(ctx, "bias"),
                trainable=True,
            ),
        ]

    def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        assert context.has_input("lhs") is True
        assert context.has_input("rhs") is True
        assert context.has_input("scale") is False
        assert context.has_parameter("scale") is True
        assert context.has_param("bias") is True
        assert context.has_output("missing") is False
        assert context.device_num == 0

        with pytest.raises(RuntimeError, match="has no feature input named 'missing'"):
            context.input_tensor("missing")
        with pytest.raises(RuntimeError, match="has no parameter named 'missing'"):
            context.parameter_tensor("missing")
        with pytest.raises(RuntimeError, match="has no feature input named 'scale'"):
            context.input("scale")

        lhs_tensor = context.input_tensor("lhs")
        rhs_tensor = context.input_tensor("rhs")
        scale_tensor = context.param_tensor("scale")
        bias_tensor = context.parameter_tensor("bias")
        lhs_dims = lhs_tensor.get_dimensions()
        rhs_dims = rhs_tensor.get_dimensions()
        scale_dims = scale_tensor.get_dimensions()
        bias_dims = bias_tensor.get_dimensions()

        assert lhs_dims == rhs_dims
        assert scale_dims == [lhs_dims[1]]
        assert bias_dims == [lhs_dims[1]]
        assert set(context.inputs.keys()) == {"lhs", "rhs"}
        assert set(context.input_tensors.keys()) == {"lhs", "rhs"}
        assert set(context.parameters.keys()) == {"scale", "bias"}
        assert set(context.parameter_tensors.keys()) == {"scale", "bias"}
        assert set(context.param_tensors.keys()) == {"scale", "bias"}

        output_dims: dict[str, list[int]] = {}
        if context.has_output("sum_output"):
            output_dims["sum_output"] = context.output_tensor("sum_output").get_dimensions()
        if context.has_output("affine_output"):
            output_dims["affine_output"] = context.output_tensor("affine_output").get_dimensions()

        self.build_contexts.append(
            {
                "input_dims": {
                    "lhs": lhs_dims,
                    "rhs": rhs_dims,
                },
                "parameter_dims": {
                    "scale": scale_dims,
                    "bias": bias_dims,
                },
                "output_dims": output_dims,
                "output_names": sorted(context.outputs.keys()),
                "stream_gpu_num": context.stream.get_gpu_num(),
            })

        lhs = context.input("lhs")
        rhs = context.input("rhs")
        scale = context.param("scale")
        bias = context.param("bias")

        shared = lhs + rhs
        return {
            "sum_output": shared,
            "affine_output": shared * scale + bias,
        }


class NoParameterMimoLayer(thor.layers.CustomLayer):

    def __init__(self, network: thor.Network, lhs: thor.Tensor, rhs: thor.Tensor):
        super().__init__(
            network=network,
            inputs={
                "lhs": lhs,
                "rhs": rhs
            },
            output_names=["sum"],
        )

    def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        return {
            "sum": context.input("lhs") + context.input("rhs")
        }


def _network_input(network: thor.Network, name: str, shape: list[int], dtype: thor.DataType) -> thor.Tensor:
    return thor.layers.NetworkInput(network, name, shape, dtype).get_feature_output()


def _place_for_custom_layer_test(
    network: thor.Network,
    *,
    batch_size: int = 2,
    inference_only: bool = True,
) -> thor.runtime.PlacedNetwork:
    return network.place(
        batch_size,
        inference_only=inference_only,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, shape)
    return thor.physical.PhysicalTensor(placement, descriptor)


def _cpu_tensor_from_numpy(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    tensor = _cpu_tensor(list(values.shape), dtype)
    tensor.numpy()[...] = values
    return tensor


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


def test_python_cuda_kernel_layer_wraps_kernel_expression_and_infers_outputs():
    network = thor.Network("cuda-kernel-layer-logical")
    x = thor.Tensor([8], thor.DataType.fp32)
    kernel = (
        thor.physical.CudaKernelExpression.builder("py_layer_square")
        .source(
            r"""
extern "C" __global__
void py_layer_square_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * x[i];
}
"""
        )
        .entry("py_layer_square_kernel")
        .input("x", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, thor.physical.CudaKernelExpression.numel("y"))
        .launch_grid_1d(thor.physical.CudaKernelExpression.numel("y"), block_size=128)
        .build()
    )

    layer = thor.layers.CudaKernelLayer(network=network, inputs={"x": x}, kernel=kernel)

    assert isinstance(layer, thor.layers.CustomLayer)
    assert layer.get_input_names() == ["x"]
    assert layer.get_output_names() == ["y"]
    assert layer["y"].get_dimensions() == [8]
    assert layer["y"].get_data_type() == thor.DataType.fp32


def test_python_cuda_kernel_layer_rejects_non_kernel():
    network = thor.Network("cuda-kernel-layer-rejects-non-kernel")
    x = thor.Tensor([8], thor.DataType.fp32)

    with pytest.raises(TypeError, match="CudaKernelLayer kernel must be"):
        thor.layers.CudaKernelLayer(network=network, inputs={"x": x}, kernel=object())


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


@pytest.mark.parametrize("getter", ["getitem", "get_output"])
def test_python_custom_layer_rejects_unknown_output_name(getter: str):
    network = thor.Network("custom-layer-missing-output")
    x = thor.Tensor([5], thor.DataType.fp16)
    layer = FusedLinear(network, x, 3, has_bias=False)

    with pytest.raises(RuntimeError, match="CustomLayer has no output named 'missing'"):
        if getter == "getitem":
            _ = layer["missing"]
        else:
            layer.get_output("missing")


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


def test_python_custom_layer_builds_logical_mimo_interface():
    network = thor.Network("custom-layer-logical-mimo")
    lhs = thor.Tensor([5], thor.DataType.fp32)
    rhs = thor.Tensor([5], thor.DataType.fp32)

    layer = MultiInputMultiOutputAffine(network, lhs, rhs)

    assert layer.get_input_names() == ["lhs", "rhs"]
    assert layer.get_output_names() == ["sum_output", "affine_output"]
    assert set(layer.outputs.keys()) == {"sum_output", "affine_output"}
    assert layer["sum_output"].get_dimensions() == [5]
    assert layer["affine_output"].get_dimensions() == [5]
    assert layer["sum_output"].get_data_type() == thor.DataType.fp32
    assert layer["affine_output"].get_data_type() == thor.DataType.fp32
    assert [parameter.name for parameter in layer.get_parameters()] == ["scale", "bias"]


@pytest.mark.cuda
def test_python_custom_layer_place_invokes_build_with_physical_context():
    network = thor.Network("custom-layer-place-physical-context")
    network_input = thor.layers.NetworkInput(network, "input", [5], thor.DataType.fp16)

    layer = FusedLinear(network, network_input.get_feature_output(), 3, has_bias=True)
    thor.layers.NetworkOutput(network, "output", layer["feature_output"], thor.DataType.fp16)

    placed = _place_for_custom_layer_test(network, batch_size=2, inference_only=True)

    assert placed.get_num_stamps() >= 1

    weights: thor.parameters.ParameterSpecification = layer.get_parameters()[0]
    assert weights.name == "weights"
    assert weights.trainable is True
    assert weights.is_trainable() is True
    assert weights.is_training_initially_enabled() is True
    assert weights.has_optimizer() is False

    biases: thor.parameters.ParameterSpecification = layer.get_parameters()[1]
    assert biases.name == "biases"
    assert biases.trainable is True
    assert biases.is_trainable() is True
    assert biases.is_training_initially_enabled() is True
    assert biases.has_optimizer() is False

    bound_weights: thor.parameters.BoundParameter = layer.get_bound_parameter(placed, "weights")
    assert bound_weights.name == "weights"
    assert bound_weights.trainable is True
    assert bound_weights.is_trainable() is True
    assert bound_weights.has_optimizer() is False
    assert bound_weights.is_training_enabled() is False
    bound_weights.set_training_enabled(True)
    assert bound_weights.is_training_enabled() is False  # Was compiled for inference only, cannot enable training
    bound_weights.set_training_enabled(False)
    assert bound_weights.is_training_enabled() is False

    bound_biases: thor.parameters.BoundParameter = layer.get_bound_parameters(placed)[1]
    assert bound_biases.name == "biases"
    assert bound_biases.is_training_enabled() is False
    bound_biases.set_training_enabled(True)
    assert bound_biases.is_training_enabled() is False  # Was compiled for inference only, cannot enable training
    bound_biases.set_training_enabled(False)
    assert bound_biases.is_training_enabled() is False


@pytest.mark.cuda
def test_python_custom_layer_mimo_place_exposes_named_physical_contexts_and_parameters():
    network = thor.Network("custom-layer-mimo-place-context")
    lhs = _network_input(network, "lhs", [5], thor.DataType.fp32)
    rhs = _network_input(network, "rhs", [5], thor.DataType.fp32)

    layer = MultiInputMultiOutputAffine(network, lhs, rhs)
    thor.layers.NetworkOutput(network, "sum", layer["sum_output"], thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "affine", layer["affine_output"], thor.DataType.fp32)

    placed = _place_for_custom_layer_test(network, batch_size=3, inference_only=True)

    assert placed.get_num_stamps() >= 1
    assert layer.storage_context_input_names
    assert all(names == ["lhs", "rhs"] for names in layer.storage_context_input_names)
    assert (3, 5) in {tuple(record["lhs"]) for record in layer.storage_context_input_dims}
    assert (3, 5) in {tuple(record["rhs"]) for record in layer.storage_context_input_dims}

    assert layer.build_contexts
    assert any(ctx["input_dims"] == {
        "lhs": [3, 5],
        "rhs": [3, 5]
    } for ctx in layer.build_contexts)
    assert any(ctx["parameter_dims"] == {
        "scale": [5],
        "bias": [5]
    } for ctx in layer.build_contexts)
    assert any(ctx["output_dims"] == {
        "sum_output": [3, 5],
        "affine_output": [3, 5]
    } for ctx in layer.build_contexts)
    assert any(ctx["output_names"] == ["affine_output", "sum_output"] for ctx in layer.build_contexts)
    assert all(ctx["stream_gpu_num"] == 0 for ctx in layer.build_contexts)

    bound_parameters = layer.get_bound_parameters(placed)
    assert [parameter.name for parameter in bound_parameters] == ["scale", "bias"]
    assert all(parameter.is_training_enabled() is False for parameter in bound_parameters)


@pytest.mark.cuda
def test_python_custom_layer_mimo_without_parameters_places_and_builds():
    network = thor.Network("custom-layer-mimo-no-parameters")
    lhs = _network_input(network, "lhs", [4], thor.DataType.fp16)
    rhs = _network_input(network, "rhs", [4], thor.DataType.fp16)

    layer = NoParameterMimoLayer(network, lhs, rhs)
    thor.layers.NetworkOutput(network, "sum", layer["sum"], thor.DataType.fp16)

    placed = _place_for_custom_layer_test(network, batch_size=2, inference_only=True)

    assert placed.get_num_stamps() >= 1
    assert layer.get_parameters() == []
    assert layer.get_bound_parameters(placed) == []


@pytest.mark.cuda
def test_python_custom_layer_supports_inherited_python_build_and_parameters():

    class BaseAffine(thor.layers.CustomLayer):

        def __init__(self, network: thor.Network, x: thor.Tensor):
            super().__init__(
                network=network,
                inputs=x,
            )

        def parameters(self):
            return [
                thor.parameters.ParameterSpecification(
                    name="bias",
                    create_storage_from_context=lambda ctx: thor.parameters.ParameterSpecification.allocate_storage(
                        ctx.get_feature_input(),
                        shape=[ctx.get_feature_input().get_descriptor().get_dimensions()[1]],
                        dtype=ctx.get_feature_input().get_descriptor().get_data_type(),
                    ),
                    trainable=True,
                )
            ]

        def build(self, context):
            return {
                "feature_output": context.input("feature_input") + context.param("bias")
            }

    class DerivedAffine(BaseAffine):
        pass

    network = thor.Network("custom-layer-inherited-python-overrides")
    x = _network_input(network, "input", [3], thor.DataType.fp32)
    layer = DerivedAffine(network, x)
    thor.layers.NetworkOutput(network, "output", layer["feature_output"], thor.DataType.fp32)

    placed = _place_for_custom_layer_test(network)

    assert placed.get_num_stamps() >= 1
    assert [parameter.name for parameter in layer.get_parameters()] == ["bias"]
    assert [parameter.name for parameter in layer.get_bound_parameters(placed)] == ["bias"]


def test_python_custom_layer_parameters_must_return_list_not_dict():

    class BadParametersDict(thor.layers.CustomLayer):

        def __init__(self):
            network = thor.Network("custom-layer-bad-parameters-dict")
            x = thor.Tensor([3], thor.DataType.fp32)
            super().__init__(
                network=network,
                inputs=x,
            )

        def parameters(self):
            return {
                "weights": thor.parameters.ParameterSpecification(name="weights", shape=[3])
            }

        def build(self, context):
            return {
                "feature_output": context.input("feature_input")
            }

    with pytest.raises(TypeError, match=r"parameters\(\) must return list\[thor.parameters.ParameterSpecification\], not dict"):
        BadParametersDict()


def test_python_custom_layer_parameters_must_return_list():

    class BadParametersScalar(thor.layers.CustomLayer):

        def __init__(self):
            network = thor.Network("custom-layer-bad-parameters-scalar")
            x = thor.Tensor([3], thor.DataType.fp32)
            super().__init__(
                network=network,
                inputs=x,
            )

        def parameters(self):
            return 123

        def build(self, context):
            return {
                "feature_output": context.input("feature_input")
            }

    with pytest.raises(TypeError, match=r"parameters\(\) return value: expected list\[thor.parameters.ParameterSpecification\]"):
        BadParametersScalar()


def test_python_custom_layer_rejects_duplicate_parameter_names():

    class DuplicateParameterNames(thor.layers.CustomLayer):

        def __init__(self):
            network = thor.Network("custom-layer-duplicate-parameter-names")
            x = thor.Tensor([3], thor.DataType.fp32)
            super().__init__(
                network=network,
                inputs=x,
            )

        def parameters(self):
            return [
                thor.parameters.ParameterSpecification(name="weights", shape=[3]),
                thor.parameters.ParameterSpecification(name="weights", shape=[3]),
            ]

        def build(self, context):
            return {
                "feature_output": context.input("feature_input")
            }

    with pytest.raises(RuntimeError, match="duplicate Parameter name 'weights'"):
        DuplicateParameterNames()


def test_python_custom_layer_rejects_parameter_name_that_conflicts_with_feature_input():

    class ParameterInputNameCollision(thor.layers.CustomLayer):

        def __init__(self):
            network = thor.Network("custom-layer-parameter-input-name-collision")
            x = thor.Tensor([3], thor.DataType.fp32)
            super().__init__(
                network=network,
                inputs=x,
            )

        def parameters(self):
            return [thor.parameters.ParameterSpecification(name="feature_input", shape=[3])]

        def build(self, context):
            return {
                "feature_output": context.input("feature_input")
            }

    with pytest.raises(RuntimeError, match="conflicts with a feature input name"):
        ParameterInputNameCollision()


@pytest.mark.cuda
def test_python_custom_layer_place_rejects_parameter_name_that_conflicts_with_output():

    class ParameterOutputNameCollision(thor.layers.CustomLayer):

        def __init__(self, network: thor.Network, x: thor.Tensor):
            super().__init__(
                network=network,
                inputs=x,
            )

        def parameters(self):
            return [
                thor.parameters.ParameterSpecification(
                    name="feature_output",
                    create_storage_from_context=lambda ctx: thor.parameters.ParameterSpecification.allocate_storage(
                        ctx.get_feature_input(),
                        shape=[ctx.get_feature_input().get_descriptor().get_dimensions()[1]],
                        dtype=ctx.get_feature_input().get_descriptor().get_data_type(),
                    ),
                    trainable=True,
                ),
            ]

        def build(self, context):
            return {
                "feature_output": context.input("feature_input")
            }

    network = thor.Network("custom-layer-parameter-output-name-collision")
    x = _network_input(network, "input", [3], thor.DataType.fp32)
    layer = ParameterOutputNameCollision(network, x)
    thor.layers.NetworkOutput(network, "output", layer["feature_output"], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="parameter name collides with an output port name"):
        _place_for_custom_layer_test(network)


def test_python_custom_layer_rejects_empty_inputs_and_outputs():
    network = thor.Network("custom-layer-empty-input-output")
    x = thor.Tensor([3], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="requires at least one input"):
        thor.layers.CustomLayer(
            network=network,
            inputs={},
            output_names=["feature_output"],
            build=lambda context: {
                "feature_output": context.input("feature_input")
            },
        )

    with pytest.raises(ValueError, match="requires at least one output name"):
        thor.layers.CustomLayer(
            network=network,
            inputs={
                "feature_input": x
            },
            output_names=[],
            build=lambda context: {
                "feature_output": context.input("feature_input")
            },
        )


@pytest.mark.cuda
def test_python_custom_layer_place_rejects_build_returning_non_dict():

    class NonDictBuild(thor.layers.CustomLayer):

        def __init__(self, network: thor.Network, x: thor.Tensor):
            super().__init__(
                network=network,
                inputs=x,
            )

        def build(self, context):
            return [context.input("feature_input")]

    network = thor.Network("custom-layer-build-non-dict")
    x = _network_input(network, "input", [3], thor.DataType.fp32)

    with pytest.raises(TypeError, match=r"build\(context\) return value: expected dict"):
        NonDictBuild(network, x)


@pytest.mark.cuda
def test_python_custom_layer_place_rejects_build_returning_non_expression_value():

    class NonExpressionBuildValue(thor.layers.CustomLayer):

        def __init__(self, network: thor.Network, x: thor.Tensor):
            super().__init__(
                network=network,
                inputs=x,
            )

        def build(self, context):
            return {
                "feature_output": object()
            }

    network = thor.Network("custom-layer-build-non-expression")
    x = _network_input(network, "input", [3], thor.DataType.fp32)

    with pytest.raises(TypeError, match=r"build\(context\) return dict value: expected thor\.physical\.Expression"):
        NonExpressionBuildValue(network, x)


@pytest.mark.cuda
def test_python_custom_layer_place_rejects_missing_declared_output_from_build():

    class MissingDeclaredOutput(thor.layers.CustomLayer):

        def __init__(self, network: thor.Network, x: thor.Tensor):
            super().__init__(
                network=network,
                inputs=x,
                output_names=["first", "second"],
            )

        def build(self, context):
            return {
                "first": context.input("feature_input")
            }

    network = thor.Network("custom-layer-build-missing-output")
    x = _network_input(network, "input", [3], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="CustomLayer forward output mismatch"):
        MissingDeclaredOutput(network, x)


@pytest.mark.cuda
def test_python_custom_layer_place_rejects_unreferenced_declared_parameter():

    class DropsParameter(thor.layers.CustomLayer):

        def __init__(self, network: thor.Network, x: thor.Tensor):
            super().__init__(
                network=network,
                inputs=x,
            )

        def parameters(self):
            return [
                thor.parameters.ParameterSpecification(
                    name="scale",
                    create_storage_from_context=lambda ctx: thor.parameters.ParameterSpecification.allocate_storage(
                        ctx.get_feature_input(),
                        shape=[ctx.get_feature_input().get_descriptor().get_dimensions()[1]],
                        dtype=ctx.get_feature_input().get_descriptor().get_data_type(),
                    ),
                    trainable=True,
                ),
            ]

        def build(self, context):
            return {
                "feature_output": context.input("feature_input")
            }

    network = thor.Network("custom-layer-build-drops-parameter")
    x = _network_input(network, "input", [3], thor.DataType.fp32)

    with pytest.raises(RuntimeError, match="Unexpected input"):
        DropsParameter(network, x)


@pytest.mark.cuda
def test_python_custom_layer_bound_parameter_rejects_unknown_name_after_place():
    network = thor.Network("custom-layer-bound-parameter-missing")
    x = _network_input(network, "input", [5], thor.DataType.fp16)
    layer = FusedLinear(network, x, 3, has_bias=False)
    thor.layers.NetworkOutput(network, "output", layer["feature_output"], thor.DataType.fp16)
    placed = _place_for_custom_layer_test(network, batch_size=2, inference_only=True)

    with pytest.raises(RuntimeError, match="is not present on this parameterizable"):
        layer.get_bound_parameter(placed, "missing")


def test_python_custom_layer_direct_construction_stitches_activation_into_expression_build():
    network = thor.Network("custom-layer-direct-activation")
    x = thor.Tensor([6], thor.DataType.fp32)

    build_calls: list[dict[str, object]] = []

    def build(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        build_calls.append(
            {
                "input_dims": context.input_tensor("feature_input").get_dimensions(),
                "output_names": sorted(context.outputs.keys()),
            })
        return {
            "feature_output": context.input("feature_input") - 1.0,
        }

    layer = thor.layers.CustomLayer(
        network=network,
        inputs=x,
        build=build,
        activation=thor.activations.Relu(),
    )

    y = layer["feature_output"]
    assert y.get_dimensions() == [6]
    assert y.get_data_type() == thor.DataType.fp32
    assert layer.get_input_names() == ["feature_input"]
    assert layer.get_output_names() == ["feature_output"]
    assert build_calls
    assert build_calls[0]["input_dims"] == [1, 6]


def test_python_custom_layer_activation_rejects_dynamic_expression_build_return():
    network = thor.Network("custom-layer-activation-dynamic-build")
    x = thor.Tensor([3], thor.DataType.fp32)

    def build(context: thor.layers.CustomLayerBuildContext) -> thor.physical.DynamicExpressionBuild:
        expression_outputs = thor.physical.Expression.outputs({
            "feature_output": context.input("feature_input"),
        })
        equation = expression_outputs.compile(context.device_num)
        return thor.physical.DynamicExpressionBuild(
            equation=equation,
            stamp_inputs=context.inputs,
            tensor_scalar_inputs={},
            preallocated_outputs=context.outputs,
            requested_output_shapes={},
        )

    with pytest.raises(RuntimeError, match="activation can only be stitched"):
        thor.layers.CustomLayer(
            network=network,
            inputs=x,
            build=build,
            activation=thor.activations.Relu(),
        )


def test_python_custom_layer_direct_construction_builds_logical_output_interface():
    network = thor.Network("custom-layer-direct-logical")
    x = thor.Tensor([6], thor.DataType.fp32)

    build_calls: list[dict[str, object]] = []

    def build(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        build_calls.append(
            {
                "input_dims": context.input_tensor("feature_input").get_dimensions(),
                "output_names": sorted(context.outputs.keys()),
            })
        return {
            "feature_output": context.input("feature_input") * 2.0,
        }

    layer = thor.layers.CustomLayer(
        network=network,
        inputs=x,
        build=build,
    )

    y = layer["feature_output"]
    assert y.get_dimensions() == [6]
    assert y.get_data_type() == thor.DataType.fp32
    assert layer.outputs["feature_output"].get_dimensions() == [6]
    assert layer.get_input_names() == ["feature_input"]
    assert layer.get_output_names() == ["feature_output"]
    assert layer.get_parameters() == []
    assert build_calls
    assert build_calls[0]["input_dims"] == [1, 6]


def test_python_custom_layer_callable_dict_build_is_serializable(tmp_path):
    network = thor.Network("custom-layer-direct-dict-serializable")
    x = thor.layers.NetworkInput(network, "input", [6], thor.DataType.fp32).get_feature_output()

    def build(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        return {
            "feature_output": context.input("feature_input") * 2.0,
        }

    layer = thor.layers.CustomLayer(
        network=network,
        inputs=x,
        build=build,
    )
    thor.layers.NetworkOutput(network, "output", layer["feature_output"], thor.DataType.fp32)

    architecture = json.loads(network.get_architecture_json())
    custom_layers = [layer_json for layer_json in architecture["layers"] if layer_json["layer_type"] == "custom_layer"]
    assert len(custom_layers) == 1
    assert custom_layers[0]["expression"]["type"] == "thor.expression"

    save_dir = tmp_path / "custom_layer_direct_dict"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network("custom-layer-direct-dict-serializable")
    loaded.load(str(save_dir))
    loaded_architecture = json.loads(loaded.get_architecture_json())
    loaded_custom_layers = [
        layer_json for layer_json in loaded_architecture["layers"] if layer_json["layer_type"] == "custom_layer"
    ]
    assert len(loaded_custom_layers) == 1
    assert loaded_custom_layers[0]["expression"]["type"] == "thor.expression"


@pytest.mark.cuda
def test_python_custom_layer_callable_dict_build_runs_numerically_after_save_load(tmp_path):
    dtype = thor.DataType.fp32
    batch_size = 3
    network = thor.Network("custom-layer-direct-dict-numerical-round-trip")
    x = thor.layers.NetworkInput(network, "input", [6], dtype).get_feature_output()

    def build(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        feature_input = context.input("feature_input")
        return {
            "feature_output": feature_input * 2.0 + 1.25,
        }

    layer = thor.layers.CustomLayer(
        network=network,
        inputs=x,
        build=build,
    )
    thor.layers.NetworkOutput(network, "output", layer["feature_output"], dtype)

    values = np.array(
        [
            [0.0, 1.0, -2.0, 3.5, -4.5, 8.0],
            [9.0, -10.0, 11.5, -12.5, 13.0, -14.0],
            [0.25, -0.75, 1.25, -1.5, 2.5, -3.25],
        ],
        dtype=np.float32,
    )
    expected = values * np.float32(2.0) + np.float32(1.25)

    placed = _place_for_custom_layer_test(network, batch_size=batch_size, inference_only=True)
    outputs = placed.infer({"input": _cpu_tensor_from_numpy(values, dtype)})
    assert set(outputs.keys()) == {"output"}
    np.testing.assert_allclose(np.array(outputs["output"].numpy(), copy=True), expected, rtol=0.0, atol=0.0)

    architecture = json.loads(network.get_architecture_json())
    custom_layers = [layer_json for layer_json in architecture["layers"] if layer_json["layer_type"] == "custom_layer"]
    assert len(custom_layers) == 1
    expression = custom_layers[0]["expression"]
    assert expression["type"] == "thor.expression"
    assert expression["schema_version"] == 1
    assert expression["expected_input_names"] == ["feature_input"]
    assert expression["expected_output_names"] == ["feature_output"]
    assert expression["canonical_hash"].startswith("fnv1a64:")

    save_dir = tmp_path / "custom_layer_direct_dict_numerical"
    network.save(str(save_dir), overwrite=False)

    loaded = thor.Network("custom-layer-direct-dict-numerical-round-trip")
    loaded.load(str(save_dir))
    loaded_architecture = json.loads(loaded.get_architecture_json())
    loaded_custom_layers = [
        layer_json for layer_json in loaded_architecture["layers"] if layer_json["layer_type"] == "custom_layer"
    ]
    assert len(loaded_custom_layers) == 1
    assert loaded_custom_layers[0]["expression"] == expression

    loaded_placed = _place_for_custom_layer_test(loaded, batch_size=batch_size, inference_only=True)
    loaded_outputs = loaded_placed.infer({"input": _cpu_tensor_from_numpy(values, dtype)})
    assert set(loaded_outputs.keys()) == {"output"}
    np.testing.assert_allclose(np.array(loaded_outputs["output"].numpy(), copy=True), expected, rtol=0.0, atol=0.0)


def test_python_custom_layer_uint32_strided_view_can_infer_outputs():
    network = thor.Network("custom-layer-uint32-strided-view")
    x = thor.layers.NetworkInput(network, "x", [9], thor.DataType.uint32).get_feature_output()

    class Uint32Split(thor.layers.CustomLayer):
        def __init__(self, network: thor.Network, packed: thor.Tensor):
            super().__init__(network=network, inputs={"packed": packed}, output_names=["head", "tail"])

        def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
            packed_tensor = context.input_tensor("packed")
            packed_dims = packed_tensor.get_dimensions()
            assert packed_dims in ([1, 9], [2, 9])
            packed = context.input("packed", output_dtype=thor.DataType.uint32, compute_dtype=thor.DataType.uint32)
            return {
                "head": packed.strided_view([packed_dims[0], 4], [packed_dims[1], 1], 0),
                "tail": packed.strided_view([packed_dims[0], 4], [packed_dims[1], 1], 5),
            }

    split = Uint32Split(network, x)
    assert split["head"].get_dimensions() == [4]
    assert split["tail"].get_dimensions() == [4]
    assert split["head"].get_data_type() == thor.DataType.uint32
    assert split["tail"].get_data_type() == thor.DataType.uint32


def test_python_custom_layer_output_specs_are_logical_and_batch_free():
    network = thor.Network("custom-layer-logical-output-specs")
    x = thor.layers.NetworkInput(network, "x", [6, 2], thor.DataType.fp32).get_feature_output()
    seen_specs: list[list[int]] = []
    seen_physical_shapes: list[list[int]] = []

    class TailWindow(thor.layers.CustomLayer):
        def __init__(self, network: thor.Network, sequence: thor.Tensor):
            super().__init__(network=network, inputs={"sequence": sequence}, output_names=["feature_output"])

        def output_specs(self, context: thor.layers.CustomLayerSpecContext) -> dict[str, thor.layers.TensorSpec]:
            sequence = context.input_spec("sequence")
            seen_specs.append(list(sequence.shape))
            return {
                "feature_output": thor.layers.TensorSpec(shape=[3, sequence.shape[1]], dtype=sequence.dtype),
            }

        def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
            dimensions = context.input_tensor("sequence").get_dimensions()
            seen_physical_shapes.append(list(dimensions))
            batch_size, sequence_steps, channels = dimensions
            return {
                "feature_output": context.input("sequence").strided_view(
                    [batch_size, 3, channels],
                    [sequence_steps * channels, channels, 1],
                    (sequence_steps - 3) * channels,
                )
            }

    tail = TailWindow(network, x)
    assert tail["feature_output"].get_dimensions() == [3, 2]
    assert seen_specs == [[6, 2]]
    assert seen_physical_shapes == [[1, 6, 2], [2, 6, 2]]


def _find_expression_nodes_with_field(value: object, field: str) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    if isinstance(value, dict):
        if field in value:
            found.append(value)
        for child in value.values():
            found.extend(_find_expression_nodes_with_field(child, field))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_expression_nodes_with_field(child, field))
    return found


@pytest.mark.cuda
def test_python_custom_layer_batch_dependent_view_is_symbolically_serialized_and_reloaded(tmp_path):
    network_name = "custom-layer-batch-polymorphic-view-round-trip"
    network = thor.Network(network_name)
    x = thor.layers.NetworkInput(network, "x", [6, 2], thor.DataType.fp32).get_feature_output()

    class TailWindow(thor.layers.CustomLayer):
        def __init__(self, network: thor.Network, sequence: thor.Tensor):
            super().__init__(network=network, inputs={"sequence": sequence}, output_names=["feature_output"])

        def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
            batch_size, sequence_steps, channels = context.input_tensor("sequence").get_dimensions()
            return {
                "feature_output": context.input("sequence").strided_view(
                    [batch_size, 3, channels],
                    [sequence_steps * channels, channels, 1],
                    (sequence_steps - 3) * channels,
                )
            }

    tail = TailWindow(network, x)
    thor.layers.NetworkOutput(network, "tail", tail["feature_output"], thor.DataType.fp32)

    architecture = json.loads(network.get_architecture_json())
    custom_layer = next(layer for layer in architecture["layers"] if layer["layer_type"] == "custom_layer")
    view_nodes = _find_expression_nodes_with_field(custom_layer["expression"], "view_dims")
    assert any(node["view_dims"] == [0, 3, 2] for node in view_nodes)

    save_dir = tmp_path / "batch_polymorphic_tail"
    network.save(str(save_dir), overwrite=False)
    tail_ref = weakref.ref(tail)
    del tail
    del x
    del network
    gc.collect()
    assert tail_ref() is None

    loaded = thor.Network(network_name)
    loaded.load(str(save_dir))
    placed = _place_for_custom_layer_test(loaded, batch_size=4, inference_only=True)
    values = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    outputs = placed.infer({"x": _cpu_tensor_from_numpy(values, thor.DataType.fp32)})
    np.testing.assert_array_equal(np.array(outputs["tail"].numpy(), copy=True), values[:, -3:, :])


def test_python_custom_layer_rejects_fixed_inference_batch_definition_during_construction():
    network = thor.Network("custom-layer-fixed-batch-definition-rejected")
    x = thor.layers.NetworkInput(network, "x", [6, 2], thor.DataType.fp32).get_feature_output()

    def output_specs(context: thor.layers.CustomLayerSpecContext) -> dict[str, thor.layers.TensorSpec]:
        sequence = context.input_spec("sequence")
        return {"feature_output": thor.layers.TensorSpec([3, sequence.shape[1]], sequence.dtype)}

    def build(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        _, sequence_steps, channels = context.input_tensor("sequence").get_dimensions()
        return {
            "feature_output": context.input("sequence").strided_view(
                [1, 3, channels],
                [sequence_steps * channels, channels, 1],
                (sequence_steps - 3) * channels,
            )
        }

    with pytest.raises(RuntimeError, match="CustomLayer construction rejected.*batch-2.*logical TensorSpec"):
        thor.layers.CustomLayer(
            network=network,
            inputs={"sequence": x},
            output_names=["feature_output"],
            output_specs=output_specs,
            build=build,
        )


def test_python_custom_layer_without_serializable_definition_is_rejected_during_construction():
    network = thor.Network("custom-layer-requires-serializable-definition")
    x = thor.layers.NetworkInput(network, "x", [6], thor.DataType.fp32).get_feature_output()

    def output_specs(context: thor.layers.CustomLayerSpecContext) -> dict[str, thor.layers.TensorSpec]:
        spec = context.input_spec("feature_input")
        return {"feature_output": thor.layers.TensorSpec(spec.shape, spec.dtype)}

    def build(context: thor.layers.CustomLayerBuildContext) -> thor.physical.DynamicExpressionBuild:
        expression_outputs = thor.physical.Expression.outputs({
            "feature_output": context.input("feature_input") + 1.0,
        })
        return thor.physical.DynamicExpressionBuild(
            equation=expression_outputs.compile(context.device_num),
            stamp_inputs=context.inputs,
            tensor_scalar_inputs={},
            preallocated_outputs=context.outputs,
            requested_output_shapes={},
        )

    with pytest.raises(RuntimeError, match="CustomLayer construction rejected.*did not provide an ExpressionDefinition"):
        thor.layers.CustomLayer(network=network, inputs=x, build=build, output_specs=output_specs)


@pytest.mark.cuda
def test_temporary_python_custom_layer_strided_view_uses_placement_batch_size():
    network = thor.Network("temporary-custom-layer-strided-view-runtime-batch")
    x = thor.layers.NetworkInput(network, "x", [6, 2], thor.DataType.fp32).get_feature_output()
    build_input_shapes: list[list[int]] = []

    class TailWindow(thor.layers.CustomLayer):
        def __init__(self, network: thor.Network, sequence: thor.Tensor):
            super().__init__(network=network, inputs={"sequence": sequence}, output_names=["feature_output"])

        def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
            dimensions = context.input_tensor("sequence").get_dimensions()
            build_input_shapes.append(list(dimensions))
            batch_size, sequence_steps, channels = dimensions
            sequence = context.input("sequence")
            return {
                "feature_output": sequence.strided_view(
                    [batch_size, 3, channels],
                    [sequence_steps * channels, channels, 1],
                    (sequence_steps - 3) * channels,
                )
            }

    # Match common graph-building code that keeps only the output tensor. The Network
    # must retain the Python CustomLayer owner so build(context) can be called again
    # with placement-time dimensions instead of reusing the batch-1 inference shape.
    tail = TailWindow(network, x)["feature_output"]
    gc.collect()
    assert tail.get_dimensions() == [3, 2]

    thor.layers.NetworkOutput(network, "tail", tail, thor.DataType.fp32)
    placed = _place_for_custom_layer_test(network, batch_size=4, inference_only=True)

    values = np.arange(4 * 6 * 2, dtype=np.float32).reshape(4, 6, 2)
    outputs = placed.infer({"x": _cpu_tensor_from_numpy(values, thor.DataType.fp32)})
    actual = np.array(outputs["tail"].numpy(), copy=True)
    np.testing.assert_array_equal(actual, values[:, -3:, :])

    assert [1, 6, 2] in build_input_shapes
    assert [4, 6, 2] in build_input_shapes


@pytest.mark.cuda
def test_python_custom_layer_direct_construction_places_with_named_inputs_and_parameters():
    network = thor.Network("custom-layer-direct-mimo-place")
    lhs = _network_input(network, "lhs", [5], thor.DataType.fp32)
    rhs = _network_input(network, "rhs", [5], thor.DataType.fp32)

    storage_context_input_names: list[list[str]] = []
    storage_context_input_dims: list[dict[str, list[int]]] = []
    build_contexts: list[dict[str, object]] = []

    def allocate_vector(context: thor.parameters.ParameterSpecification.StorageContext) -> thor.physical.PhysicalTensor:
        assert context.input_names() == ["lhs", "rhs"]
        lhs_tensor = context.get_input("lhs")
        rhs_tensor = context.get_input("rhs")
        lhs_dims = lhs_tensor.get_descriptor().get_dimensions()
        rhs_dims = rhs_tensor.get_descriptor().get_dimensions()
        assert lhs_dims == rhs_dims

        storage_context_input_names.append(context.input_names())
        storage_context_input_dims.append({
            "lhs": lhs_dims,
            "rhs": rhs_dims,
        })

        return thor.parameters.ParameterSpecification.allocate_storage(
            lhs_tensor,
            shape=[lhs_dims[1]],
            dtype=lhs_tensor.get_descriptor().get_data_type(),
        )

    scale = thor.parameters.ParameterSpecification(
        name="scale",
        create_storage_from_context=allocate_vector,
        trainable=True,
    )
    bias = thor.parameters.ParameterSpecification(
        name="bias",
        create_storage_from_context=allocate_vector,
        trainable=True,
    )

    def build(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        lhs_tensor = context.input_tensor("lhs")
        rhs_tensor = context.input_tensor("rhs")
        scale_tensor = context.parameter_tensor("scale")
        bias_tensor = context.parameter_tensor("bias")

        output_dims: dict[str, list[int]] = {}
        if context.has_output("sum_output"):
            output_dims["sum_output"] = context.output_tensor("sum_output").get_dimensions()
        if context.has_output("affine_output"):
            output_dims["affine_output"] = context.output_tensor("affine_output").get_dimensions()

        build_contexts.append(
            {
                "input_dims": {
                    "lhs": lhs_tensor.get_dimensions(),
                    "rhs": rhs_tensor.get_dimensions(),
                },
                "parameter_dims": {
                    "scale": scale_tensor.get_dimensions(),
                    "bias": bias_tensor.get_dimensions(),
                },
                "output_dims": output_dims,
                "output_names": sorted(context.outputs.keys()),
                "stream_gpu_num": context.stream.get_gpu_num(),
            })

        shared = context.input("lhs") + context.input("rhs")
        return {
            "sum_output": shared,
            "affine_output": shared * context.param("scale") + context.param("bias"),
        }

    layer = thor.layers.CustomLayer(
        network=network,
        inputs={
            "lhs": lhs,
            "rhs": rhs,
        },
        output_names=["sum_output", "affine_output"],
        parameters=[scale, bias],
        build=build,
            )

    assert layer.get_input_names() == ["lhs", "rhs"]
    assert layer.get_output_names() == ["sum_output", "affine_output"]
    assert [parameter.name for parameter in layer.get_parameters()] == ["scale", "bias"]
    assert layer["sum_output"].get_dimensions() == [5]
    assert layer["affine_output"].get_dimensions() == [5]

    thor.layers.NetworkOutput(network, "sum", layer["sum_output"], thor.DataType.fp32)
    thor.layers.NetworkOutput(network, "affine", layer["affine_output"], thor.DataType.fp32)

    placed = _place_for_custom_layer_test(network, batch_size=3, inference_only=True)

    assert placed.get_num_stamps() >= 1
    assert storage_context_input_names
    assert all(names == ["lhs", "rhs"] for names in storage_context_input_names)
    assert {tuple(record["lhs"]) for record in storage_context_input_dims} == {(1, 5), (2, 5), (3, 5)}
    assert {tuple(record["rhs"]) for record in storage_context_input_dims} == {(1, 5), (2, 5), (3, 5)}

    assert build_contexts
    assert any(ctx["input_dims"] == {
        "lhs": [3, 5],
        "rhs": [3, 5]
    } for ctx in build_contexts)
    assert any(ctx["parameter_dims"] == {
        "scale": [5],
        "bias": [5]
    } for ctx in build_contexts)
    assert any(ctx["output_dims"] == {
        "sum_output": [3, 5],
        "affine_output": [3, 5]
    } for ctx in build_contexts)
    assert any(ctx["output_names"] == ["affine_output", "sum_output"] for ctx in build_contexts)
    assert all(ctx["stream_gpu_num"] == 0 for ctx in build_contexts)

    bound_parameters = layer.get_bound_parameters(placed)
    assert [parameter.name for parameter in bound_parameters] == ["scale", "bias"]
    assert all(parameter.is_training_enabled() is False for parameter in bound_parameters)


def test_python_custom_layer_direct_construction_rejects_noncallable_build():
    network = thor.Network("custom-layer-direct-bad-build")
    x = thor.Tensor([3], thor.DataType.fp32)

    with pytest.raises(TypeError, match="CustomLayer build: expected callable"):
        thor.layers.CustomLayer(
            network=network,
            inputs=x,
            build=123,
        )


def test_python_custom_layer_tokenwise_linear_preserves_sequence_axis():
    class TokenwiseLinearForTest(thor.layers.CustomLayer):
        def __init__(self, network: thor.Network, x: thor.Tensor, output_features: int):
            self.input_features = x.get_dimensions()[1]
            self.output_features = output_features
            self.parameter_dtype = x.get_data_type()
            super().__init__(network=network, inputs=x, output_names=["feature_output"])

        def parameters(self) -> list[thor.parameters.ParameterSpecification]:
            return [
                thor.parameters.ParameterSpecification(
                    name="weights",
                    shape=[self.input_features, self.output_features],
                    dtype=self.parameter_dtype,
                    trainable=True,
                ),
                thor.parameters.ParameterSpecification(
                    name="biases",
                    shape=[self.output_features],
                    dtype=self.parameter_dtype,
                    trainable=True,
                ),
            ]

        def build(self, context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
            x_tensor = context.input_tensor("feature_input")
            batch_dim, sequence_dim, hidden_dim = x_tensor.get_dimensions()
            x = context.input("feature_input")
            w = context.param("weights")
            b = context.param("biases")
            flat = x.reshape([batch_dim * sequence_dim, hidden_dim])
            logits = (flat @ w + b).reshape([batch_dim, sequence_dim, self.output_features])
            return {"feature_output": logits}

    network = thor.Network("python_custom_layer_tokenwise_linear_preserves_sequence_axis")
    feature_input = thor.layers.NetworkInput(network, "features", [5, 11], thor.DataType.fp16).get_feature_output()
    layer = TokenwiseLinearForTest(network, feature_input, 7)

    assert layer["feature_output"].get_dimensions() == [5, 7]
    assert layer["feature_output"].get_data_type() == thor.DataType.fp16

    architecture = json.loads(network.get_architecture_json())
    custom_layer = next(layer_json for layer_json in architecture["layers"] if layer_json["layer_type"] == "custom_layer")
    reshape_nodes = _find_expression_nodes_with_field(custom_layer["expression"], "reshape_dims")
    assert any(node["reshape_dims"] == [thor.physical.INFER_DIM, 11] for node in reshape_nodes)
    assert any(node["reshape_dims"] == [thor.physical.INFER_DIM, 5, 7] for node in reshape_nodes)
