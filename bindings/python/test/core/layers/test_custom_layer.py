import numpy as np
import pytest

import thor
from thor.physical import Expression as ex


class FusedLinear(thor.layers.CustomLayer):

    def __init__(self, network: thor.Network, x: thor.Tensor, num_output_features: int, has_bias: bool):
        self.num_output_features: int = num_output_features
        self.has_bias: bool = has_bias

        super().__init__(
            network=network,
            inputs={"feature_input": x},          # API tensor
            # FIXME: Is it ok to require outputs from here or are they a layer compile time thing?
            #        Check on the implementation side how they are defined.
            #        Also if they are defined here, then maybe I don't need custom layer to infer the output tensor.
            outputs={
                "feature_output": thor.Tensor([num_output_features], x.get_data_type())
            },
        )

    def parameters(self) -> list[thor.Parameter]:

        num_output_features = self.num_output_features

        def create_weights_storage_from_context(context: thor.Parameter.StorageContext) -> thor.physical.PhysicalTensor:
            input_tensor = context.get_feature_input()
            batch_size = input_tensor.get_descriptor().get_dimensions()[0]
            num_input_features = input_tensor.get_descriptor().get_total_num_elements() // batch_size
            return thor.Parameter.allocate_storage(
                input_tensor,
                shape=[num_input_features, num_output_features],
                dtype=input_tensor.get_descriptor().get_data_type(),
            )

        weights = thor.Parameter(
            name="weights",
            create_storage_from_context=create_weights_storage_from_context,
            trainable=True,
        )

        params: list[thor.Parameter] = [weights]
        if self.has_bias:

            def create_biases_storage_from_context(
                    context: thor.Parameter.StorageContext) -> thor.physical.PhysicalTensor:
                input_tensor = context.get_feature_input()
                return thor.Parameter.allocate_storage(
                    input_tensor,
                    shape=[num_output_features],
                    dtype=input_tensor.get_descriptor().get_data_type(),
                )

            biases = thor.Parameter(
                name="biases",
                create_storage_from_context=create_biases_storage_from_context,
                trainable=True,
            )
            params.append(biases)
        return params

    def build(
        self,
        context: thor.layers.CustomLayerBuildContext,
    ) -> dict[str, thor.physical.Expression]:
        # Physical tensors carry the information that is only known after placement:
        # batch-inflated shapes, placement, and parameter storage shapes. Expressions
        # are only the symbolic graph values that will be compiled.
        x_tensor = context.input_tensor("feature_input")
        w_tensor = context.parameter_tensor("weights")

        assert len(x_tensor.get_dimensions()) == 2
        assert len(w_tensor.get_dimensions()) == 2
        assert x_tensor.get_dimensions()[1] == w_tensor.get_dimensions()[0]

        # Ensure the allocated output tensor is the right size for the expression that will fill it.
        if context.has_output("feature_output"):
            feature_output_tensor: thor.physical.PhysicalTensor = context.output_tensor("feature_output")
            assert len(feature_output_tensor.get_dimensions()) == 2
            assert feature_output_tensor.get_dimensions()[1] == w_tensor.get_dimensions()[1]

        # Define the expression whose result will be placed in the output tensor.
        x = context.input("feature_input")
        w = context.param("weights")
        feature_output = x @ w
        if self.has_bias:
            b = context.param("biases")
            feature_output = feature_output + b

        # Bind the output tensor as the memory that the expression will fill.
        return {
            "feature_output": feature_output,
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


@pytest.mark.cuda
def test_python_custom_layer_place_invokes_build_with_physical_context():
    network = thor.Network("custom-layer-place-physical-context")
    network_input = thor.layers.NetworkInput(network, "input", [5], thor.DataType.fp16)

    layer = FusedLinear(network, network_input.get_feature_output(), 3, has_bias=True)
    thor.layers.NetworkOutput(network, "output", layer["feature_output"], thor.DataType.fp16)

    placed = network.place(
        2,
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )

    assert placed.get_num_stamps() >= 1

    weights = layer.get_parameters()[0]
    print(weights)
    # assert set(weights.storage_context_input_names) == {"feature_input"}

    assert weights.name == "weights"
    assert weights.trainable is True
    assert weights.is_trainable() is True
    assert weights.is_training_enabled() is True
    weights.set_training_enabled(False)
    assert weights.is_training_enabled is False
    # parameter.def("has_optimizer", &Parameter::hasOptimizer);
