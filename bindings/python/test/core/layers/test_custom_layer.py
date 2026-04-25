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

    def create_storage(self, ctx):
        x = ctx.get_input("feature_input")
        dims = x.get_descriptor().get_dimensions()
        batch = dims[0]
        in_features = x.get_descriptor().get_total_num_elements() // batch
        return self.createStorage(
            x,
            shape=[in_features, self.out_features],
            dtype=x.get_descriptor().get_data_type(),
        )


class FusedLinear(thor.layers.CustomLayer):

    def __init__(self, network, x, out_features, has_bias, optimizer=None):
        self.out_features = out_features
        self.has_bias = has_bias
        self.parameters_ctx_seen = False
        self.parameters_ctx_input_dims = None
        self.parameters_ctx_output_dims = None

        super().__init__(
            network=network,
            inputs={
                "feature_input": x,
            },
            outputs={
                "feature_output": thor.Tensor([out_features], x.get_data_type()),
            },
            optimizer=optimizer,
        )

    def parameters(self, ctx: thor.layers.CustomLayerApiContext) -> list[thor.Parameter]:
        self.parameters_ctx_seen = True

        feature_input = ctx.input("feature_input")
        feature_output = ctx.output("feature_output")

        self.parameters_ctx_input_dims = feature_input.get_dimensions()
        self.parameters_ctx_output_dims = feature_output.get_dimensions()

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

        feature_output = x @ w
        if self.has_bias:
            b = ctx.param("biases")
            feature_output = feature_output + b

        return {
            "feature_output": feature_output,
        }


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

    def parameters(self, ctx: thor.layers.CustomLayerApiContext) -> dict[str, thor.Parameter]:
        assert ctx.has_input("feature_input")
        assert ctx.has_output("feature_output")
        assert not ctx.has_input("missing")
        assert not ctx.has_output("missing")

        return {
            "weights": Weights(self.out_features),
            "biases": thor.Parameter(
                name="biases", shape=[self.out_features], dtype=thor.DataType.fp32, trainable=True)
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

    def parameters(self, ctx: thor.layers.CustomLayerApiContext) -> list[thor.Parameter]:
        return [
            Weights(self.out_features),
            thor.Parameter(
                name="biases",
                shape=[self.out_features],
                dtype=thor.DataType.fp32,
                trainable=True,
            )
        ]

    def build(
        self,
        ctx: thor.layers.CustomLayerBuildContext,
    ) -> dict[str, thor.physical.Expression]:
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


def _parameter_names(layer) -> list[str]:
    return [parameter.name for parameter in layer.get_parameters()]


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

    assert _parameter_names(layer) == ["weights"]

    assert layer.parameters_ctx_seen
    assert layer.parameters_ctx_input_dims == [5]
    assert layer.parameters_ctx_output_dims == [3]


def test_python_custom_layer_builds_logical_output_interface_with_bias():
    network = thor.Network("custom-layer-smoke-with-bias")
    x = thor.Tensor([7], thor.DataType.fp32)

    layer = FusedLinear(network, x, 4, has_bias=True)

    y = layer.get_output("feature_output")
    assert y.get_dimensions() == [4]
    assert y.get_data_type() == thor.DataType.fp32

    assert layer.get_input_names() == ["feature_input"]
    assert layer.get_output_names() == ["feature_output"]

    assert _parameter_names(layer) == ["weights", "biases"]

    assert layer.parameters_ctx_seen
    assert layer.parameters_ctx_input_dims == [7]
    assert layer.parameters_ctx_output_dims == [4]


def test_python_custom_layer_accepts_dict_returned_from_parameters():
    network = thor.Network("custom-layer-dict-parameters")
    x = thor.Tensor([6], thor.DataType.fp16)

    layer = DictParameterLayer(network, x, 2)

    assert layer["feature_output"].get_dimensions() == [2]
    assert layer["feature_output"].get_data_type() == thor.DataType.fp16
    assert _parameter_names(layer) == ["weights", "biases"]


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


def test_python_custom_layer_api_context_reports_missing_names():
    network = thor.Network("custom-layer-api-context-missing-name")
    x = thor.Tensor([5], thor.DataType.fp16)

    class BadParameterContextLayer(thor.layers.CustomLayer):

        def __init__(self):
            super().__init__(
                network=network,
                inputs={
                    "feature_input": x,
                },
                outputs={
                    "feature_output": thor.Tensor([3], x.get_data_type()),
                },
            )

        def parameters(self, ctx: thor.layers.CustomLayerApiContext):
            ctx.input("not_an_input")
            return []

        def build(self, ctx: thor.layers.CustomLayerBuildContext):
            return {
                "feature_output": ctx.input("feature_input"),
            }

    with pytest.raises(RuntimeError, match="not_an_input"):
        BadParameterContextLayer()


def test_python_custom_layer_requires_build_override():
    network = thor.Network("custom-layer-build-required")
    x = thor.Tensor([5], thor.DataType.fp16)

    class MissingBuildLayer(thor.layers.CustomLayer):

        def __init__(self, network_arg, x_arg):
            super().__init__(
                network=network_arg,
                inputs={
                    "feature_input": x_arg,
                },
                outputs={
                    "feature_output": thor.Tensor([5], x_arg.get_data_type()),
                },
            )

        def parameters(self, ctx: thor.layers.CustomLayerApiContext):
            return []

    layer = MissingBuildLayer(network, x)

    assert layer["feature_output"].get_dimensions() == [5]
    assert layer.get_input_names() == ["feature_input"]
    assert layer.get_output_names() == ["feature_output"]
