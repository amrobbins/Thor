import pytest
import thor


def _net():
    return thor.Network("test_net_network_output")


def _tensor_1d(size: int = 1, dtype=thor.DataType.fp32):
    # API tensor: dims + dtype
    return thor.Tensor([size], dtype)


def test_network_output_constructs_and_returns_feature_output():
    n = _net()

    # Use a NetworkInput to produce a connected tensor, which is the most realistic input_tensor.
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16)
    x = ni.get_feature_output()

    no = thor.layers.NetworkOutput(n, "output", x, thor.DataType.fp16)

    assert no is not None
    assert isinstance(no, thor.layers.NetworkOutput)
    assert no.is_external()

    out = no.get_feature_output()
    assert out is not None
    assert isinstance(out, thor.Tensor)


def test_network_output_external_flag():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16)
    no = thor.layers.NetworkOutput(n, "internal_output", ni.get_feature_output(), thor.DataType.fp16, external=False)
    assert not no.is_external()


def test_network_output_rejects_empty_name():
    n = _net()
    x = _tensor_1d(1, thor.DataType.fp16)

    with pytest.raises(ValueError, match=r"name must have non-zero length"):
        thor.layers.NetworkOutput(n, "", x, thor.DataType.fp16)


def test_network_output_rejects_wrong_types_and_arity():
    n = _net()
    x = _tensor_1d(1, thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput()  # missing args

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, "out", x)  # missing data_type

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, "out", x, thor.DataType.fp16, True, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput("not a network", "out", x, thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, 123, x, thor.DataType.fp16)  # name must be str

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, "out", "not a tensor", thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkOutput(n, "out", x, "fp16")  # data_type must be enum

def test_ragged_network_output_exposes_one_logical_output():
    n = thor.Network("test_ragged_network_output")
    tokens = thor.layers.RaggedNetworkInput(
        n,
        "tokens",
        thor.DataType.fp32,
        [2],
        max_total_values=6,
        batch_size=2,
    )

    output = thor.layers.RaggedNetworkOutput(n, "tokens_out", tokens)

    assert isinstance(output, thor.layers.RaggedNetworkOutput)
    assert isinstance(output.get_input(), thor.RaggedTensor)
    assert isinstance(output.get_feature_output(), thor.RaggedTensor)
    assert output.get_feature_output().values.get_dimensions() == [6, 2]
    assert output.get_feature_output().offsets.get_dimensions() == [3]

    import json

    arch = json.loads(n.get_architecture_json())
    assert len(arch["ragged_network_outputs"]) == 1
    logical = arch["ragged_network_outputs"][0]
    assert logical["name"] == "tokens_out"
    assert logical["values_output_name"] == "__thor_ragged_output.tokens_out.values"
    assert logical["offsets_output_name"] == "__thor_ragged_output.tokens_out.offsets"

    component_outputs = [
        layer for layer in arch["layers"] if layer["layer_type"] == "network_output" and layer["name"].startswith("__thor_ragged_output.")
    ]
    assert len(component_outputs) == 2
    assert all(layer["external"] is False for layer in component_outputs)
