import pytest
import thor


def _net():
    return thor.Network("test_net_network_input")


def test_network_input_constructs_and_returns_feature_output():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16)

    assert ni is not None
    assert isinstance(ni, thor.layers.NetworkInput)
    assert ni.is_external()

    out = ni.get_feature_output()
    assert out is not None
    assert isinstance(out, thor.Tensor)


def test_network_input_rejects_empty_name():
    n = _net()
    with pytest.raises(ValueError, match=r"name must have non-zero length"):
        thor.layers.NetworkInput(n, "", [16], thor.DataType.fp16)


def test_network_input_rejects_empty_dimensions():
    n = _net()
    with pytest.raises(ValueError, match=r"dimensions must be non-zero"):
        thor.layers.NetworkInput(n, "input", [], thor.DataType.fp16)


def test_network_input_external_flag():
    n = _net()
    ni = thor.layers.NetworkInput(n, "internal", [16], thor.DataType.fp16, external=False)
    assert not ni.is_external()


def test_network_input_accepts_multi_dimensional_shape():
    n = _net()
    ni = thor.layers.NetworkInput(n, "img", [3, 224, 224], thor.DataType.fp16)
    assert isinstance(ni, thor.layers.NetworkInput)
    out = ni.get_feature_output()
    assert isinstance(out, thor.Tensor)


def test_network_input_rejects_wrong_types_and_arity():
    n = _net()

    with pytest.raises(TypeError):
        thor.layers.NetworkInput()  # missing args

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", [16])  # missing data_type

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", [16], thor.DataType.fp16, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.NetworkInput("not a network", "input", [16], thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, 123, [16], thor.DataType.fp16)  # name must be str

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", "not a list", thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", [16], "fp16")  # data_type must be enum

    with pytest.raises(TypeError):
        thor.layers.NetworkInput(n, "input", [16.5], thor.DataType.fp16)


def test_ragged_network_input_returns_one_logical_ragged_tensor():
    n = _net()
    labels = thor.layers.RaggedNetworkInput(
        n,
        "labels",
        thor.DataType.int32,
        [],
        max_total_values=64,
        batch_size=8,
        offsets_data_type=thor.DataType.uint64,
        max_values_per_row=12,
    )

    assert isinstance(labels, thor.RaggedTensor)
    assert labels.values.get_dimensions() == [64]
    assert labels.offsets.get_dimensions() == [9]
    assert labels.values_data_type == thor.DataType.int32
    assert labels.offsets_data_type == thor.DataType.uint64
    assert labels.batch_size == 8
    assert labels.max_total_values == 64
    assert labels.max_values_per_row == 12


def test_ragged_network_input_rejects_invalid_shape_contract():
    n = _net()
    with pytest.raises(ValueError, match="max_total_values"):
        thor.layers.RaggedNetworkInput(
            n,
            "labels",
            thor.DataType.int32,
            [],
            max_total_values=0,
            batch_size=8,
        )

    with pytest.raises(ValueError, match="trailing_dimensions"):
        thor.layers.RaggedNetworkInput(
            n,
            "labels2",
            thor.DataType.fp16,
            [4, 0],
            max_total_values=64,
            batch_size=8,
        )

    with pytest.raises((ValueError, RuntimeError)):
        thor.layers.RaggedNetworkInput(
            n,
            "labels3",
            thor.DataType.fp16,
            [4],
            max_total_values=64,
            batch_size=8,
            max_values_per_row=65,
        )
