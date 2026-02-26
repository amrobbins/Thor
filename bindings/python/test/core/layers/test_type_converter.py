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
