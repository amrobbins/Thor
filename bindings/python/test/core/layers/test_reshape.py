import pytest
import thor


def _net():
    return thor.Network("test_net_reshape")


def _input_tensor(n: thor.Network, dims, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", list(dims), dtype)
    return ni.get_feature_output()


def test_reshape_constructs_and_output_matches_new_dimensions_and_dtype():
    n = _net()
    x = _input_tensor(n, [2, 3, 4], thor.DataType.fp16)  # 24 elems

    r = thor.layers.Reshape(n, x, [6, 4])  # 24 elems
    assert r is not None
    assert isinstance(r, thor.layers.Reshape)

    y = r.get_feature_output()
    assert y is not None
    assert isinstance(y, thor.Tensor)

    assert y.get_dimensions() == [6, 4]
    assert y.get_data_type() == x.get_data_type()


def test_reshape_rejects_element_count_mismatch():
    n = _net()
    x = _input_tensor(n, [2, 3, 4], thor.DataType.fp16)  # 24 elems

    with pytest.raises(ValueError, match=r"number of elements must match"):
        thor.layers.Reshape(n, x, [5, 5])  # 25 elems


def test_reshape_rejects_empty_new_dimensions():
    n = _net()
    x = _input_tensor(n, [2, 3, 4], thor.DataType.fp16)

    with pytest.raises(ValueError, match=r"new_dimensions must be non-empty"):
        thor.layers.Reshape(n, x, [])


def test_reshape_rejects_zero_in_new_dimensions():
    n = _net()
    x = _input_tensor(n, [2, 3, 4], thor.DataType.fp16)

    with pytest.raises(ValueError, match=r"new_dimensions must all be > 0"):
        thor.layers.Reshape(n, x, [6, 0, 4])


def test_reshape_rejects_wrong_types_and_arity():
    n = _net()
    x = _input_tensor(n, [2, 3, 4], thor.DataType.fp16)

    with pytest.raises(TypeError):
        thor.layers.Reshape()  # missing args

    with pytest.raises(TypeError):
        thor.layers.Reshape(n, x)  # missing new_dimensions

    with pytest.raises(TypeError):
        thor.layers.Reshape(n, x, [6, 4], 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.Reshape("not a network", x, [6, 4])

    with pytest.raises(TypeError):
        thor.layers.Reshape(n, "not a tensor", [6, 4])

    with pytest.raises(TypeError):
        thor.layers.Reshape(n, x, "not a list")
