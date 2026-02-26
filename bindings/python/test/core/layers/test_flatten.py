import pytest
import thor


def _net():
    return thor.Network("test_net_flatten")


def _tensor(dims, dtype=thor.DataType.fp32):
    return thor.Tensor(list(dims), dtype)


def test_flatten_constructs_valid_num_output_dimensions_1():
    n = _net()
    x = _tensor([2, 3, 4], thor.DataType.fp32)

    layer = thor.layers.Flatten(n, x, 1)
    assert layer is not None
    assert isinstance(layer, thor.layers.Flatten)


def test_flatten_constructs_valid_num_output_dimensions_rank_minus_1():
    n = _net()
    x = _tensor([2, 3, 4], thor.DataType.fp32)  # rank=3

    layer = thor.layers.Flatten(n, x, 2)  # rank-1
    assert isinstance(layer, thor.layers.Flatten)


def test_flatten_rejects_num_output_dimensions_zero():
    n = _net()
    x = _tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"num_output_dimensions must be >= 1"):
        thor.layers.Flatten(n, x, 0)


def test_flatten_rejects_num_output_dimensions_out_of_range_equal_rank():
    n = _net()
    x = _tensor([2, 3, 4], thor.DataType.fp32)  # rank=3

    with pytest.raises(ValueError, match=r"num_output_dimensions must be < rank"):
        thor.layers.Flatten(n, x, 3)


def test_flatten_rejects_num_output_dimensions_out_of_range_greater_than_rank():
    n = _net()
    x = _tensor([2, 3, 4], thor.DataType.fp32)  # rank=3

    with pytest.raises(ValueError, match=r"num_output_dimensions must be < rank"):
        thor.layers.Flatten(n, x, 4)


def test_flatten_rejects_wrong_types_and_arity():
    n = _net()
    x = _tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(TypeError):
        thor.layers.Flatten()  # missing args

    with pytest.raises(TypeError):
        thor.layers.Flatten(n, x)  # missing num_output_dimensions

    with pytest.raises(TypeError):
        thor.layers.Flatten(n, x, 1, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.Flatten("not a network", x, 1)

    with pytest.raises(TypeError):
        thor.layers.Flatten(n, "not a tensor", 1)
