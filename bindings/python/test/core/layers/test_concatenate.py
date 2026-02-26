import pytest
import thor


def _net():
    return thor.Network("test_net_concatenate")


def _tensor(dims, dtype=thor.DataType.fp32):
    return thor.Tensor(list(dims), dtype)


def test_concatenate_constructs_valid_axis0():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([4, 3], thor.DataType.fp32)  # axis0 can differ
    layer = thor.layers.Concatenate(n, [t1, t2], 0)
    assert layer is not None
    assert isinstance(layer, thor.layers.Concatenate)


def test_concatenate_constructs_valid_axis1():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([2, 5], thor.DataType.fp32)  # axis1 can differ
    layer = thor.layers.Concatenate(n, [t1, t2], 1)
    assert isinstance(layer, thor.layers.Concatenate)


def test_concatenate_rejects_empty_list():
    n = _net()
    with pytest.raises(ValueError, match=r"feature_inputs must be a non-empty"):
        thor.layers.Concatenate(n, [], 0)


def test_concatenate_rejects_axis_out_of_range():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([2, 3], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"concatenation_axis .* out of range"):
        thor.layers.Concatenate(n, [t1, t2], 2)  # rank=2, valid axes are 0/1


def test_concatenate_rejects_rank_mismatch():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([2, 3, 4], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"same number of dimensions"):
        thor.layers.Concatenate(n, [t1, t2], 0)


def test_concatenate_rejects_dtype_mismatch():
    n = _net()
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([2, 3], thor.DataType.fp16)

    with pytest.raises(ValueError, match=r"same data type"):
        thor.layers.Concatenate(n, [t1, t2], 0)


def test_concatenate_rejects_dim_mismatch_non_axis():
    n = _net()
    # axis=0 => dim1 must match, but we make it mismatch
    t1 = _tensor([2, 3], thor.DataType.fp32)
    t2 = _tensor([4, 5], thor.DataType.fp32)

    with pytest.raises(ValueError, match=r"dimensions must match except along concatenation_axis"):
        thor.layers.Concatenate(n, [t1, t2], 0)
