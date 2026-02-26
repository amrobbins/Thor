import pytest
import thor


def _net():
    return thor.Network("test_net_dropout")


def _tensor_1d(size: int = 1, dtype=thor.DataType.fp32):
    # API tensor: dims + dtype
    return thor.Tensor([size], dtype)


def test_dropout_constructs_and_reports_drop_proportion():
    n = _net()
    x = _tensor_1d(1, thor.DataType.fp32)

    layer = thor.layers.DropOut(n, x, 0.25)
    assert layer is not None
    assert isinstance(layer, thor.layers.DropOut)

    # Should return exactly what we passed (float compare)
    assert layer.get_drop_proportion() == pytest.approx(0.25)


def test_dropout_allows_endpoints():
    n = _net()
    x = _tensor_1d()

    l0 = thor.layers.DropOut(n, x, 0.0)
    assert l0.get_drop_proportion() == pytest.approx(0.0)

    l1 = thor.layers.DropOut(n, x, 1.0)
    assert l1.get_drop_proportion() == pytest.approx(1.0)


def test_dropout_rejects_out_of_range_values():
    n = _net()
    x = _tensor_1d()

    with pytest.raises(ValueError, match=r"0 <= drop_proportion <= 1"):
        thor.layers.DropOut(n, x, -0.01)

    with pytest.raises(ValueError, match=r"0 <= drop_proportion <= 1"):
        thor.layers.DropOut(n, x, 1.01)


def test_dropout_rejects_wrong_types_and_arity():
    n = _net()
    x = _tensor_1d()

    with pytest.raises(TypeError):
        thor.layers.DropOut()  # missing args

    with pytest.raises(TypeError):
        thor.layers.DropOut(n, x)  # missing drop_proportion

    with pytest.raises(TypeError):
        thor.layers.DropOut(n, x, 0.5, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.DropOut("not a network", x, 0.5)

    with pytest.raises(TypeError):
        thor.layers.DropOut(n, "not a tensor", 0.5)

    with pytest.raises(TypeError):
        thor.layers.DropOut(n, x, "0.5")
