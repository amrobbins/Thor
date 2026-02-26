# test/test_layers_pooling.py
import pytest
import thor


def _net():
    return thor.Network("test_net_pooling")


def _chw(c, h, w, dtype=thor.DataType.fp16):
    # API tensor: NCHW without batch => [C, H, W]
    return thor.Tensor([c, h, w], dtype)


def test_pooling_constructs_valid_max():
    n = _net()
    x = _chw(8, 32, 32, thor.DataType.fp16)

    p = thor.layers.Pooling(
        n,
        x,
        thor.layers.Pooling.Type.max,
        window_height=2,
        window_width=2,
        vertical_stride=2,
        horizontal_stride=2,
        vertical_padding=0,
        horizontal_padding=0,
    )
    assert p is not None
    assert isinstance(p, thor.layers.Pooling)
    assert p.get_pooling_type() == thor.layers.Pooling.Type.max
    assert p.get_window_height() == 2
    assert p.get_window_width() == 2


def test_pooling_constructs_valid_average_with_padding():
    n = _net()
    x = _chw(3, 5, 5, thor.DataType.fp16)

    p = thor.layers.Pooling(
        n,
        x,
        thor.layers.Pooling.Type.average,
        window_height=3,
        window_width=3,
        vertical_stride=1,
        horizontal_stride=1,
        vertical_padding=1,
        horizontal_padding=1,
    )
    assert isinstance(p, thor.layers.Pooling)
    assert p.get_pooling_type() == thor.layers.Pooling.Type.average
    assert p.get_vertical_padding() == 1
    assert p.get_horizontal_padding() == 1


def test_pooling_rejects_rank_not_3():
    n = _net()
    x = thor.Tensor([32, 32], thor.DataType.fp16)  # rank 2

    with pytest.raises(ValueError, match=r"must be a 3D NCHW tensor"):
        thor.layers.Pooling(n, x, thor.layers.Pooling.Type.max, 2, 2)


def test_pooling_rejects_zero_window_or_stride():
    n = _net()
    x = _chw(8, 32, 32)

    with pytest.raises(ValueError, match=r"window_height and window_width must be >= 1"):
        thor.layers.Pooling(n, x, thor.layers.Pooling.Type.max, 0, 2)

    with pytest.raises(ValueError, match=r"window_height and window_width must be >= 1"):
        thor.layers.Pooling(n, x, thor.layers.Pooling.Type.max, 2, 0)

    with pytest.raises(ValueError, match=r"vertical_stride and horizontal_stride must be >= 1"):
        thor.layers.Pooling(n, x, thor.layers.Pooling.Type.max, 2, 2, 0, 1)

    with pytest.raises(ValueError, match=r"vertical_stride and horizontal_stride must be >= 1"):
        thor.layers.Pooling(n, x, thor.layers.Pooling.Type.max, 2, 2, 1, 0)


def test_pooling_rejects_window_larger_than_padded_input():
    n = _net()
    x = _chw(3, 4, 4)

    # No padding: window too big
    with pytest.raises(ValueError, match=r"window_height .* larger than padded input height"):
        thor.layers.Pooling(n, x, thor.layers.Pooling.Type.max, 5, 2)

    with pytest.raises(ValueError, match=r"window_width .* larger than padded input width"):
        thor.layers.Pooling(n, x, thor.layers.Pooling.Type.max, 2, 5)

    # Padding makes it feasible (sanity)
    p = thor.layers.Pooling(
        n,
        x,
        thor.layers.Pooling.Type.max,
        window_height=5,
        window_width=5,
        vertical_stride=1,
        horizontal_stride=1,
        vertical_padding=1,
        horizontal_padding=1)
    assert isinstance(p, thor.layers.Pooling)


def test_pooling_rejects_wrong_types_and_arity():
    n = _net()
    x = _chw(3, 4, 4)

    with pytest.raises(TypeError):
        thor.layers.Pooling()  # missing args

    with pytest.raises(TypeError):
        thor.layers.Pooling(n, x, thor.layers.Pooling.Type.max, 2)  # missing window_width

    with pytest.raises(TypeError):
        thor.layers.Pooling(n, x, "max", 2, 2)  # type must be enum

    with pytest.raises(TypeError):
        thor.layers.Pooling("not a network", x, thor.layers.Pooling.Type.max, 2, 2)

    with pytest.raises(TypeError):
        thor.layers.Pooling(n, "not a tensor", thor.layers.Pooling.Type.max, 2, 2)


def test_pooling_getters_round_trip_max_no_padding():
    n = _net()
    x = _chw(8, 32, 32, thor.DataType.fp16)

    p = thor.layers.Pooling(
        n,
        x,
        thor.layers.Pooling.Type.max,
        window_height=2,
        window_width=3,
        vertical_stride=2,
        horizontal_stride=4,
        vertical_padding=0,
        horizontal_padding=0,
    )

    # Values
    assert p.get_pooling_type() == thor.layers.Pooling.Type.max
    assert p.get_window_height() == 2
    assert p.get_window_width() == 3
    assert p.get_vertical_stride() == 2
    assert p.get_horizontal_stride() == 4
    assert p.get_vertical_padding() == 0
    assert p.get_horizontal_padding() == 0

    # Types (don’t overfit: nanobind will usually return Python int)
    assert isinstance(p.get_window_height(), int)
    assert isinstance(p.get_window_width(), int)
    assert isinstance(p.get_vertical_stride(), int)
    assert isinstance(p.get_horizontal_stride(), int)
    assert isinstance(p.get_vertical_padding(), int)
    assert isinstance(p.get_horizontal_padding(), int)

    # Output dimensions sanity: should be a sequence of ints.
    out_dims = p.get_output_dimensions()
    assert isinstance(out_dims, (list, tuple))
    assert all(isinstance(d, int) for d in out_dims)

    # For effH=32, windowH=2, stride=2 => (32-2)//2 + 1 = 16
    # For effW=32, windowW=3, stride=4 => (32-3)//4 + 1 = 8
    assert out_dims == [8, 16, 8]


def test_pooling_getters_round_trip_average_with_padding():
    n = _net()
    x = _chw(3, 5, 5, thor.DataType.fp16)

    p = thor.layers.Pooling(
        n,
        x,
        thor.layers.Pooling.Type.average,
        window_height=3,
        window_width=3,
        vertical_stride=1,
        horizontal_stride=1,
        vertical_padding=1,
        horizontal_padding=2,
    )

    assert p.get_pooling_type() == thor.layers.Pooling.Type.average
    assert p.get_window_height() == 3
    assert p.get_window_width() == 3
    assert p.get_vertical_stride() == 1
    assert p.get_horizontal_stride() == 1
    assert p.get_vertical_padding() == 1
    assert p.get_horizontal_padding() == 2

    out_dims = p.get_output_dimensions()
    assert isinstance(out_dims, (list, tuple))
    assert all(isinstance(d, int) for d in out_dims)

    # effH = 5 + 2*1 = 7 => (7-3)//1 + 1 = 5
    # effW = 5 + 2*2 = 9 => (9-3)//1 + 1 = 7
    assert out_dims == [3, 5, 7]
