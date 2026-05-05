import pytest

import thor


def _net():
    return thor.Network("test_net_conv2d")


def _chw_input(n: thor.Network, c: int, h: int, w: int, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", [c, h, w], dtype)
    return ni.get_feature_output()


def test_conv2d_constructs_and_output_shape_dtype_round_trip():
    n = _net()
    x = _chw_input(n, 3, 32, 32, thor.DataType.fp16)

    conv = thor.layers.Convolution2d(
        n,
        x,
        num_output_channels=16,
        filter_height=3,
        filter_width=3,
        vertical_stride=2,
        horizontal_stride=2,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=True,
        activation=None,
    )

    assert conv is not None
    assert isinstance(conv, thor.layers.Convolution2d)

    y = conv.get_feature_output()
    assert y is not None
    assert isinstance(y, thor.Tensor)

    # dtype should match input at API level (adjust if your builder forces fp32)
    assert y.get_data_type() == x.get_data_type()

    # Output dims for CHW:
    # effH=32+2*1=34, outH=(34-3)//2 + 1 = 16
    # effW=32+2*1=34, outW=(34-3)//2 + 1 = 16
    assert y.get_dimensions() == [16, 16, 16]


def test_conv2d_rejects_feature_input_wrong_rank():
    n = _net()
    x = thor.Tensor([32, 32], thor.DataType.fp16)  # rank 2

    with pytest.raises(ValueError, match=r"feature_input must be a 3D CHW tensor"):
        thor.layers.Convolution2d(n, x, 16, 3, 3)


def test_conv2d_rejects_zero_or_invalid_params():
    n = _net()
    x = _chw_input(n, 3, 8, 8)

    with pytest.raises(ValueError, match=r"num_output_channels must be > 0"):
        thor.layers.Convolution2d(n, x, 0, 3, 3)

    with pytest.raises(ValueError, match=r"filter_height and filter_width must be >= 1"):
        thor.layers.Convolution2d(n, x, 8, 0, 3)

    with pytest.raises(ValueError, match=r"filter_height and filter_width must be >= 1"):
        thor.layers.Convolution2d(n, x, 8, 3, 0)

    with pytest.raises(ValueError, match=r"vertical_stride and horizontal_stride must be >= 1"):
        thor.layers.Convolution2d(n, x, 8, 3, 3, 0, 1)

    with pytest.raises(ValueError, match=r"vertical_stride and horizontal_stride must be >= 1"):
        thor.layers.Convolution2d(n, x, 8, 3, 3, 1, 0)


def test_conv2d_rejects_filter_larger_than_padded_input():
    n = _net()
    x = _chw_input(n, 3, 4, 4)

    with pytest.raises(ValueError, match=r"filter_height .* larger than padded input height"):
        thor.layers.Convolution2d(n, x, 8, 5, 3)

    with pytest.raises(ValueError, match=r"filter_width .* larger than padded input width"):
        thor.layers.Convolution2d(n, x, 8, 3, 5)

    # Padding can make it feasible
    conv = thor.layers.Convolution2d(
        n,
        x,
        num_output_channels=8,
        filter_height=5,
        filter_width=5,
        vertical_stride=1,
        horizontal_stride=1,
        vertical_padding=1,
        horizontal_padding=1,
    )
    assert isinstance(conv, thor.layers.Convolution2d)


def test_conv2d_rejects_wrong_types_and_arity():
    n = _net()
    x = _chw_input(n, 3, 8, 8)

    with pytest.raises(TypeError):
        thor.layers.Convolution2d()  # missing args

    with pytest.raises(TypeError):
        thor.layers.Convolution2d(n, x)  # missing required ints

    with pytest.raises(TypeError):
        thor.layers.Convolution2d("not a network", x, 8, 3, 3)

    with pytest.raises(TypeError):
        thor.layers.Convolution2d(n, "not a tensor", 8, 3, 3)


def test_conv2d_accepts_stitched_activation_without_extra_api_layer():
    n = _net()
    x = _chw_input(n, 3, 8, 8, thor.DataType.fp16)

    conv = thor.layers.Convolution2d(
        n,
        x,
        num_output_channels=4,
        filter_height=3,
        filter_width=3,
        vertical_stride=1,
        horizontal_stride=1,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=True,
        activation=thor.activations.Tanh(),
    )

    y = conv.get_feature_output()
    assert y.get_dimensions() == [4, 8, 8]
    assert y.get_data_type() == thor.DataType.fp16
