import pytest

import thor


def _net():
    return thor.Network("test_net_conv3d")


def _cdhw_input(n: thor.Network, c: int, d: int, h: int, w: int, dtype=thor.DataType.fp16):
    ni = thor.layers.NetworkInput(n, "input", [c, d, h, w], dtype)
    return ni.get_feature_output()


def test_conv3d_constructs_and_output_shape_dtype_round_trip():
    n = _net()
    x = _cdhw_input(n, 3, 8, 16, 16, thor.DataType.fp16)

    conv = thor.layers.Convolution3d(
        n,
        x,
        num_output_channels=6,
        filter_depth=3,
        filter_height=3,
        filter_width=3,
        depth_stride=2,
        vertical_stride=2,
        horizontal_stride=2,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=True,
        activation=None,
    )

    y = conv.get_feature_output()
    assert y.get_data_type() == thor.DataType.fp16
    assert y.get_dimensions() == [6, 4, 8, 8]


def test_conv3d_accepts_stitched_activation_without_extra_api_layer():
    n = _net()
    x = _cdhw_input(n, 3, 6, 8, 8, thor.DataType.fp16)

    conv = thor.layers.Convolution3d(
        n,
        x,
        num_output_channels=4,
        filter_depth=3,
        filter_height=3,
        filter_width=3,
        depth_stride=1,
        vertical_stride=1,
        horizontal_stride=1,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=True,
        activation=thor.activations.Tanh(),
    )

    y = conv.get_feature_output()
    assert y.get_dimensions() == [4, 6, 8, 8]
    assert y.get_data_type() == thor.DataType.fp16


def test_conv3d_rejects_feature_input_wrong_rank():
    n = _net()
    x = thor.Tensor([3, 8, 8], thor.DataType.fp16)

    with pytest.raises(ValueError, match=r"feature_input must be a 4D CDHW tensor"):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3)


def test_conv3d_rejects_zero_or_invalid_params():
    n = _net()
    x = _cdhw_input(n, 3, 6, 8, 8)

    with pytest.raises(ValueError, match=r"num_output_channels must be > 0"):
        thor.layers.Convolution3d(n, x, 0, 3, 3, 3)

    with pytest.raises(ValueError, match=r"filter_depth, filter_height, and filter_width"):
        thor.layers.Convolution3d(n, x, 4, 0, 3, 3)

    with pytest.raises(ValueError, match=r"depth_stride, vertical_stride, and horizontal_stride"):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3, 0, 1, 1)


def test_conv3d_rejects_filter_larger_than_padded_input():
    n = _net()
    x = _cdhw_input(n, 3, 4, 4, 4)

    with pytest.raises(ValueError, match=r"filter is larger than padded input"):
        thor.layers.Convolution3d(n, x, 8, 5, 3, 3)

    conv = thor.layers.Convolution3d(
        n,
        x,
        num_output_channels=8,
        filter_depth=5,
        filter_height=5,
        filter_width=5,
        depth_stride=1,
        vertical_stride=1,
        horizontal_stride=1,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
    )
    assert isinstance(conv, thor.layers.Convolution3d)
