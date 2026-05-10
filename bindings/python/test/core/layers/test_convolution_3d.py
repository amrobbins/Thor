import json

import pytest

import thor


def _net(name: str = "test_net_conv3d"):
    return thor.Network(name)


def _cdhw_input(n: thor.Network, c: int, d: int, h: int, w: int, dtype=thor.DataType.fp16, name: str = "input"):
    ni = thor.layers.NetworkInput(n, name, [c, d, h, w], dtype)
    return ni.get_feature_output()


def _architecture(n: thor.Network):
    return json.loads(n.get_architecture_json())


def _layers(n: thor.Network, layer_type: str):
    return [layer for layer in _architecture(n)["layers"] if layer["layer_type"] == layer_type]


def _only_layer_architecture(n: thor.Network, layer_type: str):
    layers = _layers(n, layer_type)
    assert len(layers) == 1
    return layers[0]


def _conv3d_output_shape(cdhw, k: int, fd: int, fh: int, fw: int, sd: int = 1, sh: int = 1, sw: int = 1, pd: int = 0, ph: int = 0, pw: int = 0):
    _, d, h, w = cdhw
    od = (d + 2 * pd - fd) // sd + 1
    oh = (h + 2 * ph - fh) // sh + 1
    ow = (w + 2 * pw - fw) // sw + 1
    return [k, od, oh, ow]


def _floating_convolution_dtypes():
    names = ["fp8_e4m3", "fp8_e5m2", "fp16", "bf16", "fp32"]
    return [(name, getattr(thor.DataType, name)) for name in names if hasattr(thor.DataType, name)]


def _assert_parameter_shape(arch, name: str, shape):
    assert name in arch["parameters"]
    assert arch["parameters"][name]["name"] == name
    assert arch["parameters"][name]["shape"] == shape
    assert arch["parameters"][name]["trainable"] is True
    assert arch["parameters"][name]["training_enabled"] is True


def test_conv3d_constructs_defaults_architecture_parameters_and_output_shape_dtype():
    n = _net()
    x = _cdhw_input(n, 3, 8, 16, 16, thor.DataType.fp16)

    conv = thor.layers.Convolution3d(n, x, num_output_channels=6, filter_depth=3, filter_height=3, filter_width=3)

    assert isinstance(conv, thor.layers.Convolution3d)
    y = conv.get_feature_output()
    assert isinstance(y, thor.Tensor)
    assert y.get_dimensions() == [6, 6, 14, 14]
    assert y.get_data_type() == thor.DataType.fp16

    arch = _only_layer_architecture(n, "convolution_3d")
    assert arch["factory"] == "learning"
    assert arch["layer_type"] == "convolution_3d"
    assert arch["data_layout"] == "NCDHW"
    assert arch["filter_depth"] == 3
    assert arch["filter_height"] == 3
    assert arch["filter_width"] == 3
    assert arch["depth_stride"] == 1
    assert arch["vertical_stride"] == 1
    assert arch["horizontal_stride"] == 1
    assert arch["depth_padding"] == 0
    assert arch["vertical_padding"] == 0
    assert arch["horizontal_padding"] == 0
    assert arch["num_output_channels"] == 6
    assert arch["has_bias"] is True
    assert arch["activation"]["layer_type"] == "soft_plus"
    assert arch["epilogue"] is None
    assert len(arch["inputs"]) == 1
    assert len(arch["outputs"]) == 1
    _assert_parameter_shape(arch, "weights", [6, 3, 3, 3, 3])
    _assert_parameter_shape(arch, "biases", [6])


@pytest.mark.parametrize(
    ("input_shape", "num_output_channels", "filter_shape", "stride", "padding"),
    [
        ((1, 4, 5, 7), 2, (1, 1, 1), (1, 1, 1), (0, 0, 0)),
        ((3, 6, 8, 9), 4, (3, 3, 2), (1, 2, 2), (1, 1, 0)),
        ((2, 7, 11, 13), 5, (4, 5, 3), (2, 3, 2), (2, 1, 1)),
        ((4, 5, 6, 6), 7, (5, 5, 5), (1, 1, 1), (2, 2, 2)),
    ],
)
def test_conv3d_output_shape_matches_convolution_formula(input_shape, num_output_channels, filter_shape, stride, padding):
    n = _net("test_net_conv3d_shape")
    x = _cdhw_input(n, *input_shape, dtype=thor.DataType.fp16)
    fd, fh, fw = filter_shape
    sd, sh, sw = stride
    pd, ph, pw = padding

    conv = thor.layers.Convolution3d(
        n,
        x,
        num_output_channels=num_output_channels,
        filter_depth=fd,
        filter_height=fh,
        filter_width=fw,
        depth_stride=sd,
        vertical_stride=sh,
        horizontal_stride=sw,
        depth_padding=pd,
        vertical_padding=ph,
        horizontal_padding=pw,
        activation=None,
    )

    assert conv.get_feature_output().get_dimensions() == _conv3d_output_shape(input_shape, num_output_channels, fd, fh, fw, sd, sh, sw, pd, ph, pw)
    arch = _only_layer_architecture(n, "convolution_3d")
    assert arch["filter_depth"] == fd
    assert arch["filter_height"] == fh
    assert arch["filter_width"] == fw
    assert arch["depth_stride"] == sd
    assert arch["vertical_stride"] == sh
    assert arch["horizontal_stride"] == sw
    assert arch["depth_padding"] == pd
    assert arch["vertical_padding"] == ph
    assert arch["horizontal_padding"] == pw


@pytest.mark.parametrize(("dtype_name", "dtype"), _floating_convolution_dtypes())
def test_conv3d_accepts_cudnn_frontend_floating_dtype_footprint_at_api_boundary(dtype_name: str, dtype: thor.DataType):
    n = _net(f"test_net_conv3d_dtype_{dtype_name}")
    x = _cdhw_input(n, 3, 4, 8, 8, dtype)

    conv = thor.layers.Convolution3d(
        n,
        x,
        num_output_channels=4,
        filter_depth=3,
        filter_height=3,
        filter_width=3,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        activation=None,
    )

    y = conv.get_feature_output()
    assert y.get_dimensions() == [4, 4, 8, 8]
    assert y.get_data_type() == dtype


@pytest.mark.parametrize("has_bias", [True, False])
def test_conv3d_has_bias_controls_architecture_and_parameter_set(has_bias: bool):
    n = _net(f"test_net_conv3d_bias_{has_bias}")
    x = _cdhw_input(n, 3, 4, 8, 8)

    conv = thor.layers.Convolution3d(
        n,
        x,
        4,
        3,
        3,
        3,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        has_bias=has_bias,
        activation=None,
    )

    assert conv.get_feature_output().get_dimensions() == [4, 4, 8, 8]
    arch = _only_layer_architecture(n, "convolution_3d")
    assert arch["has_bias"] is has_bias
    _assert_parameter_shape(arch, "weights", [4, 3, 3, 3, 3])
    if has_bias:
        _assert_parameter_shape(arch, "biases", [4])
    else:
        assert "biases" not in arch["parameters"]


def test_conv3d_omitted_activation_defaults_to_softplus_and_none_disables_activation():
    default_net = _net("test_net_conv3d_default_activation")
    default_x = _cdhw_input(default_net, 3, 4, 8, 8, thor.DataType.fp16)
    thor.layers.Convolution3d(default_net, default_x, 4, 3, 3, 3)
    default_arch = _only_layer_architecture(default_net, "convolution_3d")
    assert default_arch["activation"]["layer_type"] == "soft_plus"

    linear_net = thor.Network("test_net_conv3d_linear")
    linear_x = _cdhw_input(linear_net, 3, 4, 8, 8, thor.DataType.fp16)
    thor.layers.Convolution3d(linear_net, linear_x, 4, 3, 3, 3, activation=None)
    linear_arch = _only_layer_architecture(linear_net, "convolution_3d")
    assert linear_arch["activation"] is None


def test_conv3d_accepts_stitched_activation_without_extra_api_layer():
    n = _net("test_net_conv3d_tanh")
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
    arch = _only_layer_architecture(n, "convolution_3d")
    assert arch["activation"]["layer_type"] == "tanh"
    assert len(_layers(n, "tanh")) == 0


def test_conv3d_accepts_custom_initializers_and_serializes_parameter_initializers():
    n = _net("test_net_conv3d_initializers")
    x = _cdhw_input(n, 3, 4, 8, 8)
    weights_initializer = thor.initializers.Glorot(thor.initializers.Glorot.Mode.UNIFORM)
    biases_initializer = thor.initializers.Glorot(thor.initializers.Glorot.Mode.NORMAL)

    thor.layers.Convolution3d(
        n,
        x,
        4,
        3,
        3,
        3,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        weights_initializer=weights_initializer,
        biases_initializer=biases_initializer,
        activation=None,
    )

    arch = _only_layer_architecture(n, "convolution_3d")
    assert arch["parameters"]["weights"]["initializer"] is not None
    assert arch["parameters"]["biases"]["initializer"] is not None


def test_conv3d_accepts_epilogue_expression_and_serializes_it():
    n = thor.Network("test_net_conv3d_epilogue")
    x = _cdhw_input(n, 3, 4, 8, 8, thor.DataType.fp16)

    epilogue_input = thor.layers.Convolution3d.epilogue_input(
        output_dtype=thor.DataType.fp32,
        compute_dtype=thor.DataType.fp32,
    )
    epilogue = epilogue_input * 2.0 + 1.0

    conv = thor.layers.Convolution3d(
        n,
        x,
        num_output_channels=4,
        filter_depth=3,
        filter_height=3,
        filter_width=3,
        depth_padding=1,
        vertical_padding=1,
        horizontal_padding=1,
        activation=None,
        epilogue=epilogue,
    )

    assert conv.get_feature_output().get_dimensions() == [4, 4, 8, 8]
    arch = _only_layer_architecture(n, "convolution_3d")
    assert arch["activation"] is None
    assert arch["epilogue"] is not None
    assert arch["epilogue"]["expected_input_names"] == ["__convolution_3d_epilogue_input"]
    assert arch["epilogue"]["expected_output_names"] == ["__convolution_3d_epilogue_output"]


@pytest.mark.parametrize(("output_dtype", "compute_dtype"), [(None, None), (thor.DataType.fp16, thor.DataType.fp16), (thor.DataType.fp32, thor.DataType.fp32)])
def test_conv3d_epilogue_input_accepts_default_and_explicit_dtypes(output_dtype, compute_dtype):
    kwargs = {}
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    if compute_dtype is not None:
        kwargs["compute_dtype"] = compute_dtype

    epilogue_input = thor.layers.Convolution3d.epilogue_input(**kwargs)
    assert epilogue_input is not None


def test_conv3d_rejects_feature_input_wrong_rank():
    n = _net("test_net_conv3d_bad_rank")
    x = thor.Tensor([3, 8, 8], thor.DataType.fp16)

    with pytest.raises(ValueError, match=r"feature_input must be a 4D CDHW tensor"):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3)


def test_conv3d_rejects_zero_or_invalid_params():
    n = _net("test_net_conv3d_bad_params")
    x = _cdhw_input(n, 3, 6, 8, 8)

    with pytest.raises(ValueError, match=r"num_output_channels must be > 0"):
        thor.layers.Convolution3d(n, x, 0, 3, 3, 3)

    with pytest.raises(ValueError, match=r"filter_depth, filter_height, and filter_width"):
        thor.layers.Convolution3d(n, x, 4, 0, 3, 3)

    with pytest.raises(ValueError, match=r"filter_depth, filter_height, and filter_width"):
        thor.layers.Convolution3d(n, x, 4, 3, 0, 3)

    with pytest.raises(ValueError, match=r"filter_depth, filter_height, and filter_width"):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 0)

    with pytest.raises(ValueError, match=r"depth_stride, vertical_stride, and horizontal_stride"):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3, 0, 1, 1)

    with pytest.raises(ValueError, match=r"depth_stride, vertical_stride, and horizontal_stride"):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3, 1, 0, 1)

    with pytest.raises(ValueError, match=r"depth_stride, vertical_stride, and horizontal_stride"):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3, 1, 1, 0)


def test_conv3d_rejects_filter_larger_than_padded_input_and_accepts_when_padding_makes_it_fit():
    n = _net("test_net_conv3d_filter_fit")
    x = _cdhw_input(n, 3, 4, 4, 4)

    with pytest.raises(ValueError, match=r"filter is larger than padded input"):
        thor.layers.Convolution3d(n, x, 8, 5, 3, 3)

    with pytest.raises(ValueError, match=r"filter is larger than padded input"):
        thor.layers.Convolution3d(n, x, 8, 3, 5, 3)

    with pytest.raises(ValueError, match=r"filter is larger than padded input"):
        thor.layers.Convolution3d(n, x, 8, 3, 3, 5)

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
        activation=None,
    )
    assert isinstance(conv, thor.layers.Convolution3d)
    assert conv.get_feature_output().get_dimensions() == [8, 2, 2, 2]


def test_conv3d_rejects_wrong_types_and_arity():
    n = _net("test_net_conv3d_wrong_types")
    x = _cdhw_input(n, 3, 4, 8, 8)

    with pytest.raises(TypeError):
        thor.layers.Convolution3d()

    with pytest.raises(TypeError):
        thor.layers.Convolution3d(n, x)

    with pytest.raises(TypeError):
        thor.layers.Convolution3d("not a network", x, 4, 3, 3, 3)

    with pytest.raises(TypeError):
        thor.layers.Convolution3d(n, "not a tensor", 4, 3, 3, 3)

    with pytest.raises(TypeError):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3, has_bias="yes")

    with pytest.raises(TypeError):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3, activation=123)

    with pytest.raises(TypeError):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3, weights_initializer=123)


def test_conv3d_rejects_wrong_epilogue_type():
    n = _net("test_net_conv3d_bad_epilogue")
    x = _cdhw_input(n, 3, 4, 8, 8)

    with pytest.raises(TypeError, match="epilogue must be"):
        thor.layers.Convolution3d(n, x, 4, 3, 3, 3, epilogue=123)
