import json

import pytest

import thor


def _net(name: str = "test_net_conv1d"):
    return thor.Network(name)


def _cw_input(n: thor.Network, c: int, w: int, dtype=thor.DataType.fp16, name: str = "input"):
    ni = thor.layers.NetworkInput(n, name, [c, w], dtype)
    return ni.get_feature_output()


def _only_arch(n: thor.Network):
    arch = json.loads(n.get_architecture_json())
    layers = [layer for layer in arch["layers"] if layer["layer_type"] == "convolution_1d"]
    assert len(layers) == 1
    return layers[0]


def _output_width(input_width, filter_width, stride, dilation, left, right):
    effective = dilation * (filter_width - 1) + 1
    return (input_width + left + right - effective) // stride + 1

def test_conv1d_defaults_shape_parameters_and_architecture():
    n = _net()
    x = _cw_input(n, 3, 16)
    conv = thor.layers.Convolution1d(n, x, num_output_channels=5, filter_width=3)

    assert conv.get_feature_output().get_dimensions() == [5, 14]
    arch = _only_arch(n)
    assert arch["version"] == "2.0.0"
    assert arch["data_layout"] == "NCW"
    assert arch["filter_width"] == 3
    assert arch["stride"] == 1
    assert arch["dilation"] == 1
    assert arch["padding_mode"] == "valid"
    assert arch["padding_left"] == 0
    assert arch["padding_right"] == 0
    assert arch["num_output_channels"] == 5
    assert arch["groups"] == 1
    assert arch["has_bias"] is True
    assert arch["activation"]["layer_type"] == "gelu"
    assert arch["parameters"]["weights"]["shape"] == [5, 3, 3]
    assert arch["parameters"]["biases"]["shape"] == [5]


@pytest.mark.parametrize(
    ("padding", "input_width", "filter_width", "stride", "dilation", "expected_mode", "expected_padding"),
    [
        ("valid", 11, 3, 2, 1, "valid", (0, 0)),
        ((2, 1), 11, 3, 2, 2, "explicit", (2, 1)),
        ("same", 8, 4, 2, 2, "same_upper", (2, 3)),
        ("causal", 8, 4, 2, 2, "causal", (6, 0)),
    ],
)
def test_conv1d_padding_modes_resolve_geometry(
        padding, input_width, filter_width, stride, dilation, expected_mode, expected_padding):
    n = _net(f"test_conv1d_{expected_mode}")
    x = _cw_input(n, 2, input_width)
    conv = thor.layers.Convolution1d(
        n,
        x,
        num_output_channels=4,
        filter_width=filter_width,
        stride=stride,
        dilation=dilation,
        padding=padding,
        activation=None,
    )
    arch = _only_arch(n)
    assert arch["padding_mode"] == expected_mode
    assert (arch["padding_left"], arch["padding_right"]) == expected_padding
    expected_width = _output_width(input_width, filter_width, stride, dilation, *expected_padding)
    assert conv.get_feature_output().get_dimensions() == [4, expected_width]

def test_conv1d_causal_stride_one_preserves_logical_width():
    n = _net("test_conv1d_causal_width")
    x = _cw_input(n, 7, 23)
    conv = thor.layers.Convolution1d(
        n, x, num_output_channels=9, filter_width=5, stride=1, dilation=3, padding="causal", activation=None)
    assert conv.get_feature_output().get_dimensions() == [9, 23]
    arch = _only_arch(n)
    assert (arch["padding_left"], arch["padding_right"]) == (12, 0)


@pytest.mark.parametrize("padding", ["bogus", "explicit", (1,), (1, 2, 3), -1, (0, -1)])
def test_conv1d_rejects_invalid_padding(padding):
    n = _net("test_conv1d_bad_padding")
    x = _cw_input(n, 2, 8)
    with pytest.raises((TypeError, ValueError)):
        thor.layers.Convolution1d(n, x, 3, 3, padding=padding)


@pytest.mark.parametrize("kwargs", [{"filter_width": 0}, {"stride": 0}, {"dilation": 0}, {"num_output_channels": 0}])
def test_conv1d_rejects_non_positive_geometry(kwargs):
    n = _net("test_conv1d_bad_geometry")
    x = _cw_input(n, 2, 8)
    args = dict(num_output_channels=3, filter_width=3)
    args.update(kwargs)
    with pytest.raises(ValueError):
        thor.layers.Convolution1d(n, x, **args)

def test_conv1d_rejects_non_cw_input():
    n = _net("test_conv1d_bad_rank")
    x = thor.layers.NetworkInput(n, "input", [2, 3, 8], thor.DataType.fp16).get_feature_output()
    with pytest.raises(ValueError, match="2D CW"):
        thor.layers.Convolution1d(n, x, 4, 3)


@pytest.mark.cuda
def test_conv1d_save_load_preserves_causal_semantics(tmp_path):
    name = "test_conv1d_save_load"
    n = thor.Network(name)
    x = _cw_input(n, 4, 17)
    conv = thor.layers.Convolution1d(
        n,
        x,
        num_output_channels=6,
        filter_width=4,
        stride=2,
        dilation=3,
        padding="causal",
        groups=2,
        has_bias=False,
        activation=None,
    )
    thor.layers.NetworkOutput(n, "output", conv.get_feature_output(), thor.DataType.fp16)

    save_dir = tmp_path / "model"
    n.save(str(save_dir), overwrite=False)
    loaded = thor.Network(name)
    loaded.load(str(save_dir))

    loaded_arch = json.loads(loaded.get_architecture_json())
    layers = [layer for layer in loaded_arch["layers"] if layer["layer_type"] == "convolution_1d"]
    assert len(layers) == 1
    arch = layers[0]
    assert arch["padding_mode"] == "causal"
    assert arch["padding_left"] == 9
    assert arch["padding_right"] == 0
    assert arch["stride"] == 2
    assert arch["dilation"] == 3
    assert arch["groups"] == 2
    assert arch["parameters"]["weights"]["shape"] == [6, 2, 4]
    assert "biases" not in arch["parameters"]


def test_conv1d_grouped_and_depthwise_parameter_shapes():
    n = _net("test_conv1d_grouped")
    x = _cw_input(n, 8, 16)
    thor.layers.Convolution1d(n, x, num_output_channels=12, filter_width=3, groups=4, activation=None)
    arch = _only_arch(n)
    assert arch["groups"] == 4
    assert arch["parameters"]["weights"]["shape"] == [12, 2, 3]

    n2 = _net("test_conv1d_depthwise")
    x2 = _cw_input(n2, 8, 16)
    thor.layers.Convolution1d(n2, x2, num_output_channels=8, filter_width=5, groups=8, padding="causal", activation=None)
    assert _only_arch(n2)["parameters"]["weights"]["shape"] == [8, 1, 5]


@pytest.mark.parametrize("groups", [0, 3, 5])
def test_conv1d_rejects_invalid_groups(groups):
    n = _net(f"test_conv1d_bad_groups_{groups}")
    x = _cw_input(n, 8, 16)
    with pytest.raises(ValueError, match="groups"):
        thor.layers.Convolution1d(n, x, num_output_channels=12, filter_width=3, groups=groups)
