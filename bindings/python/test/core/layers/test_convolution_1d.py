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
    assert arch["version"] == "1.0.0"
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
    assert conv.get_compute_data_type() == thor.DataType.fp32
    assert arch["compute_data_type"] == "fp32"
    assert arch["activation"]["layer_type"] == "gelu"
    assert arch["parameters"]["weights"]["shape"] == [5, 3, 3]
    assert arch["parameters"]["biases"]["shape"] == [5]


def test_conv1d_explicit_tf32_compute_requires_fp32_storage():
    n = _net("test_conv1d_tf32")
    x = _cw_input(n, 4, 16, thor.DataType.fp32)
    conv = thor.layers.Convolution1d(
        n, x, 8, 3, activation=None, compute_data_type=thor.DataType.tf32)
    assert conv.get_compute_data_type() == thor.DataType.tf32
    assert _only_arch(n)["compute_data_type"] == "tf32"

    n_bad = _net("test_conv1d_tf32_bad_storage")
    x_bad = _cw_input(n_bad, 4, 16, thor.DataType.fp16)
    with pytest.raises(ValueError, match="TF32 compute requires FP32"):
        thor.layers.Convolution1d(
            n_bad, x_bad, 8, 3, activation=None, compute_data_type=thor.DataType.tf32)


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


def _ragged_input(
    n: thor.Network,
    channels: int = 8,
    *,
    max_total_values: int = 96,
    batch_size: int = 3,
    max_values_per_row: int | None = 48,
    name: str = "tokens",
):
    return thor.layers.RaggedNetworkInput(
        n,
        name,
        thor.DataType.fp32,
        [channels],
        max_total_values=max_total_values,
        batch_size=batch_size,
        offsets_data_type=thor.DataType.uint32,
        max_values_per_row=max_values_per_row,
    )


def test_conv1d_accepts_ragged_input_and_preserves_capacity_contract():
    n = _net("test_conv1d_ragged_contract")
    x = _ragged_input(n, channels=8, max_total_values=96, batch_size=3, max_values_per_row=48)

    conv = thor.layers.Convolution1d(
        n,
        x,
        num_output_channels=12,
        filter_width=5,
        padding="causal",
        dilation=7,
        groups=4,
        activation=None,
    )
    y = conv.get_feature_output()

    assert conv.get_use_ragged() is True
    assert isinstance(y, thor.RaggedTensor)
    assert y.values.get_dimensions() == [96, 12]
    assert y.offsets == x.offsets
    assert y.batch_size == x.batch_size == 3
    assert y.max_total_values == x.max_total_values == 96
    assert y.max_values_per_row == x.max_values_per_row == 48

    arch = _only_arch(n)
    assert arch["use_ragged"] is True
    assert arch["dilation"] == 7
    assert arch["groups"] == 4
    assert arch["padding_mode"] == "causal"
    assert arch["ragged_input"]["offsets"]["id"] == arch["ragged_output"]["offsets"]["id"]
    assert arch["ragged_input"]["max_values_per_row"] == 48
    assert arch["ragged_output"]["max_values_per_row"] == 48
    assert arch["parameters"]["weights"]["shape"] == [12, 2, 5]


@pytest.mark.parametrize("dilation", [1, 7, 28])
def test_conv1d_ragged_python_surface_preserves_dilation(dilation):
    n = _net(f"test_conv1d_ragged_dilation_{dilation}")
    x = _ragged_input(n, channels=4)
    conv = thor.layers.Convolution1d(
        n,
        x,
        num_output_channels=6,
        filter_width=3,
        padding="causal",
        dilation=dilation,
        activation=None,
    )

    assert isinstance(conv.get_feature_output(), thor.RaggedTensor)
    arch = _only_arch(n)
    assert arch["dilation"] == dilation
    assert arch["padding_left"] == dilation * 2
    assert arch["padding_right"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stride": 2, "padding": "causal"}, "stride=1"),
        ({"padding": "valid"}, "padding='causal'"),
        ({"padding": "same"}, "padding='causal'"),
        ({"padding": (2, 0)}, "padding='causal'"),
    ],
)
def test_conv1d_ragged_rejects_unsupported_geometry(kwargs, message):
    n = _net("test_conv1d_ragged_bad_geometry")
    x = _ragged_input(n, channels=4)
    with pytest.raises(ValueError, match=message):
        thor.layers.Convolution1d(n, x, num_output_channels=6, filter_width=3, activation=None, **kwargs)


def test_conv1d_ragged_requires_max_values_per_row():
    n = _net("test_conv1d_ragged_missing_max_row")
    x = _ragged_input(n, channels=4, max_values_per_row=None)
    with pytest.raises(ValueError, match="max_values_per_row"):
        thor.layers.Convolution1d(n, x, num_output_channels=6, filter_width=3, padding="causal", activation=None)


def test_conv1d_rejects_non_tensor_feature_input():
    n = _net("test_conv1d_bad_feature_input_type")
    with pytest.raises(TypeError, match="Tensor or thor.RaggedTensor"):
        thor.layers.Convolution1d(n, object(), num_output_channels=6, filter_width=3, padding="causal", activation=None)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("channels", "num_output_channels", "groups"),
    [
        (8, 12, 4),
        (8, 8, 8),
    ],
)
def test_conv1d_ragged_grouped_and_depthwise_place(channels, num_output_channels, groups):
    n = _net(f"test_conv1d_ragged_place_g{groups}")
    x = _ragged_input(n, channels=channels, max_total_values=96, batch_size=3, max_values_per_row=48)
    conv = thor.layers.Convolution1d(
        n,
        x,
        num_output_channels=num_output_channels,
        filter_width=3,
        padding="causal",
        dilation=7,
        groups=groups,
        activation=None,
    )
    thor.layers.RaggedNetworkOutput(n, "output", conv.get_feature_output())

    placed = n.place(3, inference_only=True, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert isinstance(placed, thor.runtime.PlacedNetwork)
    assert placed.has_network_input("tokens")


@pytest.mark.cuda
def test_conv1d_ragged_places_for_training_with_network_default_optimizer():
    n = _net("test_conv1d_ragged_training_place")
    thor.optimizers.Sgd(initial_learning_rate=0.01, network=n)
    x = _ragged_input(n, channels=8, max_total_values=96, batch_size=3, max_values_per_row=48)
    conv = thor.layers.Convolution1d(
        n,
        x,
        num_output_channels=12,
        filter_width=3,
        padding="causal",
        dilation=7,
        groups=4,
        activation=None,
    )
    thor.layers.RaggedNetworkOutput(n, "output", conv.get_feature_output())

    placed = n.place(3, inference_only=False, forced_devices=[0], forced_num_stamps_per_gpu=1)
    assert isinstance(placed, thor.runtime.PlacedNetwork)
    assert placed.get_num_trainable_layers() == 1
    assert {parameter.name for parameter in conv.get_bound_parameters(placed)} == {"weights", "biases"}


def test_conv1d_ragged_f2_style_public_layer_chain_preserves_partition_end_to_end():
    n = _net("test_conv1d_ragged_f2_public_chain")
    x = _ragged_input(
        n,
        channels=6,
        max_total_values=24,
        batch_size=3,
        max_values_per_row=8,
        name="history",
    )
    root_offsets = x.offsets

    stages = []
    x = thor.layers.FullyConnected(n, x, 10, True, activation=None).get_feature_output()
    stages.append((x, 10))
    x = thor.layers.RMSNorm(n, x, normalized_shape=[10], epsilon=1e-5).get_feature_output()
    stages.append((x, 10))
    x = thor.layers.FullyConnected(n, x, 8, True, activation=None).get_feature_output()
    stages.append((x, 8))

    for dilation in (1, 7):
        x = thor.layers.Convolution1d(
            n,
            x,
            num_output_channels=8,
            filter_width=3,
            padding="causal",
            dilation=dilation,
            activation=None,
        ).get_feature_output()
        stages.append((x, 8))
        x = thor.activations.Relu().add_to_network(n, x)
        stages.append((x, 8))

    x = thor.layers.Convolution1d(
        n,
        x,
        num_output_channels=8,
        filter_width=3,
        padding="causal",
        dilation=28,
        activation=None,
    ).get_feature_output()
    stages.append((x, 8))
    x = thor.layers.FullyConnected(n, x, 6, True, activation=None).get_feature_output()
    stages.append((x, 6))
    thor.layers.RaggedNetworkOutput(n, "temporal_output", x)

    for stage, channels in stages:
        assert isinstance(stage, thor.RaggedTensor)
        assert stage.values.get_dimensions() == [24, channels]
        assert stage.offsets == root_offsets
        assert stage.batch_size == 3
        assert stage.max_total_values == 24
        assert stage.max_values_per_row == 8

    arch = json.loads(n.get_architecture_json())
    convs = [layer for layer in arch["layers"] if layer["layer_type"] == "convolution_1d"]
    assert [layer["dilation"] for layer in convs] == [1, 7, 28]
    assert all(layer["use_ragged"] for layer in convs)
    assert all(layer["padding_mode"] == "causal" and layer["stride"] == 1 for layer in convs)
    assert all(
        layer["ragged_input"]["offsets"]["id"] == layer["ragged_output"]["offsets"]["id"]
        for layer in convs
    )
