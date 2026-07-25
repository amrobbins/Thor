import pytest
import thor


@pytest.mark.parametrize(
    "cls,args",
    [
        (thor.activations.Glu, ()),
        (thor.activations.Reglu, ()),
        (thor.activations.Geglu, ()),
        (thor.activations.Swiglu, ()),
        (thor.activations.BilinearGlu, ()),
        (thor.activations.Mish, ()),
        (thor.activations.Relu6, ()),
        (thor.activations.HardSwish, ()),
        (thor.activations.HardTanh, ()),
        (thor.activations.HardTanh, (-0.5, 0.5)),
        (thor.activations.Threshold, ()),
        (thor.activations.Threshold, (0.25, -1.0)),
    ],
)
def test_new_activation_constructs(cls, args):
    activation = cls(*args)
    assert activation is not None
    assert isinstance(activation, cls)
    Activation = getattr(thor.activations, "Activation", None)
    if Activation is not None:
        assert isinstance(activation, Activation)


def test_hard_tanh_rejects_invalid_range():
    with pytest.raises(ValueError):
        thor.activations.HardTanh(1.0, -1.0)


@pytest.mark.parametrize(
    "cls",
    [
        thor.activations.Glu,
        thor.activations.Reglu,
        thor.activations.Geglu,
        thor.activations.Swiglu,
        thor.activations.BilinearGlu,
    ],
)
def test_standalone_glu_add_to_network_halves_final_feature_dimension(cls):
    network = thor.Network(f"python_{cls.__name__}_standalone_shape")
    feature_input = thor.layers.NetworkInput(
        network,
        "feature_input",
        [53, 512],
        thor.DataType.bf16,
    ).get_feature_output()

    feature_output = cls().add_to_network(network, feature_input)

    assert feature_output.get_dimensions() == [53, 256]
    assert feature_output.get_data_type() == thor.DataType.bf16


def test_standalone_swiglu_rejects_epilogue_until_shape_changing_epilogues_are_supported():
    network = thor.Network("python_swiglu_epilogue_rejected")
    feature_input = thor.layers.NetworkInput(
        network,
        "feature_input",
        [53, 512],
        thor.DataType.bf16,
    ).get_feature_output()
    epilogue_input = thor.activations.Activation.epilogue_input(
        output_dtype=thor.DataType.bf16,
        compute_dtype=thor.DataType.fp32,
    )

    with pytest.raises(ValueError, match="do not currently support activation epilogues"):
        thor.activations.Swiglu().add_to_network(
            network,
            feature_input,
            epilogue=epilogue_input,
        )
