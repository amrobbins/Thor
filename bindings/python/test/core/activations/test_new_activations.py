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
