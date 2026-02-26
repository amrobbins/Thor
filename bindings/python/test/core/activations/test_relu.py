import pytest
import thor


def test_relu_constructs():
    a = thor.activations.Relu()
    assert a is not None
    assert isinstance(a, thor.activations.Relu)


def test_relu_rejects_unexpected_args():
    # __new__ takes only cls, so passing args should TypeError at the Python binding level
    with pytest.raises(TypeError):
        thor.activations.Relu(123)

    with pytest.raises(TypeError):
        thor.activations.Relu(alpha=1.0)


def test_relu_is_activation_subclass_if_exposed():
    Activation = getattr(thor.activations, "Activation", None)
    if Activation is None:
        pytest.skip("thor.activations.Activation not exposed in Python")
    assert isinstance(thor.activations.Relu(), Activation)
