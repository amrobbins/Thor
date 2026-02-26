import pytest
import thor


def test_sigmoid_constructs():
    a = thor.activations.Sigmoid()
    assert a is not None
    assert isinstance(a, thor.activations.Sigmoid)


def test_sigmoid_rejects_unexpected_args():
    # __new__ takes only cls, so passing args should TypeError at the Python binding level
    with pytest.raises(TypeError):
        thor.activations.Sigmoid(123)

    with pytest.raises(TypeError):
        thor.activations.Sigmoid(alpha=1.0)


def test_sigmoid_is_activation_subclass_if_exposed():
    Activation = getattr(thor.activations, "Activation", None)
    if Activation is None:
        pytest.skip("thor.activations.Activation not exposed in Python")
    assert isinstance(thor.activations.Sigmoid(), Activation)
