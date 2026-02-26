import pytest
import thor


def test_softmax_constructs():
    a = thor.activations.Softmax()
    assert a is not None
    assert isinstance(a, thor.activations.Softmax)


def test_softmax_rejects_unexpected_args():
    # __new__ takes only cls, so passing args should TypeError at the Python binding level
    with pytest.raises(TypeError):
        thor.activations.Softmax(123)

    with pytest.raises(TypeError):
        thor.activations.Softmax(alpha=1.0)


def test_softmax_is_activation_subclass_if_exposed():
    Activation = getattr(thor.activations, "Activation", None)
    if Activation is None:
        pytest.skip("thor.activations.Activation not exposed in Python")
    assert isinstance(thor.activations.Softmax(), Activation)
