import pytest
import thor


def test_swish_constructs():
    a = thor.activations.Swish()
    assert a is not None
    assert isinstance(a, thor.activations.Swish)


def test_swish_rejects_unexpected_args():
    # __new__ takes only cls, so passing args should TypeError at the Python binding level
    with pytest.raises(TypeError):
        thor.activations.Swish(123)

    with pytest.raises(TypeError):
        thor.activations.Swish(alpha=1.0)


def test_swish_is_activation_subclass_if_exposed():
    Activation = getattr(thor.activations, "Activation", None)
    if Activation is None:
        pytest.skip("thor.activations.Activation not exposed in Python")
    assert isinstance(thor.activations.Swish(), Activation)
