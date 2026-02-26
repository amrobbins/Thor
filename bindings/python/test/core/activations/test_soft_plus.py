import pytest
import thor


def test_softPlus_constructs():
    a = thor.activations.SoftPlus()
    assert a is not None
    assert isinstance(a, thor.activations.SoftPlus)


def test_softPlus_rejects_unexpected_args():
    # __new__ takes only cls, so passing args should TypeError at the Python binding level
    with pytest.raises(TypeError):
        thor.activations.SoftPlus(123)

    with pytest.raises(TypeError):
        thor.activations.SoftPlus(alpha=1.0)


def test_softPlus_is_activation_subclass_if_exposed():
    Activation = getattr(thor.activations, "Activation", None)
    if Activation is None:
        pytest.skip("thor.activations.Activation not exposed in Python")
    assert isinstance(thor.activations.SoftPlus(), Activation)
