import pytest
import thor


def test_selu_constructs():
    a = thor.activations.Selu()
    assert a is not None
    assert isinstance(a, thor.activations.Selu)


def test_selu_rejects_unexpected_args():
    # __new__ takes only cls, so passing args should TypeError at the Python binding level
    with pytest.raises(TypeError):
        thor.activations.Selu(123)

    with pytest.raises(TypeError):
        thor.activations.Selu(alpha=1.0)


def test_selu_is_activation_subclass_if_exposed():
    Activation = getattr(thor.activations, "Activation", None)
    if Activation is None:
        pytest.skip("thor.activations.Activation not exposed in Python")
    assert isinstance(thor.activations.Selu(), Activation)
