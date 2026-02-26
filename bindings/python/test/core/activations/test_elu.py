import pytest
import thor


def test_elu_constructs_default():
    a = thor.activations.Elu()
    assert a is not None
    assert isinstance(a, thor.activations.Elu)


def test_elu_constructs_with_alpha_positive():
    a = thor.activations.Elu(1.5)
    assert isinstance(a, thor.activations.Elu)


def test_elu_constructs_with_alpha_zero_allowed():
    a = thor.activations.Elu(0.0)
    assert isinstance(a, thor.activations.Elu)


def test_elu_rejects_negative_alpha():
    with pytest.raises(ValueError, match=r"alpha must be >= 0"):
        thor.activations.Elu(-1.0)


def test_elu_rejects_negative_alpha_includes_value():
    # Optional: verify the message includes the numeric value (your C++ includes it)
    with pytest.raises(ValueError, match=r"alpha == -2(\.0+)?"):
        thor.activations.Elu(-2.0)


def test_elu_is_activation_subclass_if_exposed():
    # If you expose Activation in Python, this should hold.
    # If you don't, this test will be skipped gracefully.
    Activation = getattr(thor.activations, "Activation", None)
    if Activation is None:
        pytest.skip("thor.activations.Activation not exposed in Python")
    assert isinstance(thor.activations.Elu(), Activation)
