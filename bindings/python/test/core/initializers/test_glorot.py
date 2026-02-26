# test/test_initializers_glorot.py
import pytest
import thor


def test_glorot_mode_enum_exists():
    assert hasattr(thor.initializers, "Glorot")
    assert hasattr(thor.initializers.Glorot, "Mode")
    assert hasattr(thor.initializers.Glorot.Mode, "UNIFORM")
    assert hasattr(thor.initializers.Glorot.Mode, "NORMAL")


def test_glorot_constructs_default():
    g = thor.initializers.Glorot()
    assert g is not None
    assert isinstance(g, thor.initializers.Glorot)


def test_glorot_constructs_uniform():
    g = thor.initializers.Glorot(thor.initializers.Glorot.Mode.UNIFORM)
    assert isinstance(g, thor.initializers.Glorot)

    g2 = thor.initializers.Glorot(mode=thor.initializers.Glorot.Mode.UNIFORM)
    assert isinstance(g2, thor.initializers.Glorot)


def test_glorot_constructs_normal():
    g = thor.initializers.Glorot(thor.initializers.Glorot.Mode.NORMAL)
    assert isinstance(g, thor.initializers.Glorot)

    g2 = thor.initializers.Glorot(mode=thor.initializers.Glorot.Mode.NORMAL)
    assert isinstance(g2, thor.initializers.Glorot)


def test_glorot_rejects_wrong_types():
    with pytest.raises(TypeError):
        thor.initializers.Glorot("UNIFORM")  # must be enum, not str

    with pytest.raises(TypeError):
        thor.initializers.Glorot(123)  # must be enum, not int


def test_glorot_rejects_wrong_arity_and_kwargs():
    with pytest.raises(TypeError):
        thor.initializers.Glorot(thor.initializers.Glorot.Mode.UNIFORM, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.initializers.Glorot(bogus=thor.initializers.Glorot.Mode.UNIFORM)  # wrong kw

    with pytest.raises(TypeError):
        thor.initializers.Glorot(mode=thor.initializers.Glorot.Mode.UNIFORM, extra=1)  # extra kw


def test_glorot_is_initializer_subclass_if_exposed():
    Initializer = getattr(thor.initializers, "Initializer", None)
    if Initializer is None:
        pytest.skip("thor.initializers.Initializer not exposed in Python")
    assert isinstance(thor.initializers.Glorot(), Initializer)
