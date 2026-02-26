# test/test_initializers_uniform_random.py
import pytest
import thor


def test_uniform_random_constructs():
    init = thor.initializers.UniformRandom(-0.1, 0.1)
    assert init is not None
    assert isinstance(init, thor.initializers.UniformRandom)


def test_uniform_random_allows_equal_min_max():
    init = thor.initializers.UniformRandom(0.25, 0.25)
    assert isinstance(init, thor.initializers.UniformRandom)


def test_uniform_random_rejects_min_greater_than_max():
    with pytest.raises(ValueError, match=r"min_value must be <= max_value"):
        thor.initializers.UniformRandom(1.0, 0.0)


def test_uniform_random_rejects_min_greater_than_max_includes_values():
    # Optional: verify the message includes both values (your C++ prints both)
    with pytest.raises(ValueError, match=r"min_value:\s*-?1(\.0+)?\s*max_value:\s*-?2(\.0+)?"):
        thor.initializers.UniformRandom(1.0, -2.0)


def test_uniform_random_rejects_wrong_arity_and_kwargs():
    with pytest.raises(TypeError):
        thor.initializers.UniformRandom()  # missing args

    with pytest.raises(TypeError):
        thor.initializers.UniformRandom(0.0)  # missing max_value

    with pytest.raises(TypeError):
        thor.initializers.UniformRandom(0.0, 1.0, 2.0)  # extra arg

    with pytest.raises(TypeError):
        thor.initializers.UniformRandom(min_value=0.0, max_value=1.0, extra=123)


def test_uniform_random_is_initializer_subclass_if_exposed():
    Initializer = getattr(thor.initializers, "Initializer", None)
    if Initializer is None:
        pytest.skip("thor.initializers.Initializer not exposed in Python")
    assert isinstance(thor.initializers.UniformRandom(0.0, 1.0), Initializer)
