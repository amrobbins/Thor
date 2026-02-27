import pytest
import thor


def _net():
    return thor.Network("test_net_batch_norm")


def _input_tensor(n: thor.Network):
    ni = thor.layers.NetworkInput(n, "input", [8, 32, 32], thor.DataType.fp16)
    return ni.get_feature_output()


def test_batch_norm_constructs_and_getters_round_trip():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [8, 32, 32], thor.DataType.fp16)
    x = ni.get_feature_output()

    bn = thor.layers.BatchNormalization(
        n,
        x,
        exponential_running_average_factor=0.1,
        epsilon=1e-4,
    )

    assert bn is not None
    assert isinstance(bn, thor.layers.BatchNormalization)

    assert bn.get_exponential_running_average_factor() == pytest.approx(0.1)
    assert bn.get_epsilon() == pytest.approx(1e-4)

    out = bn.get_feature_output()
    assert out is not None
    assert isinstance(out, thor.Tensor)


def test_batch_norm_rejects_bad_exponential_running_average_factor():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [8, 32, 32], thor.DataType.fp16)
    x = ni.get_feature_output()

    with pytest.raises(ValueError, match=r"exponential_running_average_factor.*0 < factor <= 1"):
        thor.layers.BatchNormalization(n, x, exponential_running_average_factor=0.0)

    with pytest.raises(ValueError, match=r"exponential_running_average_factor.*0 < factor <= 1"):
        thor.layers.BatchNormalization(n, x, exponential_running_average_factor=-0.1)

    with pytest.raises(ValueError, match=r"exponential_running_average_factor.*0 < factor <= 1"):
        thor.layers.BatchNormalization(n, x, exponential_running_average_factor=1.01)


def test_batch_norm_rejects_bad_epsilon():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [8, 32, 32], thor.DataType.fp16)
    x = ni.get_feature_output()

    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.layers.BatchNormalization(n, x, epsilon=0.0)

    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        thor.layers.BatchNormalization(n, x, epsilon=-1e-6)


def test_batch_norm_rejects_wrong_types_and_arity():
    n = _net()
    ni = thor.layers.NetworkInput(n, "input", [8, 32, 32], thor.DataType.fp16)
    x = ni.get_feature_output()

    with pytest.raises(TypeError):
        thor.layers.BatchNormalization()  # missing args

    with pytest.raises(TypeError):
        thor.layers.BatchNormalization(n)  # missing feature_input

    with pytest.raises(TypeError):
        thor.layers.BatchNormalization(n, x, 0.1, 1e-4, 123)  # extra arg

    with pytest.raises(TypeError):
        thor.layers.BatchNormalization("not a network", x)

    with pytest.raises(TypeError):
        thor.layers.BatchNormalization(n, "not a tensor")

    with pytest.raises(TypeError):
        thor.layers.BatchNormalization(n, x, exponential_running_average_factor="0.1")

    with pytest.raises(TypeError):
        thor.layers.BatchNormalization(n, x, epsilon="1e-4")


def test_batch_norm_get_feature_output_returns_tensor():
    n = _net()
    x = _input_tensor(n)

    bn = thor.layers.BatchNormalization(n, x, exponential_running_average_factor=0.1, epsilon=1e-4)
    out = bn.get_feature_output()

    assert out is not None
    assert isinstance(out, thor.Tensor)
    assert out.get_dimensions() == x.get_dimensions()
    assert out.get_data_type() == x.get_data_type()


def test_batch_norm_getters_default_to_none_or_float():
    """
    Depending on your C++ Optional semantics, defaults may be stored as:
      - not-present (=> Python None), OR
      - present with default value (=> Python float).

    This test accepts either, but still enforces that the types are correct.
    """
    n = _net()
    x = _input_tensor(n)

    bn = thor.layers.BatchNormalization(n, x)  # use defaults

    eraf = bn.get_exponential_running_average_factor()
    eps = bn.get_epsilon()

    assert (eraf is None) or isinstance(eraf, float)
    assert (eps is None) or isinstance(eps, float)

    # If present, they should be reasonable (match your binding defaults)
    if eraf is not None:
        assert eraf == pytest.approx(0.05)
    if eps is not None:
        assert eps == pytest.approx(1e-4)
