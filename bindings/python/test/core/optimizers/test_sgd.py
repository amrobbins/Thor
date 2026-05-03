# test/test_optimizers_sgd.py
import pytest
import thor


def _net():
    return thor.Network("test_net_sgd")


def test_sgd_constructs_defaults():
    n = _net()
    opt = thor.optimizers.Sgd(network=n)
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Sgd)


def test_sgd_constructs_defaults_without_network():
    n = _net()
    opt = thor.optimizers.Sgd()
    assert opt is not None
    assert isinstance(opt, thor.optimizers.Sgd)


def test_sgd_constructs_custom_params():
    n = _net()
    opt = thor.optimizers.Sgd(
        network=n,
        initial_learning_rate=0.1,
        decay=0.25,
        momentum=0.9,
        nesterov_momentum=True,
    )
    assert isinstance(opt, thor.optimizers.Sgd)


def test_sgd_allows_decay_endpoints():
    n = _net()
    assert isinstance(thor.optimizers.Sgd(decay=0.0), thor.optimizers.Sgd)
    assert isinstance(thor.optimizers.Sgd(network=n, decay=1.0), thor.optimizers.Sgd)


def test_sgd_rejects_non_positive_initial_learning_rate():
    n = _net()
    with pytest.raises(ValueError, match=r"initial_learning_rate must be > 0"):
        thor.optimizers.Sgd(network=n, initial_learning_rate=0.0)

    with pytest.raises(ValueError, match=r"initial_learning_rate must be > 0"):
        thor.optimizers.Sgd(initial_learning_rate=-1.0)


def test_sgd_rejects_decay_out_of_range():
    n = _net()
    with pytest.raises(ValueError, match=r"0 <= decay <= 1"):
        thor.optimizers.Sgd(network=n, decay=-0.01)

    with pytest.raises(ValueError, match=r"0 <= decay <= 1"):
        thor.optimizers.Sgd(network=n, decay=1.01)


def test_sgd_rejects_negative_momentum():
    n = _net()
    with pytest.raises(ValueError, match=r"momentum must be >= 0"):
        thor.optimizers.Sgd(momentum=-0.1)


def test_sgd_rejects_wrong_types():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.Sgd("not a network")

    with pytest.raises(TypeError):
        thor.optimizers.Sgd(network=n, initial_learning_rate="0.1")

    with pytest.raises(TypeError):
        thor.optimizers.Sgd(network=n, decay="0.0")

    with pytest.raises(TypeError):
        thor.optimizers.Sgd(network=n, momentum="0.0")

    with pytest.raises(TypeError):
        thor.optimizers.Sgd(network=n, nesterov_momentum="false")


def test_sgd_rejects_wrong_arity_and_kwargs():
    n = _net()

    with pytest.raises(TypeError):
        thor.optimizers.Sgd(0.1, 0.0, 0.0, False, 123, network=n)  # extra positional

    with pytest.raises(TypeError):
        thor.optimizers.Sgd(network=n, bogus=123)  # wrong kw

    with pytest.raises(TypeError):
        thor.optimizers.Sgd(network=n, initial_learning_rate=0.1, extra=123)  # extra kw


def test_sgd_is_optimizer_subclass_if_exposed():
    Optimizer = getattr(thor.optimizers, "Optimizer", None)
    if Optimizer is None:
        pytest.skip("thor.optimizers.Optimizer not exposed in Python")
    n = _net()
    assert isinstance(thor.optimizers.Sgd(network=n), Optimizer)
