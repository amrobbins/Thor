# import pytest
# import thor
#
#
# def _net():
#     return thor.Network("test_net_adam")
#
#
# def test_adam_constructs_defaults():
#     n = _net()
#     opt = thor.optimizers.Adam(network=n)
#     assert opt is not None
#     assert isinstance(opt, thor.optimizers.Adam)
#
#
# def test_adam_constructs_defaults_without_network():
#     n = _net()
#     opt = thor.optimizers.Adam()
#     assert opt is not None
#     assert isinstance(opt, thor.optimizers.Adam)
#
#
# def test_adam_constructs_custom_params():
#     n = _net()
#     opt = thor.optimizers.Adam(
#         network=n,
#         alpha=1e-3,
#         beta1=0.9,
#         beta2=0.999,
#         epsilon=1e-7,
#     )
#     assert isinstance(opt, thor.optimizers.Adam)
#
#
# def test_adam_rejects_non_positive_alpha():
#     n = _net()
#     with pytest.raises(ValueError, match=r"alpha must be > 0"):
#         thor.optimizers.Adam(network=n, alpha=0.0)
#     with pytest.raises(ValueError, match=r"alpha must be > 0"):
#         thor.optimizers.Adam(network=n, alpha=-1.0)
#
#
# def test_adam_rejects_beta1_out_of_range():
#     n = _net()
#     with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
#         thor.optimizers.Adam(network=n, beta1=-0.01)
#     with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
#         thor.optimizers.Adam(beta1=1.0)
#     with pytest.raises(ValueError, match=r"0 <= beta1 < 1"):
#         thor.optimizers.Adam(network=n, beta1=1.01)
#
#
# def test_adam_rejects_beta2_out_of_range():
#     n = _net()
#     with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
#         thor.optimizers.Adam(beta2=-0.01)
#     with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
#         thor.optimizers.Adam(network=n, beta2=1.0)
#     with pytest.raises(ValueError, match=r"0 <= beta2 < 1"):
#         thor.optimizers.Adam(network=n, beta2=1.01)
#
#
# def test_adam_rejects_non_positive_epsilon():
#     n = _net()
#     with pytest.raises(ValueError, match=r"epsilon must be > 0"):
#         thor.optimizers.Adam(network=n, epsilon=0.0)
#     with pytest.raises(ValueError, match=r"epsilon must be > 0"):
#         thor.optimizers.Adam(network=n, epsilon=-1e-7)
#
#
# def test_adam_rejects_wrong_types():
#     n = _net()
#
#     with pytest.raises(TypeError):
#         thor.optimizers.Adam("not a network")
#
#     with pytest.raises(TypeError):
#         thor.optimizers.Adam(network=n, alpha="1e-3")
#
#     with pytest.raises(TypeError):
#         thor.optimizers.Adam(network=n, beta1="0.9")
#
#     with pytest.raises(TypeError):
#         thor.optimizers.Adam(network=n, beta2="0.999")
#
#     with pytest.raises(TypeError):
#         thor.optimizers.Adam(network=n, epsilon="1e-7")
#
#
# def test_adam_rejects_wrong_arity_and_kwargs():
#     n = _net()
#
#     with pytest.raises(TypeError):
#         thor.optimizers.Adam(1e-3, 0.9, 0.999, 1e-7, 123, network=n)  # extra positional
#
#     with pytest.raises(TypeError):
#         thor.optimizers.Adam(bogus=123, network=n)  # wrong kw
#
#     with pytest.raises(TypeError):
#         thor.optimizers.Adam(alpha=1e-3, extra=123, network=n)  # extra kw
#
#
# def test_adam_is_optimizer_subclass_if_exposed():
#     Optimizer = getattr(thor.optimizers, "Optimizer", None)
#     if Optimizer is None:
#         pytest.skip("thor.optimizers.Optimizer not exposed in Python")
#     n = _net()
#     assert isinstance(thor.optimizers.Adam(network=n), Optimizer)
#
#
# def test_adam_multiple_optimizers_on_same_network_throws():
#     n = _net()
#     opt = thor.optimizers.Adam(network=n)
#     assert opt is not None
#     assert isinstance(opt, thor.optimizers.Adam)
#
#     with pytest.raises(RuntimeError, match=r".*Multiple default optimizers.*"):
#         thor.optimizers.Adam(network=n)
