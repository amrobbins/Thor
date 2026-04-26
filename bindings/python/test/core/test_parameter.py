import pytest
import thor


def _cpu_input(dtype=thor.DataType.fp16, dims=(8, 16)):
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    desc = thor.physical.PhysicalTensor.Descriptor(dtype, list(dims))
    return thor.physical.PhysicalTensor(placement, desc)


def test_fixed_shape_parameter_allocate_storage_uses_input_placement_and_requested_descriptor():
    input_tensor = _cpu_input()

    shape = [16, 32]
    dtype = thor.DataType.fp32

    parameter = thor.Parameter(
        name="weights",
        shape=shape,
        dtype=dtype,
        trainable=True,
    )

    storage = parameter.allocate_storage(input_tensor, shape=shape, dtype=dtype)

    assert parameter.name == "weights"
    assert parameter.trainable is True
    assert parameter.is_trainable() is True
    assert storage.get_placement() == input_tensor.get_placement()
    assert storage.get_descriptor().get_data_type() == dtype
    assert storage.get_descriptor().get_dimensions() == shape


def test_fixed_shape_parameter_constructor_defaults_dtype_to_fp32():
    parameter = thor.Parameter(
        name="biases",
        shape=[7],
    )

    assert parameter.name == "biases"
    assert parameter.trainable is True
    assert parameter.has_optimizer() is False


def test_storage_context_accepts_single_tensor_and_uses_feature_input_name():
    input_tensor = _cpu_input(dims=(4, 7))
    ctx = thor.Parameter.StorageContext(input_tensor)

    assert ctx.input_names() == ["feature_input"]
    assert ctx.has_input("feature_input") is True
    assert ctx.get_feature_input().get_descriptor().get_dimensions() == [4, 7]
    assert ctx.get_input("feature_input").get_descriptor().get_dimensions() == [4, 7]


def test_storage_context_exposes_named_inputs_from_mapping():
    x = _cpu_input(dims=(4, 7))
    y = _cpu_input(dims=(4, 9))
    ctx = thor.Parameter.StorageContext({
        "y": y,
        "x": x,
    })

    assert ctx.has_input("x") is True
    assert ctx.has_input("y") is True
    assert ctx.input_names() == ["x", "y"]
    assert ctx.input_names_string() == '["x", "y"]'
    assert set(ctx.inputs.keys()) == {"x", "y"}
    assert ctx.get_input("x").get_descriptor().get_dimensions() == [4, 7]
    assert ctx.get_input("y").get_descriptor().get_dimensions() == [4, 9]


def test_storage_context_get_input_raises_for_missing_name():
    ctx = thor.Parameter.StorageContext(_cpu_input(dims=(4, 7)))

    with pytest.raises(RuntimeError, match='No input named "missing"'):
        ctx.get_input("missing")


def test_storage_context_get_feature_input_raises_when_multiple_inputs_are_present():
    ctx = thor.Parameter.StorageContext({
        "x": _cpu_input(dims=(4, 7)),
        "y": _cpu_input(dims=(4, 9)),
    })

    with pytest.raises(RuntimeError, match="There is not exactly 1 input available"):
        ctx.get_feature_input()


def test_dynamic_parameter_constructor_accepts_context_factory_callable():
    parameter = thor.Parameter(
        name="weights",
        create_storage_from_context=lambda ctx: thor.Parameter.allocate_storage(
            ctx.get_feature_input(),
            shape=[ctx.get_feature_input().get_descriptor().get_dimensions()[-1]],
            dtype=ctx.get_feature_input().get_descriptor().get_data_type(),
        ),
        trainable=False,
    )

    assert parameter.name == "weights"
    assert parameter.trainable is False
    assert parameter.is_trainable() is False
    assert parameter.has_optimizer() is False


def test_dynamic_parameter_constructor_rejects_none_factory():
    with pytest.raises(TypeError, match="incompatible function arguments"):
        thor.Parameter(
            name="weights",
            create_storage_from_context=None,
        )


def test_dynamic_parameter_constructor_rejects_non_callable_factory():
    with pytest.raises(RuntimeError, match="create_storage_from_context must be callable"):
        thor.Parameter(
            name="weights",
            create_storage_from_context=123,
        )
