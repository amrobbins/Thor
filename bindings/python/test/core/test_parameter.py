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

    parameter = thor.parameters.ParameterSpecification(
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
    parameter = thor.parameters.ParameterSpecification(
        name="biases",
        shape=[7],
    )

    assert parameter.name == "biases"
    assert parameter.trainable is True
    assert parameter.has_optimizer() is False


def test_api_parameter_does_not_expose_training_enabled_setter():
    parameter = thor.parameters.ParameterSpecification(
        name="biases",
        shape=[7],
    )

    assert hasattr(parameter, "is_training_initially_enabled") is True
    assert hasattr(parameter, "set_training_enabled") is False


def test_storage_context_accepts_single_tensor_and_uses_feature_input_name():
    input_tensor = _cpu_input(dims=(4, 7))
    ctx = thor.parameters.ParameterSpecification.StorageContext(input_tensor)

    assert ctx.input_names() == ["feature_input"]
    assert ctx.has_input("feature_input") is True
    assert ctx.get_feature_input().get_descriptor().get_dimensions() == [4, 7]
    assert ctx.get_input("feature_input").get_descriptor().get_dimensions() == [4, 7]


def test_storage_context_exposes_named_inputs_from_mapping():
    x = _cpu_input(dims=(4, 7))
    y = _cpu_input(dims=(4, 9))
    ctx = thor.parameters.ParameterSpecification.StorageContext({
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
    ctx = thor.parameters.ParameterSpecification.StorageContext(_cpu_input(dims=(4, 7)))

    with pytest.raises(RuntimeError, match='No input named "missing"'):
        ctx.get_input("missing")


def test_storage_context_get_feature_input_raises_when_multiple_inputs_are_present():
    ctx = thor.parameters.ParameterSpecification.StorageContext({
        "x": _cpu_input(dims=(4, 7)),
        "y": _cpu_input(dims=(4, 9)),
    })

    with pytest.raises(RuntimeError, match="There is not exactly 1 input available"):
        ctx.get_feature_input()


def test_dynamic_parameter_constructor_accepts_context_factory_callable():
    parameter = thor.parameters.ParameterSpecification(
        name="weights",
        create_storage_from_context=lambda ctx: thor.parameters.ParameterSpecification.allocate_storage(
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
        thor.parameters.ParameterSpecification(
            name="weights",
            create_storage_from_context=None,
        )


def test_dynamic_parameter_constructor_rejects_non_callable_factory():
    with pytest.raises(TypeError, match="create_storage_from_context.*expected callable"):
        thor.parameters.ParameterSpecification(
            name="weights",
            create_storage_from_context=123,
        )


def test_non_negative_parameter_constraint_constructs_and_serializes():
    constraint = thor.constraints.NonNegative()

    assert isinstance(constraint, thor.constraints.ParameterConstraint)
    assert constraint.constraint_type == "non_negative"
    assert '"constraint_type":"non_negative"' in constraint.get_architecture_json()


def test_parameter_spec_accepts_single_constraint_and_exposes_it():
    parameter = thor.parameters.ParameterSpecification(
        name="weights",
        shape=[4, 7],
        dtype=thor.DataType.fp32,
        constraints=thor.constraints.NonNegative(),
    )

    assert parameter.has_constraints() is True
    constraints = parameter.get_constraints()
    assert len(constraints) == 1
    assert constraints[0].constraint_type == "non_negative"
    assert '"constraint_type":"non_negative"' in parameter.get_architecture_json()


def test_parameter_spec_accepts_constraint_sequence():
    parameter = thor.parameters.ParameterSpecification(
        name="weights",
        shape=[4, 7],
        dtype=thor.DataType.fp32,
        constraints=[thor.constraints.NonNegative()],
    )

    assert parameter.has_constraints() is True
    assert len(parameter.get_constraints()) == 1


def test_parameter_spec_rejects_invalid_constraint_object():
    with pytest.raises(TypeError, match="constraints"):
        thor.parameters.ParameterSpecification(
            name="weights",
            shape=[4, 7],
            dtype=thor.DataType.fp32,
            constraints=123,
        )


def test_additional_parameter_constraints_construct_and_serialize():
    cases = [
        (thor.constraints.NonPositive(), "non_positive"),
        (thor.constraints.Min(-0.25), "min"),
        (thor.constraints.Max(0.75), "max"),
        (thor.constraints.MinMax(-0.5, 0.5), "min_max"),
    ]

    for constraint, constraint_type in cases:
        assert isinstance(constraint, thor.constraints.ParameterConstraint)
        assert constraint.constraint_type == constraint_type
        assert f'"constraint_type":"{constraint_type}"' in constraint.get_architecture_json()

    assert thor.constraints.Min(-0.25).min_value == pytest.approx(-0.25)
    assert thor.constraints.Max(0.75).max_value == pytest.approx(0.75)
    min_max = thor.constraints.MinMax(-0.5, 0.5)
    assert min_max.min_value == pytest.approx(-0.5)
    assert min_max.max_value == pytest.approx(0.5)


def test_min_max_parameter_constraint_rejects_reversed_bounds():
    with pytest.raises(RuntimeError, match="min_value <= max_value"):
        thor.constraints.MinMax(1.0, -1.0)


def test_parameter_spec_accepts_additional_constraint_sequence():
    parameter = thor.parameters.ParameterSpecification(
        name="weights",
        shape=[4, 7],
        dtype=thor.DataType.fp32,
        constraints=[
            thor.constraints.Min(-0.5),
            thor.constraints.Max(0.75),
        ],
    )

    assert parameter.has_constraints() is True
    constraints = parameter.get_constraints()
    assert [constraint.constraint_type for constraint in constraints] == ["min", "max"]
    architecture_json = parameter.get_architecture_json()
    assert '"constraint_type":"min"' in architecture_json
    assert '"constraint_type":"max"' in architecture_json
    assert '"min_value":-0.5' in architecture_json
    assert '"max_value":0.75' in architecture_json
