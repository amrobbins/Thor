import thor


def test_top_level_api_is_curated():
    expected = {
        "DataType",
        "Network",
        "Tensor",
        "__git_version__",
        "__version__",
        "activations",
        "constraints",
        "data",
        "initializers",
        "layers",
        "losses",
        "metrics",
        "optimizers",
        "parameters",
        "physical",
        "random",
        "runtime",
        "training",
    }

    assert set(dir(thor)) == expected

    for leaked_name in [
        "BoundParameter",
        "ParameterReference",
        "ParameterSpecification",
        "ParameterConstraint",
        "NonNegativeParameterConstraint",
        "NonPositiveParameterConstraint",
        "MinParameterConstraint",
        "MaxParameterConstraint",
        "MinMaxParameterConstraint",
        "PlacedNetwork",
        "StatusCode",
        "_thor",
        "Path",
        "ctypes",
        "os",
    ]:
        assert not hasattr(thor, leaked_name)


def test_parameter_and_constraint_namespaces_export_public_types():
    parameter = thor.parameters.ParameterSpecification(name="weights", shape=[2, 3])
    constraint = thor.constraints.NonNegative()

    assert isinstance(parameter, thor.parameters.ParameterSpecification)
    assert isinstance(constraint, thor.constraints.ParameterConstraint)
    assert constraint.constraint_type == "non_negative"

    assert "ParameterSpecification" in dir(thor.parameters)
    assert "NonNegative" in dir(thor.constraints)
    assert "NonNegativeParameterConstraint" not in dir(thor.constraints)


def test_runtime_namespace_exports_runtime_types():
    assert thor.runtime.StatusCode.success is thor.Network.StatusCode.success
    assert "PlacedNetwork" in dir(thor.runtime)
