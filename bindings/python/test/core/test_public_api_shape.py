import thor


def test_top_level_api_is_curated():
    expected = {
        "DataType",
        "EnsembleModel",
        "Network",
        "Tensor",
        "__git_version__",
        "__version__",
        "activations",
        "constraints",
        "data",
        "ensembles",
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
        "EnsembleAggregation",
        "EnsembleMemberSpec",
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
        "TensorSpec",
        "_thor",
        "Path",
        "ctypes",
        "os",
    ]:
        assert not hasattr(thor, leaked_name)


def test_custom_layer_tensor_spec_is_namespaced_under_layers():
    assert "TensorSpec" in dir(thor.layers)
    assert not hasattr(thor, "TensorSpec")


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


def test_ensembles_namespace_exports_manifest_types():
    assert thor.EnsembleModel is thor.ensembles.EnsembleModel
    assert "EnsembleModel" in dir(thor.ensembles)
    assert "EnsembleMemberSpec" in dir(thor.ensembles)
    assert "EnsembleAggregation" in dir(thor.ensembles)
