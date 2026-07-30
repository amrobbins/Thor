import pytest
import thor


def test_loss_shape_exposes_only_semantic_names():
    loss_shape = thor.losses.LossShape

    assert [shape.name for shape in loss_shape] == ["none", "batch", "per_example", "per_output", "raw"]
    assert loss_shape["per_example"] is loss_shape.per_example
    assert loss_shape["per_output"] is loss_shape.per_output


@pytest.mark.parametrize("legacy_name", ["elementwise", "classwise"])
def test_loss_shape_does_not_expose_legacy_names(legacy_name):
    loss_shape = thor.losses.LossShape

    assert not hasattr(loss_shape, legacy_name)
    with pytest.raises(KeyError):
        loss_shape[legacy_name]


@pytest.mark.parametrize(
    "factory",
    [
        lambda n, predictions, labels: thor.losses.BinaryCrossEntropy(
            n, predictions, labels, reports_per_example_loss=True
        ),
        lambda n, predictions, labels: thor.losses.MSE(n, predictions, labels, reports_per_example_loss=True),
        lambda n, predictions, labels: thor.losses.MAE(n, predictions, labels, reports_per_example_loss=True),
        lambda n, predictions, labels: thor.losses.MAPE(n, predictions, labels, reports_per_example_loss=True),
        lambda n, predictions, labels: thor.losses.MeanPowerError(
            n, predictions, labels, reports_per_example_loss=True
        ),
    ],
)
def test_standardized_loss_constructors_reject_removed_reports_per_example_keyword(factory):
    network = thor.Network("test_standardized_loss_constructor_rejects_legacy_keyword")
    predictions = thor.Tensor([3], thor.DataType.fp32)
    labels = thor.Tensor([3], thor.DataType.fp32)

    with pytest.raises(TypeError):
        factory(network, predictions, labels)


def test_none_disables_the_user_facing_loss_report():
    network = thor.Network("test_none_disables_the_user_facing_loss_report")
    predictions = thor.Tensor([3], thor.DataType.fp32)
    labels = thor.Tensor([3], thor.DataType.fp32)

    loss = thor.losses.MSE(
        network, predictions, labels, reported_loss_shape=thor.losses.LossShape.none
    )

    with pytest.raises(RuntimeError, match="does not expose a reported loss tensor"):
        loss.get_loss()
