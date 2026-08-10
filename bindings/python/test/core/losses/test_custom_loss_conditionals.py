import pytest
import thor
from thor.physical import Expression as ex


def _dynamic_from_outputs(outputs):
    definition = thor.physical.ExpressionDefinition.from_outputs(outputs)
    return thor.physical.DynamicExpression.from_expression_definition(definition)


def _conditional_squared_error_loss():
    dtype = thor.DataType.fp32
    predictions = ex.input("predictions", compute_dtype=dtype, output_dtype=dtype)
    labels = ex.input("labels", compute_dtype=dtype, output_dtype=dtype)
    diff = predictions - labels
    predicate = ex.reduce_sum(labels) > ex.constant_scalar(0.0)
    return _dynamic_from_outputs(
        ex.if_else(
            predicate,
            ex.outputs({"loss": diff * diff}),
            ex.outputs({"loss": (diff * diff) * ex.constant_scalar(3.0)}),
        )
    )


def _conditional_squared_error_gradient():
    dtype = thor.DataType.fp32
    predictions = ex.input("predictions", compute_dtype=dtype, output_dtype=dtype)
    labels = ex.input("labels", compute_dtype=dtype, output_dtype=dtype)
    base_gradient = (predictions - labels) * ex.constant_scalar(2.0)
    predicate = ex.reduce_sum(labels) > ex.constant_scalar(0.0)
    return _dynamic_from_outputs(
        ex.if_else(
            predicate,
            ex.outputs({"predictions_grad": base_gradient}),
            ex.outputs({"predictions_grad": base_gradient * ex.constant_scalar(3.0)}),
        )
    )


@pytest.mark.cuda
def test_conditional_custom_loss_training_graph_places_from_python():
    network = thor.Network("python_conditional_custom_loss_training_graph")
    features = thor.layers.NetworkInput(network, "features", [1], thor.DataType.fp32).get_feature_output()
    optimizer = thor.optimizers.Sgd(initial_learning_rate=0.1, momentum=0.0)
    prediction = thor.layers.FullyConnected(
        network,
        features,
        1,
        has_bias=False,
        activation=None,
        weights_optimizer=optimizer,
        weights_data_type=thor.DataType.fp32,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.fp32,
    )
    labels = thor.layers.NetworkInput(network, "labels", [1], thor.DataType.fp32).get_feature_output()

    loss = thor.losses.CustomLoss(
        network,
        _conditional_squared_error_loss(),
        _conditional_squared_error_gradient(),
        prediction.get_feature_output(),
        labels,
        thor.DataType.fp32,
        thor.losses.LossShape.raw,
        loss_weight=0.5,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)

    placed = network.place(
        batch_size=2,
        inference_only=False,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    assert placed is not None
    assert placed.get_num_trainable_layers() >= 1
