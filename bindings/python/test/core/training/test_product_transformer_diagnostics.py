"""Focused Thor placement diagnostics for the ProductTransformerForecaster topology.

These tests intentionally split the production graph into small rungs.  When a
real TrainingRuns failure loses its exception, the first failing test identifies
whether the problem is in the bounded dense head, the ragged encoder block, or
the dense-query/ragged-KV decoder block.
"""

import math

import pytest
import thor


BATCH_SIZE = 2
HISTORY_CAPACITY = 32
MODEL_WIDTH = 16
FFN_WIDTH = 32
FUTURE_DAYS = 100
HISTORY_BOUNDARY = 819


def _bound_log_rate_residual(network: thor.Network, raw: thor.Tensor, bound: float) -> thor.Tensor:
    inverse_bound = 1.0 / float(bound)

    def normalize(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        scale = thor.physical.Expression.constant_scalar(inverse_bound)
        return {"normalized": context.input("raw") * scale}

    normalized = thor.layers.CustomLayer(
        network=network,
        inputs={"raw": raw},
        output_names=["normalized"],
        build=normalize,
        parameters=[],
    )["normalized"]
    unit_bounded = thor.activations.Tanh().add_to_network(network, normalized)

    def rescale(context: thor.layers.CustomLayerBuildContext) -> dict[str, thor.physical.Expression]:
        scale = thor.physical.Expression.constant_scalar(float(bound))
        return {"bounded": context.input("unit_bounded") * scale}

    return thor.layers.CustomLayer(
        network=network,
        inputs={"unit_bounded": unit_bounded},
        output_names=["bounded"],
        build=rescale,
        parameters=[],
    )["bounded"]


def _diagnostic_optimizer():
    # Direct training placement requires every trainable parameter to already
    # have an optimizer. Use an effectively no-op SGD here: these diagnostics
    # exercise graph construction/backward placement, not optimizer behavior.
    return thor.optimizers.Sgd(initial_learning_rate=1.0e-12, momentum=0.0)


def _place_training(network: thor.Network, batch_size: int = BATCH_SIZE):
    placed = network.place(
        batch_size,
        inference_only=False,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    assert placed is not None
    return placed


def _ffn_residual(network: thor.Network, feature_input, optimizer):
    norm = thor.layers.RMSNorm(
        network,
        feature_input,
        epsilon=1.0e-5,
        parameter_data_type=thor.DataType.fp32,
        weights_optimizer=optimizer,
    ).get_feature_output()
    gate_and_value = thor.layers.FullyConnected(
        network,
        norm,
        2 * FFN_WIDTH,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        weights_optimizer=optimizer,
        biases_optimizer=optimizer,
    ).get_feature_output()
    hidden = thor.activations.Swiglu().add_to_network(network, gate_and_value)
    hidden = thor.layers.DropOut(network, hidden, 0.10).get_feature_output()
    return thor.layers.FullyConnected(
        network,
        hidden,
        MODEL_WIDTH,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        output_dropout_probability=0.10,
        output_dropout_seed=41,
        residual_input=feature_input,
        weights_optimizer=optimizer,
        biases_optimizer=optimizer,
    ).get_feature_output()


@pytest.mark.cuda
def test_product_transformer_bounded_dense_head_places_for_training_with_poisson_backward():
    """Isolate FC -> scalar expression -> Tanh -> scalar expression -> Poisson backward."""

    network = thor.Network("diagnostic_product_transformer_bounded_dense_head")
    optimizer = _diagnostic_optimizer()
    features = thor.layers.NetworkInput(
        network, "features", [FUTURE_DAYS, MODEL_WIDTH], thor.DataType.bf16
    ).get_feature_output()
    raw = thor.layers.FullyConnected(
        network,
        features,
        1,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.fp32,
        weights_optimizer=optimizer,
        biases_optimizer=optimizer,
    ).get_feature_output()
    bounded = _bound_log_rate_residual(network, raw, math.log(8.0))
    log_rates = thor.layers.Reshape(network, bounded, [FUTURE_DAYS]).get_feature_output()
    labels = thor.layers.NetworkInput(
        network, "labels", [FUTURE_DAYS], thor.DataType.fp32
    ).get_feature_output()
    loss = thor.losses.distribution.PoissonNLLLoss(
        network,
        log_rates,
        labels,
        log_input=True,
        full=False,
        eps=1.0e-8,
        loss_data_type=thor.DataType.fp32,
        loss_weight=1.0 / FUTURE_DAYS,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)

    placed = _place_training(network)
    assert set(placed.get_network_input_names()) == {"features", "labels"}




def _attach_ragged_mean_loss(network: thor.Network, ragged_values, label_name: str = "labels"):
    """Attach a dense per-row loss so training placement must differentiate through ragged_values."""
    pooled = thor.layers.SegmentedReduction(
        network, ragged_values, thor.layers.SegmentedReduction.Type.mean
    ).get_feature_output()
    labels = thor.layers.NetworkInput(
        network, label_name, [MODEL_WIDTH], thor.DataType.bf16
    ).get_feature_output()
    loss = thor.losses.MSE(
        network,
        pooled,
        labels,
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)


def _ragged_attention_for_diagnostic(
    network: thor.Network,
    history,
    origins,
    optimizer,
    *,
    use_residual: bool,
    output_dropout_probability: float,
    sdpa_dropout_probability: float = 0.10,
):
    kwargs = dict(
        num_heads=2,
        head_dim=MODEL_WIDTH // 2,
        output_features=MODEL_WIDTH,
        has_bias=False,
        mask_kind="none",
        use_rope=True,
        rope_rotary_dim=MODEL_WIDTH // 2,
        rope_query_position_offsets=origins,
        rope_key_position_offsets=origins,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        rope_in_place=True,
        sdpa_dropout_probability=sdpa_dropout_probability,
        sdpa_dropout_seed=101,
        output_dropout_probability=output_dropout_probability,
        output_dropout_seed=103,
        optimizer=optimizer,
    )
    if use_residual:
        kwargs["residual_input"] = history
    return thor.layers.Attention(network, history, **kwargs).get_feature_output()


@pytest.mark.cuda
def test_product_transformer_ragged_segmented_reduction_backward_baseline_places_for_training():
    """Prove the ragged reduction/loss backward boundary is not the failing component."""

    network = thor.Network("diagnostic_product_transformer_ragged_reduction_baseline")
    optimizer = _diagnostic_optimizer()
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.bf16,
        [MODEL_WIDTH],
        max_total_values=HISTORY_CAPACITY,
        batch_size=BATCH_SIZE,
        offsets_data_type=thor.DataType.uint32,
    )
    projected = thor.layers.FullyConnected(
        network,
        history,
        MODEL_WIDTH,
        False,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        weights_optimizer=optimizer,
    ).get_feature_output()
    _attach_ragged_mean_loss(network, projected)
    _place_training(network)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("use_residual", "sdpa_dropout_probability", "output_dropout_probability"),
    [
        pytest.param(False, 0.0, 0.0, id="plain"),
        pytest.param(False, 0.10, 0.0, id="sdpa_dropout_only"),
        pytest.param(True, 0.0, 0.0, id="residual_only"),
        pytest.param(False, 0.0, 0.10, id="output_dropout_only"),
        pytest.param(True, 0.0, 0.10, id="residual_plus_output_dropout"),
        pytest.param(True, 0.10, 0.10, id="production_all"),
    ],
)
def test_product_transformer_ragged_attention_postop_variants_place_for_training(
    use_residual, sdpa_dropout_probability, output_dropout_probability
):
    """Pinpoint whether ragged Attention residual/output-dropout fusion breaks backward placement."""

    suffix = f"residual_{int(use_residual)}_dropout_{int(output_dropout_probability > 0.0)}"
    network = thor.Network(f"diagnostic_product_transformer_ragged_attention_{suffix}")
    optimizer = _diagnostic_optimizer()
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.bf16,
        [MODEL_WIDTH],
        max_total_values=HISTORY_CAPACITY,
        batch_size=BATCH_SIZE,
        offsets_data_type=thor.DataType.uint32,
    )
    origins = thor.layers.NetworkInput(
        network, "history_origins", [1], thor.DataType.int32
    ).get_feature_output()
    encoded = _ragged_attention_for_diagnostic(
        network,
        history,
        origins,
        optimizer,
        use_residual=use_residual,
        output_dropout_probability=output_dropout_probability,
        sdpa_dropout_probability=sdpa_dropout_probability,
    )
    _attach_ragged_mean_loss(network, encoded)
    _place_training(network)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("hidden_dropout_probability", "output_dropout_probability"),
    [
        pytest.param(0.0, 0.0, id="residual_only"),
        pytest.param(0.10, 0.0, id="hidden_dropout_plus_residual"),
        pytest.param(0.10, 0.10, id="production_ffn"),
    ],
)
def test_product_transformer_ragged_ffn_fused_fc_postop_places_for_training(
    hidden_dropout_probability, output_dropout_probability
):
    """Exercise the ragged SwiGLU/dropout/FC residual path independently of Attention."""

    suffix = f"dropout_{int(output_dropout_probability > 0.0)}"
    network = thor.Network(f"diagnostic_product_transformer_ragged_ffn_{suffix}")
    optimizer = _diagnostic_optimizer()
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.bf16,
        [MODEL_WIDTH],
        max_total_values=HISTORY_CAPACITY,
        batch_size=BATCH_SIZE,
        offsets_data_type=thor.DataType.uint32,
    )
    gate_and_value = thor.layers.FullyConnected(
        network,
        history,
        2 * FFN_WIDTH,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        weights_optimizer=optimizer,
        biases_optimizer=optimizer,
    ).get_feature_output()
    hidden = thor.activations.Swiglu().add_to_network(network, gate_and_value)
    if hidden_dropout_probability > 0.0:
        hidden = thor.layers.DropOut(network, hidden, hidden_dropout_probability).get_feature_output()
    projected = thor.layers.FullyConnected(
        network,
        hidden,
        MODEL_WIDTH,
        True,
        activation=None,
        preserve_prefix_dimensions=True,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        output_dropout_probability=output_dropout_probability,
        output_dropout_seed=107,
        residual_input=history,
        weights_optimizer=optimizer,
        biases_optimizer=optimizer,
    ).get_feature_output()
    _attach_ragged_mean_loss(network, projected)
    _place_training(network)


@pytest.mark.cuda
def test_product_transformer_ragged_encoder_block_with_fused_residual_dropout_places_for_training():
    """Exercise the exact ragged self-attention + FFN residual/dropout topology."""

    network = thor.Network("diagnostic_product_transformer_ragged_encoder_block")
    optimizer = _diagnostic_optimizer()
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.bf16,
        [MODEL_WIDTH],
        max_total_values=HISTORY_CAPACITY,
        batch_size=BATCH_SIZE,
        offsets_data_type=thor.DataType.uint32,
    )
    origins = thor.layers.NetworkInput(
        network, "history_origins", [1], thor.DataType.int32
    ).get_feature_output()

    attention_norm = thor.layers.RMSNorm(
        network,
        history,
        epsilon=1.0e-5,
        parameter_data_type=thor.DataType.fp32,
        weights_optimizer=optimizer,
    ).get_feature_output()
    encoded = thor.layers.Attention(
        network,
        attention_norm,
        num_heads=2,
        head_dim=MODEL_WIDTH // 2,
        output_features=MODEL_WIDTH,
        has_bias=False,
        mask_kind="none",
        use_rope=True,
        rope_rotary_dim=MODEL_WIDTH // 2,
        rope_query_position_offsets=origins,
        rope_key_position_offsets=origins,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        rope_in_place=True,
        sdpa_dropout_probability=0.10,
        sdpa_dropout_seed=17,
        output_dropout_probability=0.10,
        output_dropout_seed=19,
        residual_input=history,
        optimizer=optimizer,
    ).get_feature_output()
    encoded = _ffn_residual(network, encoded, optimizer)
    pooled = thor.layers.SegmentedReduction(
        network, encoded, thor.layers.SegmentedReduction.Type.mean
    ).get_feature_output()
    labels = thor.layers.NetworkInput(
        network, "labels", [MODEL_WIDTH], thor.DataType.bf16
    ).get_feature_output()
    loss = thor.losses.MSE(
        network,
        pooled,
        labels,
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)

    _place_training(network)


@pytest.mark.cuda
def test_product_transformer_dense_query_ragged_kv_decoder_with_fused_residual_dropout_places_for_training():
    """Exercise future self-attention and dense-Q/ragged-KV cross-attention in training."""

    network = thor.Network("diagnostic_product_transformer_dense_query_ragged_kv_decoder")
    optimizer = _diagnostic_optimizer()
    history = thor.layers.RaggedNetworkInput(
        network,
        "history",
        thor.DataType.bf16,
        [MODEL_WIDTH],
        max_total_values=HISTORY_CAPACITY,
        batch_size=BATCH_SIZE,
        offsets_data_type=thor.DataType.uint32,
    )
    future = thor.layers.NetworkInput(
        network, "future", [FUTURE_DAYS, MODEL_WIDTH], thor.DataType.bf16
    ).get_feature_output()
    origins = thor.layers.NetworkInput(
        network, "history_origins", [1], thor.DataType.int32
    ).get_feature_output()

    self_norm = thor.layers.RMSNorm(
        network,
        future,
        epsilon=1.0e-5,
        parameter_data_type=thor.DataType.fp32,
        weights_optimizer=optimizer,
    ).get_feature_output()
    future_x = thor.layers.Attention(
        network,
        self_norm,
        num_heads=2,
        head_dim=MODEL_WIDTH // 2,
        output_features=MODEL_WIDTH,
        has_bias=False,
        mask_kind="none",
        use_rope=True,
        rope_rotary_dim=MODEL_WIDTH // 2,
        rope_position_offset=HISTORY_BOUNDARY,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        rope_in_place=True,
        sdpa_dropout_probability=0.10,
        sdpa_dropout_seed=23,
        output_dropout_probability=0.10,
        output_dropout_seed=29,
        residual_input=future,
        optimizer=optimizer,
    ).get_feature_output()

    cross_norm = thor.layers.RMSNorm(
        network,
        future_x,
        epsilon=1.0e-5,
        parameter_data_type=thor.DataType.fp32,
        weights_optimizer=optimizer,
    ).get_feature_output()
    future_x = thor.layers.Attention(
        network,
        cross_norm,
        num_heads=2,
        head_dim=MODEL_WIDTH // 2,
        output_features=MODEL_WIDTH,
        has_bias=False,
        mask_kind="none",
        use_rope=True,
        rope_rotary_dim=MODEL_WIDTH // 2,
        rope_query_position_offset=HISTORY_BOUNDARY,
        rope_key_position_offsets=origins,
        context_input=history,
        weights_data_type=thor.DataType.bf16,
        compute_data_type=thor.DataType.fp32,
        output_data_type=thor.DataType.bf16,
        rope_in_place=True,
        sdpa_dropout_probability=0.10,
        sdpa_dropout_seed=31,
        output_dropout_probability=0.10,
        output_dropout_seed=37,
        residual_input=future_x,
        optimizer=optimizer,
    ).get_feature_output()
    future_x = _ffn_residual(network, future_x, optimizer)

    labels = thor.layers.NetworkInput(
        network, "labels", [FUTURE_DAYS, MODEL_WIDTH], thor.DataType.bf16
    ).get_feature_output()
    loss = thor.losses.MSE(
        network,
        future_x,
        labels,
        thor.DataType.fp32,
        thor.losses.LossShape.batch,
    )
    thor.layers.NetworkOutput(network, "loss", loss.get_loss(), thor.DataType.fp32)

    _place_training(network)
