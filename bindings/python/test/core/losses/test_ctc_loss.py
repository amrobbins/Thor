import pytest
import thor


def _net(name="test_ctc_loss"):
    return thor.Network(name)


def _input_tensor(network, name, dims, dtype):
    return thor.layers.NetworkInput(network, name, dims, dtype).get_feature_output()


def test_ctc_loss_uses_canonical_ragged_labels():
    n = _net("test_ctc_loss_canonical_ragged_labels")
    logits = _input_tensor(n, "logits", [12, 20], thor.DataType.fp32)
    labels = thor.layers.RaggedNetworkInput(
        n,
        "labels",
        thor.DataType.int32,
        [],
        max_total_values=24,
        batch_size=4,
        offsets_data_type=thor.DataType.uint64,
    )
    input_lengths = _input_tensor(n, "input_lengths", [1], thor.DataType.int32)

    loss = thor.losses.CtcLoss(
        n,
        logits,
        labels,
        input_lengths,
        reported_loss_shape=thor.losses.LossShape.per_example,
        out_of_bounds_gradients="skip",
    )

    assert isinstance(loss, thor.losses.CtcLoss)
    assert isinstance(loss.get_labels(), thor.RaggedTensor)
    assert loss.get_labels().values == labels.values
    assert loss.get_labels().offsets == labels.offsets
    assert loss.get_input_lengths() == input_lengths
    assert loss.get_out_of_bounds_gradients() == "skip"
    assert loss.get_loss().get_dimensions() == [1]


def test_ctc_loss_rejects_noncanonical_public_inputs():
    n = _net("test_ctc_loss_rejects_noncanonical_public_inputs")
    logits = _input_tensor(n, "logits", [12, 20], thor.DataType.fp32)
    dense_labels = _input_tensor(n, "dense_labels", [8], thor.DataType.int32)
    input_lengths = _input_tensor(n, "input_lengths", [1], thor.DataType.int32)

    with pytest.raises(TypeError):
        thor.losses.CtcLoss(n, logits, dense_labels, input_lengths)

    labels = thor.layers.RaggedNetworkInput(
        n,
        "labels",
        thor.DataType.int32,
        [],
        max_total_values=24,
        batch_size=4,
    )
    with pytest.raises(ValueError, match="out_of_bounds_gradients"):
        thor.losses.CtcLoss(n, logits, labels, input_lengths, out_of_bounds_gradients="invalid")

    with pytest.raises(ValueError, match="per_output"):
        thor.losses.CtcLoss(n, logits, labels, input_lengths, reported_loss_shape=thor.losses.LossShape.per_output)
