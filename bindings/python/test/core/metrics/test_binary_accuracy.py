# test/test_metrics_binary_accuracy.py
import pytest

import thor


def _make_binary_vectors(n: int = 1):
    net = thor.Network("test_net_binary_accuracy")

    preds = thor.Tensor([n], thor.DataType.fp32)  # shape [n]
    labs = thor.Tensor([n], thor.DataType.fp32)  # or bool / int, depending on your API

    return net, preds, labs


def test_binary_accuracy_constructs():
    net, preds, labs = _make_binary_vectors()

    m = thor.metrics.BinaryAccuracy(net, preds, labs)

    assert m is not None
    assert isinstance(m, thor.metrics.BinaryAccuracy)


def test_binary_accuracy_rejects_wrong_arity():
    net, preds, labs = _make_binary_vectors()

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy(net, preds)  # missing labels

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy(net, preds, labs, 123)  # extra arg


def test_binary_accuracy_rejects_wrong_types():
    net, preds, labs = _make_binary_vectors()

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy("not a network", preds, labs)

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy(net, "not a tensor", labs)

    with pytest.raises(TypeError):
        thor.metrics.BinaryAccuracy(net, preds, "not a tensor")
