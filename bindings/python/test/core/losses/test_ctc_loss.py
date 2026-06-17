import math

import numpy as np
import pytest
import thor


_FP32 = thor.DataType.fp32
_INT32 = thor.DataType.int32


def _net(name="test_net_ctc_loss"):
    return thor.Network(name)


def _tensor(dims, dtype):
    return thor.Tensor(list(dims), dtype)


def _cpu_tensor(values: np.ndarray, dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    values = np.asarray(values, dtype=thor.physical.numpy_dtypes.from_thor(dtype), order="C")
    placement = thor.physical.Placement(thor.physical.DeviceType.cpu, 0)
    descriptor = thor.physical.PhysicalTensor.Descriptor(dtype, list(values.shape))
    tensor = thor.physical.PhysicalTensor(placement, descriptor)
    tensor.numpy()[...] = values
    return tensor


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=-1, keepdims=True)


def _log_add_exp(a: float, b: float) -> float:
    if math.isinf(a) and a < 0:
        return b
    if math.isinf(b) and b < 0:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _ctc_loss_one_sample(probabilities: np.ndarray, labels: np.ndarray, blank: int = 0) -> float:
    """Log-domain CTC reference for one already-softmaxed sample."""
    t_steps = probabilities.shape[0]
    extended = [blank]
    for label in labels.tolist():
        extended.append(int(label))
        extended.append(blank)
    states = len(extended)

    log_alpha = np.full((t_steps, states), -np.inf, dtype=np.float64)
    log_alpha[0, 0] = math.log(probabilities[0, blank])
    if states > 1:
        log_alpha[0, 1] = math.log(probabilities[0, extended[1]])

    for t in range(1, t_steps):
        for s in range(states):
            total = log_alpha[t - 1, s]
            if s > 0:
                total = _log_add_exp(total, log_alpha[t - 1, s - 1])
            if s > 1 and extended[s] != blank and extended[s] != extended[s - 2]:
                total = _log_add_exp(total, log_alpha[t - 1, s - 2])
            log_alpha[t, s] = total + math.log(probabilities[t, extended[s]])

    if states == 1:
        log_likelihood = log_alpha[t_steps - 1, 0]
    else:
        log_likelihood = _log_add_exp(log_alpha[t_steps - 1, states - 1], log_alpha[t_steps - 1, states - 2])
    return -log_likelihood


def _ctc_reference(logits: np.ndarray, labels: np.ndarray, label_lengths: np.ndarray, input_lengths: np.ndarray) -> np.ndarray:
    probabilities = _softmax(logits)
    losses = []
    for b in range(logits.shape[0]):
        valid_t = int(input_lengths[b, 0])
        valid_l = int(label_lengths[b, 0])
        losses.append(_ctc_loss_one_sample(probabilities[b, :valid_t], labels[b, :valid_l]))
    return np.asarray(losses, dtype=np.float32).reshape(logits.shape[0], 1)


def _reduce_loss(raw: np.ndarray, reported_loss_shape: thor.losses.LossShape) -> np.ndarray:
    if reported_loss_shape in (thor.losses.LossShape.raw, thor.losses.LossShape.elementwise):
        return raw.astype(np.float32)
    if reported_loss_shape == thor.losses.LossShape.batch:
        return np.array([[np.sum(raw) / raw.shape[0]]], dtype=np.float32)
    raise AssertionError(f"Unhandled loss shape: {reported_loss_shape}")


def test_ctc_loss_is_exposed_in_loss_namespace():
    assert hasattr(thor.losses, "CTCLoss")
    assert hasattr(thor.losses, "CTCOobGradientMode")


def test_ctc_loss_constructs_defaults_and_exposes_accessors():
    n = _net()
    logits = _tensor([4, 3], _FP32)
    labels = _tensor([2], _INT32)
    label_lengths = _tensor([1], _INT32)
    input_lengths = _tensor([1], _INT32)

    loss = thor.losses.CTCLoss(n, logits, labels, label_lengths, input_lengths)

    assert isinstance(loss, thor.losses.CTCLoss)
    assert isinstance(loss, thor.losses.Loss)
    assert loss.get_predictions() == logits
    assert loss.get_labels() == labels
    assert loss.get_label_lengths() == label_lengths
    assert loss.get_input_lengths() == input_lengths
    assert loss.max_label_length == 2
    assert loss.loss_weight is None
    assert loss.oob_gradient_mode == thor.losses.CTCOobGradientMode.zero


def test_ctc_loss_constructs_with_raw_shape_loss_weight_and_skip_oob_mode():
    n = _net()
    logits = _tensor([5, 4], _FP32)
    labels = _tensor([3], _INT32)
    label_lengths = _tensor([1], _INT32)
    input_lengths = _tensor([1], _INT32)

    loss = thor.losses.CTCLoss(
        n,
        logits,
        labels,
        label_lengths,
        input_lengths,
        None,
        thor.losses.LossShape.raw,
        thor.losses.CTCOobGradientMode.skip,
        loss_weight=2.5,
    )

    assert isinstance(loss, thor.losses.CTCLoss)
    assert loss.max_label_length == 3
    assert loss.loss_weight == pytest.approx(2.5)
    assert loss.oob_gradient_mode == thor.losses.CTCOobGradientMode.skip


@pytest.mark.parametrize("shape", [thor.losses.LossShape.batch, thor.losses.LossShape.elementwise, thor.losses.LossShape.raw])
def test_ctc_loss_reported_loss_shape_variants_construct(shape):
    n = _net(f"test_net_ctc_loss_shape_{shape}")
    loss = thor.losses.CTCLoss(
        n,
        _tensor([4, 3], _FP32),
        _tensor([2], _INT32),
        _tensor([1], _INT32),
        _tensor([1], _INT32),
        None,
        shape,
    )
    assert isinstance(loss, thor.losses.CTCLoss)


def test_ctc_loss_rejects_classwise_reported_loss_shape():
    n = _net()
    with pytest.raises(ValueError, match=r"reported_loss_shape must be batch, elementwise, or raw"):
        thor.losses.CTCLoss(
            n,
            _tensor([4, 3], _FP32),
            _tensor([2], _INT32),
            _tensor([1], _INT32),
            _tensor([1], _INT32),
            None,
            thor.losses.LossShape.classwise,
        )


def test_ctc_loss_rejects_unsupported_shapes_and_dtypes():
    n = _net()
    labels = _tensor([2], _INT32)
    label_lengths = _tensor([1], _INT32)
    input_lengths = _tensor([1], _INT32)

    with pytest.raises(ValueError, match=r"logits must use fp32 dtype"):
        thor.losses.CTCLoss(n, _tensor([4, 3], thor.DataType.fp16), labels, label_lengths, input_lengths)

    with pytest.raises(ValueError, match=r"logits must have dimensions \[time, classes\]"):
        thor.losses.CTCLoss(n, _tensor([4], _FP32), labels, label_lengths, input_lengths)

    with pytest.raises(ValueError, match=r"labels must use int32 dtype"):
        thor.losses.CTCLoss(n, _tensor([4, 3], _FP32), _tensor([2], thor.DataType.uint32), label_lengths, input_lengths)

    with pytest.raises(ValueError, match=r"labels must have dimensions \[max_label_length\]"):
        thor.losses.CTCLoss(n, _tensor([4, 3], _FP32), _tensor([1, 1], _INT32), label_lengths, input_lengths)

    with pytest.raises(ValueError, match=r"label_lengths must have dimensions \[1\]"):
        thor.losses.CTCLoss(n, _tensor([4, 3], _FP32), labels, _tensor([2], _INT32), input_lengths)

    with pytest.raises(ValueError, match=r"input_lengths must use int32 dtype"):
        thor.losses.CTCLoss(n, _tensor([4, 3], _FP32), labels, label_lengths, _tensor([1], thor.DataType.uint32))

    with pytest.raises(ValueError, match=r"loss_data_type must be fp32"):
        thor.losses.CTCLoss(n, _tensor([4, 3], _FP32), labels, label_lengths, input_lengths, thor.DataType.fp16)


def test_ctc_loss_rejects_label_length_greater_than_time_and_duplicate_tensors():
    n = _net()
    logits = _tensor([2, 3], _FP32)
    labels = _tensor([3], _INT32)
    label_lengths = _tensor([1], _INT32)
    input_lengths = _tensor([1], _INT32)

    with pytest.raises(ValueError, match=r"max_label_length 3 must be less than or equal to logits time dimension 2"):
        thor.losses.CTCLoss(n, logits, labels, label_lengths, input_lengths)

    n = _net("test_net_ctc_duplicate")
    shared_lengths = _tensor([1], _INT32)
    with pytest.raises(ValueError, match=r"must be distinct tensors"):
        thor.losses.CTCLoss(n, _tensor([4, 3], _FP32), _tensor([2], _INT32), shared_lengths, shared_lengths)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "reported_loss_shape",
    [thor.losses.LossShape.raw, thor.losses.LossShape.elementwise, thor.losses.LossShape.batch],
)
def test_ctc_loss_python_api_forward_matches_cpu_reference(reported_loss_shape):
    logits = np.array(
        [
            [
                [2.0, -1.0, 0.3],
                [0.1, 1.7, -0.4],
                [-0.2, 0.4, 1.5],
                [1.0, 0.1, -0.7],
            ],
            [
                [0.3, 1.2, -0.8],
                [1.4, -0.2, 0.1],
                [-0.5, 0.2, 1.1],
                [9.0, -7.0, 3.0],  # ignored because input_lengths[1] == 3
            ],
        ],
        dtype=np.float32,
    )
    labels = np.array(
        [
            [1, 2],
            [2, 99],  # padding is ignored because label_lengths[1] == 1
        ],
        dtype=np.int32,
    )
    label_lengths = np.array([[2], [1]], dtype=np.int32)
    input_lengths = np.array([[4], [3]], dtype=np.int32)

    n = _net(f"test_net_ctc_loss_python_forward_{reported_loss_shape}")
    logits_input = thor.layers.NetworkInput(n, "logits", list(logits.shape[1:]), _FP32)
    labels_input = thor.layers.NetworkInput(n, "labels", list(labels.shape[1:]), _INT32)
    label_lengths_input = thor.layers.NetworkInput(n, "label_lengths", [1], _INT32)
    input_lengths_input = thor.layers.NetworkInput(n, "input_lengths", [1], _INT32)

    loss = thor.losses.CTCLoss(
        n,
        logits_input.get_feature_output(),
        labels_input.get_feature_output(),
        label_lengths_input.get_feature_output(),
        input_lengths_input.get_feature_output(),
        None,
        reported_loss_shape,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), _FP32)

    placed = n.place(
        logits.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer(
        {
            "logits": _cpu_tensor(logits, _FP32),
            "labels": _cpu_tensor(labels, _INT32),
            "label_lengths": _cpu_tensor(label_lengths, _INT32),
            "input_lengths": _cpu_tensor(input_lengths, _INT32),
        }
    )
    actual = np.array(outputs["loss"].numpy(), copy=True)
    expected = _reduce_loss(_ctc_reference(logits, labels, label_lengths, input_lengths), reported_loss_shape)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-5)


def _run_python_ctc_forward(
    *,
    test_name: str,
    logits: np.ndarray,
    labels: np.ndarray,
    label_lengths: np.ndarray,
    input_lengths: np.ndarray,
    reported_loss_shape: thor.losses.LossShape = thor.losses.LossShape.raw,
    oob_gradient_mode: thor.losses.CTCOobGradientMode = thor.losses.CTCOobGradientMode.zero,
    loss_weight: float | None = None,
) -> np.ndarray:
    n = _net(test_name)
    logits_input = thor.layers.NetworkInput(n, "logits", list(logits.shape[1:]), _FP32)
    labels_input = thor.layers.NetworkInput(n, "labels", list(labels.shape[1:]), _INT32)
    label_lengths_input = thor.layers.NetworkInput(n, "label_lengths", [1], _INT32)
    input_lengths_input = thor.layers.NetworkInput(n, "input_lengths", [1], _INT32)

    loss = thor.losses.CTCLoss(
        n,
        logits_input.get_feature_output(),
        labels_input.get_feature_output(),
        label_lengths_input.get_feature_output(),
        input_lengths_input.get_feature_output(),
        None,
        reported_loss_shape,
        oob_gradient_mode,
        loss_weight=loss_weight,
    )
    thor.layers.NetworkOutput(n, "loss", loss.get_loss(), _FP32)

    placed = n.place(
        logits.shape[0],
        inference_only=True,
        forced_devices=[0],
        forced_num_stamps_per_gpu=1,
    )
    outputs = placed.infer(
        {
            "logits": _cpu_tensor(logits, _FP32),
            "labels": _cpu_tensor(labels, _INT32),
            "label_lengths": _cpu_tensor(label_lengths, _INT32),
            "input_lengths": _cpu_tensor(input_lengths, _INT32),
        }
    )
    return np.array(outputs["loss"].numpy(), copy=True)


def _ctc_reference_with_options(
    logits: np.ndarray,
    labels: np.ndarray,
    label_lengths: np.ndarray,
    input_lengths: np.ndarray,
    *,
    reported_loss_shape: thor.losses.LossShape = thor.losses.LossShape.raw,
    loss_weight: float | None = None,
) -> np.ndarray:
    raw = _ctc_reference(logits, labels, label_lengths, input_lengths)
    if loss_weight is not None:
        raw = raw * np.float32(loss_weight)
    return _reduce_loss(raw, reported_loss_shape)


def _ctc_reference_scalar_sum(
    logits: np.ndarray,
    labels: np.ndarray,
    label_lengths: np.ndarray,
    input_lengths: np.ndarray,
    *,
    loss_weight: float = 1.0,
) -> float:
    probabilities = _softmax(np.asarray(logits, dtype=np.float64))
    total = 0.0
    for b in range(probabilities.shape[0]):
        valid_t = int(input_lengths[b, 0])
        valid_l = int(label_lengths[b, 0])
        total += _ctc_loss_one_sample(probabilities[b, :valid_t], labels[b, :valid_l])
    return float(total * loss_weight)


def _finite_difference_ctc_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    label_lengths: np.ndarray,
    input_lengths: np.ndarray,
    *,
    loss_weight: float = 1.0,
    epsilon: float = 1.0e-3,
) -> np.ndarray:
    logits64 = np.asarray(logits, dtype=np.float64).copy()
    gradient = np.zeros_like(logits64)
    for index in np.ndindex(*logits64.shape):
        original = logits64[index]
        logits64[index] = original + epsilon
        plus = _ctc_reference_scalar_sum(logits64, labels, label_lengths, input_lengths, loss_weight=loss_weight)
        logits64[index] = original - epsilon
        minus = _ctc_reference_scalar_sum(logits64, labels, label_lengths, input_lengths, loss_weight=loss_weight)
        logits64[index] = original
        gradient[index] = (plus - minus) / (2.0 * epsilon)
    return gradient.astype(np.float32)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "single_timestep_cross_entropy_equivalent",
            "logits": np.array([[[0.0, 0.0]]], dtype=np.float32),
            "labels": np.array([[1]], dtype=np.int32),
            "label_lengths": np.array([[1]], dtype=np.int32),
            "input_lengths": np.array([[1]], dtype=np.int32),
        },
        {
            "name": "repeated_labels_need_blanks",
            "logits": np.array(
                [
                    [
                        [2.1, -0.5, 0.3],
                        [0.2, 1.4, -0.6],
                        [1.2, -0.1, 0.4],
                        [-0.3, 1.7, 0.2],
                        [0.7, 0.1, -0.2],
                    ]
                ],
                dtype=np.float32,
            ),
            "labels": np.array([[1, 1]], dtype=np.int32),
            "label_lengths": np.array([[2]], dtype=np.int32),
            "input_lengths": np.array([[5]], dtype=np.int32),
        },
        {
            "name": "two_sample_variable_input_and_label_lengths",
            "logits": np.array(
                [
                    [
                        [1.2, -0.4, 0.6, -1.0],
                        [0.1, 1.5, -0.7, 0.2],
                        [-0.3, 0.4, 1.8, -0.5],
                        [0.9, -0.2, 0.1, 1.1],
                        [1.4, 0.2, -1.3, 0.0],
                    ],
                    [
                        [0.4, 1.1, -0.8, 0.3],
                        [1.6, -0.5, 0.2, -0.1],
                        [-0.6, 0.5, 1.0, -0.4],
                        [8.0, -7.0, 4.0, 3.0],
                        [-6.0, 5.0, -4.0, 2.0],
                    ],
                ],
                dtype=np.float32,
            ),
            "labels": np.array([[1, 2, 3], [2, 99, 77]], dtype=np.int32),
            "label_lengths": np.array([[3], [1]], dtype=np.int32),
            "input_lengths": np.array([[5], [3]], dtype=np.int32),
        },
    ],
    ids=lambda c: c["name"],
)
def test_ctc_loss_python_api_forward_numerical_cases(case):
    for reported_loss_shape in (
        thor.losses.LossShape.raw,
        thor.losses.LossShape.elementwise,
        thor.losses.LossShape.batch,
    ):
        actual = _run_python_ctc_forward(
            test_name=f"test_net_ctc_python_numerical_{case['name']}_{reported_loss_shape}",
            logits=case["logits"],
            labels=case["labels"],
            label_lengths=case["label_lengths"],
            input_lengths=case["input_lengths"],
            reported_loss_shape=reported_loss_shape,
        )
        expected = _ctc_reference_with_options(
            case["logits"],
            case["labels"],
            case["label_lengths"],
            case["input_lengths"],
            reported_loss_shape=reported_loss_shape,
        )
        np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-5)


@pytest.mark.cuda
def test_ctc_loss_python_api_forward_ignores_padded_label_suffix_values():
    logits = np.array(
        [
            [
                [0.3, 1.1, -0.2],
                [1.4, -0.4, 0.3],
                [-0.5, 0.6, 1.2],
                [0.9, -0.1, 0.0],
            ],
            [
                [0.1, -0.7, 1.2],
                [1.3, 0.2, -0.5],
                [-0.4, 1.0, 0.1],
                [0.5, -0.2, 0.8],
            ],
        ],
        dtype=np.float32,
    )
    labels_a = np.array([[1, 2, 99], [2, 88, 77]], dtype=np.int32)
    labels_b = np.array([[1, 2, 11], [2, 22, 33]], dtype=np.int32)
    label_lengths = np.array([[2], [1]], dtype=np.int32)
    input_lengths = np.array([[4], [3]], dtype=np.int32)

    actual_a = _run_python_ctc_forward(
        test_name="test_net_ctc_python_padding_a",
        logits=logits,
        labels=labels_a,
        label_lengths=label_lengths,
        input_lengths=input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    actual_b = _run_python_ctc_forward(
        test_name="test_net_ctc_python_padding_b",
        logits=logits,
        labels=labels_b,
        label_lengths=label_lengths,
        input_lengths=input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    expected = _ctc_reference_with_options(
        logits,
        labels_a,
        label_lengths,
        input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
    )

    np.testing.assert_allclose(actual_a, expected, rtol=2.0e-4, atol=2.0e-5)
    np.testing.assert_allclose(actual_b, expected, rtol=2.0e-4, atol=2.0e-5)
    np.testing.assert_allclose(actual_a, actual_b, rtol=2.0e-4, atol=2.0e-5)


@pytest.mark.cuda
@pytest.mark.parametrize("loss_weight", [0.25, 2.5])
def test_ctc_loss_python_api_loss_weight_scales_forward_numerically(loss_weight):
    logits = np.array(
        [
            [
                [2.0, -0.3, 0.5],
                [0.1, 1.7, -0.4],
                [0.8, -0.2, 1.1],
            ]
        ],
        dtype=np.float32,
    )
    labels = np.array([[1, 2]], dtype=np.int32)
    label_lengths = np.array([[2]], dtype=np.int32)
    input_lengths = np.array([[3]], dtype=np.int32)

    actual = _run_python_ctc_forward(
        test_name=f"test_net_ctc_python_loss_weight_{loss_weight}",
        logits=logits,
        labels=labels,
        label_lengths=label_lengths,
        input_lengths=input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
        loss_weight=loss_weight,
    )
    expected = _ctc_reference_with_options(
        logits,
        labels,
        label_lengths,
        input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
        loss_weight=loss_weight,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-5)


@pytest.mark.cuda
def test_ctc_loss_python_api_oob_gradient_modes_match_forward_for_valid_examples():
    logits = np.array(
        [
            [
                [0.3, 0.9, -0.6],
                [1.0, -0.4, 0.2],
                [-0.2, 1.1, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    labels = np.array([[1, 1]], dtype=np.int32)
    label_lengths = np.array([[2]], dtype=np.int32)
    input_lengths = np.array([[3]], dtype=np.int32)

    zero = _run_python_ctc_forward(
        test_name="test_net_ctc_python_oob_zero",
        logits=logits,
        labels=labels,
        label_lengths=label_lengths,
        input_lengths=input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
        oob_gradient_mode=thor.losses.CTCOobGradientMode.zero,
    )
    skip = _run_python_ctc_forward(
        test_name="test_net_ctc_python_oob_skip",
        logits=logits,
        labels=labels,
        label_lengths=label_lengths,
        input_lengths=input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
        oob_gradient_mode=thor.losses.CTCOobGradientMode.skip,
    )
    expected = _ctc_reference_with_options(
        logits,
        labels,
        label_lengths,
        input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
    )

    np.testing.assert_allclose(zero, expected, rtol=2.0e-4, atol=2.0e-5)
    np.testing.assert_allclose(skip, expected, rtol=2.0e-4, atol=2.0e-5)


def test_ctc_cpu_reference_backward_matches_single_timestep_cross_entropy_gradient():
    logits = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, -0.5, 0.25]],
        ],
        dtype=np.float32,
    )
    labels = np.array([[1], [2]], dtype=np.int32)
    label_lengths = np.array([[1], [1]], dtype=np.int32)
    input_lengths = np.array([[1], [1]], dtype=np.int32)
    loss_weight = 1.75

    actual = _finite_difference_ctc_gradient(
        logits,
        labels,
        label_lengths,
        input_lengths,
        loss_weight=loss_weight,
        epsilon=1.0e-3,
    )
    probabilities = _softmax(logits)
    expected = probabilities.copy()
    expected[0, 0, 1] -= 1.0
    expected[1, 0, 2] -= 1.0
    expected *= loss_weight

    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=2.5e-3, atol=2.5e-3)


def test_ctc_cpu_reference_backward_zeroes_trailing_timesteps_by_input_length():
    logits = np.array(
        [
            [
                [0.4, 1.0, -0.2],
                [1.2, -0.5, 0.3],
                [-2.0, 3.0, 0.7],  # ignored by input_lengths
            ]
        ],
        dtype=np.float32,
    )
    labels = np.array([[1]], dtype=np.int32)
    label_lengths = np.array([[1]], dtype=np.int32)
    input_lengths = np.array([[2]], dtype=np.int32)

    actual = _finite_difference_ctc_gradient(
        logits,
        labels,
        label_lengths,
        input_lengths,
        epsilon=1.0e-3,
    )

    assert np.any(np.abs(actual[:, :2, :]) > 1.0e-5)
    np.testing.assert_allclose(actual[:, 2:, :], 0.0, rtol=0.0, atol=1.0e-7)


@pytest.mark.cuda
def test_ctc_loss_python_api_forward_directional_finite_difference_matches_cpu_reference():
    logits = np.array(
        [
            [
                [0.5, 1.1, -0.8],
                [1.3, -0.4, 0.2],
                [-0.2, 0.7, 1.4],
                [0.9, -0.3, 0.1],
            ]
        ],
        dtype=np.float32,
    )
    direction = np.array(
        [
            [
                [0.2, -0.1, 0.3],
                [-0.4, 0.2, 0.1],
                [0.1, 0.3, -0.2],
                [-0.3, 0.1, 0.2],
            ]
        ],
        dtype=np.float32,
    )
    labels = np.array([[1, 2]], dtype=np.int32)
    label_lengths = np.array([[2]], dtype=np.int32)
    input_lengths = np.array([[4]], dtype=np.int32)
    epsilon = 1.0e-3

    actual_plus = _run_python_ctc_forward(
        test_name="test_net_ctc_python_directional_plus",
        logits=logits + epsilon * direction,
        labels=labels,
        label_lengths=label_lengths,
        input_lengths=input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    actual_minus = _run_python_ctc_forward(
        test_name="test_net_ctc_python_directional_minus",
        logits=logits - epsilon * direction,
        labels=labels,
        label_lengths=label_lengths,
        input_lengths=input_lengths,
        reported_loss_shape=thor.losses.LossShape.raw,
    )
    actual_directional = (float(actual_plus[0, 0]) - float(actual_minus[0, 0])) / (2.0 * epsilon)

    expected_gradient = _finite_difference_ctc_gradient(logits, labels, label_lengths, input_lengths, epsilon=epsilon)
    expected_directional = float(np.sum(expected_gradient * direction))

    assert actual_directional == pytest.approx(expected_directional, rel=3.0e-3, abs=3.0e-3)
