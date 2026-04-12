import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes

FP16_RTOL = 5e-2
FP16_ATOL = 5e-2
FP16_BWD_RTOL = 2e-1
FP16_BWD_ATOL = 1.5e-1

FORWARD_CASES = [
    {
        "name": "multibatch_multichannel_stride2",
        "x_shape": (2, 3, 5, 4),
        "w_shape": (4, 3, 3, 2),
        "stride_h": 2,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 0,
        "x_scale": 50.0,
        "w_scale": 40.0,
    },
    {
        "name": "same_padding_square_kernel",
        "x_shape": (1, 2, 4, 4),
        "w_shape": (3, 2, 3, 3),
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "x_scale": 20.0,
        "w_scale": 30.0,
    },
    {
        "name": "rectangular_kernel_and_padding",
        "x_shape": (1, 1, 6, 5),
        "w_shape": (2, 1, 2, 3),
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 1,
        "pad_w": 1,
        "x_scale": 25.0,
        "w_scale": 18.0,
    },
]

BACKWARD_CASES = [
    {
        "name": "single_channel_basic",
        "x": np.array([[[[0.2, -0.1, 0.4], [0.0, 0.3, -0.2], [0.1, -0.4, 0.5]]]], dtype=np.float32),
        "w": np.array([[[[0.25, -0.5], [0.75, 0.1]]]], dtype=np.float32),
        "grad": np.array([[[[1.0, -0.5], [0.25, 0.75]]]], dtype=np.float32),
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 0,
        "pad_w": 0,
        "eps": 1e-2,
    },
    {
        "name":
            "multichannel_stride_padding",
        "x":
            np.array(
                [
                    [
                        [
                            [0.10, -0.20, 0.05, 0.30],
                            [-0.15, 0.25, -0.05, 0.10],
                            [0.20, 0.00, -0.10, 0.15],
                            [0.05, -0.30, 0.35, -0.25],
                        ],
                        [
                            [0.00, 0.10, -0.20, 0.15],
                            [0.25, -0.05, 0.30, -0.10],
                            [-0.15, 0.20, 0.05, -0.30],
                            [0.10, -0.25, 0.15, 0.05],
                        ]
                    ]
                ],
                dtype=np.float32,
            ),
        "w":
            np.array(
                [
                    [[
                        [0.20, -0.10],
                        [0.05, 0.30],
                        [-0.15, 0.10],
                    ], [
                        [0.00, 0.25],
                        [-0.20, 0.15],
                        [0.10, -0.05],
                    ]], [[
                        [-0.10, 0.05],
                        [0.20, -0.25],
                        [0.15, 0.00],
                    ], [
                        [0.30, -0.15],
                        [0.05, 0.10],
                        [-0.20, 0.25],
                    ]]
                ],
                dtype=np.float32,
            ),
        "grad":
            np.array(
                [[[
                    [0.50, -0.25, 0.75],
                    [0.10, 0.20, -0.40],
                ], [
                    [-0.30, 0.15, 0.25],
                    [0.60, -0.10, 0.05],
                ]]],
                dtype=np.float32,
            ),
        "stride_h":
            2,
        "stride_w":
            1,
        "pad_h":
            1,
        "pad_w":
            0,
        "eps":
            1e-2,
    },
]

FAN_CASES = [
    {
        "w_shape": (5, 3, 3, 2),
        "expected_fan_in": 18,
        "expected_fan_out": 30
    },
    {
        "w_shape": (7, 4, 1, 5),
        "expected_fan_in": 20,
        "expected_fan_out": 35
    },
]


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


def _numpy_storage_dtype(dtype: thor.DataType) -> np.dtype:
    return numpy_dtypes.from_thor(dtype)


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream, gpu_num: int = 0) -> PhysicalTensor:
    cpu = Placement(DeviceType.cpu, 0)
    gpu = Placement(DeviceType.gpu, gpu_num)
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host = PhysicalTensor(cpu, desc)
    host.numpy()[...] = arr
    device = PhysicalTensor(gpu, desc)
    device.copy_from_async(host, stream)
    return device


def _copy_to_host_fp32(device_tensor: PhysicalTensor, stream: Stream) -> np.ndarray:
    host = _cpu_tensor(list(device_tensor.dimensions), thor.DataType.fp32)
    host.copy_from_async(device_tensor, stream)
    stream.synchronize()
    return host.numpy().copy().astype(np.float32)


def _conv2d_output_shape(
        x_shape: tuple[int, int, int, int], w_shape: tuple[int, int, int, int], stride_h: int, stride_w: int,
        pad_h: int, pad_w: int) -> tuple[int, int, int, int]:
    n, _, h, width = x_shape
    k, _, r, s = w_shape
    out_h = (h + 2 * pad_h - r) // stride_h + 1
    out_w = (width + 2 * pad_w - s) // stride_w + 1
    return n, k, out_h, out_w


def _conv2d_nchw_ref(
        x: np.ndarray,
        w: np.ndarray,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0) -> np.ndarray:
    """Reference matching Thor's current GpuConvolution path: true convolution with spatial kernel flip."""
    x32 = x.astype(np.float32)
    w32 = w.astype(np.float32)

    n, c, h, width = x32.shape
    k, c2, r, s = w32.shape
    assert c == c2

    out_h = (h + 2 * pad_h - r) // stride_h + 1
    out_w = (width + 2 * pad_w - s) // stride_w + 1

    xpad = np.pad(x32, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    wflip = np.flip(w32, axis=(2, 3))

    out = np.zeros((n, k, out_h, out_w), dtype=np.float32)
    for ni in range(n):
        for ko in range(k):
            for oh in range(out_h):
                ih = oh * stride_h
                for ow in range(out_w):
                    iw = ow * stride_w
                    window = xpad[ni, :, ih:ih + r, iw:iw + s]
                    out[ni, ko, oh, ow] = np.sum(window * wflip[ko])
    return out


def _conv2d_loss_with_upstream_ref(
        x: np.ndarray,
        w: np.ndarray,
        grad: np.ndarray,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0) -> float:
    return float(
        np.sum(
            _conv2d_nchw_ref(x, w, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w) *
            grad.astype(np.float32)))


def _finite_difference_conv2d_grads(
        x: np.ndarray,
        w: np.ndarray,
        grad: np.ndarray,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0,
        eps: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    x32 = x.astype(np.float32).copy()
    w32 = w.astype(np.float32).copy()
    grad32 = grad.astype(np.float32)

    x_grad = np.zeros_like(x32, dtype=np.float32)
    w_grad = np.zeros_like(w32, dtype=np.float32)

    for idx in np.ndindex(*x32.shape):
        xp = x32.copy()
        xm = x32.copy()
        xp[idx] += eps
        xm[idx] -= eps
        lp = _conv2d_loss_with_upstream_ref(
            xp, w32, grad32, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w)
        lm = _conv2d_loss_with_upstream_ref(
            xm, w32, grad32, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w)
        x_grad[idx] = (lp - lm) / (2.0 * eps)

    for idx in np.ndindex(*w32.shape):
        wp = w32.copy()
        wm = w32.copy()
        wp[idx] += eps
        wm[idx] -= eps
        lp = _conv2d_loss_with_upstream_ref(
            x32, wp, grad32, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w)
        lm = _conv2d_loss_with_upstream_ref(
            x32, wm, grad32, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w)
        w_grad[idx] = (lp - lm) / (2.0 * eps)

    return x_grad, w_grad


def _conv_pointwise_loss_with_upstream_ref(
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        grad: np.ndarray,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0) -> float:
    conv = _conv2d_nchw_ref(x, w, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w)
    y = np.exp(conv + b.astype(np.float32))
    return float(np.sum(y * grad.astype(np.float32)))


def _finite_difference_conv_pointwise_grads(
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        grad: np.ndarray,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0,
        eps: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x32 = x.astype(np.float32).copy()
    w32 = w.astype(np.float32).copy()
    b32 = b.astype(np.float32).copy()
    grad32 = grad.astype(np.float32)

    x_grad = np.zeros_like(x32, dtype=np.float32)
    w_grad = np.zeros_like(w32, dtype=np.float32)
    b_grad = np.zeros_like(b32, dtype=np.float32)

    for idx in np.ndindex(*x32.shape):
        xp = x32.copy()
        xm = x32.copy()
        xp[idx] += eps
        xm[idx] -= eps
        lp = _conv_pointwise_loss_with_upstream_ref(xp, w32, b32, grad32, stride_h, stride_w, pad_h, pad_w)
        lm = _conv_pointwise_loss_with_upstream_ref(xm, w32, b32, grad32, stride_h, stride_w, pad_h, pad_w)
        x_grad[idx] = (lp - lm) / (2.0 * eps)

    for idx in np.ndindex(*w32.shape):
        wp = w32.copy()
        wm = w32.copy()
        wp[idx] += eps
        wm[idx] -= eps
        lp = _conv_pointwise_loss_with_upstream_ref(x32, wp, b32, grad32, stride_h, stride_w, pad_h, pad_w)
        lm = _conv_pointwise_loss_with_upstream_ref(x32, wm, b32, grad32, stride_h, stride_w, pad_h, pad_w)
        w_grad[idx] = (lp - lm) / (2.0 * eps)

    for idx in np.ndindex(*b32.shape):
        bp = b32.copy()
        bm = b32.copy()
        bp[idx] += eps
        bm[idx] -= eps
        lp = _conv_pointwise_loss_with_upstream_ref(x32, w32, bp, grad32, stride_h, stride_w, pad_h, pad_w)
        lm = _conv_pointwise_loss_with_upstream_ref(x32, w32, bm, grad32, stride_h, stride_w, pad_h, pad_w)
        b_grad[idx] = (lp - lm) / (2.0 * eps)

    return x_grad, w_grad, b_grad


def _fp16_exp_expected(pre_exp: np.ndarray) -> np.ndarray:
    pre_exp32 = pre_exp.astype(np.float32)
    fp16_log_max = np.log(np.finfo(np.float16).max).astype(np.float32)
    expected = np.full(pre_exp32.shape, np.inf, dtype=np.float32)
    finite_mask = pre_exp32 <= fp16_log_max
    expected[finite_mask] = np.exp(pre_exp32[finite_mask]).astype(np.float32)
    return expected


@pytest.mark.cuda
def test_conv2d_binding_exists_and_returns_expression():
    x = ex.input("x")
    w = ex.input("w")
    out = ex.conv2d(x, w)
    assert isinstance(out, thor.physical.Expression)


@pytest.mark.parametrize(
    ("stride_h", "stride_w", "pad_h", "pad_w", "match"),
    [
        (0, 1, 0, 0, "stride must be positive"),
        (1, 0, 0, 0, "stride must be positive"),
        (1, 1, -1, 0, "padding must be non-negative"),
        (1, 1, 0, -1, "padding must be non-negative"),
    ],
)
def test_conv2d_invalid_stride_or_padding_rejected(stride_h: int, stride_w: int, pad_h: int, pad_w: int, match: str):
    x = ex.input("x")
    w = ex.input("w")
    with pytest.raises(RuntimeError, match=match):
        ex.conv2d(x, w, stride_h=stride_h, stride_w=stride_w, pad_h=pad_h, pad_w=pad_w)


@pytest.mark.cuda
@pytest.mark.parametrize("case", FORWARD_CASES, ids=[case["name"] for case in FORWARD_CASES])
def test_conv2d_forward_numerical_fp16(case: dict):
    dtype = thor.DataType.fp16
    x = ex.input("x")
    w = ex.input("w")
    out = ex.conv2d(
        x,
        w,
        stride_h=case["stride_h"],
        stride_w=case["stride_w"],
        pad_h=case["pad_h"],
        pad_w=case["pad_w"],
    )
    eq = ex.compile(out, device_num=0)

    x_np = (np.arange(np.prod(case["x_shape"]), dtype=np.float32).reshape(case["x_shape"]) / case["x_scale"]).astype(
        np.float16)
    w_np = (np.arange(np.prod(case["w_shape"]), dtype=np.float32).reshape(case["w_shape"]) / case["w_scale"]).astype(
        np.float16)
    expected = _conv2d_nchw_ref(
        x_np,
        w_np,
        stride_h=case["stride_h"],
        stride_w=case["stride_w"],
        pad_h=case["pad_h"],
        pad_w=case["pad_w"],
    )

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host_fp32(stamped.output(), stream)

    assert got.shape == _conv2d_output_shape(
        case["x_shape"], case["w_shape"], case["stride_h"], case["stride_w"], case["pad_h"], case["pad_w"])
    np.testing.assert_allclose(got, expected.astype(np.float32), rtol=FP16_RTOL, atol=FP16_ATOL)


@pytest.mark.cuda
def test_conv2d_followed_by_finite_pointwise_broadcast_numerical_fp16():
    dtype = thor.DataType.fp16
    x = ex.input("x")
    w = ex.input("w")
    b = ex.input("b")

    out = (ex.conv2d(x, w, stride_h=1, stride_w=1, pad_h=1, pad_w=1) + b) * 0.5 - 0.125
    eq = ex.compile(out, device_num=0)

    x_np = np.linspace(-0.25, 0.30, num=1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4).astype(np.float16)
    w_np = np.linspace(-0.20, 0.20, num=3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3).astype(np.float16)
    b_np = np.array([0.10, -0.05, 0.02], dtype=np.float32).reshape(1, 3, 1, 1).astype(np.float16)

    conv_ref = _conv2d_nchw_ref(x_np, w_np, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    expected = ((conv_ref + b_np.astype(np.float32)) * 0.5 - 0.125).astype(np.float16).astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host_fp32(stamped.output(), stream)

    np.testing.assert_allclose(got, expected, rtol=FP16_RTOL, atol=FP16_ATOL)


@pytest.mark.cuda
def test_conv2d_followed_by_exp_matches_fp16_overflow_semantics():
    dtype = thor.DataType.fp16
    x = ex.input("x")
    w = ex.input("w")
    b = ex.input("b")

    out = ex.exp(ex.conv2d(x, w, stride_h=1, stride_w=1, pad_h=1, pad_w=1) + b)
    eq = ex.compile(out, device_num=0)

    x_np = ((np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4) - 8.0) / 20.0).astype(np.float16)
    w_np = ((np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3) - 10.0) / 30.0).astype(np.float16)
    b_np = np.array([0.1, -0.2, 0.05], dtype=np.float32).reshape(1, 3, 1, 1).astype(np.float16)

    conv_ref = _conv2d_nchw_ref(x_np, w_np, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    expected = _fp16_exp_expected(conv_ref + b_np.astype(np.float32))

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
    }

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()
    got = _copy_to_host_fp32(stamped.output(), stream)

    np.testing.assert_allclose(got, expected, rtol=7e-2, atol=7e-2)


@pytest.mark.cuda
@pytest.mark.parametrize("case", FAN_CASES, ids=[f"wshape_{'x'.join(map(str, case['w_shape']))}" for case in FAN_CASES])
def test_conv2d_parameter_fan_override_filter(case: dict):
    x = ex.input("x")
    w = ex.input("w")
    out = ex.conv2d(x, w, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    eq = ex.compile(out, device_num=0)

    stream = Stream(gpu_num=0)
    x_shape = (2, case["w_shape"][1], 8, 7)
    x_np = np.zeros(x_shape, dtype=np.float16)
    w_np = np.zeros(case["w_shape"], dtype=np.float16)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, thor.DataType.fp16, stream),
        "w": _host_to_gpu(w_np, thor.DataType.fp16, stream),
    }

    fan = eq.get_parameter_fan_overrides(inputs_gpu, ["w"])
    assert fan["w"]["fan_in"] == case["expected_fan_in"]
    assert fan["w"]["fan_out"] == case["expected_fan_out"]


@pytest.mark.cuda
@pytest.mark.parametrize("case", BACKWARD_CASES, ids=[case["name"] for case in BACKWARD_CASES])
def test_conv2d_backward_numerical_fp16(case: dict):
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"

    x = ex.input("x")
    w = ex.input("w")
    out = ex.conv2d(
        x,
        w,
        stride_h=case["stride_h"],
        stride_w=case["stride_w"],
        pad_h=case["pad_h"],
        pad_w=case["pad_w"],
    )

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "w"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad", "w_grad"]

    x_np = case["x"].astype(np.float16)
    w_np = case["w"].astype(np.float16)
    grad_np = case["grad"].astype(np.float16)

    expected_x, expected_w = _finite_difference_conv2d_grads(
        x_np,
        w_np,
        grad_np,
        stride_h=case["stride_h"],
        stride_w=case["stride_w"],
        pad_h=case["pad_h"],
        pad_w=case["pad_w"],
        eps=case["eps"],
    )
    expected_x = expected_x.astype(np.float16).astype(np.float32)
    expected_w = expected_w.astype(np.float16).astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_x = _copy_to_host_fp32(stamped.output("x_grad"), stream)
    got_w = _copy_to_host_fp32(stamped.output("w_grad"), stream)

    np.testing.assert_allclose(got_x, expected_x, rtol=FP16_BWD_RTOL, atol=FP16_BWD_ATOL)
    np.testing.assert_allclose(got_w, expected_w, rtol=FP16_BWD_RTOL, atol=FP16_BWD_ATOL)


@pytest.mark.cuda
def test_conv2d_backward_through_pointwise_chain_numerical_fp16():
    dtype = thor.DataType.fp16
    upstream_name = "__grad_output"

    x = ex.input("x")
    w = ex.input("w")
    b = ex.input("b")
    out = ex.exp(ex.conv2d(x, w, stride_h=1, stride_w=1, pad_h=0, pad_w=0) + b)

    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x", "w", "b"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad", "w_grad", "b_grad"]

    x_np = np.array([[[[0.10, -0.05, 0.15], [0.00, 0.20, -0.10], [0.05, -0.15, 0.25]]]],
                    dtype=np.float32).astype(np.float16)
    w_np = np.array(
        [[[[0.10, -0.20], [0.30, 0.05]]], [[[0.05, 0.10], [-0.15, 0.20]]]],
        dtype=np.float32,
    ).astype(np.float16)
    b_np = np.array([0.02, -0.03], dtype=np.float32).reshape(1, 2, 1, 1).astype(np.float16)
    grad_np = np.array(
        [[[[0.20, -0.10], [0.05, 0.15]], [[-0.05, 0.10], [0.25, -0.20]]]],
        dtype=np.float32,
    ).astype(np.float16)

    expected_x, expected_w, expected_b = _finite_difference_conv_pointwise_grads(
        x_np, w_np, b_np, grad_np, stride_h=1, stride_w=1, pad_h=0, pad_w=0, eps=1e-2)
    expected_x = expected_x.astype(np.float16).astype(np.float32)
    expected_w = expected_w.astype(np.float16).astype(np.float32)
    expected_b = expected_b.astype(np.float16).astype(np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "w": _host_to_gpu(w_np, dtype, stream),
        "b": _host_to_gpu(b_np, dtype, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got_x = _copy_to_host_fp32(stamped.output("x_grad"), stream)
    got_w = _copy_to_host_fp32(stamped.output("w_grad"), stream)
    got_b = _copy_to_host_fp32(stamped.output("b_grad"), stream)

    np.testing.assert_allclose(got_x, expected_x, rtol=FP16_BWD_RTOL, atol=FP16_BWD_ATOL)
    np.testing.assert_allclose(got_w, expected_w, rtol=FP16_BWD_RTOL, atol=FP16_BWD_ATOL)
    np.testing.assert_allclose(got_b, expected_b, rtol=FP16_BWD_RTOL, atol=FP16_BWD_ATOL)
