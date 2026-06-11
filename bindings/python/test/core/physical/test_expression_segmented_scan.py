import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, ScanOp, Stream, numpy_dtypes


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


def _copy_to_host(tensor: PhysicalTensor, dtype: thor.DataType, stream: Stream) -> np.ndarray:
    host = _cpu_tensor(list(tensor.dimensions), dtype)
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return host.numpy().copy()


def _segmented_scan_reference(values: np.ndarray, offsets: np.ndarray, op: ScanOp, inclusive_mode: bool) -> np.ndarray:
    out = np.empty_like(values)
    for segment in range(len(offsets) - 1):
        begin = int(offsets[segment])
        end = int(offsets[segment + 1])
        segment_values = values[begin:end]
        if op == ScanOp.sum:
            inclusive = np.cumsum(segment_values)
            identity = np.array(0, dtype=values.dtype)
        elif op == ScanOp.min:
            inclusive = np.minimum.accumulate(segment_values)
            identity = np.iinfo(values.dtype).max if np.issubdtype(values.dtype, np.integer) else np.inf
        elif op == ScanOp.max:
            inclusive = np.maximum.accumulate(segment_values)
            identity = np.iinfo(values.dtype).min if np.issubdtype(values.dtype, np.integer) else -np.inf
        elif op == ScanOp.product:
            inclusive = np.cumprod(segment_values)
            identity = np.array(1, dtype=values.dtype)
        else:
            raise AssertionError(f"Unhandled ScanOp: {op}")

        if inclusive_mode:
            out[begin:end] = inclusive
        else:
            if end > begin:
                out[begin] = identity
                out[begin + 1 : end] = inclusive[:-1]
    return out


@pytest.mark.cuda
@pytest.mark.parametrize("op", [ScanOp.sum, ScanOp.min, ScanOp.max, ScanOp.product])
@pytest.mark.parametrize("inclusive", [True, False])
def test_segmented_scan_expression_matches_numpy_reference(op: ScanOp, inclusive: bool):
    x = ex.input("x")
    offsets = ex.input("offsets")
    out = x.segmented_scan(offsets, op=op, inclusive=inclusive)
    eq = ex.compile(out, device_num=0)

    values_np = np.array([8, 6, 7, 5, 3, 2, 9], dtype=np.uint32)
    offsets_np = np.array([0, 2, 5, 7], dtype=np.uint32)
    expected = _segmented_scan_reference(values_np, offsets_np, op, inclusive)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(values_np, thor.DataType.uint32, stream),
        "offsets": _host_to_gpu(offsets_np, thor.DataType.uint32, stream),
    }
    assert eq._debug_stage_kinds(inputs_gpu) == ["Scan"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), thor.DataType.uint32, stream)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("inclusive", [True, False])
def test_segmented_sum_scan_backward_uses_ragged_reverse_scan(inclusive: bool):
    x = ex.input("x")
    offsets = ex.input("offsets")
    upstream_name = "__grad_output"

    out = x.segmented_scan(offsets, op=ScanOp.sum, inclusive=inclusive)
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(thor.DataType.fp32)
    x_np = np.array([1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5], dtype=storage_dtype)
    offsets_np = np.array([0, 3, 3, 7], dtype=np.uint32)
    grad_np = np.array([0.5, -1.0, 0.25, 2.0, 1.25, -0.5, 1.5], dtype=storage_dtype)

    expected = np.empty_like(grad_np)
    for segment in range(len(offsets_np) - 1):
        begin = int(offsets_np[segment])
        end = int(offsets_np[segment + 1])
        rev = grad_np[begin:end][::-1]
        if inclusive:
            expected[begin:end] = np.cumsum(rev)[::-1]
        else:
            exclusive = np.empty_like(rev)
            if len(rev) > 0:
                exclusive[0] = 0.0
                exclusive[1:] = np.cumsum(rev[:-1])
            expected[begin:end] = exclusive[::-1]

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, thor.DataType.fp32, stream),
        "offsets": _host_to_gpu(offsets_np, thor.DataType.uint32, stream),
        upstream_name: _host_to_gpu(grad_np, thor.DataType.fp32, stream),
    }
    assert bwd_eq._debug_stage_kinds(inputs_gpu) == ["Scan"]

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output("x_grad"), thor.DataType.fp32, stream)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def _segmented_minmax_scan_backward_reference(values: np.ndarray,
                                              offsets: np.ndarray,
                                              grad: np.ndarray,
                                              op: ScanOp,
                                              inclusive: bool) -> np.ndarray:
    expected = np.zeros_like(values, dtype=np.float32)
    values32 = values.astype(np.float32)
    grad32 = grad.astype(np.float32)
    for segment in range(len(offsets) - 1):
        begin = int(offsets[segment])
        end = int(offsets[segment + 1])
        for i in range(begin, end):
            prefix_end = i + 1 if inclusive else i
            if prefix_end <= begin:
                continue
            winner = begin
            winner_value = values32[winner]
            for j in range(begin + 1, prefix_end):
                candidate = values32[j]
                if (op == ScanOp.min and candidate < winner_value) or (op == ScanOp.max and candidate > winner_value):
                    winner = j
                    winner_value = candidate
            expected[winner] += grad32[i]
    return expected


@pytest.mark.cuda
@pytest.mark.parametrize("op", [ScanOp.min, ScanOp.max])
@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("dtype", [thor.DataType.fp16, thor.DataType.bf16, thor.DataType.fp32])
def test_segmented_minmax_scan_backward_uses_arg_scan_and_flat_scatter_add(op: ScanOp, inclusive: bool, dtype: thor.DataType):
    x = ex.input("x")
    offsets = ex.input("offsets")
    upstream_name = "__grad_output"

    out = x.segmented_scan(offsets, op=op, inclusive=inclusive)
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    storage_dtype = _numpy_storage_dtype(dtype)
    x_np = np.array([3.0, 1.0, 1.0, 4.0, 2.0, 2.0, 0.0], dtype=np.float32).astype(storage_dtype)
    offsets_np = np.array([0, 3, 3, 7], dtype=np.uint32)
    grad_np = np.array([0.5, -1.0, 0.25, 2.0, 1.25, -0.5, 1.5], dtype=np.float32).astype(storage_dtype)
    expected = _segmented_minmax_scan_backward_reference(x_np, offsets_np, grad_np, op, inclusive)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, dtype, stream),
        "offsets": _host_to_gpu(offsets_np, thor.DataType.uint32, stream),
        upstream_name: _host_to_gpu(grad_np, dtype, stream),
    }
    stage_kinds = bwd_eq._debug_stage_kinds(inputs_gpu)
    assert stage_kinds.count("ScanMinMaxBackward") == 1

    stamped = bwd_eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output("x_grad"), thor.DataType.fp32, stream)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
def test_segmented_scan_rejects_unsupported_uint64_offsets_forward():
    x = ex.input("x")
    offsets = ex.input("offsets")
    out = x.segmented_scan(offsets, op=ScanOp.max, inclusive=False)
    eq = ex.compile(out, device_num=0)

    values_np = np.array([8, 6, 7, 5, 3, 2, 9], dtype=np.uint32)
    offsets_np = np.array([0, 2, 5, 7], dtype=np.uint64)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(values_np, thor.DataType.uint32, stream),
        "offsets": _host_to_gpu(offsets_np, thor.DataType.uint64, stream),
    }
    with pytest.raises(RuntimeError, match="segmented_scan offsets dtype is not supported"):
        eq._debug_stage_kinds(inputs_gpu)


@pytest.mark.cuda
def test_segmented_sum_scan_backward_rejects_unsupported_uint64_offsets():
    x = ex.input("x")
    offsets = ex.input("offsets")
    upstream_name = "__grad_output"

    out = x.segmented_scan(offsets, op=ScanOp.sum, inclusive=True)
    fwd_eq = ex.compile(out, device_num=0)
    bwd_eq = fwd_eq.compile_backward(["x"], error_input_name=upstream_name)
    assert bwd_eq.output_names() == ["x_grad"]

    x_np = np.array([1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5], dtype=np.float32)
    offsets_np = np.array([0, 3, 3, 7], dtype=np.uint64)
    grad_np = np.array([0.5, -1.0, 0.25, 2.0, 1.25, -0.5, 1.5], dtype=np.float32)

    stream = Stream(gpu_num=0)
    inputs_gpu = {
        "x": _host_to_gpu(x_np, thor.DataType.fp32, stream),
        "offsets": _host_to_gpu(offsets_np, thor.DataType.uint64, stream),
        upstream_name: _host_to_gpu(grad_np, thor.DataType.fp32, stream),
    }
    with pytest.raises(RuntimeError, match="segmented_scan offsets dtype is not supported"):
        bwd_eq._debug_stage_kinds(inputs_gpu)
