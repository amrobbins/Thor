import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, ScanOp, Stream


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> thor.physical.PhysicalTensor:
    placement = Placement(DeviceType.cpu, 0)
    descriptor = PhysicalTensor.Descriptor(dtype, shape)
    return PhysicalTensor(placement, descriptor)


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


def _scan_reference(values: np.ndarray, op: ScanOp, inclusive_mode: bool) -> np.ndarray:
    rows = values.reshape((-1, values.shape[-1]))
    out = np.empty_like(rows)
    for row_idx, row in enumerate(rows):
        if op == ScanOp.sum:
            inclusive = np.cumsum(row)
            identity = np.array(0, dtype=values.dtype)
        elif op == ScanOp.min:
            inclusive = np.minimum.accumulate(row)
            identity = np.iinfo(values.dtype).max if np.issubdtype(values.dtype, np.integer) else np.inf
        elif op == ScanOp.max:
            inclusive = np.maximum.accumulate(row)
            identity = np.iinfo(values.dtype).min if np.issubdtype(values.dtype, np.integer) else -np.inf
        elif op == ScanOp.product:
            inclusive = np.cumprod(row)
            identity = np.array(1, dtype=values.dtype)
        else:
            raise AssertionError(f"Unhandled ScanOp: {op}")

        if inclusive_mode:
            out[row_idx] = inclusive
        else:
            out[row_idx, 0] = identity
            out[row_idx, 1:] = inclusive[:-1]
    return out.reshape(values.shape)


@pytest.mark.cuda
@pytest.mark.parametrize("op", [ScanOp.sum, ScanOp.min, ScanOp.max, ScanOp.product])
@pytest.mark.parametrize("inclusive", [True, False])
def test_scan_expression_matches_numpy_reference(op: ScanOp, inclusive: bool):
    x = ex.input("x")
    out = x.scan(op=op, axis=-1, inclusive=inclusive)
    eq = ex.compile(out, device_num=0)

    values_np = np.array([[2, 3, 1, 4], [5, 1, 2, 3]], dtype=np.uint32)
    expected = _scan_reference(values_np, op, inclusive)

    stream = Stream(gpu_num=0)
    inputs_gpu = {"x": _host_to_gpu(values_np, thor.DataType.uint32, stream)}
    assert eq._debug_stage_kinds(inputs_gpu) == ["Scan"]

    stamped = eq.stamp(inputs_gpu, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), thor.DataType.uint32, stream)
    np.testing.assert_array_equal(got, expected)
