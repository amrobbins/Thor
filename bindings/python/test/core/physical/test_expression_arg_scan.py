import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, Outputs, PhysicalTensor, Placement, ScanOp, Stream


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


def _prefix_arg_reference(values: np.ndarray, op: ScanOp) -> np.ndarray:
    flat = values.reshape(-1)
    rows = values.reshape((-1, values.shape[-1]))
    out = np.empty_like(rows, dtype=np.uint32)
    for row_idx, row in enumerate(rows):
        best_pos = 0
        for inner, value in enumerate(row):
            if inner == 0:
                best_pos = 0
            elif op == ScanOp.arg_min and value < row[best_pos]:
                best_pos = inner
            elif op == ScanOp.arg_max and row[best_pos] < value:
                best_pos = inner
            out[row_idx, inner] = row_idx * row.shape[0] + best_pos
    return out.reshape(values.shape)


@pytest.mark.cuda
@pytest.mark.parametrize("op", [ScanOp.arg_min, ScanOp.arg_max])
def test_scan_arg_min_max_returns_flattened_prefix_winner_indices(op: ScanOp):
    x = ex.input("x")
    out = x.scan(op=op, axis=-1, inclusive=True)
    eq = ex.compile(out, device_num=0)

    values_np = np.array([[3.0, 1.0, 4.0, 1.0], [2.0, 5.0, 0.0, 0.0]], dtype=np.float32)
    expected = _prefix_arg_reference(values_np, op)

    stream = Stream(gpu_num=0)
    stamped = eq.stamp({"x": _host_to_gpu(values_np, thor.DataType.fp32, stream)}, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), thor.DataType.uint32, stream)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
@pytest.mark.parametrize("op", [ScanOp.arg_min, ScanOp.arg_max])
def test_segmented_scan_arg_min_max_returns_flattened_prefix_winner_indices(op: ScanOp):
    x = ex.input("x")
    offsets = ex.input("offsets")
    out = x.segmented_scan(offsets, op=op, inclusive=True)
    eq = ex.compile(out, device_num=0)

    values_np = np.array([8, 6, 7, 5, 3, 2, 9], dtype=np.uint32)
    offsets_np = np.array([0, 2, 5, 7], dtype=np.uint32)
    expected = np.empty_like(values_np, dtype=np.uint32)
    for segment in range(len(offsets_np) - 1):
        begin = int(offsets_np[segment])
        end = int(offsets_np[segment + 1])
        best = begin
        for i in range(begin, end):
            if i == begin:
                best = i
            elif op == ScanOp.arg_min and values_np[i] < values_np[best]:
                best = i
            elif op == ScanOp.arg_max and values_np[best] < values_np[i]:
                best = i
            expected[i] = best

    stream = Stream(gpu_num=0)
    inputs = {
        "x": _host_to_gpu(values_np, thor.DataType.uint32, stream),
        "offsets": _host_to_gpu(offsets_np, thor.DataType.uint32, stream),
    }
    stamped = eq.stamp(inputs, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), thor.DataType.uint32, stream)
    np.testing.assert_array_equal(got, expected)


def test_arg_scan_serialization_round_trips():
    x = ex.input("x")
    out = x.scan(op=ScanOp.arg_max, axis=-1, inclusive=True)
    outputs = ex.outputs({"argmax": out})

    payload = outputs.to_json()
    assert "arg_max" in payload

    loaded = Outputs.from_json(payload)
    assert loaded.output_names() == ["argmax"]
    assert "arg_max" in loaded.to_json()


def _prefix_value_reference(values: np.ndarray, op: ScanOp) -> np.ndarray:
    rows = values.reshape((-1, values.shape[-1]))
    out = np.empty_like(rows)
    for row_idx, row in enumerate(rows):
        best_pos = 0
        for inner, value in enumerate(row):
            if inner == 0:
                best_pos = 0
            elif op == ScanOp.arg_min and value < row[best_pos]:
                best_pos = inner
            elif op == ScanOp.arg_max and row[best_pos] < value:
                best_pos = inner
            out[row_idx, inner] = row[best_pos]
    return out.reshape(values.shape)


@pytest.mark.cuda
@pytest.mark.parametrize("op_name", ["min", "max"])
def test_scan_with_indices_returns_values_and_indices_from_one_expression_pair(op_name: str):
    x = ex.input("x")
    if op_name == "min":
        values, indices = x.scan_with_indices(op=ScanOp.min, axis=-1, inclusive=True)
        arg_op = ScanOp.arg_min
    else:
        values, indices = x.scan_with_indices(op=ScanOp.max, axis=-1, inclusive=True)
        arg_op = ScanOp.arg_max
    eq = ex.outputs({"values": values, "indices": indices}).compile(device_num=0)

    values_np = np.array([[3.0, 1.0, 4.0, 1.0], [2.0, 5.0, 0.0, 0.0]], dtype=np.float32)
    expected_values = _prefix_value_reference(values_np, arg_op)
    expected_indices = _prefix_arg_reference(values_np, arg_op)

    stream = Stream(gpu_num=0)
    stamped = eq.stamp({"x": _host_to_gpu(values_np, thor.DataType.fp32, stream)}, stream)
    stamped.run()

    got = stamped.outputs()
    np.testing.assert_allclose(_copy_to_host(got["values"], thor.DataType.fp32, stream), expected_values)
    np.testing.assert_array_equal(_copy_to_host(got["indices"], thor.DataType.uint32, stream), expected_indices)


@pytest.mark.cuda
@pytest.mark.parametrize("op_name", ["min", "max"])
def test_segmented_scan_with_indices_returns_values_and_indices_from_one_expression_pair(op_name: str):
    x = ex.input("x")
    offsets = ex.input("offsets")
    if op_name == "min":
        values, indices = x.segmented_scan_with_indices(offsets, op=ScanOp.min, inclusive=True)
        is_min = True
    else:
        values, indices = x.segmented_scan_with_indices(offsets, op=ScanOp.max, inclusive=True)
        is_min = False
    eq = ex.outputs({"values": values, "indices": indices}).compile(device_num=0)

    values_np = np.array([8, 6, 7, 5, 3, 2, 9], dtype=np.uint32)
    offsets_np = np.array([0, 2, 5, 7], dtype=np.uint32)
    expected_values = np.empty_like(values_np)
    expected_indices = np.empty_like(values_np, dtype=np.uint32)
    for segment in range(len(offsets_np) - 1):
        begin = int(offsets_np[segment])
        end = int(offsets_np[segment + 1])
        best = begin
        for i in range(begin, end):
            if i == begin:
                best = i
            elif is_min and values_np[i] < values_np[best]:
                best = i
            elif (not is_min) and values_np[best] < values_np[i]:
                best = i
            expected_values[i] = values_np[best]
            expected_indices[i] = best

    stream = Stream(gpu_num=0)
    inputs = {
        "x": _host_to_gpu(values_np, thor.DataType.uint32, stream),
        "offsets": _host_to_gpu(offsets_np, thor.DataType.uint32, stream),
    }
    stamped = eq.stamp(inputs, stream)
    stamped.run()

    got = stamped.outputs()
    np.testing.assert_array_equal(_copy_to_host(got["values"], thor.DataType.uint32, stream), expected_values)
    np.testing.assert_array_equal(_copy_to_host(got["indices"], thor.DataType.uint32, stream), expected_indices)
