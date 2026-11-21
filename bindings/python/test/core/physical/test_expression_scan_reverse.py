"""reverse=True on scan, segmented_scan, scan_with_indices and segmented_scan_with_indices.

A reverse scan of a row equals the forward scan of the row flipped, flipped back. For arg scans the flipped row's
first winner is the original row's last, which is the reverse scan's tie rule (the first element it meets wins).
"""

import json

import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, ScanOp, Stream

_NO_INDEX = np.iinfo(np.uint32).max


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    return PhysicalTensor(Placement(DeviceType.cpu, 0), PhysicalTensor.Descriptor(dtype, shape))


def _host_to_gpu(arr: np.ndarray, dtype: thor.DataType, stream: Stream) -> PhysicalTensor:
    desc = PhysicalTensor.Descriptor(dtype, list(arr.shape))
    host = PhysicalTensor(Placement(DeviceType.cpu, 0), desc)
    host.numpy()[...] = arr
    device = PhysicalTensor(Placement(DeviceType.gpu, 0), desc)
    device.copy_from_async(host, stream)
    return device


def _copy_to_host(tensor: PhysicalTensor, dtype: thor.DataType, stream: Stream) -> np.ndarray:
    host = _cpu_tensor(list(tensor.dimensions), dtype)
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return host.numpy().copy()


def _forward_row(row: np.ndarray, op: ScanOp, inclusive: bool) -> tuple[np.ndarray, np.ndarray]:
    """Forward scan of one row: values and the winner's position in the row (_NO_INDEX before the first element)."""
    values = np.empty(len(row), dtype=np.float32)
    positions = np.full(len(row), _NO_INDEX, dtype=np.uint64)
    identity = {ScanOp.sum: 0.0, ScanOp.max: -np.inf, ScanOp.min: np.inf}[op]
    acc, winner = np.float32(identity), _NO_INDEX
    for i, v in enumerate(row):
        before, before_winner = acc, winner
        if op == ScanOp.sum:
            acc = np.float32(acc + v)
        elif winner == _NO_INDEX or (v > acc if op == ScanOp.max else v < acc):
            acc, winner = v, i
        values[i], positions[i] = (acc, winner) if inclusive else (before, before_winner)
    return values, positions


def _reverse_reference(x: np.ndarray, bounds: list[int], op: ScanOp, inclusive: bool) -> tuple[np.ndarray, np.ndarray]:
    values = np.empty_like(x, dtype=np.float32)
    indices = np.full(x.shape, _NO_INDEX, dtype=np.uint32)
    for begin, end in zip(bounds[:-1], bounds[1:]):
        row_values, positions = _forward_row(x[begin:end][::-1], op, inclusive)
        values[begin:end] = row_values[::-1]
        flat = [_NO_INDEX if p == _NO_INDEX else end - 1 - int(p) for p in positions[::-1]]
        indices[begin:end] = np.array(flat, dtype=np.uint32)
    return values, indices


# Rows of 7 with ties: a reverse arg scan gives a tie to the later element.
_X = np.array([3, 1, 4, 1, 5, 4, 2, 2, 2, 7, 7, 1, 0, 0, -1, 6, -3, 6, -3, 2, 2], dtype=np.float32)
_ROWS, _COLS = 3, 7
_DENSE_BOUNDS = [0, 7, 14, 21]
_OFFSETS = np.array([0, 5, 5, 13, 21], dtype=np.uint32)


def _run(outputs, inputs_np: dict, stream: Stream):
    eq = ex.outputs(outputs).compile(device_num=0)
    inputs = {name: _host_to_gpu(arr, dtype, stream) for name, (arr, dtype) in inputs_np.items()}
    stamped = eq.stamp(inputs, stream)
    stamped.run()
    return eq, inputs, stamped.outputs()


@pytest.mark.cuda
@pytest.mark.parametrize("op", [ScanOp.sum, ScanOp.min, ScanOp.max, ScanOp.arg_min, ScanOp.arg_max])
@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("segmented", [False, True])
def test_reverse_scan_runs_each_row_from_its_end(op: ScanOp, inclusive: bool, segmented: bool):
    stream = Stream(gpu_num=0)
    x = ex.input("x")
    if segmented:
        out = x.segmented_scan(ex.input("offsets"), op=op, inclusive=inclusive, reverse=True)
        inputs = {"x": (_X, thor.DataType.fp32), "offsets": (_OFFSETS, thor.DataType.uint32)}
        bounds = [int(b) for b in _OFFSETS]
    else:
        out = x.scan(op=op, axis=-1, inclusive=inclusive, reverse=True)
        inputs = {"x": (_X.reshape(_ROWS, _COLS), thor.DataType.fp32)}
        bounds = _DENSE_BOUNDS
    _, _, got = _run({"y": out}, inputs, stream)

    value_op = {ScanOp.arg_min: ScanOp.min, ScanOp.arg_max: ScanOp.max}.get(op, op)
    want_values, want_indices = _reverse_reference(_X, bounds, value_op, inclusive)
    if op in (ScanOp.arg_min, ScanOp.arg_max):
        np.testing.assert_array_equal(_copy_to_host(got["y"], thor.DataType.uint32, stream).reshape(-1), want_indices)
    else:
        np.testing.assert_array_equal(_copy_to_host(got["y"], thor.DataType.fp32, stream).reshape(-1), want_values)


@pytest.mark.cuda
@pytest.mark.parametrize("op", [ScanOp.min, ScanOp.max])
@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("segmented", [False, True])
def test_reverse_scan_with_indices_shares_one_stage(op: ScanOp, inclusive: bool, segmented: bool):
    stream = Stream(gpu_num=0)
    x = ex.input("x")
    if segmented:
        values, indices = x.segmented_scan_with_indices(ex.input("offsets"), op=op, inclusive=inclusive, reverse=True)
        inputs = {"x": (_X, thor.DataType.fp32), "offsets": (_OFFSETS, thor.DataType.uint32)}
        bounds = [int(b) for b in _OFFSETS]
    else:
        values, indices = x.scan_with_indices(op=op, axis=-1, inclusive=inclusive, reverse=True)
        inputs = {"x": (_X.reshape(_ROWS, _COLS), thor.DataType.fp32)}
        bounds = _DENSE_BOUNDS
    eq, gpu_inputs, got = _run({"values": values, "indices": indices}, inputs, stream)
    assert eq._debug_stage_kinds(gpu_inputs) == ["Scan"]

    want_values, want_indices = _reverse_reference(_X, bounds, op, inclusive)
    np.testing.assert_array_equal(_copy_to_host(got["values"], thor.DataType.fp32, stream).reshape(-1), want_values)
    np.testing.assert_array_equal(_copy_to_host(got["indices"], thor.DataType.uint32, stream).reshape(-1), want_indices)


def _scan_reverse_flags(payload: str) -> list[bool]:
    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("op") in ("scan", "segmented_scan"):
                found.append(bool(node["scan_reverse"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(payload))
    return found


def test_reverse_defaults_to_false_and_round_trips():
    x = ex.input("x")
    assert _scan_reverse_flags(ex.outputs({"y": x.scan(op=ScanOp.max)}).to_json()) == [False]
    payload = ex.outputs({"y": x.scan(op=ScanOp.max, reverse=True)}).to_json()
    assert _scan_reverse_flags(payload) == [True]
    assert _scan_reverse_flags(thor.physical.Outputs.from_json(payload).to_json()) == [True]
