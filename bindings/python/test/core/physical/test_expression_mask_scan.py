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


@pytest.mark.cuda
def test_bool_mask_prefix_count_inclusive_and_exclusive():
    x = ex.input("x")
    mask = x > 0.0
    inclusive = mask.prefix_count(inclusive=True, axis=-1)
    exclusive = mask.prefix_count(inclusive=False, axis=-1)
    eq = ex.outputs({"inclusive": inclusive, "exclusive": exclusive}).compile(device_num=0)

    values_np = np.array([[0.0, 2.0, -1.0, 3.0], [4.0, -5.0, 6.0, 7.0]], dtype=np.float32)
    mask_np = values_np > 0.0
    expected_inclusive = np.cumsum(mask_np.astype(np.uint32), axis=-1, dtype=np.uint32)
    expected_exclusive = np.concatenate(
        [np.zeros_like(expected_inclusive[..., :1], dtype=np.uint32), expected_inclusive[..., :-1]], axis=-1
    )

    stream = Stream(gpu_num=0)
    stamped = eq.stamp({"x": _host_to_gpu(values_np, thor.DataType.fp32, stream)}, stream)
    stamped.run()

    got = stamped.outputs()
    np.testing.assert_array_equal(_copy_to_host(got["inclusive"], thor.DataType.uint32, stream), expected_inclusive)
    np.testing.assert_array_equal(_copy_to_host(got["exclusive"], thor.DataType.uint32, stream), expected_exclusive)


@pytest.mark.cuda
def test_bool_mask_cast_to_uint32_can_feed_sum_scan_directly():
    x = ex.input("x")
    out = (x != 0.0).cast(thor.DataType.uint32).scan(op=ScanOp.sum, axis=-1, inclusive=True)
    eq = ex.compile(out, device_num=0)

    values_np = np.array([[1.0, 0.0, 2.0, 3.0, 0.0]], dtype=np.float32)
    expected = np.cumsum((values_np != 0.0).astype(np.uint32), axis=-1, dtype=np.uint32)

    stream = Stream(gpu_num=0)
    stamped = eq.stamp({"x": _host_to_gpu(values_np, thor.DataType.fp32, stream)}, stream)
    stamped.run()

    got = _copy_to_host(stamped.output(), thor.DataType.uint32, stream)
    np.testing.assert_array_equal(got, expected)


def test_mask_cast_and_prefix_count_serialization_round_trips():
    x = ex.input("x")
    out = (x > 0.0).prefix_count(inclusive=True, axis=-1)
    outputs = ex.outputs({"prefix_count": out})

    payload = outputs.to_json()
    assert "cast" in payload
    assert "uint32" in payload.lower()
    assert "scan" in payload

    loaded = Outputs.from_json(payload)
    assert loaded.output_names() == ["prefix_count"]
    assert "cast" in loaded.to_json()
