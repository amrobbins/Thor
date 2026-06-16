import numpy as np
import pytest
import thor
from thor.physical import DeviceType, Expression as ex, PhysicalTensor, Placement, Stream, numpy_dtypes


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    return PhysicalTensor(Placement(DeviceType.cpu, 0), PhysicalTensor.Descriptor(dtype, shape))


def _gpu_tensor(shape: list[int], dtype: thor.DataType, gpu_num: int = 0) -> PhysicalTensor:
    return PhysicalTensor(Placement(DeviceType.gpu, gpu_num), PhysicalTensor.Descriptor(dtype, shape))


def _copy_numpy_to_gpu(values: np.ndarray, stream: Stream, dtype: thor.DataType, gpu_num: int = 0) -> PhysicalTensor:
    values = np.asarray(values, dtype=numpy_dtypes.from_thor(dtype), order="C")
    cpu = _cpu_tensor(list(values.shape), dtype)
    gpu = _gpu_tensor(list(values.shape), dtype, gpu_num=gpu_num)
    cpu_view = cpu.numpy()
    assert isinstance(cpu_view, np.ndarray)
    cpu_view[...] = values
    gpu.copy_from_async(cpu, stream)
    return gpu


def _copy_gpu_to_numpy(tensor: PhysicalTensor, dtype: thor.DataType, stream: Stream) -> np.ndarray:
    cpu = _cpu_tensor(list(tensor.get_descriptor().get_dimensions()), dtype)
    cpu.copy_from_async(tensor, stream)
    stream.synchronize()
    view = cpu.numpy()
    assert isinstance(view, np.ndarray)
    return np.array(view, copy=True)


@pytest.mark.cuda
def test_embedding_lookup_uint32_forward_zeroes_padding_index():
    indices_np = np.array([[1, 0, 3], [2, 1, 0]], dtype=np.uint32)
    weights_np = np.array(
        [
            [100.0, 101.0, 102.0, 103.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    expected = weights_np[indices_np]
    expected[indices_np == 0] = 0.0

    expr = ex.embedding_lookup(ex.input("indices"), ex.input("weights"), padding_index=0)
    eq = ex.compile(expr, device_num=0)

    stream = Stream(0)
    indices_gpu = _copy_numpy_to_gpu(indices_np, stream, thor.DataType.uint32)
    weights_gpu = _copy_numpy_to_gpu(weights_np, stream, thor.DataType.fp32)
    out_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)

    stamped = eq.stamp({"indices": indices_gpu, "weights": weights_gpu}, stream, preallocated_output=out_gpu)
    assert stamped._debug_stage_kinds() == ["EmbeddingLookup"]
    stamped.run()
    got = _copy_gpu_to_numpy(out_gpu, thor.DataType.fp32, stream)

    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
def test_embedding_lookup_uint8_forward_without_padding():
    indices_np = np.array([[3, 1, 2], [0, 2, 1]], dtype=np.uint8)
    weights_np = np.arange(20, dtype=np.float32).reshape(5, 4)
    expected = weights_np[indices_np]

    expr = ex.embedding_lookup(ex.input("indices"), ex.input("weights"))
    eq = ex.compile(expr, device_num=0)

    stream = Stream(0)
    indices_gpu = _copy_numpy_to_gpu(indices_np, stream, thor.DataType.uint8)
    weights_gpu = _copy_numpy_to_gpu(weights_np, stream, thor.DataType.fp32)
    out_gpu = _gpu_tensor([2, 3, 4], thor.DataType.fp32)

    stamped = eq.stamp({"indices": indices_gpu, "weights": weights_gpu}, stream, preallocated_output=out_gpu)
    assert stamped._debug_stage_kinds() == ["EmbeddingLookup"]
    stamped.run()
    got = _copy_gpu_to_numpy(out_gpu, thor.DataType.fp32, stream)

    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
def test_embedding_lookup_uint64_forward_without_padding():
    indices_np = np.array([3, 1, 2], dtype=np.uint64)
    weights_np = np.arange(20, dtype=np.float32).reshape(5, 4)
    expected = weights_np[indices_np]

    expr = ex.embedding_lookup(ex.input("indices"), ex.input("weights"))
    eq = ex.compile(expr, device_num=0)

    stream = Stream(0)
    indices_gpu = _copy_numpy_to_gpu(indices_np, stream, thor.DataType.uint64)
    weights_gpu = _copy_numpy_to_gpu(weights_np, stream, thor.DataType.fp32)
    out_gpu = _gpu_tensor([3, 4], thor.DataType.fp32)

    stamped = eq.stamp({"indices": indices_gpu, "weights": weights_gpu}, stream, preallocated_output=out_gpu)
    assert stamped._debug_stage_kinds() == ["EmbeddingLookup"]
    stamped.run()
    got = _copy_gpu_to_numpy(out_gpu, thor.DataType.fp32, stream)

    np.testing.assert_array_equal(got, expected)


@pytest.mark.cuda
def test_embedding_lookup_root_fuses_same_shape_pointwise_epilogue():
    indices_np = np.array([3, 1, 2], dtype=np.uint32)
    weights_np = np.arange(20, dtype=np.float32).reshape(5, 4)
    bias_np = np.array(
        [
            [0.5, 1.0, 1.5, 2.0],
            [2.5, 3.0, 3.5, 4.0],
            [4.5, 5.0, 5.5, 6.0],
        ],
        dtype=np.float32,
    )
    expected = (weights_np[indices_np] + bias_np) * np.float32(2.0)

    lookup = ex.embedding_lookup(ex.input("indices"), ex.input("weights"))
    expr = (lookup + ex.input("bias")) * 2.0
    eq = ex.compile(expr, device_num=0)

    stream = Stream(0)
    indices_gpu = _copy_numpy_to_gpu(indices_np, stream, thor.DataType.uint32)
    weights_gpu = _copy_numpy_to_gpu(weights_np, stream, thor.DataType.fp32)
    bias_gpu = _copy_numpy_to_gpu(bias_np, stream, thor.DataType.fp32)
    out_gpu = _gpu_tensor([3, 4], thor.DataType.fp32)

    stamped = eq.stamp({"indices": indices_gpu, "weights": weights_gpu, "bias": bias_gpu}, stream, preallocated_output=out_gpu)
    assert stamped._debug_stage_kinds() == ["EmbeddingLookup"]
    stamped.run()
    got = _copy_gpu_to_numpy(out_gpu, thor.DataType.fp32, stream)

    np.testing.assert_allclose(got, expected, rtol=0, atol=0)
