import numpy as np
import pytest
import thor
from thor.physical import (
    CudaKernelExpression,
    CudaKernelLaunchConfig,
    DeviceType,
    Expression as ex,
    PhysicalTensor,
    Placement,
    Stream,
)


def _cpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    return PhysicalTensor(
        Placement(DeviceType.cpu, 0),
        PhysicalTensor.Descriptor(dtype, shape),
    )


def _gpu_tensor(shape: list[int], dtype: thor.DataType) -> PhysicalTensor:
    return PhysicalTensor(
        Placement(DeviceType.gpu, 0),
        PhysicalTensor.Descriptor(dtype, shape),
    )


def _copy_numpy_to_gpu(values: np.ndarray, dtype: thor.DataType, stream: Stream) -> PhysicalTensor:
    host = _cpu_tensor(list(values.shape), dtype)
    host.numpy()[:] = values.astype(thor.physical.numpy_dtypes.from_thor(dtype))
    gpu = _gpu_tensor(list(values.shape), dtype)
    gpu.copy_from_async(host, stream)
    return gpu


def _copy_gpu_to_numpy(tensor: PhysicalTensor, stream: Stream) -> np.ndarray:
    host = tensor.clone(Placement(DeviceType.cpu, 0))
    host.copy_from_async(tensor, stream)
    stream.synchronize()
    return np.array(host.numpy(), copy=True)


@pytest.mark.cuda
def test_cuda_kernel_expression_single_output_raw_pointer_kernel_runs_from_python():
    kernel = (
        CudaKernelExpression.builder("py_scale")
        .source(
            r"""
extern "C" __global__
void py_scale_kernel(const float* x, float* y, float alpha, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = alpha * x[i];
}
"""
        )
        .entry("py_scale_kernel")
        .input("x", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("alpha", thor.DataType.fp32, 2.5)
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch(lambda ctx: CudaKernelLaunchConfig.grid_1d(ctx.numel("y"), block_size=128))
        .build()
    )

    outputs = kernel.apply({"x": ex.input("x")})
    eq = outputs.compile(device_num=0, use_fast_math=False)

    stream = Stream(gpu_num=0)
    x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32, stream)

    stamped = eq.stamp({"x": x_gpu}, stream)
    stamped.run()

    y_np = _copy_gpu_to_numpy(stamped.output("y"), stream)
    np.testing.assert_allclose(y_np, x_np * 2.5, rtol=1e-6, atol=1e-6)


@pytest.mark.cuda
def test_cuda_kernel_expression_multi_output_raw_pointer_kernel_runs_from_python():
    kernel = (
        CudaKernelExpression.builder("py_split_math")
        .source(
            r"""
extern "C" __global__
void py_split_math_kernel(const float* x, float* twice, float* plus_one, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    twice[i] = 2.0f * v;
    plus_one[i] = v + 1.0f;
}
"""
        )
        .entry("py_split_math_kernel")
        .input("x", thor.DataType.fp32)
        .output_like("twice", thor.DataType.fp32, "x")
        .output_like("plus_one", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("x"))
        .launch(lambda ctx: CudaKernelLaunchConfig.grid_1d(ctx.numel("x"), block_size=64))
        .build()
    )

    split = kernel({"x": ex.input("x")})
    eq = split.compile(device_num=0, use_fast_math=False)

    stream = Stream(gpu_num=0)
    x_np = np.array([1.25, 2.5, 3.75, 5.0, 6.25], dtype=np.float32)
    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32, stream)

    stamped = eq.stamp({"x": x_gpu}, stream)
    stamped.run()

    twice_np = _copy_gpu_to_numpy(stamped.output("twice"), stream)
    plus_one_np = _copy_gpu_to_numpy(stamped.output("plus_one"), stream)
    np.testing.assert_allclose(twice_np, x_np * 2.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(plus_one_np, x_np + 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.cuda
def test_cuda_kernel_expression_output_shape_dim_expr_from_python():
    kernel = (
        CudaKernelExpression.builder("py_shape_copy")
        .source(
            r"""
extern "C" __global__
void py_shape_copy_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i];
}
"""
        )
        .entry("py_shape_copy_kernel")
        .input("x", thor.DataType.fp32)
        .output(
            "y",
            thor.DataType.fp32,
            [CudaKernelExpression.dim("x", 0), CudaKernelExpression.dim("x", 1)],
        )
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch(lambda ctx: CudaKernelLaunchConfig.grid_1d(ctx.numel("y"), block_size=128))
        .build()
    )

    outputs = kernel.apply({"x": ex.input("x")})
    eq = outputs.compile(device_num=0, use_fast_math=False)

    stream = Stream(gpu_num=0)
    x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32, stream)

    stamped = eq.stamp({"x": x_gpu}, stream)
    stamped.run()

    y_np = _copy_gpu_to_numpy(stamped.output("y"), stream)
    np.testing.assert_allclose(y_np, x_np, rtol=1e-6, atol=1e-6)


@pytest.mark.cuda
def test_cuda_kernel_expression_rejects_python_input_dtype_mismatch_before_launch():
    kernel = (
        CudaKernelExpression.builder("py_dtype_reject")
        .source(
            r"""
extern "C" __global__
void py_dtype_reject_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i];
}
"""
        )
        .entry("py_dtype_reject_kernel")
        .input("x", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("x"))
        .launch(lambda ctx: CudaKernelLaunchConfig.grid_1d(ctx.numel("x"), block_size=128))
        .build()
    )

    outputs = kernel.apply({"x": ex.input("x")})
    eq = outputs.compile(device_num=0, use_fast_math=False)

    stream = Stream(gpu_num=0)
    x_gpu = _copy_numpy_to_gpu(np.array([1.0, 2.0, 3.0], dtype=np.float32), thor.DataType.fp16, stream)

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        eq.stamp({"x": x_gpu}, stream)

@pytest.mark.cuda
def test_cuda_kernel_expression_tensor_runtime_scalar_input_from_python():
    kernel = (
        CudaKernelExpression.builder("py_runtime_scalar_scale")
        .source(
            r"""
extern "C" __global__
void py_runtime_scalar_scale_kernel(const float* x, const float* alpha, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (*alpha) * x[i];
}
"""
        )
        .entry("py_runtime_scalar_scale_kernel")
        .input("x", thor.DataType.fp32)
        .tensor_runtime_scalar_input("alpha", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch(lambda ctx: CudaKernelLaunchConfig.grid_1d(ctx.numel("y"), block_size=128))
        .build()
    )

    outputs = kernel.apply({
        "x": ex.input("x"),
        "alpha": ex.tensor_runtime_scalar("alpha", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
    })
    eq = outputs.compile(device_num=0, use_fast_math=False)

    stream = Stream(gpu_num=0)
    x_np = np.array([1.0, -2.0, 3.0, 4.5, -5.0, 6.0], dtype=np.float32)
    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32, stream)
    alpha_gpu = _copy_numpy_to_gpu(np.array([123.0, -1.5], dtype=np.float32), thor.DataType.fp32, stream)
    alpha_binding = thor.physical.TensorScalarBinding(alpha_gpu, 4, thor.DataType.fp32)

    stamped = eq.stamp({"x": x_gpu}, stream, tensor_scalar_inputs={"alpha": alpha_binding})
    stamped.run()

    y_np = _copy_gpu_to_numpy(stamped.output("y"), stream)
    np.testing.assert_allclose(y_np, x_np * -1.5, rtol=1e-6, atol=1e-6)


@pytest.mark.cuda
def test_cuda_kernel_expression_tensor_runtime_scalar_dtype_mismatch_rejected_from_python():
    kernel = (
        CudaKernelExpression.builder("py_runtime_scalar_dtype_reject")
        .source(
            r"""
extern "C" __global__
void py_runtime_scalar_dtype_reject_kernel(const float* x, const float* alpha, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (*alpha) * x[i];
}
"""
        )
        .entry("py_runtime_scalar_dtype_reject_kernel")
        .input("x", thor.DataType.fp32)
        .tensor_runtime_scalar_input("alpha", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch(lambda ctx: CudaKernelLaunchConfig.grid_1d(ctx.numel("y"), block_size=128))
        .build()
    )

    outputs = kernel.apply({
        "x": ex.input("x"),
        "alpha": ex.tensor_runtime_scalar("alpha", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
    })
    eq = outputs.compile(device_num=0, use_fast_math=False)

    stream = Stream(gpu_num=0)
    x_gpu = _copy_numpy_to_gpu(np.array([1.0, 2.0, 3.0], dtype=np.float32), thor.DataType.fp32, stream)
    alpha_gpu = _copy_numpy_to_gpu(np.array([2.0], dtype=np.float32), thor.DataType.fp32, stream)
    alpha_binding = thor.physical.TensorScalarBinding(alpha_gpu, 0, thor.DataType.fp16)

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        eq.stamp({"x": x_gpu}, stream, tensor_scalar_inputs={"alpha": alpha_binding})


@pytest.mark.cuda
def test_cuda_kernel_expression_host_runtime_scalar_input_from_python():
    kernel = (
        CudaKernelExpression.builder("py_host_runtime_scalar_scale")
        .source(
            r"""
extern "C" __global__
void py_host_runtime_scalar_scale_kernel(const float* x, float alpha, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = alpha * x[i];
}
"""
        )
        .entry("py_host_runtime_scalar_scale_kernel")
        .input("x", thor.DataType.fp32)
        .host_runtime_scalar_input("alpha", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch(lambda ctx: CudaKernelLaunchConfig.grid_1d(ctx.numel("y"), block_size=128))
        .build()
    )

    outputs = kernel.apply({
        "x": ex.input("x"),
        "alpha": ex.runtime_scalar("alpha", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32),
    })
    eq = outputs.compile(device_num=0, use_fast_math=False)

    stream = Stream(gpu_num=0)
    x_np = np.array([1.0, -2.0, 3.0, 4.5, -5.0, 6.0], dtype=np.float32)
    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32, stream)

    stamped = eq.stamp({"x": x_gpu}, stream)
    with pytest.raises(RuntimeError, match="requires runtime scalar values"):
        stamped.run()

    stamped.run({"alpha": -2.0})

    y_np = _copy_gpu_to_numpy(stamped.output("y"), stream)
    np.testing.assert_allclose(y_np, x_np * -2.0, rtol=1e-6, atol=1e-6)


def test_cuda_kernel_expression_host_runtime_scalar_rejects_non_fp32_dtype_from_python():
    builder = (
        CudaKernelExpression.builder("py_host_runtime_scalar_dtype_reject")
        .source(
            r"""
extern "C" __global__
void py_host_runtime_scalar_dtype_reject_kernel(const float* x, float alpha, float* y, int64_t n) {}
"""
        )
        .entry("py_host_runtime_scalar_dtype_reject_kernel")
        .input("x", thor.DataType.fp32)
    )

    with pytest.raises(ValueError, match="Host runtime scalars are currently bound as fp32"):
        builder.host_runtime_scalar_input("alpha", thor.DataType.int64)
