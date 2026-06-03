import json
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


def test_cuda_kernel_expression_exposes_declared_names_from_python():
    kernel = (
        CudaKernelExpression.builder("py_declared_names")
        .source(
            r"""
extern "C" __global__
void py_declared_names_kernel(const float* x, const float* scale, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = (*scale) * x[i];
}
"""
        )
        .entry("py_declared_names_kernel")
        .input("x", thor.DataType.fp32)
        .tensor_runtime_scalar_input("scale", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch_grid_1d(CudaKernelExpression.numel("y"), block_size=128)
        .build()
    )

    assert kernel.input_names() == ["x", "scale"]
    assert kernel.tensor_input_names() == ["x"]
    assert kernel.output_names() == ["y"]


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
    eq = outputs.compile(device_num=0)

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
    eq = split.compile(device_num=0)

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
    eq = outputs.compile(device_num=0)

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
    eq = outputs.compile(device_num=0)

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
    eq = outputs.compile(device_num=0)

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
    eq = outputs.compile(device_num=0)

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
    eq = outputs.compile(device_num=0)

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


@pytest.mark.cuda
def test_cuda_kernel_expression_serialized_source_is_inspectable_and_requires_opt_in_from_python():
    kernel = (
        CudaKernelExpression.builder("py_serializable_scale")
        .source(
            r"""
extern "C" __global__
void py_serializable_scale_kernel(const float* x, float* y, float alpha, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = alpha * x[i];
}
"""
        )
        .entry("py_serializable_scale_kernel")
        .input("x", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("alpha", thor.DataType.fp32, 3.0)
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch_grid_1d(CudaKernelExpression.numel("y"), block_size=128)
        .build()
    )

    kernel_source_info = kernel.source_info()
    assert kernel_source_info["name"] == "py_serializable_scale"
    assert kernel_source_info["entrypoint"] == "py_serializable_scale_kernel"
    assert "py_serializable_scale_kernel" in kernel_source_info["source"]
    assert "THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES" in kernel_source_info["compiled_source"]

    definition_to_save = thor.physical.ExpressionDefinition.from_outputs(
        kernel.apply({"x": ex.input("x", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32)})
    )
    payload = definition_to_save.to_json()
    key_sets = thor.physical.cuda_kernel_out_of_band_keys_from_json(payload)
    assert len(key_sets) == 1
    trusted_public_key = key_sets[0]["signing_public_key"]
    trusted_source_decryption_key = key_sets[0]["source_decryption_key"]

    payload_json = json.loads(payload)
    assert payload_json["cuda_kernels"][0]["name"] == "py_serializable_scale"
    assert "source" not in payload_json["cuda_kernels"][0]
    assert payload_json["cuda_kernels"][0]["encrypted_source"]
    assert payload_json["cuda_kernels"][0]["source_encryption"]["algorithm"] == "aes-256-gcm"
    # Entrypoint names remain plaintext ABI metadata; the CUDA source body must not.
    assert "y[i] = alpha * x[i]" not in payload
    signature_json = payload_json["cuda_kernel_manifest_signature"]
    assert signature_json["algorithm"] == "ed25519"
    assert "public_key" not in signature_json
    assert signature_json["public_key_fingerprint"]
    assert trusted_public_key
    assert trusted_source_decryption_key
    assert signature_json["public_key_fingerprint"] != trusted_public_key
    assert payload_json["cuda_kernels"][0]["source_encryption"]["source_decryption_key_fingerprint"] != trusted_source_decryption_key
    assert thor.physical.cuda_kernel_signing_public_keys_from_json(payload) == [trusted_public_key]

    serialized_source_info = thor.physical.cuda_kernel_source_info_from_json(payload)
    assert len(serialized_source_info) == 1
    assert serialized_source_info[0]["name"] == "py_serializable_scale"
    assert serialized_source_info[0]["entrypoint"] == "py_serializable_scale_kernel"
    assert serialized_source_info[0]["source_encrypted"] is True
    assert serialized_source_info[0]["source"] == ""
    assert serialized_source_info[0]["compiled_source"] == ""
    assert serialized_source_info[0]["source_encryption_algorithm"] == "aes-256-gcm"
    assert serialized_source_info[0]["loaded_source_compilation_allowed"] is False
    assert serialized_source_info[0]["signing_public_key_fingerprint"] == signature_json["public_key_fingerprint"]
    assert "signing_public_key" not in serialized_source_info[0]

    with pytest.raises(RuntimeError, match="trusted Ed25519 public key|required to load encrypted|source decryption key"):
        thor.physical.ExpressionDefinition.from_json(payload)

    definition = thor.physical.ExpressionDefinition.from_json(
        payload,
        trusted_cuda_kernel_public_key=trusted_public_key,
        trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
    )
    first_class_source_info = definition.cuda_kernel_source_info()
    assert len(first_class_source_info) == 1
    assert first_class_source_info[0]["name"] == "py_serializable_scale"
    assert "py_serializable_scale_kernel" in first_class_source_info[0]["source"]
    assert definition.cuda_kernel_sources() == [first_class_source_info[0]["source"]]

    source_info = json.loads(definition.cuda_kernel_source_info_json())
    assert source_info[0]["loaded_source_compilation_allowed"] is False
    assert "THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES" in source_info[0]["compiled_source"]
    assert "py_serializable_scale_kernel" in source_info[0]["compiled_source"]
    assert source_info[0]["signing_public_key_fingerprint"]
    assert "signing_public_key" not in source_info[0]

    dynamic_expression = thor.physical.DynamicExpression.from_expression_definition(definition)

    stream = Stream(gpu_num=0)
    x_np = np.array([1.0, -2.0, 3.0, 4.5, -5.0, 6.0], dtype=np.float32)
    x_gpu = _copy_numpy_to_gpu(x_np, thor.DataType.fp32, stream)

    with pytest.raises(RuntimeError, match="Refusing to compile CudaKernelExpression"):
        dynamic_expression.stamp({"x": x_gpu}, {}, stream)

    with pytest.raises(RuntimeError, match="trusted Ed25519 public key"):
        thor.physical.ExpressionDefinition.from_json(payload, allow_unsafe_loaded_cuda_kernel_source=True)
    with pytest.raises(RuntimeError, match="source decryption key"):
        thor.physical.ExpressionDefinition.from_json(payload, trusted_cuda_kernel_public_key=trusted_public_key)

    missing_signature_payload_json = json.loads(payload)
    del missing_signature_payload_json["cuda_kernel_manifest_signature"]
    with pytest.raises(RuntimeError, match="no cuda_kernel_manifest_signature"):
        thor.physical.ExpressionDefinition.from_json(
            json.dumps(missing_signature_payload_json),
            allow_unsafe_loaded_cuda_kernel_source=True,
            trusted_cuda_kernel_public_key=trusted_public_key,
            trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
        )

    encrypted_with_extra_plaintext_source_json = json.loads(payload)
    encrypted_with_extra_plaintext_source_json["cuda_kernels"][0]["source"] = kernel_source_info["source"]
    with pytest.raises(RuntimeError, match="plaintext CUDA source"):
        thor.physical.ExpressionDefinition.from_json(
            json.dumps(encrypted_with_extra_plaintext_source_json),
            allow_unsafe_loaded_cuda_kernel_source=True,
            trusted_cuda_kernel_public_key=trusted_public_key,
            trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
        )

    plaintext_only_source_json = json.loads(payload)
    plaintext_only_source_json["cuda_kernels"][0]["source"] = kernel_source_info["source"]
    del plaintext_only_source_json["cuda_kernels"][0]["encrypted_source"]
    del plaintext_only_source_json["cuda_kernels"][0]["source_encryption"]
    with pytest.raises(RuntimeError, match="plaintext CUDA source"):
        thor.physical.ExpressionDefinition.from_json(
            json.dumps(plaintext_only_source_json),
            allow_unsafe_loaded_cuda_kernel_source=True,
            trusted_cuda_kernel_public_key=trusted_public_key,
            trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
        )

    public_key_in_fingerprint_payload_json = json.loads(payload)
    public_key_in_fingerprint_payload_json["cuda_kernel_manifest_signature"]["public_key_fingerprint"] = trusted_public_key
    with pytest.raises(RuntimeError, match="public_key_fingerprint contains public key material"):
        thor.physical.ExpressionDefinition.from_json(
            json.dumps(public_key_in_fingerprint_payload_json),
            allow_unsafe_loaded_cuda_kernel_source=True,
            trusted_cuda_kernel_public_key=trusted_public_key,
            trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
        )

    wrong_key_kernel = (
        CudaKernelExpression.builder("py_wrong_key_source")
        .source(
            r"""
extern "C" __global__
void py_wrong_key_source_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i];
}
"""
        )
        .entry("py_wrong_key_source_kernel")
        .input("x", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch_grid_1d(CudaKernelExpression.numel("y"), block_size=128)
        .build()
    )
    wrong_key_definition = thor.physical.ExpressionDefinition.from_outputs(
        wrong_key_kernel.apply({"x": ex.input("x", output_dtype=thor.DataType.fp32, compute_dtype=thor.DataType.fp32)})
    )
    wrong_key_payload = wrong_key_definition.to_json()
    wrong_key_payload_json = json.loads(wrong_key_payload)
    assert "public_key" not in wrong_key_payload_json["cuda_kernel_manifest_signature"]
    wrong_key_sets = thor.physical.cuda_kernel_out_of_band_keys_from_json(wrong_key_payload)
    assert len(wrong_key_sets) == 1
    wrong_trusted_public_key = wrong_key_sets[0]["signing_public_key"]
    assert wrong_trusted_public_key != trusted_public_key
    with pytest.raises(RuntimeError, match="does not match"):
        thor.physical.ExpressionDefinition.from_json(
            payload,
            allow_unsafe_loaded_cuda_kernel_source=True,
            trusted_cuda_kernel_public_key=wrong_trusted_public_key,
            trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
        )

    tampered_payload_json = json.loads(payload)
    tampered_payload_json["cuda_kernels"][0]["encrypted_source"] += "00"
    with pytest.raises(RuntimeError, match="signature verification failed|SHA-256 hash"):
        thor.physical.ExpressionDefinition.from_json(
            json.dumps(tampered_payload_json),
            allow_unsafe_loaded_cuda_kernel_source=True,
            trusted_cuda_kernel_public_key=trusted_public_key,
            trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
        )

    tampered_launch_payload_json = json.loads(payload)
    tampered_launch_payload_json["cuda_kernels"][0]["launch"]["block"] = 256
    with pytest.raises(RuntimeError, match="signature verification failed|SHA-256 hash"):
        thor.physical.ExpressionDefinition.from_json(
            json.dumps(tampered_launch_payload_json),
            allow_unsafe_loaded_cuda_kernel_source=True,
            trusted_cuda_kernel_public_key=trusted_public_key,
            trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
        )

    allowed_definition = thor.physical.ExpressionDefinition.from_json(
        payload,
        allow_unsafe_loaded_cuda_kernel_source=True,
        trusted_cuda_kernel_public_key=trusted_public_key,
        trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
    )
    allowed_source_info = json.loads(allowed_definition.cuda_kernel_source_info_json())
    assert allowed_source_info[0]["loaded_source_compilation_allowed"] is True

    stamped = thor.physical.DynamicExpression.from_expression_definition(allowed_definition).stamp({"x": x_gpu}, {}, stream)
    stamped.run()
    y_np = _copy_gpu_to_numpy(stamped.output("y"), stream)
    np.testing.assert_allclose(y_np, x_np * 3.0, rtol=1e-6, atol=1e-6)



def _build_serializable_cuda_kernel_custom_layer_network(name: str) -> thor.Network:
    network = thor.Network(name)
    x = thor.layers.NetworkInput(network, "x", [4], thor.DataType.fp32).get_feature_output()

    kernel = (
        CudaKernelExpression.builder(f"{name}_scale")
        .source(
            r"""
extern "C" __global__
void cuda_kernel_save_key_capture_scale_kernel(const float* feature_input, float* feature_output, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    feature_output[i] = 2.0f * feature_input[i];
}
"""
        )
        .entry("cuda_kernel_save_key_capture_scale_kernel")
        .input("feature_input", thor.DataType.fp32)
        .output_like("feature_output", thor.DataType.fp32, "feature_input")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("feature_output"))
        .launch_grid_1d(CudaKernelExpression.numel("feature_output"), block_size=128)
        .build()
    )

    layer = thor.layers.CustomLayer(network=network, inputs=x, build=kernel.as_dynamic_expression())
    thor.layers.NetworkOutput(network, "out", layer["feature_output"], thor.DataType.fp32)
    return network

def test_python_defined_cuda_kernel_expression_serializes_through_custom_layer():
    network = thor.Network("py_cuda_kernel_custom_layer_serializable")
    x = thor.layers.NetworkInput(network, "x", [4], thor.DataType.fp32).get_feature_output()

    kernel = (
        CudaKernelExpression.builder("py_custom_layer_serializable_scale")
        .source(
            r"""
extern "C" __global__
void py_custom_layer_serializable_scale_kernel(const float* feature_input, float* feature_output, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    feature_output[i] = 2.0f * feature_input[i];
}
"""
        )
        .entry("py_custom_layer_serializable_scale_kernel")
        .input("feature_input", thor.DataType.fp32)
        .output_like("feature_output", thor.DataType.fp32, "feature_input")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("feature_output"))
        .launch_grid_1d(CudaKernelExpression.numel("feature_output"), block_size=128)
        .build()
    )

    layer = thor.layers.CustomLayer(network=network, inputs=x, build=kernel.as_dynamic_expression())
    thor.layers.NetworkOutput(network, "out", layer["feature_output"], thor.DataType.fp32)

    architecture_payload = network.get_architecture_json()
    architecture = json.loads(architecture_payload)
    key_sets = thor.physical.cuda_kernel_out_of_band_keys_from_json(architecture_payload)
    assert len(key_sets) == 1
    signing_public_keys = network.cuda_kernel_signing_public_keys()
    assert signing_public_keys == [key_sets[0]["signing_public_key"]]
    assert thor.physical.cuda_kernel_signing_public_keys_from_json(architecture_payload) == signing_public_keys
    assert network.get_architecture_json() == architecture_payload
    assert network.cuda_kernel_signing_public_keys() == signing_public_keys
    trusted_public_key = key_sets[0]["signing_public_key"]
    trusted_source_decryption_key = key_sets[0]["source_decryption_key"]
    custom_layers = [layer_json for layer_json in architecture["layers"] if layer_json["layer_type"] == "custom_layer"]
    assert len(custom_layers) == 1

    expression_json = custom_layers[0]["expression"]
    assert expression_json["cuda_kernels"][0]["name"] == "py_custom_layer_serializable_scale"
    assert "source" not in expression_json["cuda_kernels"][0]
    assert expression_json["cuda_kernels"][0]["encrypted_source"]
    # Entrypoint names remain plaintext ABI metadata; the CUDA source body must not.
    assert "feature_output[i] = 2.0f * feature_input[i]" not in architecture_payload
    assert expression_json["cuda_kernel_manifest_signature"]["algorithm"] == "ed25519"
    assert "public_key" not in expression_json["cuda_kernel_manifest_signature"]
    assert expression_json["cuda_kernel_manifest_signature"]["public_key_fingerprint"]

    network_source_info = network.cuda_kernel_source_info()
    assert len(network_source_info) == 1
    assert network_source_info[0]["name"] == "py_custom_layer_serializable_scale"
    assert "py_custom_layer_serializable_scale_kernel" in network_source_info[0]["source"]
    assert network.cuda_kernel_sources() == [network_source_info[0]["source"]]

    payload = json.dumps(expression_json)
    definition = thor.physical.ExpressionDefinition.from_json(
        payload,
        allow_unsafe_loaded_cuda_kernel_source=True,
        trusted_cuda_kernel_public_key=trusted_public_key,
        trusted_cuda_kernel_source_decryption_key=trusted_source_decryption_key,
    )
    assert definition.has_cuda_kernel_expressions is True


def test_network_cuda_kernel_save_key_capture_is_required_for_training_and_save(tmp_path):
    network = _build_serializable_cuda_kernel_custom_layer_network("py_cuda_kernel_save_key_capture_required")
    assert network.has_cuda_kernel_expressions() is True
    assert network.cuda_kernel_save_key_capture_configured() is False

    with pytest.raises(RuntimeError, match="Refusing to place a training network.*save-key capture"):
        network.place(batch_size=1, inference_only=False)

    with pytest.raises(RuntimeError, match="Refusing to save.*save-key capture"):
        network.save(str(tmp_path / "model_without_key_capture"), overwrite=True)

    key_path = tmp_path / "cuda_kernel_keys.json"
    network.capture_cuda_kernel_save_keys_to_file(str(key_path))
    assert network.cuda_kernel_save_key_capture_configured() is True

    pending = json.loads(key_path.read_text())
    assert pending["type"] == "thor.cuda_kernel_expression_out_of_band_keys"
    assert pending["status"] == "pending"
    assert pending["keys"] == []

    network.save(str(tmp_path / "model_with_key_capture"), overwrite=True)

    captured = json.loads(key_path.read_text())
    assert captured["status"] == "complete"
    assert len(captured["keys"]) == 1
    assert captured["keys"][0]["signing_public_key"]
    assert captured["keys"][0]["source_decryption_key"]

    with pytest.raises(RuntimeError, match="already exists"):
        network.capture_cuda_kernel_save_keys_to_file(str(key_path))


def test_cuda_kernel_expression_nonserializable_launch_callback_is_rejected_when_serializing_from_python():
    kernel = (
        CudaKernelExpression.builder("py_callback_launch_not_serializable")
        .source(
            r"""
extern "C" __global__
void py_callback_launch_not_serializable_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i];
}
"""
        )
        .entry("py_callback_launch_not_serializable_kernel")
        .input("x", thor.DataType.fp32)
        .output_like("y", thor.DataType.fp32, "x")
        .scalar("n", thor.DataType.int64, CudaKernelExpression.numel("y"))
        .launch(lambda ctx: CudaKernelLaunchConfig.grid_1d(ctx.numel("y"), block_size=128))
        .build()
    )

    with pytest.raises(RuntimeError, match="non-serializable launch callback"):
        kernel.apply({"x": ex.input("x")}).to_json()
