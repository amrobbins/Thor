#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/CudaKernelSecurity.h"
#include "Utilities/Expression/FusedEquation.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using DataType = TensorDescriptor::DataType;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t numel(const Tensor& tensor) {
    uint64_t n = 1;
    for (uint64_t d : tensor.getDimensions())
        n *= d;
    return n;
}

Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, dims));
    if (numel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    }
    auto* ptr = static_cast<float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(cpu, stream);
    return gpu;
}

std::vector<float> copyToCpuValues(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    std::vector<float> values(numel(cpu));
    const auto* ptr = static_cast<const float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];
    return values;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1.0e-5f) << "index " << i;
    }
}

}  // namespace

TEST(CudaKernelExpression, SingleOutputRawPointerKernelRunsAsCustomStage) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);

    auto op = CudaKernelExpression::builder("scale")
                  .source(R"cuda(
extern "C" __global__
void scale_kernel(const float* x, float* y, float alpha, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
)cuda")
                  .entry("scale_kernel")
                  .input("x", DataType::FP32)
                  .output("y", DataType::FP32, {CudaKernelExpression::DimExpr::dim("x", 0), CudaKernelExpression::DimExpr::dim("x", 1)})
                  .scalar("alpha", DataType::FP32, 2.5f)
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    auto plan = op.asDynamicExpression().stamp({{"x", x}}, {}, stream);
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"CudaKernel"});
    plan.run();

    Tensor y = plan.output("y");
    EXPECT_EQ(y.getDimensions(), (std::vector<uint64_t>{2, 3}));
    expectNear(copyToCpuValues(y, stream), {2.5f, -5.0f, 7.5f, 11.25f, -12.5f, 15.0f});
}

TEST(CudaKernelExpression, MultiOutputKernelReturnsNamedOutputs) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f}, stream);

    auto op = CudaKernelExpression::builder("split_math")
                  .source(R"cuda(
extern "C" __global__
void split_math_kernel(const float* x, float* twice, float* plus_one, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        twice[i] = 2.0f * x[i];
        plus_one[i] = x[i] + 1.0f;
    }
}
)cuda")
                  .entry("split_math_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("twice", DataType::FP32, "x")
                  .outputLike("plus_one", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("twice"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 64;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("twice") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    auto plan = op.stamp({{"x", x}}, {}, stream);
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"CudaKernel"});
    plan.run();

    Tensor twice = plan.output("twice");
    Tensor plusOne = plan.output("plus_one");
    EXPECT_EQ(twice.getDimensions(), (std::vector<uint64_t>{2, 2, 2}));
    EXPECT_EQ(plusOne.getDimensions(), (std::vector<uint64_t>{2, 2, 2}));

    expectNear(copyToCpuValues(twice, stream), {2.0f, 4.0f, 6.0f, 8.0f, -2.0f, -4.0f, -6.0f, -8.0f});
    expectNear(copyToCpuValues(plusOne, stream), {2.0f, 3.0f, 4.0f, 5.0f, 0.0f, -1.0f, -2.0f, -3.0f});
}

TEST(CudaKernelExpression, RejectsInputDTypeMismatchBeforeLaunch) {
    Stream stream(0);
    Tensor x = makeGpuTensor({4}, {1.0f, 2.0f, 3.0f, 4.0f}, stream);

    auto op = CudaKernelExpression::builder("reject_dtype")
                  .source(R"cuda(
extern "C" __global__
void reject_dtype_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("reject_dtype_kernel")
                  .input("x", DataType::FP16)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      return CudaKernelLaunchConfig{dim3(static_cast<uint32_t>(ctx.numel("y")), 1, 1), dim3(1, 1, 1), 0};
                  })
                  .build();

    EXPECT_THROW((void)op.stamp({{"x", x}}, {}, stream), std::runtime_error);
}
TEST(CudaKernelExpression, TensorRuntimeScalarInputPassesDevicePointerThroughStagedPath) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);
    Tensor alphaBuffer = makeGpuTensor({2}, {123.0f, -1.5f}, stream);

    auto op = CudaKernelExpression::builder("runtime_scalar_scale")
                  .source(R"cuda(
extern "C" __global__
void runtime_scalar_scale_kernel(const float* x, const float* alpha, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (*alpha) * x[i];
    }
}
)cuda")
                  .entry("runtime_scalar_scale_kernel")
                  .input("x", DataType::FP32)
                  .tensorRuntimeScalarInput("alpha", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    Outputs outputs = op.apply({
        {"x", Expression::input("x")},
        {"alpha", Expression::tensorRuntimeScalar("alpha", DataType::FP32, DataType::FP32)},
    });
    FusedEquation eq = FusedEquation::compile(outputs.physicalOutputs(), 0);

    TensorScalarBinding alphaBinding{alphaBuffer, sizeof(float), DataType::FP32};
    auto plan = eq.stamp({{"x", x}}, stream, {{"alpha", alphaBinding}});
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"CudaKernel"});
    plan.run();

    Tensor y = plan.output("y");
    expectNear(copyToCpuValues(y, stream), {-1.5f, 3.0f, -4.5f, -6.75f, 7.5f, -9.0f});
}

TEST(CudaKernelExpression, HostRuntimeScalarInputPassesByValueThroughStagedPath) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);

    auto op = CudaKernelExpression::builder("host_runtime_scalar_scale")
                  .source(R"cuda(
extern "C" __global__
void host_runtime_scalar_scale_kernel(const float* x, float alpha, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
)cuda")
                  .entry("host_runtime_scalar_scale_kernel")
                  .input("x", DataType::FP32)
                  .hostRuntimeScalarInput("alpha", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    Outputs outputs = op.apply({
        {"x", Expression::input("x")},
        {"alpha", Expression::runtimeScalar("alpha", DataType::FP32, DataType::FP32)},
    });
    FusedEquation eq = FusedEquation::compile(outputs.physicalOutputs(), 0);

    auto plan = eq.stamp({{"x", x}}, stream);
    EXPECT_EQ(plan.stageKindNames(), std::vector<std::string>{"CudaKernel"});

    EXPECT_THROW(plan.run(), std::runtime_error);
    plan.run({{"alpha", -2.0f}});

    Tensor y = plan.output("y");
    expectNear(copyToCpuValues(y, stream), {-2.0f, 4.0f, -6.0f, -9.0f, 10.0f, -12.0f});
}

TEST(CudaKernelExpression, HostRuntimeScalarInputRejectsNonFp32DType) {
    EXPECT_THROW((void)CudaKernelExpression::builder("host_runtime_scalar_dtype_reject")
                     .source(R"cuda(
extern "C" __global__
void host_runtime_scalar_dtype_reject_kernel(const float* x, float alpha, float* y, int64_t n) {}
)cuda")
                     .entry("host_runtime_scalar_dtype_reject_kernel")
                     .input("x", DataType::FP32)
                     .hostRuntimeScalarInput("alpha", DataType::INT64),
                 std::invalid_argument);
}

TEST(CudaKernelExpression, SerializedCudaSourceIsInspectableAndRequiresUnsafeOptInToRunAfterLoad) {
    Stream stream(0);
    Tensor x = makeGpuTensor({2, 3}, {1.0f, -2.0f, 3.0f, 4.5f, -5.0f, 6.0f}, stream);

    auto op = CudaKernelExpression::builder("serializable_scale")
                  .source(R"cuda(
extern "C" __global__
void serializable_scale_kernel(const float* x, float* y, float alpha, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * x[i];
    }
}
)cuda")
                  .entry("serializable_scale_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("alpha", DataType::FP32, 3.0f)
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                  .build();

    CudaKernelSourceInspection directInfo;
    {
        const auto info = op.sourceInfo();
        directInfo.name = info.name;
        directInfo.entrypoint = info.entrypoint;
        directInfo.source = info.source;
        directInfo.compiled_source = info.compiled_source;
        directInfo.compiled_source_hash = info.source_hash;
        directInfo.loaded_source_compilation_allowed = info.loaded_source_compilation_allowed;
    }
    EXPECT_EQ(directInfo.name, "serializable_scale");
    EXPECT_NE(directInfo.source.find("serializable_scale_kernel"), std::string::npos);
    EXPECT_NE(directInfo.compiled_source.find("THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES"), std::string::npos);

    Outputs outputs = op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}});
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(outputs);
    nlohmann::json payload = definition.architectureJsonWithCudaKernelManifestSignature();

    ASSERT_TRUE(payload.contains("cuda_kernels"));
    ASSERT_TRUE(payload.contains("cuda_kernel_manifest_signature"));
    ASSERT_EQ(payload.at("cuda_kernels").size(), 1u);
    EXPECT_EQ(payload.at("cuda_kernels").at(0).at("name").get<std::string>(), "serializable_scale");
    EXPECT_NE(payload.at("cuda_kernels").at(0).at("source").get<std::string>().find("serializable_scale_kernel"),
              std::string::npos);
    const nlohmann::json& signatureJson = payload.at("cuda_kernel_manifest_signature");
    EXPECT_EQ(signatureJson.at("algorithm").get<std::string>(), "ed25519");
    EXPECT_FALSE(signatureJson.contains("public_key"));
    EXPECT_FALSE(signatureJson.at("public_key_fingerprint").get<std::string>().empty());
    std::vector<std::string> signingPublicKeys = definition.cudaKernelSigningPublicKeys();
    ASSERT_EQ(signingPublicKeys.size(), 1u);
    const std::string trustedPublicKey = signingPublicKeys.front();
    EXPECT_FALSE(trustedPublicKey.empty());
    EXPECT_NE(signatureJson.at("public_key_fingerprint").get<std::string>(), trustedPublicKey);

    std::vector<CudaKernelSourceInspection> serializedInfo = collectCudaKernelSourceInfo(payload);
    ASSERT_EQ(serializedInfo.size(), 1u);
    EXPECT_EQ(serializedInfo.front().name, "serializable_scale");
    EXPECT_NE(serializedInfo.front().source.find("serializable_scale_kernel"), std::string::npos);
    EXPECT_NE(serializedInfo.front().compiled_source.find("THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES"), std::string::npos);
    EXPECT_NE(serializedInfo.front().compiled_source.find("serializable_scale_kernel"), std::string::npos);
    EXPECT_FALSE(serializedInfo.front().loaded_source_compilation_allowed);
    EXPECT_EQ(serializedInfo.front().signing_public_key_fingerprint, signatureJson.at("public_key_fingerprint").get<std::string>());

    ExpressionDefinition loadedDefault = ExpressionDefinition::deserialize(payload);
    std::vector<CudaKernelSourceInspection> firstClassSourceInfo = loadedDefault.cudaKernelSourceInfo();
    ASSERT_EQ(firstClassSourceInfo.size(), 1u);
    EXPECT_EQ(firstClassSourceInfo.front().name, "serializable_scale");
    EXPECT_NE(firstClassSourceInfo.front().source.find("serializable_scale_kernel"), std::string::npos);
    EXPECT_EQ(loadedDefault.cudaKernelSources(), std::vector<std::string>{firstClassSourceInfo.front().source});

    nlohmann::json sourceInfo = loadedDefault.cudaKernelSourceInfoJson();
    ASSERT_EQ(sourceInfo.size(), 1u);
    EXPECT_FALSE(sourceInfo.at(0).at("loaded_source_compilation_allowed").get<bool>());
    EXPECT_NE(sourceInfo.at(0).at("compiled_source").get<std::string>().find("THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES"),
              std::string::npos);
    EXPECT_NE(sourceInfo.at(0).at("compiled_source").get<std::string>().find("serializable_scale_kernel"), std::string::npos);
    EXPECT_TRUE(sourceInfo.at(0).contains("signing_public_key_fingerprint"));
    EXPECT_FALSE(sourceInfo.at(0).contains("signing_public_key"));

    EXPECT_THROW((void)DynamicExpression::fromExpressionDefinition(loadedDefault).stamp({{"x", x}}, {}, stream), std::runtime_error);

    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload, true), std::runtime_error);

    nlohmann::json missingSignature = payload;
    missingSignature.erase("cuda_kernel_manifest_signature");
    EXPECT_THROW((void)ExpressionDefinition::deserialize(missingSignature, true, trustedPublicKey), std::runtime_error);

    nlohmann::json publicKeyInFingerprint = payload;
    publicKeyInFingerprint["cuda_kernel_manifest_signature"]["public_key_fingerprint"] = trustedPublicKey;
    try {
        (void)ExpressionDefinition::deserialize(publicKeyInFingerprint, true, trustedPublicKey);
        FAIL() << "Expected manifest public_key_fingerprint containing public key material to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("public_key_fingerprint contains public key material"), std::string::npos);
    }

    auto wrongKeyOp = CudaKernelExpression::builder("wrong_key_source")
                          .source(R"cuda(
extern "C" __global__
void wrong_key_source_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                          .entry("wrong_key_source_kernel")
                          .input("x", DataType::FP32)
                          .outputLike("y", DataType::FP32, "x")
                          .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                          .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                          .build();
    ExpressionDefinition wrongKeyDefinition =
        ExpressionDefinition::fromOutputs(wrongKeyOp.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));
    nlohmann::json wrongKeyPayload = wrongKeyDefinition.architectureJsonWithCudaKernelManifestSignature();
    EXPECT_FALSE(wrongKeyPayload.at("cuda_kernel_manifest_signature").contains("public_key"));
    std::vector<std::string> wrongSigningPublicKeys = wrongKeyDefinition.cudaKernelSigningPublicKeys();
    ASSERT_EQ(wrongSigningPublicKeys.size(), 1u);
    const std::string wrongTrustedPublicKey = wrongSigningPublicKeys.front();
    ASSERT_FALSE(wrongTrustedPublicKey.empty());
    ASSERT_NE(wrongTrustedPublicKey, trustedPublicKey);
    EXPECT_THROW((void)ExpressionDefinition::deserialize(payload, true, wrongTrustedPublicKey), std::runtime_error);

    nlohmann::json tampered = payload;
    tampered["cuda_kernels"][0]["source"] = tampered["cuda_kernels"][0]["source"].get<std::string>() + "\n// tampered\n";
    EXPECT_THROW((void)ExpressionDefinition::deserialize(tampered, true, trustedPublicKey), std::runtime_error);

    nlohmann::json tamperedLaunch = payload;
    tamperedLaunch["cuda_kernels"][0]["launch"]["block"] = 256;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(tamperedLaunch, true, trustedPublicKey), std::runtime_error);

    ExpressionDefinition loadedAllowed = ExpressionDefinition::deserialize(payload, true, trustedPublicKey);
    nlohmann::json allowedSourceInfo = loadedAllowed.cudaKernelSourceInfoJson();
    EXPECT_TRUE(allowedSourceInfo.at(0).at("loaded_source_compilation_allowed").get<bool>());

    auto plan = DynamicExpression::fromExpressionDefinition(loadedAllowed).stamp({{"x", x}}, {}, stream);
    plan.run();
    expectNear(copyToCpuValues(plan.output("y"), stream), {3.0f, -6.0f, 9.0f, 13.5f, -15.0f, 18.0f});
}


TEST(CudaKernelExpression, ArchitectureJsonDoesNotMintCudaManifestSignature) {
    auto op = CudaKernelExpression::builder("unsigned_inspection_scale")
                  .source(R"cuda(
extern "C" __global__
void unsigned_inspection_scale_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("unsigned_inspection_scale_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                  .build();

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
        op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));

    nlohmann::json unsignedPayload = definition.architectureJson();
    ASSERT_TRUE(unsignedPayload.contains("cuda_kernels"));
    EXPECT_FALSE(unsignedPayload.contains("cuda_kernel_manifest_signature"));
    std::vector<CudaKernelSourceInspection> unsignedInfo = collectCudaKernelSourceInfo(unsignedPayload);
    ASSERT_EQ(unsignedInfo.size(), 1u);
    EXPECT_NE(unsignedInfo.front().compiled_source.find("THOR_CUDA_KERNEL_EXPRESSION_FIXED_WIDTH_TYPES"), std::string::npos);
    EXPECT_NE(unsignedInfo.front().compiled_source.find("unsigned_inspection_scale_kernel"), std::string::npos);

    nlohmann::json signedPayload = definition.architectureJsonWithCudaKernelManifestSignature();
    ASSERT_TRUE(signedPayload.contains("cuda_kernel_manifest_signature"));
    ASSERT_EQ(definition.cudaKernelSigningPublicKeys().size(), 1u);

    nlohmann::json loadedUnsignedPayload = signedPayload;
    loadedUnsignedPayload.erase("cuda_kernel_manifest_signature");
    ExpressionDefinition loadedUnsigned = ExpressionDefinition::deserialize(loadedUnsignedPayload);
    EXPECT_FALSE(loadedUnsigned.architectureJson().contains("cuda_kernel_manifest_signature"));
    EXPECT_THROW((void)loadedUnsigned.cudaKernelSigningPublicKeys(), std::runtime_error);
    EXPECT_THROW((void)loadedUnsigned.architectureJsonWithCudaKernelManifestSignature(), std::runtime_error);
}

TEST(CudaKernelExpression, RecursiveModelSigningUsesOnePublicKeyForAllCudaExpressions) {
    auto opA = CudaKernelExpression::builder("model_signing_scale_a")
                   .source(R"cuda(
extern "C" __global__
void model_signing_scale_a_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                   .entry("model_signing_scale_a_kernel")
                   .input("x", DataType::FP32)
                   .outputLike("y", DataType::FP32, "x")
                   .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                   .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                   .build();
    auto opB = CudaKernelExpression::builder("model_signing_scale_b")
                   .source(R"cuda(
extern "C" __global__
void model_signing_scale_b_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}
)cuda")
                   .entry("model_signing_scale_b_kernel")
                   .input("x", DataType::FP32)
                   .outputLike("y", DataType::FP32, "x")
                   .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                   .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                   .build();

    ExpressionDefinition definitionA = ExpressionDefinition::fromOutputs(
        opA.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));
    ExpressionDefinition definitionB = ExpressionDefinition::fromOutputs(
        opB.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));

    nlohmann::json modelJson{{"layers", nlohmann::json::array()}};
    modelJson["layers"].push_back(nlohmann::json{{"expression", definitionA.architectureJson()}});
    modelJson["layers"].push_back(nlohmann::json{{"expression", definitionB.architectureJson()}});
    nlohmann::json unsignedModelJson = modelJson;

    std::vector<std::string> signingPublicKeys = cudaKernelGenerateAndAttachManifestSignatures(modelJson);
    ASSERT_EQ(signingPublicKeys.size(), 1u);
    const std::string& trustedPublicKey = signingPublicKeys.front();
    ASSERT_FALSE(trustedPublicKey.empty());

    const nlohmann::json& expressionA = modelJson.at("layers").at(0).at("expression");
    const nlohmann::json& expressionB = modelJson.at("layers").at(1).at("expression");
    ASSERT_TRUE(expressionA.contains("cuda_kernel_manifest_signature"));
    ASSERT_TRUE(expressionB.contains("cuda_kernel_manifest_signature"));
    EXPECT_EQ(expressionA.at("cuda_kernel_manifest_signature").at("public_key_fingerprint").get<std::string>(),
              expressionB.at("cuda_kernel_manifest_signature").at("public_key_fingerprint").get<std::string>());

    CudaKernelSignatureVerificationResult verificationA = cudaKernelVerifyManifestSignature(expressionA, trustedPublicKey);
    CudaKernelSignatureVerificationResult verificationB = cudaKernelVerifyManifestSignature(expressionB, trustedPublicKey);
    EXPECT_TRUE(verificationA.verified) << verificationA.message;
    EXPECT_TRUE(verificationB.verified) << verificationB.message;

    nlohmann::json modelJsonAgain = unsignedModelJson;
    std::vector<std::string> signingPublicKeysAgain = cudaKernelGenerateAndAttachManifestSignatures(modelJsonAgain);
    EXPECT_EQ(signingPublicKeysAgain, signingPublicKeys);
    EXPECT_EQ(modelJsonAgain.dump(), modelJson.dump());
}

TEST(CudaKernelExpression, MalformedCudaKernelGraphNodesAreRejectedDuringDeserializeValidation) {
    auto op = CudaKernelExpression::builder("validation_scale")
                  .source(R"cuda(
extern "C" __global__
void validation_scale_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("validation_scale_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launchGrid1D(CudaKernelExpression::DimExpr::numel("y"), 128)
                  .build();

    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(
        op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}}));
    nlohmann::json payload = definition.architectureJsonWithCudaKernelManifestSignature();

    auto findCudaOutputNodeIndex = [](const nlohmann::json& expression_json) -> size_t {
        const auto& nodes = expression_json.at("nodes");
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes.at(i).at("op").get<std::string>() == "cuda_kernel_output") {
                return i;
            }
        }
        throw std::runtime_error("test payload did not contain a CUDA kernel output node");
    };

    const size_t cudaNodeIndex = findCudaOutputNodeIndex(payload);

    nlohmann::json badOutputIndex = payload;
    badOutputIndex["nodes"][cudaNodeIndex]["cuda_kernel_output_index"] = 999;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(badOutputIndex), std::runtime_error);

    nlohmann::json badInputCount = payload;
    badInputCount["nodes"][cudaNodeIndex]["cuda_kernel_input_nodes"] = nlohmann::json::array();
    EXPECT_THROW((void)ExpressionDefinition::deserialize(badInputCount), std::runtime_error);

    nlohmann::json badInputKind = payload;
    const uint32_t kernelInputNode = payload["nodes"][cudaNodeIndex]["cuda_kernel_input_nodes"].at(0).get<uint32_t>();
    badInputKind["nodes"][kernelInputNode]["op"] = "runtime_scalar";
    EXPECT_THROW((void)ExpressionDefinition::deserialize(badInputKind), std::runtime_error);

    nlohmann::json badOutputDType = payload;
    badOutputDType["nodes"][cudaNodeIndex]["output_dtype"] = DataType::FP64;
    EXPECT_THROW((void)ExpressionDefinition::deserialize(badOutputDType), std::runtime_error);
}

TEST(CudaKernelExpression, NonSerializableLaunchCallbackIsRejectedWhenSavingExpressionDefinition) {
    auto op = CudaKernelExpression::builder("callback_launch_not_serializable")
                  .source(R"cuda(
extern "C" __global__
void callback_launch_not_serializable_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}
)cuda")
                  .entry("callback_launch_not_serializable_kernel")
                  .input("x", DataType::FP32)
                  .outputLike("y", DataType::FP32, "x")
                  .scalar("n", DataType::INT64, CudaKernelExpression::DimExpr::numel("y"))
                  .launch([](const CudaKernelExpression::LaunchContext& ctx) {
                      constexpr uint32_t block = 128;
                      const uint32_t grid = static_cast<uint32_t>((ctx.numel("y") + block - 1) / block);
                      return CudaKernelLaunchConfig{dim3(grid, 1, 1), dim3(block, 1, 1), 0};
                  })
                  .build();

    Outputs outputs = op.apply({{"x", Expression::input("x", DataType::FP32, DataType::FP32)}});
    ExpressionDefinition definition = ExpressionDefinition::fromOutputs(outputs);
    EXPECT_THROW((void)definition.architectureJson(), std::runtime_error);
}
