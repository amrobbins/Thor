#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Parameter/ParameterConstraint.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Parameter/ParameterConstraint.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Api = Thor;
namespace Impl = ThorImplementation;
using namespace std;

namespace {

class NoOpOptimizer final : public Impl::Optimizer {
   public:
    NoOpOptimizer() : Impl::Optimizer(0xC01157) {}

    void compile(const Impl::Tensor& weights, Stream& gradientUpdateStream, bool materializeDenseGradient = true) override {
        (void)materializeDenseGradient;
        this->weights = weights;
        this->gradientUpdateStream = gradientUpdateStream;
        compiled = true;
    }

    void updateWeights(uint32_t batchSize) override { (void)batchSize; }

    std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) override {
        (void)epoch;
        (void)batch;
        (void)batchesPerEpoch;
        return {};
    }

    std::unordered_map<std::string, float> getAllHyperParameters() override { return {}; }

    std::shared_ptr<Impl::Optimizer> clone() const override { return std::make_shared<NoOpOptimizer>(); }
};

Impl::Tensor makeCpuFp32Tensor(const std::vector<float>& values) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(Impl::DataType::FP32, {static_cast<uint64_t>(values.size())});
    Impl::Tensor tensor(cpu, descriptor);
    std::copy(values.begin(), values.end(), tensor.getMemPtr<float>());
    return tensor;
}

std::vector<float> copyGpuFp32TensorToVector(const Impl::Tensor& gpuTensor, Stream& stream) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host(cpu, gpuTensor.getDescriptor());
    host.copyFromAsync(gpuTensor, stream);
    stream.synchronize();
    const float* begin = host.getMemPtr<float>();
    return std::vector<float>(begin, begin + host.getTotalNumElements());
}

template <typename StorageT>
Impl::Tensor makeCpuFloatingTensor(Impl::DataType dataType, const std::vector<float>& values) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::TensorDescriptor descriptor(dataType, {static_cast<uint64_t>(values.size())});
    Impl::Tensor tensor(cpu, descriptor);
    StorageT* ptr = tensor.getMemPtr<StorageT>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = StorageT(values[i]);
    }
    return tensor;
}

template <typename StorageT>
std::vector<float> copyGpuFloatingTensorToFloatVector(const Impl::Tensor& gpuTensor, Stream& stream) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host(cpu, gpuTensor.getDescriptor());
    host.copyFromAsync(gpuTensor, stream);
    stream.synchronize();
    const StorageT* begin = host.getMemPtr<StorageT>();
    std::vector<float> result;
    result.reserve(host.getTotalNumElements());
    for (uint64_t i = 0; i < host.getTotalNumElements(); ++i) {
        result.push_back(static_cast<float>(begin[i]));
    }
    return result;
}


void expectConstraintProjectsFp32(std::shared_ptr<Impl::ParameterConstraint> constraint,
                                  const std::vector<float>& inputValues,
                                  const std::vector<float>& expectedValues) {
    Impl::TensorPlacement gpu(Impl::TensorPlacement::MemDevices::GPU, 0);
    Impl::TensorDescriptor inputDescriptor(Impl::DataType::FP32, {1});
    Impl::Tensor input(gpu, inputDescriptor);
    Stream stream(0);

    Impl::PhysicalParameter parameter("weights", true, {inputValues.size()}, Impl::DataType::FP32);
    parameter.compileStorage(input);
    parameter.setOptimizer(std::make_shared<NoOpOptimizer>());
    parameter.addConstraint(constraint);
    parameter.compileOptimizer(stream, /*inferenceOnly=*/false);

    Impl::Tensor hostValues = makeCpuFp32Tensor(inputValues);
    Impl::Tensor storage = parameter.getStorage().value();
    storage.copyFromAsync(hostValues, stream);
    stream.synchronize();

    ASSERT_TRUE(parameter.applyGradient(/*batchSize=*/1));
    std::vector<float> constrained = copyGpuFp32TensorToVector(storage, stream);

    ASSERT_EQ(constrained.size(), expectedValues.size());
    for (size_t i = 0; i < expectedValues.size(); ++i) {
        EXPECT_FLOAT_EQ(constrained[i], expectedValues[i]) << "index " << i;
    }
}

}  // namespace

TEST(ParameterConstraint, NonNegativeArchitectureJsonRoundTrips) {
    Api::NonNegativeParameterConstraint constraint;
    nlohmann::json j = constraint.architectureJson();

    EXPECT_EQ(j.at("version").get<std::string>(), Api::ParameterConstraint::getVersion());
    EXPECT_EQ(j.at("constraint_type").get<std::string>(), "non_negative");

    std::shared_ptr<Api::ParameterConstraint> roundTrip = Api::ParameterConstraint::deserialize(j);
    ASSERT_NE(roundTrip, nullptr);
    EXPECT_EQ(roundTrip->getConstraintType(), "non_negative");
}

TEST(ParameterConstraint, ParameterSpecificationCarriesAndSerializesConstraints) {
    auto initializer = Api::Glorot::Builder().build();
    auto nonNegative = std::make_shared<Api::NonNegativeParameterConstraint>();

    Api::ParameterSpecification parameter = Api::ParameterSpecification::Builder()
                                               .name("weights")
                                               .shape({4, 7})
                                               .dtype(Api::DataType::FP32)
                                               .initializer(initializer)
                                               .trainable(true)
                                               .constraint(nonNegative)
                                               .build();

    EXPECT_TRUE(parameter.hasConstraints());
    ASSERT_EQ(parameter.getConstraints().size(), 1);
    EXPECT_EQ(parameter.getConstraints()[0]->getConstraintType(), "non_negative");

    nlohmann::json j = parameter.architectureJson();
    ASSERT_TRUE(j.contains("constraints"));
    ASSERT_EQ(j.at("constraints").size(), 1);
    EXPECT_EQ(j.at("constraints")[0].at("constraint_type").get<std::string>(), "non_negative");

    std::shared_ptr<Impl::PhysicalParameter> physical = parameter.stamp();
    ASSERT_NE(physical, nullptr);
    EXPECT_TRUE(physical->hasConstraints());
    ASSERT_EQ(physical->getConstraints().size(), 1);
    EXPECT_EQ(physical->getConstraints()[0]->getConstraintType(), "non_negative");
}

TEST(ParameterConstraint, NonNegativeAppliesAfterPhysicalParameterGradientUpdate) {
    Impl::TensorPlacement gpu(Impl::TensorPlacement::MemDevices::GPU, 0);
    Impl::TensorDescriptor inputDescriptor(Impl::DataType::FP32, {1});
    Impl::Tensor input(gpu, inputDescriptor);
    Stream stream(0);

    Impl::PhysicalParameter parameter("weights", true, {6}, Impl::DataType::FP32);
    parameter.compileStorage(input);
    parameter.setOptimizer(std::make_shared<NoOpOptimizer>());
    parameter.addConstraint(std::make_shared<Impl::NonNegativeParameterConstraint>());
    parameter.compileOptimizer(stream, /*inferenceOnly=*/false);

    Impl::Tensor hostValues = makeCpuFp32Tensor({-3.0f, -0.25f, 0.0f, 0.5f, 4.0f, -9.0f});
    Impl::Tensor storage = parameter.getStorage().value();
    storage.copyFromAsync(hostValues, stream);
    stream.synchronize();

    ASSERT_TRUE(parameter.applyGradient(/*batchSize=*/1));
    std::vector<float> constrained = copyGpuFp32TensorToVector(storage, stream);

    const std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.5f, 4.0f, 0.0f};
    ASSERT_EQ(constrained.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(constrained[i], expected[i]) << "index " << i;
    }
}

TEST(ParameterConstraint, NonNegativeSupportsFp8E4M3) {
    Impl::TensorPlacement gpu(Impl::TensorPlacement::MemDevices::GPU, 0);
    Impl::TensorDescriptor inputDescriptor(Impl::DataType::FP32, {1});
    Impl::Tensor input(gpu, inputDescriptor);
    Stream stream(0);

    Impl::PhysicalParameter parameter("weights", true, {6}, Impl::DataType::FP8_E4M3);
    parameter.compileStorage(input);
    parameter.setOptimizer(std::make_shared<NoOpOptimizer>());
    parameter.addConstraint(std::make_shared<Impl::NonNegativeParameterConstraint>());
    parameter.compileOptimizer(stream, /*inferenceOnly=*/false);

    Impl::Tensor hostValues = makeCpuFloatingTensor<__nv_fp8_e4m3>(Impl::DataType::FP8_E4M3,
                                                                   {-3.0f, -0.25f, 0.0f, 0.5f, 4.0f, -9.0f});
    Impl::Tensor storage = parameter.getStorage().value();
    storage.copyFromAsync(hostValues, stream);
    stream.synchronize();

    ASSERT_TRUE(parameter.applyGradient(/*batchSize=*/1));
    std::vector<float> constrained = copyGpuFloatingTensorToFloatVector<__nv_fp8_e4m3>(storage, stream);

    const std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.5f, 4.0f, 0.0f};
    ASSERT_EQ(constrained.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(constrained[i], expected[i]) << "index " << i;
    }
}

TEST(ParameterConstraint, NonNegativeSupportsFp8E5M2) {
    Impl::TensorPlacement gpu(Impl::TensorPlacement::MemDevices::GPU, 0);
    Impl::TensorDescriptor inputDescriptor(Impl::DataType::FP32, {1});
    Impl::Tensor input(gpu, inputDescriptor);
    Stream stream(0);

    Impl::PhysicalParameter parameter("weights", true, {6}, Impl::DataType::FP8_E5M2);
    parameter.compileStorage(input);
    parameter.setOptimizer(std::make_shared<NoOpOptimizer>());
    parameter.addConstraint(std::make_shared<Impl::NonNegativeParameterConstraint>());
    parameter.compileOptimizer(stream, /*inferenceOnly=*/false);

    Impl::Tensor hostValues = makeCpuFloatingTensor<__nv_fp8_e5m2>(Impl::DataType::FP8_E5M2,
                                                                   {-3.0f, -0.25f, 0.0f, 0.5f, 4.0f, -9.0f});
    Impl::Tensor storage = parameter.getStorage().value();
    storage.copyFromAsync(hostValues, stream);
    stream.synchronize();

    ASSERT_TRUE(parameter.applyGradient(/*batchSize=*/1));
    std::vector<float> constrained = copyGpuFloatingTensorToFloatVector<__nv_fp8_e5m2>(storage, stream);

    const std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.5f, 4.0f, 0.0f};
    ASSERT_EQ(constrained.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(constrained[i], expected[i]) << "index " << i;
    }
}


TEST(ParameterConstraint, AdditionalScalarBoundConstraintArchitectureJsonRoundTrips) {
    const std::vector<std::shared_ptr<Api::ParameterConstraint>> constraints = {
        std::make_shared<Api::NonPositiveParameterConstraint>(),
        std::make_shared<Api::MinParameterConstraint>(-0.25),
        std::make_shared<Api::MaxParameterConstraint>(0.75),
        std::make_shared<Api::MinMaxParameterConstraint>(-0.5, 0.5),
    };
    const std::vector<std::string> expectedTypes = {"non_positive", "min", "max", "min_max"};

    for (size_t i = 0; i < constraints.size(); ++i) {
        nlohmann::json j = constraints[i]->architectureJson();
        EXPECT_EQ(j.at("version").get<std::string>(), Api::ParameterConstraint::getVersion());
        EXPECT_EQ(j.at("constraint_type").get<std::string>(), expectedTypes[i]);

        if (expectedTypes[i] == "min") {
            EXPECT_DOUBLE_EQ(j.at("min_value").get<double>(), -0.25);
        } else if (expectedTypes[i] == "max") {
            EXPECT_DOUBLE_EQ(j.at("max_value").get<double>(), 0.75);
        } else if (expectedTypes[i] == "min_max") {
            EXPECT_DOUBLE_EQ(j.at("min_value").get<double>(), -0.5);
            EXPECT_DOUBLE_EQ(j.at("max_value").get<double>(), 0.5);
        }

        std::shared_ptr<Api::ParameterConstraint> roundTrip = Api::ParameterConstraint::deserialize(j);
        ASSERT_NE(roundTrip, nullptr);
        EXPECT_EQ(roundTrip->getConstraintType(), expectedTypes[i]);
    }
}

TEST(ParameterConstraint, MinMaxRejectsReversedBounds) {
    EXPECT_THROW(Api::MinMaxParameterConstraint(2.0, -1.0), std::runtime_error);
    EXPECT_THROW(Impl::MinMaxParameterConstraint(2.0, -1.0), std::runtime_error);
}

TEST(ParameterConstraint, AdditionalScalarBoundConstraintsApplyAfterPhysicalParameterGradientUpdate) {
    const std::vector<float> input = {-2.0f, -0.25f, 0.0f, 0.5f, 2.0f};

    expectConstraintProjectsFp32(std::make_shared<Impl::NonPositiveParameterConstraint>(),
                                 input,
                                 std::vector<float>({-2.0f, -0.25f, 0.0f, 0.0f, 0.0f}));
    expectConstraintProjectsFp32(std::make_shared<Impl::MinParameterConstraint>(-0.5),
                                 input,
                                 std::vector<float>({-0.5f, -0.25f, 0.0f, 0.5f, 2.0f}));
    expectConstraintProjectsFp32(std::make_shared<Impl::MaxParameterConstraint>(0.75),
                                 input,
                                 std::vector<float>({-2.0f, -0.25f, 0.0f, 0.5f, 0.75f}));
    expectConstraintProjectsFp32(std::make_shared<Impl::MinMaxParameterConstraint>(-0.5, 0.75),
                                 input,
                                 std::vector<float>({-0.5f, -0.25f, 0.0f, 0.5f, 0.75f}));
}

TEST(ParameterConstraint, AdditionalScalarBoundConstraintsSupportDenseExpressionFusion) {
    EXPECT_TRUE(Impl::NonPositiveParameterConstraint().supportsDenseExpressionFusion());
    EXPECT_TRUE(Impl::MinParameterConstraint(-1.0).supportsDenseExpressionFusion());
    EXPECT_TRUE(Impl::MaxParameterConstraint(1.0).supportsDenseExpressionFusion());
    EXPECT_TRUE(Impl::MinMaxParameterConstraint(-1.0, 1.0).supportsDenseExpressionFusion());
}
