#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/Concatenate.h"
#include "DeepLearning/Implementation/Layers/Utility/DeviceCrossing.h"
#include "DeepLearning/Implementation/Layers/Utility/FiniteCheck.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Layers/Utility/Reshape.h"
#include "DeepLearning/Implementation/Layers/Utility/Slice.h"
#include "DeepLearning/Implementation/Layers/Utility/Split.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

using std::pair;
using std::set;
using std::vector;

using namespace ThorImplementation;
using namespace std;

class BackpropDescriptorSinkLayer : public ThorImplementation::Layer {
   public:
    std::optional<ThorImplementation::Tensor> connectToPreviousLayer(ThorImplementation::Layer *previousLayer,
                                                                     std::optional<ThorImplementation::Tensor> featureInput,
                                                                     Stream stream,
                                                                     bool backPropagateError,
                                                                     int loaderConnectionType = 0) override {
        (void)loaderConnectionType;
        THOR_THROW_IF_FALSE(!compiled);
        THOR_THROW_IF_FALSE(!this->previousLayer.has_value());
        THOR_THROW_IF_FALSE(!this->featureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.has_value());

        this->previousLayer = previousLayer;
        this->stream = stream;
        this->featureInput = featureInput;
        if (backPropagateError)
            this->errorOutput = featureInput.value().clone();
        else
            this->errorOutput = std::nullopt;
        return this->errorOutput;
    }

    void infer(std::optional<ThorImplementation::Tensor> inputTensor,
               std::optional<ThorImplementation::Tensor> outputTensor,
               Stream stream) override {
        (void)inputTensor;
        (void)outputTensor;
        (void)stream;
    }

    void backProp(std::optional<ThorImplementation::Tensor> dataIn,
                  std::optional<ThorImplementation::Tensor> errorIn,
                  std::optional<ThorImplementation::Tensor> errorOut,
                  Stream stream) override {
        (void)dataIn;
        (void)errorIn;
        (void)errorOut;
        (void)stream;
    }
};

template <typename T>
struct TensorFanoutSumDTypeTraits;

template <>
struct TensorFanoutSumDTypeTraits<__nv_bfloat16> {
    static constexpr DataType dtype = DataType::BF16;
};

template <>
struct TensorFanoutSumDTypeTraits<__nv_fp8_e4m3> {
    static constexpr DataType dtype = DataType::FP8_E4M3;
};

template <>
struct TensorFanoutSumDTypeTraits<__nv_fp8_e5m2> {
    static constexpr DataType dtype = DataType::FP8_E5M2;
};

template <typename T>
void writeTensorFanoutSumValues(Tensor &tensor, const std::vector<float> &values) {
    T *mem = reinterpret_cast<T *>(tensor.getMemPtr());
    for (uint32_t i = 0; i < values.size(); ++i) {
        mem[i] = T(values[i]);
    }
}

template <typename T>
float readTensorFanoutSumValue(const Tensor &tensor, uint32_t index) {
    const T *mem = reinterpret_cast<const T *>(tensor.getMemPtr());
    return static_cast<float>(mem[index]);
}

template <typename T>
void writeUtilityLayerValue(Tensor &tensor, uint64_t index, float value) {
    reinterpret_cast<T *>(tensor.getMemPtr())[index] = T(value);
}

template <typename T>
float readUtilityLayerValue(const Tensor &tensor, uint64_t index) {
    return static_cast<float>(reinterpret_cast<const T *>(tensor.getMemPtr())[index]);
}

template <typename T>
void expectTensorFanoutSumsBackwardErrorsForDType() {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const std::vector<unsigned long> dimensions{2, 4};
    const uint32_t elementCount = 8;
    TensorDescriptor descriptor(TensorFanoutSumDTypeTraits<T>::dtype, dimensions);

    Tensor firstGradientCpu(cpuPlacement, descriptor);
    Tensor secondGradientCpu(cpuPlacement, descriptor);
    Tensor summedGradientCpu(cpuPlacement, descriptor);

    const std::vector<float> firstGradient{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
    const std::vector<float> secondGradient{-3.0f, 5.0f, -1.0f, 8.0f, -2.0f, 10.0f, -4.0f, 12.0f};
    writeTensorFanoutSumValues<T>(firstGradientCpu, firstGradient);
    writeTensorFanoutSumValues<T>(secondGradientCpu, secondGradient);

    auto input = make_shared<NetworkInput>(gpuPlacement, TensorFanoutSumDTypeTraits<T>::dtype, dimensions);
    auto upstream = make_shared<NoOpLayer>();
    auto fanout = make_shared<TensorFanout>();
    auto firstSink = make_shared<BackpropDescriptorSinkLayer>();
    auto secondSink = make_shared<BackpropDescriptorSinkLayer>();

    input->connectToNextLayer(upstream.get());
    upstream->connectToNextLayer(fanout.get());
    fanout->connectToNextLayer(firstSink.get());
    fanout->connectToNextLayer(secondSink.get());

    std::vector<shared_ptr<Layer>> layers{input, upstream, fanout, firstSink, secondSink};
    LayerTestHelper::initializeNetwork(layers);

    std::vector<std::optional<Tensor>> errorInputs = fanout->getErrorInputs();
    std::vector<std::optional<Tensor>> errorOutputs = fanout->getErrorOutputs();
    ASSERT_EQ(errorInputs.size(), 2U);
    ASSERT_EQ(errorOutputs.size(), 1U);
    ASSERT_TRUE(errorInputs[0].has_value());
    ASSERT_TRUE(errorInputs[1].has_value());
    ASSERT_TRUE(errorOutputs[0].has_value());

    std::vector<Stream> streams = fanout->getStreams();
    ASSERT_EQ(streams.size(), 2U);
    errorInputs[0].value().copyFromAsync(firstGradientCpu, streams[0]);
    errorInputs[1].value().copyFromAsync(secondGradientCpu, streams[1]);

    fanout->backward(errorInputs[0], static_cast<uint32_t>(dimensions[0]));
    fanout->backward(errorInputs[1], static_cast<uint32_t>(dimensions[0]));

    summedGradientCpu.copyFromAsync(errorOutputs[0].value(), streams[0]);
    cudaError_t cudaStatus = cudaStreamSynchronize(streams[0].getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (uint32_t i = 0; i < elementCount; ++i) {
        const float expected = firstGradient[i] + secondGradient[i];
        const float actual = readTensorFanoutSumValue<T>(summedGradientCpu, i);
        ASSERT_NEAR(actual, expected, 1.0e-3f) << "element " << i;
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(InOut, NoOpWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 1; ++test) {
        unsigned int numDimensions = 2;
        vector<unsigned long> dimensions;
        for (unsigned int i = 0; i < numDimensions; ++i)
            dimensions.push_back((rand() % 100) + 1);
        unsigned int rows = dimensions[0];
        unsigned int cols = dimensions[1];
        unsigned int numElements = rows * cols;
        TensorDescriptor descriptor(DataType::FP16, dimensions);
        Tensor cpuSource(cpuPlacement, descriptor);
        Tensor gpuSource(gpuPlacement, descriptor);
        Tensor cpuDest(cpuPlacement, descriptor);

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(gpuSource));
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        Stream stream = layers.front()->getStream();

        half *sourceMem = (half *)cpuSource.getMemPtr();
        for (unsigned int i = 0; i < numElements; ++i)
            sourceMem[i] = (half)(float)i;

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor gpuOutput = layers.back()->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(cpuSource, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        cpuDest.copyFromAsync(gpuOutput, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)cpuDest.getMemPtr();
        for (unsigned int i = 0; i < numElements; ++i)
            ASSERT_EQ(sourceMem[i], destMem[i]);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Flatten, BackwardAliasPreservesInputDescriptor) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor sourceDescriptor(DataType::FP16, {2, 3, 4, 5});
    Tensor sourceGpu(gpuPlacement, sourceDescriptor);

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<Flatten>(2));
    layers.push_back(make_shared<BackpropDescriptorSinkLayer>());

    LayerTestHelper::connectNetwork(layers);

    auto upstream = dynamic_pointer_cast<NoOpLayer>(layers[1]);
    auto flatten = dynamic_pointer_cast<Flatten>(layers[2]);
    auto sink = dynamic_pointer_cast<BackpropDescriptorSinkLayer>(layers[3]);
    ASSERT_NE(upstream, nullptr);
    ASSERT_NE(flatten, nullptr);
    ASSERT_NE(sink, nullptr);

    // This must already be true immediately after connection, before compile().
    // Upstream CustomLayer compileImpl() snapshots expected backward tensor ids,
    // so a postCompile-only alias rewrite is too late.
    ASSERT_TRUE(upstream->getErrorInput().has_value());
    ASSERT_TRUE(sink->getErrorOutput().has_value());
    EXPECT_EQ(upstream->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(upstream->getErrorInput().value().getMemPtr(), sink->getErrorOutput().value().getMemPtr());

    LayerTestHelper::initializeNetwork(layers);

    ASSERT_TRUE(flatten->getFeatureInput().has_value());
    ASSERT_TRUE(flatten->getFeatureOutput().has_value());
    ASSERT_TRUE(flatten->getErrorInput().has_value());
    ASSERT_TRUE(flatten->getErrorOutput().has_value());
    ASSERT_TRUE(upstream->getErrorInput().has_value());
    ASSERT_TRUE(sink->getErrorOutput().has_value());

    EXPECT_EQ(flatten->getFeatureInput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(flatten->getFeatureOutput().value().getDimensions(), (vector<unsigned long>{2, 60}));
    EXPECT_EQ(sink->getErrorOutput().value().getDimensions(), (vector<unsigned long>{2, 60}));
    EXPECT_EQ(flatten->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 60}));
    EXPECT_EQ(flatten->getErrorOutput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(upstream->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));

    // Flatten backward is metadata-only: the upstream error tensor must alias the
    // downstream flattened error storage, but expose the original input descriptor.
    EXPECT_EQ(flatten->getErrorInput().value().getMemPtr(), sink->getErrorOutput().value().getMemPtr());
    EXPECT_EQ(flatten->getErrorOutput().value().getMemPtr(), flatten->getErrorInput().value().getMemPtr());
    EXPECT_EQ(upstream->getErrorInput().value().getMemPtr(), flatten->getErrorInput().value().getMemPtr());

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(Flatten, BackwardAliasReplacementPreservesInputDescriptor) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Tensor sourceGpu(gpuPlacement, TensorDescriptor(DataType::FP16, {2, 3, 4, 5}));

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<Flatten>(2));
    layers.push_back(make_shared<BackpropDescriptorSinkLayer>());

    LayerTestHelper::connectNetwork(layers);

    auto upstream = dynamic_pointer_cast<NoOpLayer>(layers[1]);
    auto flatten = dynamic_pointer_cast<Flatten>(layers[2]);
    ASSERT_NE(upstream, nullptr);
    ASSERT_NE(flatten, nullptr);
    ASSERT_TRUE(flatten->getErrorInput().has_value());
    ASSERT_TRUE(flatten->getErrorOutput().has_value());

    // TensorFanout performs this kind of replacement during compile when only
    // one downstream consumer carries a gradient. The replacement tensor uses
    // the flattened/output descriptor; Flatten must re-expose the original
    // input descriptor before forwarding that storage upstream.
    Tensor replacement = flatten->getErrorInput().value().clone();
    const auto oldErrorInput = flatten->getErrorInput();
    flatten->replaceErrorInput(oldErrorInput, replacement);

    ASSERT_TRUE(flatten->getErrorInput().has_value());
    ASSERT_TRUE(flatten->getErrorOutput().has_value());
    ASSERT_TRUE(upstream->getErrorInput().has_value());
    EXPECT_EQ(flatten->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 60}));
    EXPECT_EQ(flatten->getErrorOutput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(upstream->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(flatten->getErrorInput().value().getMemPtr(), replacement.getMemPtr());
    EXPECT_EQ(flatten->getErrorOutput().value().getMemPtr(), replacement.getMemPtr());
    EXPECT_EQ(upstream->getErrorInput().value().getMemPtr(), replacement.getMemPtr());

    LayerTestHelper::initializeNetwork(layers);
    LayerTestHelper::tearDownNetwork(layers);
}


TEST(Reshape, BackwardAliasPreservesInputDescriptor) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor sourceDescriptor(DataType::FP16, {2, 3, 4, 5});
    Tensor sourceGpu(gpuPlacement, sourceDescriptor);

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<Reshape>(vector<unsigned long>{2, 12, 5}));
    layers.push_back(make_shared<BackpropDescriptorSinkLayer>());

    LayerTestHelper::connectNetwork(layers);

    auto upstream = dynamic_pointer_cast<NoOpLayer>(layers[1]);
    auto reshape = dynamic_pointer_cast<Reshape>(layers[2]);
    auto sink = dynamic_pointer_cast<BackpropDescriptorSinkLayer>(layers[3]);
    ASSERT_NE(upstream, nullptr);
    ASSERT_NE(reshape, nullptr);
    ASSERT_NE(sink, nullptr);

    // This must already be true immediately after connection, before compile().
    // Upstream CustomLayer compileImpl() snapshots expected backward tensor ids,
    // so a postCompile-only alias rewrite is too late.
    ASSERT_TRUE(upstream->getErrorInput().has_value());
    ASSERT_TRUE(sink->getErrorOutput().has_value());
    EXPECT_EQ(upstream->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(upstream->getErrorInput().value().getMemPtr(), sink->getErrorOutput().value().getMemPtr());

    LayerTestHelper::initializeNetwork(layers);

    ASSERT_TRUE(reshape->getFeatureInput().has_value());
    ASSERT_TRUE(reshape->getFeatureOutput().has_value());
    ASSERT_TRUE(reshape->getErrorInput().has_value());
    ASSERT_TRUE(reshape->getErrorOutput().has_value());
    ASSERT_TRUE(upstream->getErrorInput().has_value());
    ASSERT_TRUE(sink->getErrorOutput().has_value());

    EXPECT_EQ(reshape->getFeatureInput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(reshape->getFeatureOutput().value().getDimensions(), (vector<unsigned long>{2, 12, 5}));
    EXPECT_EQ(sink->getErrorOutput().value().getDimensions(), (vector<unsigned long>{2, 12, 5}));
    EXPECT_EQ(reshape->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 12, 5}));
    EXPECT_EQ(reshape->getErrorOutput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(upstream->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));

    EXPECT_EQ(reshape->getErrorInput().value().getMemPtr(), sink->getErrorOutput().value().getMemPtr());
    EXPECT_EQ(reshape->getErrorOutput().value().getMemPtr(), reshape->getErrorInput().value().getMemPtr());
    EXPECT_EQ(upstream->getErrorInput().value().getMemPtr(), reshape->getErrorInput().value().getMemPtr());

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(Reshape, BackwardAliasReplacementPreservesInputDescriptor) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Tensor sourceGpu(gpuPlacement, TensorDescriptor(DataType::FP16, {2, 3, 4, 5}));

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<Reshape>(vector<unsigned long>{2, 12, 5}));
    layers.push_back(make_shared<BackpropDescriptorSinkLayer>());

    LayerTestHelper::connectNetwork(layers);

    auto upstream = dynamic_pointer_cast<NoOpLayer>(layers[1]);
    auto reshape = dynamic_pointer_cast<Reshape>(layers[2]);
    ASSERT_NE(upstream, nullptr);
    ASSERT_NE(reshape, nullptr);
    ASSERT_TRUE(reshape->getErrorInput().has_value());
    ASSERT_TRUE(reshape->getErrorOutput().has_value());

    // Reproduce the late replacement performed by TensorFanout::compileImpl().
    // The downstream replacement has the reshape output descriptor, but the
    // upstream layer must continue to receive the original input descriptor.
    Tensor replacement = reshape->getErrorInput().value().clone();
    const auto oldErrorInput = reshape->getErrorInput();
    reshape->replaceErrorInput(oldErrorInput, replacement);

    ASSERT_TRUE(reshape->getErrorInput().has_value());
    ASSERT_TRUE(reshape->getErrorOutput().has_value());
    ASSERT_TRUE(upstream->getErrorInput().has_value());
    EXPECT_EQ(reshape->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 12, 5}));
    EXPECT_EQ(reshape->getErrorOutput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(upstream->getErrorInput().value().getDimensions(), (vector<unsigned long>{2, 3, 4, 5}));
    EXPECT_EQ(reshape->getErrorInput().value().getMemPtr(), replacement.getMemPtr());
    EXPECT_EQ(reshape->getErrorOutput().value().getMemPtr(), replacement.getMemPtr());
    EXPECT_EQ(upstream->getErrorInput().value().getMemPtr(), replacement.getMemPtr());

    LayerTestHelper::initializeNetwork(layers);
    LayerTestHelper::tearDownNetwork(layers);
}


TEST(Flatten, FlattensCorrectly) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 10; ++test) {
        int numDimensions = (rand() % 4) + 2;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 10) + 1);
            numElements *= dimensions.back();
        }
        int numOutputDimensions = (rand() % (numDimensions - 1)) + 1;

        int lastDimensionSize = 1;
        vector<unsigned long> outputDimensions;
        for (int i = numOutputDimensions - 1; i < numDimensions; ++i)
            lastDimensionSize *= dimensions[i];
        for (int i = 0; i < numOutputDimensions; ++i) {
            if (i == numOutputDimensions - 1)
                outputDimensions.push_back(lastDimensionSize);
            else
                outputDimensions.push_back(dimensions[i]);
        }

        TensorDescriptor sourceDescriptor(DataType::FP16, dimensions);
        TensorDescriptor destDescriptor(DataType::FP16, outputDimensions);
        Tensor sourceCpu(cpuPlacement, sourceDescriptor);
        Tensor sourceGpu(gpuPlacement, sourceDescriptor);
        Tensor destCpu(cpuPlacement, destDescriptor);

        half *sourceMem = (half *)sourceCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));
        layers.push_back(make_shared<Flatten>(numOutputDimensions));
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        ASSERT_TRUE(outputGpu.getDescriptor() == destDescriptor);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)destMem[i], (float)sourceMem[i]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Reshape, ReshapesCorrectly) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    vector<unsigned long> originalDimensions, newDimensions;
    originalDimensions.push_back(4);
    originalDimensions.push_back(10);
    originalDimensions.push_back(10);
    newDimensions.push_back(2);
    newDimensions.push_back(2);
    newDimensions.push_back(20);
    newDimensions.push_back(5);

    TensorDescriptor sourceDescriptor(DataType::FP16, originalDimensions);
    TensorDescriptor destDescriptor(DataType::FP16, newDimensions);
    Tensor sourceCpu(cpuPlacement, sourceDescriptor);
    Tensor sourceGpu(gpuPlacement, sourceDescriptor);
    Tensor destCpu(cpuPlacement, destDescriptor);

    int numElements = 400;
    half *sourceMem = (half *)sourceCpu.getMemPtr();
    for (int i = 0; i < numElements; ++i) {
        sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    layers.push_back(make_shared<Reshape>(newDimensions));
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    Stream stream = layers.front()->getStream();

    LayerTestHelper::connectAndInitializeNetwork(layers);
    Tensor outputGpu = layers.back()->getFeatureOutput().value();

    // Network is runnable here
    layers[0]->forward(sourceCpu, false);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
    destCpu.copyFromAsync(outputGpu, stream);

    cudaStatus = cudaStreamSynchronize(stream.getStream());
    assert(cudaStatus == cudaSuccess);

    ASSERT_TRUE(outputGpu.getDescriptor() == destDescriptor);

    half *destMem = (half *)destCpu.getMemPtr();
    for (int i = 0; i < numElements; ++i) {
        ASSERT_EQ((float)destMem[i], (float)sourceMem[i]);
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(TensorFanout, CreatesFanout) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    vector<unsigned long> dimensions;
    dimensions.push_back(10);
    dimensions.push_back(3);
    dimensions.push_back(6);
    int numElements = 10 * 3 * 6;
    TensorDescriptor descriptor(DataType::FP32, dimensions);
    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu(gpuPlacement, descriptor);
    Tensor destCpu0(cpuPlacement, descriptor);
    Tensor destCpu1(cpuPlacement, descriptor);

    float *sourceMem = (float *)sourceCpu.getMemPtr();
    for (int i = 0; i < numElements; ++i) {
        sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    shared_ptr<TensorFanout> tensorFanout = make_shared<TensorFanout>();
    layers.push_back(tensorFanout);
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    Stream stream = layers.front()->getStream();

    LayerTestHelper::connectAndInitializeNetwork(layers);

    // Special case, two outputs
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));
    layers[1]->connectToNextLayer(layers[3].get());
    layers[3]->compile();
    layers[3]->initialize();

    Tensor outputGpu0 = layers[2]->getFeatureOutput().value();
    Tensor outputGpu1 = layers[3]->getFeatureOutput().value();

    // Network is runnable here
    layers[0]->forward(sourceCpu, false);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers[2])->getOutputReadyEvent());
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers[3])->getOutputReadyEvent());
    destCpu0.copyFromAsync(outputGpu0, stream);
    destCpu1.copyFromAsync(outputGpu1, stream);

    cudaStatus = cudaStreamSynchronize(stream.getStream());
    assert(cudaStatus == cudaSuccess);

    float *destMem = (float *)destCpu0.getMemPtr();
    for (int i = 0; i < numElements; ++i) {
        ASSERT_EQ(destMem[i], sourceMem[i]);
    }
    destMem = (float *)destCpu1.getMemPtr();
    for (int i = 0; i < numElements; ++i) {
        ASSERT_EQ(destMem[i], sourceMem[i]);
    }

    ASSERT_EQ(tensorFanout->getStreams().size(), 2U);

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(TensorFanout, BackwardSumsBF16Errors) { expectTensorFanoutSumsBackwardErrorsForDType<__nv_bfloat16>(); }

TEST(TensorFanout, BackwardSumsFp8E4M3Errors) { expectTensorFanoutSumsBackwardErrorsForDType<__nv_fp8_e4m3>(); }

TEST(TensorFanout, BackwardSumsFp8E5M2Errors) { expectTensorFanoutSumsBackwardErrorsForDType<__nv_fp8_e5m2>(); }

inline void computeIndex(int flatIndex, int index[], int numDimensions, long stridePerDimension[]) {
    for (int i = 0; i < numDimensions; ++i) {
        index[i] = flatIndex / stridePerDimension[i];
        flatIndex -= index[i] * stridePerDimension[i];
    }
    assert(flatIndex == 0);
}

#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
inline int computeFlatIndex(int index[], long stridePerDimension[], int numDimensions) {
    int flatIndex = 0;
    for (int i = 0; i < numDimensions; ++i)
        flatIndex += index[i] * stridePerDimension[i];
    return flatIndex;
}
#pragma GCC diagnostic pop

TEST(Concatenate, Concatenates) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    long axisElementsPerSourceArray[10];
    long stridePerDestDimension[10];
    long stridePerSourceDimension[5 * 10];

    for (int test = 0; test < 50; ++test) {
        vector<Tensor> partsCpu;
        vector<Tensor> partsGpu;
        Tensor wholeCpu;
        Tensor wholeGpu;

        int numSplitTensors = (rand() % 5) + 2;
        int numDimensions = (rand() % 6) + 1;
        int axis = rand() % numDimensions;

        vector<unsigned long> wholeDimensions;
        for (int d = 0; d < numDimensions; d++) {
            if (d == axis)
                wholeDimensions.push_back(0);
            else
                wholeDimensions.push_back((rand() % 10) + 1);
        }

        for (int t = 0; t < numSplitTensors; ++t) {
            vector<unsigned long> splitArrayDimensions = wholeDimensions;
            splitArrayDimensions[axis] = (rand() % 5) + 1;
            axisElementsPerSourceArray[t] = splitArrayDimensions[axis];
            wholeDimensions[axis] += splitArrayDimensions[axis];

            TensorDescriptor partDescriptor(DataType::FP16, splitArrayDimensions);
            partsCpu.emplace_back(cpuPlacement, partDescriptor);
            partsGpu.emplace_back(gpuPlacement, partDescriptor);
        }
        TensorDescriptor wholeDescriptor(DataType::FP16, wholeDimensions);
        wholeCpu = Tensor(cpuPlacement, wholeDescriptor);

        stridePerDestDimension[numDimensions - 1] = 1;
        for (int dest = 0; dest < numSplitTensors; dest++)
            stridePerSourceDimension[dest * numDimensions + numDimensions - 1] = 1;
        for (int i = numDimensions - 2; i >= 0; --i) {
            stridePerDestDimension[i] = stridePerDestDimension[i + 1] * wholeDimensions[i + 1];
            for (int dest = 0; dest < numSplitTensors; ++dest)
                if (i + 1 == axis)
                    stridePerSourceDimension[dest * numDimensions + i] =
                        stridePerSourceDimension[dest * numDimensions + i + 1] * axisElementsPerSourceArray[dest];
                else
                    stridePerSourceDimension[dest * numDimensions + i] =
                        stridePerSourceDimension[dest * numDimensions + i + 1] * wholeDimensions[i + 1];
        }

        long numElements = wholeCpu.getDescriptor().getTotalNumElements();

        vector<shared_ptr<Layer>> layers;

        for (unsigned int i = 0; i < partsGpu.size(); ++i)
            layers.push_back(make_shared<NetworkInput>(partsGpu[i]));
        layers.push_back(make_shared<Concatenate>(axis, numSplitTensors, DataType::FP16));
        for (unsigned int i = 0; i < layers.size() - 1; ++i)
            layers[i]->connectToNextLayer(layers.back().get(), 0, static_cast<int>(i));
        layers.push_back(make_shared<NoOpLayer>());
        layers[layers.size() - 2]->connectToNextLayer(layers.back().get());
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));
        layers[layers.size() - 2]->connectToNextLayer(layers.back().get());

        LayerTestHelper::initializeNetwork(layers);

        wholeGpu = layers.back()->getFeatureOutput().value();

        // Network is runnable here
        for (int i = 0; i < numSplitTensors; ++i) {
            long numElements = partsCpu[i].getDescriptor().getTotalNumElements();
            half *mem = (half *)partsCpu[i].getMemPtr();
            for (int i = 0; i < numElements; ++i) {
                mem[i] = ((rand() % 100) / 10.0f) - 5.0f;
            }
            layers[i]->forward(partsCpu[i], false);
        }

        layers[0]->getStream().waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        wholeCpu.copyFromAsync(wholeGpu, layers[0]->getStream());
        layers[0]->getStream().synchronize();

        half *wholeMem = (half *)wholeCpu.getMemPtr();
        vector<int> sourceTensorAxisIndexStart;
        sourceTensorAxisIndexStart.push_back(0);
        for (int i = 1; i < numSplitTensors; ++i)
            sourceTensorAxisIndexStart.push_back(sourceTensorAxisIndexStart.back() + axisElementsPerSourceArray[i - 1]);

        int destIndex[10];
        int sourceIndex[10];
        for (int destFlatIndex = 0; destFlatIndex < numElements; ++destFlatIndex) {
            computeIndex(destFlatIndex, destIndex, numDimensions, stridePerDestDimension);
            int source = 0;
            for (; destIndex[axis] >= sourceTensorAxisIndexStart[source] + axisElementsPerSourceArray[source]; ++source)
                ;
            for (int i = 0; i < numDimensions; ++i) {
                if (i == axis)
                    sourceIndex[i] = destIndex[i] - sourceTensorAxisIndexStart[source];
                else
                    sourceIndex[i] = destIndex[i];
            }
            int sourceFlatIndex = computeFlatIndex(sourceIndex, stridePerSourceDimension + source * numDimensions, numDimensions);

            half *partsMem = (half *)partsCpu[source].getMemPtr();
            // printf("sourceFlatIndex%d %d destFlatIndex %d sourceIndex%d[%d][%d] destIndex[%d][%d] source %f dest %f\n",
            //       source,
            //       sourceFlatIndex,
            //       destFlatIndex,
            //       source,
            //       sourceIndex[0],
            //       sourceIndex[1],
            //       destIndex[0],
            //       destIndex[1],
            //       (float)partsMem[sourceFlatIndex],
            //       (float)wholeMem[destFlatIndex]);
            //
            // printf("whole[%d] %f  parts[%d][%d] %f\n", destFlatIndex, (float)wholeMem[destFlatIndex], source, sourceFlatIndex,
            // (float)partsMem[sourceFlatIndex]);
            ASSERT_EQ((float)wholeMem[destFlatIndex], (float)partsMem[sourceFlatIndex]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Split, CreatesOneStreamPerOutput) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Tensor wholeGpu(gpuPlacement, TensorDescriptor(DataType::FP16, {6, 4}));

    shared_ptr<NetworkInput> inputLayer = make_shared<NetworkInput>(wholeGpu);
    shared_ptr<Split> splitLayer = make_shared<Split>(0, vector<unsigned long>{1, 2, 3}, vector<DataType>(3, DataType::FP16));
    inputLayer->connectToNextLayer(splitLayer.get());

    vector<shared_ptr<NoOpLayer>> outputLayers;
    for (int output = 0; output < 3; ++output) {
        outputLayers.push_back(make_shared<NoOpLayer>());
        splitLayer->connectToNextLayer(outputLayers.back().get(), output);
    }

    const vector<Stream> splitStreams = splitLayer->getStreams();
    ASSERT_EQ(splitStreams.size(), 3U);
    for (size_t output = 0; output < splitStreams.size(); ++output) {
        for (size_t priorOutput = 0; priorOutput < output; ++priorOutput) {
            ASSERT_NE(splitStreams[output].getId(), splitStreams[priorOutput].getId());
        }
    }
}

TEST(Split, Splits) {
    const char *seedEnvironment = std::getenv("THOR_SPLIT_TEST_SEED");
    const unsigned int seed =
        seedEnvironment != nullptr ? static_cast<unsigned int>(std::stoul(seedEnvironment)) : static_cast<unsigned int>(std::time(nullptr));

    const char *runCountEnvironment = std::getenv("THOR_SPLIT_TEST_RUNS");
    const int runCount = runCountEnvironment != nullptr ? std::stoi(runCountEnvironment) : 50;

    srand(seed);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < runCount; ++test) {
        vector<Tensor> partsCpu;
        vector<Tensor> partsGpu;

        Tensor wholeCpu;
        Tensor wholeGpu;

        const int numSplitTensors = (rand() % 5) + 2;
        const int numDimensions = (rand() % 6) + 1;
        const int axis = rand() % numDimensions;

        vector<long> axisElementsPerDestArray(numSplitTensors);
        vector<long> stridePerSourceDimension(numDimensions);
        vector<long> stridePerDestDimension(numSplitTensors * numDimensions);

        vector<unsigned long> wholeDimensions;
        wholeDimensions.reserve(numDimensions);

        for (int dimension = 0; dimension < numDimensions; ++dimension) {
            if (dimension == axis)
                wholeDimensions.push_back(0);
            else
                wholeDimensions.push_back((rand() % 10) + 1);
        }

        partsCpu.reserve(numSplitTensors);

        for (int output = 0; output < numSplitTensors; ++output) {
            vector<unsigned long> splitDimensions = wholeDimensions;

            splitDimensions[axis] = (rand() % 5) + 1;
            axisElementsPerDestArray[output] = static_cast<long>(splitDimensions[axis]);

            wholeDimensions[axis] += splitDimensions[axis];

            TensorDescriptor partDescriptor(DataType::FP16, splitDimensions);
            partsCpu.emplace_back(cpuPlacement, partDescriptor);
        }

        TensorDescriptor wholeDescriptor(DataType::FP16, wholeDimensions);

        wholeCpu = Tensor(cpuPlacement, wholeDescriptor);
        wholeGpu = Tensor(gpuPlacement, wholeDescriptor);

        const long numElements = wholeCpu.getDescriptor().getTotalNumElements();

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(wholeGpu));

        Stream stream = layers.front()->getStream();

        stridePerSourceDimension[numDimensions - 1] = 1;

        for (int output = 0; output < numSplitTensors; ++output) {
            stridePerDestDimension[output * numDimensions + numDimensions - 1] = 1;
        }

        for (int dimension = numDimensions - 2; dimension >= 0; --dimension) {
            stridePerSourceDimension[dimension] =
                stridePerSourceDimension[dimension + 1] * static_cast<long>(wholeDimensions[dimension + 1]);

            for (int output = 0; output < numSplitTensors; ++output) {
                const long followingDimensionSize =
                    dimension + 1 == axis ? axisElementsPerDestArray[output] : static_cast<long>(wholeDimensions[dimension + 1]);

                stridePerDestDimension[output * numDimensions + dimension] =
                    stridePerDestDimension[output * numDimensions + dimension + 1] * followingDimensionSize;
            }
        }

        half *wholeInitialization = static_cast<half *>(wholeCpu.getMemPtr());

        for (long element = 0; element < numElements; ++element) {
            wholeInitialization[element] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<unsigned long> axisElements;
        axisElements.reserve(numSplitTensors);

        for (int output = 0; output < numSplitTensors; ++output) {
            axisElements.push_back(static_cast<unsigned long>(axisElementsPerDestArray[output]));
        }

        shared_ptr<Split> splitLayer = make_shared<Split>(axis, axisElements, vector<DataType>(axisElements.size(), DataType::FP16));

        layers.back()->connectToNextLayer(splitLayer.get());
        layers.push_back(splitLayer);

        vector<shared_ptr<Layer>> outputLayers;
        outputLayers.reserve(numSplitTensors);
        partsGpu.reserve(numSplitTensors);

        for (int output = 0; output < numSplitTensors; ++output) {
            shared_ptr<Layer> noOpLayer = make_shared<NoOpLayer>();

            layers.push_back(noOpLayer);
            splitLayer->connectToNextLayer(noOpLayer.get(), output);

            shared_ptr<Layer> networkOutputLayer = make_shared<NetworkOutput>(gpuPlacement);

            layers.push_back(networkOutputLayer);
            noOpLayer->connectToNextLayer(networkOutputLayer.get());

            partsGpu.push_back(networkOutputLayer->getFeatureOutput().value());
            outputLayers.push_back(networkOutputLayer);
        }

        LayerTestHelper::initializeNetwork(layers);

        layers[0]->forward(wholeCpu, false);

        for (size_t output = 0; output < partsGpu.size(); ++output) {
            stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(outputLayers[output])->getOutputReadyEvent());

            partsCpu[output].copyFromAsync(partsGpu[output], stream);
        }

        stream.synchronize();

        half *wholeMem = static_cast<half *>(wholeCpu.getMemPtr());

        vector<int> destTensorAxisIndexStart;
        destTensorAxisIndexStart.reserve(numSplitTensors);
        destTensorAxisIndexStart.push_back(0);

        for (int output = 1; output < numSplitTensors; ++output) {
            destTensorAxisIndexStart.push_back(destTensorAxisIndexStart.back() + static_cast<int>(axisElementsPerDestArray[output - 1]));
        }

        vector<int> sourceIndex(numDimensions);
        vector<int> destIndex(numDimensions);

        for (long sourceFlatIndex = 0; sourceFlatIndex < numElements; ++sourceFlatIndex) {
            computeIndex(sourceFlatIndex, sourceIndex.data(), numDimensions, stridePerSourceDimension.data());

            int destination = 0;

            while (sourceIndex[axis] >= destTensorAxisIndexStart[destination] + axisElementsPerDestArray[destination]) {
                ++destination;
            }

            ASSERT_LT(destination, numSplitTensors);

            for (int dimension = 0; dimension < numDimensions; ++dimension) {
                if (dimension == axis) {
                    destIndex[dimension] = sourceIndex[dimension] - destTensorAxisIndexStart[destination];
                } else {
                    destIndex[dimension] = sourceIndex[dimension];
                }
            }

            const int destFlatIndex =
                computeFlatIndex(destIndex.data(), stridePerDestDimension.data() + destination * numDimensions, numDimensions);

            half *partsMem = static_cast<half *>(partsCpu[destination].getMemPtr());

            const float expected = static_cast<float>(wholeMem[sourceFlatIndex]);

            const float observed = static_cast<float>(partsMem[destFlatIndex]);

            if (expected == observed)
                continue;

            std::ostringstream dimensionsText;
            dimensionsText << "[";
            for (size_t dimension = 0; dimension < wholeDimensions.size(); ++dimension) {
                if (dimension != 0)
                    dimensionsText << ",";
                dimensionsText << wholeDimensions[dimension];
            }
            dimensionsText << "]";

            std::ostringstream axisElementsText;
            axisElementsText << "[";
            for (int output = 0; output < numSplitTensors; ++output) {
                if (output != 0)
                    axisElementsText << ",";
                axisElementsText << axisElementsPerDestArray[output];
            }
            axisElementsText << "]";

            LayerTestHelper::tearDownNetwork(layers);
            FAIL() << "Split mismatch. Reproduce with THOR_SPLIT_TEST_SEED=" << seed
                   << " THOR_SPLIT_TEST_RUNS=" << runCount << ". randomized_case=" << test
                   << " rank=" << numDimensions << " dimensions=" << dimensionsText.str() << " axis=" << axis
                   << " axis_elements=" << axisElementsText.str() << " source_flat_index=" << sourceFlatIndex
                   << " destination=" << destination << " destination_flat_index=" << destFlatIndex
                   << " expected=" << expected << " observed=" << observed;
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

// An FP32 input split on axis 1 into FP32, BF16, FP32 and INT32 pieces, connected out of order. Forward: each piece
// converted. Backward: the first two pieces' gradients go back (the BF16 one converted to FP32); the third piece's
// next layer asks for none and the INT32 piece takes none, so their columns of the input's gradient are zeros.
TEST(Split, ConvertsEachOutputAndGivesZerosWhereAnOutputTakesNoGradient) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint32_t rows = 4;
    constexpr uint64_t inputWidth = 6;
    const vector<unsigned long> widths{2, 2, 1, 1};
    const vector<DataType> types{DataType::FP32, DataType::BF16, DataType::FP32, DataType::INT32};
    const vector<uint32_t> firstColumn{0, 2, 4, 5};
    // Every value is exact in each type it passes through (the INT32 column holds whole numbers).
    auto inputValue = [](uint32_t row, uint32_t column) -> float {
        return column == 5 ? row * 100.0f - 7.0f : row * 10.0f + column + 0.25f;
    };
    Tensor inputCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {rows, inputWidth}));
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < inputWidth; ++c) writeUtilityLayerValue<float>(inputCpu, r * inputWidth + c, inputValue(r, c));
    const vector<int> connectionOrder{2, 0, 3, 1};

    {
        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(gpuPlacement, DataType::FP32, vector<unsigned long>{rows, inputWidth}));
        auto split = make_shared<Split>(1, widths, types);
        layers.push_back(split);
        layers[0]->connectToNextLayer(split.get());
        vector<shared_ptr<Layer>> outputs(widths.size());
        for (const int output : connectionOrder) {
            outputs[output] = make_shared<NetworkOutput>(gpuPlacement);
            layers.push_back(outputs[output]);
            split->connectToNextLayer(outputs[output].get(), output);
        }
        LayerTestHelper::initializeNetwork(layers);
        layers[0]->forward(inputCpu, false);
        for (uint32_t output = 0; output < widths.size(); ++output) {
            Tensor pieceGpu = outputs[output]->getFeatureOutput().value();
            ASSERT_EQ(pieceGpu.getDescriptor().getDataType(), types[output]);
            ASSERT_EQ(pieceGpu.getDimensions(), (vector<uint64_t>{rows, widths[output]}));
            Tensor pieceCpu(cpuPlacement, pieceGpu.getDescriptor());
            Stream stream = layers[0]->getStream();
            stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(outputs[output])->getOutputReadyEvent());
            pieceCpu.copyFromAsync(pieceGpu, stream);
            stream.synchronize();
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < widths[output]; ++c) {
                    const uint64_t index = r * widths[output] + c;
                    const float expected = inputValue(r, firstColumn[output] + c);
                    if (types[output] == DataType::INT32) {
                        EXPECT_EQ(pieceCpu.getMemPtr<int32_t>()[index], static_cast<int32_t>(expected)) << "row " << r;
                    } else if (types[output] == DataType::BF16) {
                        EXPECT_EQ(readUtilityLayerValue<__nv_bfloat16>(pieceCpu, index), expected) << "row " << r << " column " << c;
                    } else {
                        EXPECT_EQ(readUtilityLayerValue<float>(pieceCpu, index), expected) << "row " << r << " column " << c;
                    }
                }
            }
        }
        LayerTestHelper::tearDownNetwork(layers);
    }

    {
        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(gpuPlacement, DataType::FP32, vector<unsigned long>{rows, inputWidth}));
        auto upstream = make_shared<NoOpLayer>();
        layers.push_back(upstream);
        layers[0]->connectToNextLayer(upstream.get());
        auto split = make_shared<Split>(1, widths, types);
        layers.push_back(split);
        upstream->connectToNextLayer(split.get());
        vector<shared_ptr<Layer>> outputs(widths.size());
        for (const int output : connectionOrder) {
            if (output == 2)
                outputs[output] = make_shared<NetworkOutput>(gpuPlacement);  // asks for no gradient
            else
                outputs[output] = make_shared<BackpropDescriptorSinkLayer>();
            layers.push_back(outputs[output]);
            split->connectToNextLayer(outputs[output].get(), output);
        }
        LayerTestHelper::initializeNetwork(layers);

        const vector<std::optional<Tensor>> errorInputs = split->getErrorInputs();
        const vector<std::optional<Tensor>> errorOutputs = split->getErrorOutputs();
        ASSERT_EQ(errorInputs.size(), widths.size());
        ASSERT_TRUE(errorInputs[0].has_value());
        ASSERT_TRUE(errorInputs[1].has_value());
        EXPECT_FALSE(errorInputs[2].has_value());
        EXPECT_FALSE(errorInputs[3].has_value()) << "an INT32 output takes no gradient";
        ASSERT_EQ(errorOutputs.size(), 1U);
        ASSERT_TRUE(errorOutputs[0].has_value());
        EXPECT_EQ(errorInputs[1].value().getDescriptor().getDataType(), DataType::BF16);

        auto gradient = [](uint32_t row, uint32_t column) { return (row + 1) * (column + 1) * 0.5f; };
        Stream stream = split->getStreams()[0];
        for (uint32_t output = 0; output < 2; ++output) {
            Tensor gradientCpu(cpuPlacement, errorInputs[output].value().getDescriptor());
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t c = 0; c < widths[output]; ++c) {
                    const float value = gradient(r, firstColumn[output] + c);
                    if (types[output] == DataType::BF16)
                        writeUtilityLayerValue<__nv_bfloat16>(gradientCpu, r * widths[output] + c, value);
                    else
                        writeUtilityLayerValue<float>(gradientCpu, r * widths[output] + c, value);
                }
            }
            Tensor gradientGpu = errorInputs[output].value();
            gradientGpu.copyFromAsync(gradientCpu, stream);
        }
        // Stale bytes where the zeros must land, so a missing span would show.
        Tensor staleCpu(cpuPlacement, errorOutputs[0].value().getDescriptor());
        for (uint64_t i = 0; i < rows * inputWidth; ++i) writeUtilityLayerValue<float>(staleCpu, i, -99.0f);
        Tensor inputGradientGpu = errorOutputs[0].value();
        inputGradientGpu.copyFromAsync(staleCpu, stream);
        stream.synchronize();
        // The output that arrives first waits; the second runs the concatenate.
        split->backward(errorInputs[1], rows);
        split->backward(errorInputs[0], rows);
        Tensor inputGradientCpu(cpuPlacement, errorOutputs[0].value().getDescriptor());
        inputGradientCpu.copyFromAsync(inputGradientGpu, stream);
        stream.synchronize();
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < inputWidth; ++c) {
                const float expected = c < 4 ? gradient(r, c) : 0.0f;
                EXPECT_EQ(readUtilityLayerValue<float>(inputGradientCpu, r * inputWidth + c), expected) << "row " << r << " column " << c;
            }
        }
        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Split, SameTypeGivesZerosWhereAnOutputTakesNoGradient) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint32_t rows = 5;
    constexpr uint64_t inputWidth = 8;
    const vector<unsigned long> widths{3, 2, 3};
    const vector<DataType> types(widths.size(), DataType::FP32);
    const vector<uint32_t> firstColumn{0, 3, 5};

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(gpuPlacement, DataType::FP32, vector<unsigned long>{rows, inputWidth}));
    auto upstream = make_shared<NoOpLayer>();
    layers.push_back(upstream);
    layers[0]->connectToNextLayer(upstream.get());
    auto split = make_shared<Split>(1, widths, types);
    layers.push_back(split);
    upstream->connectToNextLayer(split.get());
    vector<shared_ptr<Layer>> outputs(widths.size());
    for (uint32_t output = 0; output < widths.size(); ++output) {
        if (output == 1)
            outputs[output] = make_shared<NetworkOutput>(gpuPlacement);  // asks for no gradient
        else
            outputs[output] = make_shared<BackpropDescriptorSinkLayer>();
        layers.push_back(outputs[output]);
        split->connectToNextLayer(outputs[output].get(), output);
    }
    LayerTestHelper::initializeNetwork(layers);

    const vector<std::optional<Tensor>> errorInputs = split->getErrorInputs();
    const vector<std::optional<Tensor>> errorOutputs = split->getErrorOutputs();
    ASSERT_EQ(errorInputs.size(), widths.size());
    ASSERT_TRUE(errorInputs[0].has_value());
    EXPECT_FALSE(errorInputs[1].has_value());
    ASSERT_TRUE(errorInputs[2].has_value());
    ASSERT_EQ(errorOutputs.size(), 1U);
    ASSERT_TRUE(errorOutputs[0].has_value());

    auto gradient = [](uint32_t row, uint32_t column) { return (row + 1) * 16.0f + column + 0.5f; };
    Stream stream = split->getStreams()[0];
    for (const uint32_t output : {0U, 2U}) {
        Tensor gradientCpu(cpuPlacement, errorInputs[output].value().getDescriptor());
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < widths[output]; ++c)
                writeUtilityLayerValue<float>(gradientCpu, r * widths[output] + c, gradient(r, firstColumn[output] + c));
        Tensor gradientGpu = errorInputs[output].value();
        gradientGpu.copyFromAsync(gradientCpu, stream);
    }
    // Stale bytes where the zeros must land, so a missing span would show.
    Tensor staleCpu(cpuPlacement, errorOutputs[0].value().getDescriptor());
    for (uint64_t i = 0; i < rows * inputWidth; ++i) writeUtilityLayerValue<float>(staleCpu, i, -99.0f);
    Tensor inputGradientGpu = errorOutputs[0].value();
    inputGradientGpu.copyFromAsync(staleCpu, stream);
    stream.synchronize();
    split->backward(errorInputs[2], rows);
    split->backward(errorInputs[0], rows);
    Tensor inputGradientCpu(cpuPlacement, errorOutputs[0].value().getDescriptor());
    inputGradientCpu.copyFromAsync(inputGradientGpu, stream);
    stream.synchronize();
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < inputWidth; ++c) {
            const float expected = (c >= 3 && c < 5) ? 0.0f : gradient(r, c);
            EXPECT_EQ(readUtilityLayerValue<float>(inputGradientCpu, r * inputWidth + c), expected) << "row " << r << " column " << c;
        }
    }
    LayerTestHelper::tearDownNetwork(layers);
}

namespace {

// Values exact in FP32, BF16 and FP8_E5M2 (two mantissa bits); whole numbers only when asked (an integral output).
float sliceTestValue(uint64_t index, bool wholeNumbers = false) {
    static const float magnitudes[16] = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f,
                                         2.5f, 3.0f, 3.5f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float magnitude =
        wholeNumbers || index % 2 == 0 ? static_cast<float>(index % 7 + 1) : magnitudes[(index * 5) % 16];
    return index % 3 == 1 ? -magnitude : magnitude;
}

void writeSliceTestValue(Tensor &tensor, uint64_t index, float value) {
    switch (tensor.getDescriptor().getDataType()) {
        case DataType::FP32: writeUtilityLayerValue<float>(tensor, index, value); break;
        case DataType::BF16: writeUtilityLayerValue<__nv_bfloat16>(tensor, index, value); break;
        case DataType::FP8_E5M2: writeUtilityLayerValue<__nv_fp8_e5m2>(tensor, index, value); break;
        case DataType::INT32: tensor.getMemPtr<int32_t>()[index] = static_cast<int32_t>(value); break;
        default: FAIL() << "unexpected type";
    }
}

float readSliceTestValue(const Tensor &tensor, uint64_t index) {
    switch (tensor.getDescriptor().getDataType()) {
        case DataType::FP32: return readUtilityLayerValue<float>(tensor, index);
        case DataType::BF16: return readUtilityLayerValue<__nv_bfloat16>(tensor, index);
        case DataType::FP8_E5M2: return readUtilityLayerValue<__nv_fp8_e5m2>(tensor, index);
        case DataType::INT32: return static_cast<float>(tensor.getMemPtr<int32_t>()[index]);
        default: ADD_FAILURE() << "unexpected type"; return 0.0f;
    }
}

}  // namespace

// Input [4, 3, 5, 2], the physical axis 2, entries [1, 4): the window of every outer slice is 6 elements at element 2
// of its 10. Forward runs at 3 of 4 examples, so the last example's output keeps its stale bytes.
TEST(Slice, CopiesTheWindowOfTheValidExamplesInItsOutputType) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint32_t rows = 4;
    constexpr uint32_t validRows = 3;
    const vector<uint64_t> inputDimensions{rows, 3, 5, 2};
    constexpr uint64_t inputElements = 4 * 3 * 5 * 2;
    constexpr uint64_t outerSlices = 4 * 3;
    for (const DataType outputType : {DataType::FP32, DataType::BF16, DataType::FP8_E5M2, DataType::INT32}) {
        SCOPED_TRACE(TensorDescriptor::getElementTypeName(outputType));
        const bool whole = outputType == DataType::INT32;
        Tensor inputCpu(cpuPlacement, TensorDescriptor(DataType::FP32, inputDimensions));
        for (uint64_t i = 0; i < inputElements; ++i) writeUtilityLayerValue<float>(inputCpu, i, sliceTestValue(i, whole));

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(gpuPlacement, DataType::FP32, inputDimensions));
        auto slice = make_shared<ThorImplementation::Slice>(2, 1, 3, outputType);
        layers.push_back(slice);
        layers[0]->connectToNextLayer(slice.get());
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));
        slice->connectToNextLayer(layers.back().get());
        LayerTestHelper::initializeNetwork(layers);

        Tensor outputGpu = slice->getFeatureOutputs()[0].value();
        ASSERT_EQ(outputGpu.getDescriptor().getDataType(), outputType);
        ASSERT_EQ(outputGpu.getDimensions(), (vector<uint64_t>{rows, 3, 3, 2}));
        Stream stream = slice->getStreams()[0];
        Tensor staleCpu(cpuPlacement, outputGpu.getDescriptor());
        for (uint64_t i = 0; i < outerSlices * 6; ++i) writeSliceTestValue(staleCpu, i, -64.0f);
        outputGpu.copyFromAsync(staleCpu, stream);
        stream.synchronize();

        layers[0]->forward(inputCpu, false, validRows);
        Tensor outputCpu(cpuPlacement, outputGpu.getDescriptor());
        outputCpu.copyFromAsync(outputGpu, stream);
        stream.synchronize();
        for (uint64_t outer = 0; outer < outerSlices; ++outer) {
            for (uint64_t j = 0; j < 6; ++j) {
                const float expected = outer < validRows * 3 ? sliceTestValue(outer * 10 + 2 + j, whole) : -64.0f;
                EXPECT_EQ(readSliceTestValue(outputCpu, outer * 6 + j), expected) << "outer " << outer << " element " << j;
            }
        }
        LayerTestHelper::tearDownNetwork(layers);
    }
}

// Backward writes the window's gradient, converted to the input's type, and zeros on both sides of it, for the valid
// examples only; an integral output takes no gradient, so the input's is pruned.
TEST(Slice, BackwardWritesTheWindowsGradientAndZerosAroundIt) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint32_t rows = 4;
    constexpr uint32_t validRows = 3;
    const vector<uint64_t> inputDimensions{rows, 3, 5, 2};
    constexpr uint64_t outerSlices = 4 * 3;
    struct Case {
        uint64_t start;
        uint64_t length;
        DataType outputType;
    };
    // A window in the middle, at the start and at the end of the axis, and the whole axis.
    for (const Case c : {Case{1, 3, DataType::FP32}, Case{1, 3, DataType::BF16}, Case{0, 2, DataType::FP8_E5M2},
                         Case{3, 2, DataType::FP32}, Case{0, 5, DataType::BF16}}) {
        SCOPED_TRACE(std::to_string(c.start) + "+" + std::to_string(c.length) + " " +
                     TensorDescriptor::getElementTypeName(c.outputType));
        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(gpuPlacement, DataType::FP32, inputDimensions));
        auto upstream = make_shared<NoOpLayer>();
        layers.push_back(upstream);
        layers[0]->connectToNextLayer(upstream.get());
        auto slice = make_shared<ThorImplementation::Slice>(2, c.start, c.length, c.outputType);
        layers.push_back(slice);
        upstream->connectToNextLayer(slice.get());
        layers.push_back(make_shared<BackpropDescriptorSinkLayer>());
        slice->connectToNextLayer(layers.back().get());
        LayerTestHelper::initializeNetwork(layers);

        ASSERT_TRUE(slice->getErrorInputs()[0].has_value());
        ASSERT_TRUE(slice->getErrorOutputs()[0].has_value());
        Tensor errorInput = slice->getErrorInputs()[0].value();
        Tensor errorOutput = slice->getErrorOutputs()[0].value();
        ASSERT_EQ(errorInput.getDescriptor().getDataType(), c.outputType);
        const uint64_t window = c.length * 2;
        Stream stream = slice->getStreams()[0];
        Tensor gradientCpu(cpuPlacement, errorInput.getDescriptor());
        for (uint64_t i = 0; i < outerSlices * window; ++i) writeSliceTestValue(gradientCpu, i, sliceTestValue(i + 3));
        errorInput.copyFromAsync(gradientCpu, stream);
        Tensor staleCpu(cpuPlacement, errorOutput.getDescriptor());
        for (uint64_t i = 0; i < outerSlices * 10; ++i) writeUtilityLayerValue<float>(staleCpu, i, -99.0f);
        errorOutput.copyFromAsync(staleCpu, stream);
        stream.synchronize();

        slice->backward(errorInput, validRows);
        Tensor errorOutputCpu(cpuPlacement, errorOutput.getDescriptor());
        errorOutputCpu.copyFromAsync(errorOutput, stream);
        stream.synchronize();
        for (uint64_t outer = 0; outer < outerSlices; ++outer) {
            for (uint64_t j = 0; j < 10; ++j) {
                float expected = -99.0f;
                if (outer < validRows * 3) {
                    const bool inWindow = j >= c.start * 2 && j < c.start * 2 + window;
                    expected = inWindow ? sliceTestValue(outer * window + j - c.start * 2 + 3) : 0.0f;
                }
                EXPECT_EQ(readUtilityLayerValue<float>(errorOutputCpu, outer * 10 + j), expected)
                    << "outer " << outer << " element " << j;
            }
        }
        LayerTestHelper::tearDownNetwork(layers);
    }

    {
        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(gpuPlacement, DataType::FP32, inputDimensions));
        auto upstream = make_shared<NoOpLayer>();
        layers.push_back(upstream);
        layers[0]->connectToNextLayer(upstream.get());
        auto slice = make_shared<ThorImplementation::Slice>(2, 1, 3, DataType::INT32);
        layers.push_back(slice);
        upstream->connectToNextLayer(slice.get());
        layers.push_back(make_shared<BackpropDescriptorSinkLayer>());
        slice->connectToNextLayer(layers.back().get());
        LayerTestHelper::initializeNetwork(layers);
        EXPECT_FALSE(slice->getErrorInputs()[0].has_value());
        EXPECT_FALSE(slice->getErrorOutputs()[0].has_value()) << "an INT32 output takes no gradient";
        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(DeviceCrossing, Crosses) {
    if (MachineEvaluator::instance().getNumGpus() < 2)
        return;

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    vector<unsigned long> dimensions;
    dimensions.push_back(10);
    dimensions.push_back(3);
    dimensions.push_back(6);
    int numElements = 10 * 3 * 6;
    TensorDescriptor descriptor(DataType::FP32, dimensions);
    Tensor sourceCpu(cpuPlacement, descriptor);
    Tensor sourceGpu0(gpu0Placement, descriptor);
    Tensor destCpu(cpuPlacement, descriptor);

    float *sourceMem = (float *)sourceCpu.getMemPtr();
    for (int i = 0; i < numElements; ++i) {
        sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu0));
    layers.push_back(make_shared<DeviceCrossing>(gpu0Placement, gpu1Placement));
    shared_ptr<NetworkOutput> networkOutput = make_shared<NetworkOutput>(gpu1Placement);
    layers.push_back(networkOutput);
    Stream stream = layers.front()->getStream();

    LayerTestHelper::connectAndInitializeNetwork(layers);

    Tensor outputGpu = layers.back()->getFeatureOutput().value();

    // Network is runnable here. Repeat the crossing so DeviceCrossing must
    // re-record each owner-scoped dependency event many times; E1 rejects
    // accidentally reusing one event across producer GPUs.
    Stream outputStream = networkOutput->getStream();
    float *destMem = (float *)destCpu.getMemPtr();
    for (int repetition = 0; repetition < 16; ++repetition) {
        for (int i = 0; i < numElements; ++i) {
            sourceMem[i] = ((repetition + 1) * 1000.0f) + static_cast<float>(i);
        }

        layers[0]->forward(sourceCpu, false);
        destCpu.copyFromAsync(outputGpu, outputStream);
        outputStream.synchronize();

        ASSERT_TRUE(outputGpu.getPlacement() == gpu1Placement);
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ(destMem[i], sourceMem[i]);
        }
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(FiniteCheck, ForwardReportsCpuNonFiniteValues) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorDescriptor descriptor(DataType::FP32, {2, 4});
    Tensor sourceCpu(cpuPlacement, descriptor);

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceCpu));
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<FiniteCheck>("after_projection", 10001, 9001, true, true, true, 4, true));
    layers.push_back(make_shared<BackpropDescriptorSinkLayer>());
    LayerTestHelper::connectAndInitializeNetwork(layers);

    auto finiteCheck = dynamic_pointer_cast<FiniteCheck>(layers[2]);
    ASSERT_NE(finiteCheck, nullptr);
    ASSERT_TRUE(finiteCheck->getFeatureInput().has_value());
    Tensor checkedTensor = finiteCheck->getFeatureInput().value();
    float *values = checkedTensor.getMemPtr<float>();
    values[0] = 1.0f;
    values[1] = std::numeric_limits<float>::quiet_NaN();
    values[2] = std::numeric_limits<float>::infinity();
    values[3] = -std::numeric_limits<float>::infinity();
    values[4] = 5.0f;
    values[5] = 6.0f;
    values[6] = 7.0f;
    values[7] = 8.0f;

    try {
        finiteCheck->forward(checkedTensor, false, 2);
        FAIL() << "Expected FiniteCheck to throw";
    } catch (const std::runtime_error &error) {
        const string message = error.what();
        EXPECT_NE(message.find("FiniteCheck detected non-finite values"), string::npos);
        EXPECT_NE(message.find("label=\"after_projection\""), string::npos);
        EXPECT_NE(message.find("direction=forward"), string::npos);
        EXPECT_NE(message.find("dtype=fp32"), string::npos);
        EXPECT_NE(message.find("shape=[2, 4]"), string::npos);
        EXPECT_NE(message.find("non_finite=3"), string::npos);
        EXPECT_NE(message.find("nan=1"), string::npos);
        EXPECT_NE(message.find("positive_infinity=1"), string::npos);
        EXPECT_NE(message.find("negative_infinity=1"), string::npos);
        EXPECT_NE(message.find("flat_index=1"), string::npos);
        EXPECT_NE(message.find("index=[0, 1]"), string::npos);
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(FiniteCheck, DisabledLayerPassesNonFiniteValuesWithoutChecking) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorDescriptor descriptor(DataType::FP32, {2, 4});
    Tensor sourceCpu(cpuPlacement, descriptor);

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceCpu));
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<FiniteCheck>("disabled_check", 10003, 9003, true, true, true, 4, false));
    layers.push_back(make_shared<BackpropDescriptorSinkLayer>());
    LayerTestHelper::connectAndInitializeNetwork(layers);

    auto finiteCheck = dynamic_pointer_cast<FiniteCheck>(layers[2]);
    ASSERT_NE(finiteCheck, nullptr);
    ASSERT_TRUE(finiteCheck->getFeatureInput().has_value());
    ASSERT_TRUE(finiteCheck->getFeatureOutput().has_value());
    EXPECT_EQ(finiteCheck->getFeatureInput().value().getMemPtr(), finiteCheck->getFeatureOutput().value().getMemPtr());

    Tensor checkedTensor = finiteCheck->getFeatureInput().value();
    float *values = checkedTensor.getMemPtr<float>();
    values[0] = std::numeric_limits<float>::quiet_NaN();
    values[1] = std::numeric_limits<float>::infinity();
    values[2] = -std::numeric_limits<float>::infinity();

    EXPECT_NO_THROW(finiteCheck->forward(checkedTensor, false, 2));

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(FiniteCheck, BackwardAliasesAndReportsGpuGradient) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::FP32, {2, 4});
    Tensor sourceGpu(gpuPlacement, descriptor);

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<FiniteCheck>("projection_gradient", 10002, 9002, true, true, true, 4, true));
    layers.push_back(make_shared<BackpropDescriptorSinkLayer>());
    LayerTestHelper::connectNetwork(layers);

    auto upstream = dynamic_pointer_cast<NoOpLayer>(layers[1]);
    auto finiteCheck = dynamic_pointer_cast<FiniteCheck>(layers[2]);
    auto sink = dynamic_pointer_cast<BackpropDescriptorSinkLayer>(layers[3]);
    ASSERT_NE(upstream, nullptr);
    ASSERT_NE(finiteCheck, nullptr);
    ASSERT_NE(sink, nullptr);
    ASSERT_TRUE(finiteCheck->getErrorInput().has_value());
    ASSERT_TRUE(finiteCheck->getErrorOutput().has_value());
    ASSERT_TRUE(upstream->getErrorInput().has_value());
    ASSERT_TRUE(sink->getErrorOutput().has_value());
    EXPECT_EQ(finiteCheck->getErrorInput().value().getMemPtr(), finiteCheck->getErrorOutput().value().getMemPtr());
    EXPECT_EQ(upstream->getErrorInput().value().getMemPtr(), sink->getErrorOutput().value().getMemPtr());

    LayerTestHelper::initializeNetwork(layers);

    Tensor gradientCpu(cpuPlacement, descriptor);
    float *gradientValues = gradientCpu.getMemPtr<float>();
    for (uint32_t i = 0; i < 8; ++i)
        gradientValues[i] = static_cast<float>(i + 1);
    gradientValues[6] = std::numeric_limits<float>::quiet_NaN();

    Tensor incomingGradient = sink->getErrorOutput().value();
    incomingGradient.copyFromAsync(gradientCpu, finiteCheck->getStream());

    try {
        finiteCheck->backward(incomingGradient, 2);
        FAIL() << "Expected FiniteCheck backward to throw";
    } catch (const std::runtime_error &error) {
        const string message = error.what();
        EXPECT_NE(message.find("label=\"projection_gradient\""), string::npos);
        EXPECT_NE(message.find("direction=backward"), string::npos);
        EXPECT_NE(message.find("tensor_role=incoming_gradient"), string::npos);
        EXPECT_NE(message.find("dtype=fp32"), string::npos);
        EXPECT_NE(message.find("non_finite=1"), string::npos);
        EXPECT_NE(message.find("nan=1"), string::npos);
        EXPECT_NE(message.find("flat_index=6"), string::npos);
        EXPECT_NE(message.find("index=[1, 2]"), string::npos);
    }

    LayerTestHelper::tearDownNetwork(layers);
}

namespace {
class BatchCardinalityCaptureLayer final : public NoOpLayer {
   public:
    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t validExampleCount = 0) override {
        observedValidExampleCounts.push_back(validExampleCount);
        NoOpLayer::forward(featureInput, validationPass, validExampleCount);
    }

    std::vector<uint32_t> observedValidExampleCounts;
};
}  // namespace

TEST(Concatenate, PropagatesPartialBatchCardinalityAndRejectsMismatchedInputs) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor inputDescriptor(DataType::FP32, {4, 1});
    Tensor firstCpu(cpuPlacement, inputDescriptor);
    Tensor secondCpu(cpuPlacement, inputDescriptor);

    auto firstInput = make_shared<NetworkInput>(gpuPlacement, DataType::FP32, vector<unsigned long>{4, 1});
    auto secondInput = make_shared<NetworkInput>(gpuPlacement, DataType::FP32, vector<unsigned long>{4, 1});
    auto concatenate = make_shared<Concatenate>(1, 2, DataType::FP32);
    auto capture = make_shared<BatchCardinalityCaptureLayer>();
    auto output = make_shared<NetworkOutput>(cpuPlacement);
    vector<shared_ptr<Layer>> layers{firstInput, secondInput, concatenate, capture, output};

    firstInput->connectToNextLayer(concatenate.get(), 0, 0);
    secondInput->connectToNextLayer(concatenate.get(), 0, 1);
    concatenate->connectToNextLayer(capture.get());
    capture->connectToNextLayer(output.get());
    LayerTestHelper::initializeNetwork(layers);

    firstInput->forward(firstCpu, false, 2);
    secondInput->forward(secondCpu, false, 2);
    ASSERT_EQ(capture->observedValidExampleCounts, (vector<uint32_t>{2}));

    firstInput->forward(firstCpu, false, 2);
    EXPECT_THROW(secondInput->forward(secondCpu, false, 3), std::logic_error);

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(Concatenate, BatchAxisAcceptsOnlyFullCapacityInputs) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor firstDescriptor(DataType::FP32, {2, 1});
    TensorDescriptor secondDescriptor(DataType::FP32, {3, 1});
    Tensor firstCpu(cpuPlacement, firstDescriptor);
    Tensor secondCpu(cpuPlacement, secondDescriptor);

    auto firstInput = make_shared<NetworkInput>(gpuPlacement, DataType::FP32, vector<unsigned long>{2, 1});
    auto secondInput = make_shared<NetworkInput>(gpuPlacement, DataType::FP32, vector<unsigned long>{3, 1});
    auto concatenate = make_shared<Concatenate>(0, 2, DataType::FP32);
    auto capture = make_shared<BatchCardinalityCaptureLayer>();
    auto output = make_shared<NetworkOutput>(cpuPlacement);
    vector<shared_ptr<Layer>> layers{firstInput, secondInput, concatenate, capture, output};

    firstInput->connectToNextLayer(concatenate.get(), 0, 0);
    secondInput->connectToNextLayer(concatenate.get(), 0, 1);
    concatenate->connectToNextLayer(capture.get());
    capture->connectToNextLayer(output.get());
    LayerTestHelper::initializeNetwork(layers);

    firstInput->forward(firstCpu, false);
    secondInput->forward(secondCpu, false);
    ASSERT_EQ(capture->observedValidExampleCounts, (vector<uint32_t>{5}));

    firstInput->forward(firstCpu, false);
    EXPECT_THROW(secondInput->forward(secondCpu, false, 2), std::logic_error);

    LayerTestHelper::tearDownNetwork(layers);
}

// Inputs of FP32, BF16 and INT32 into an FP32 output: each input is converted as it moves, the gradient goes back to
// each floating-point input in its own type, and the INT32 input takes none.
TEST(Concatenate, ConvertsEachInputToTheOutputTypeAndTheGradientBack) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    constexpr uint32_t rows = 4;
    const vector<DataType> dataTypes{DataType::FP32, DataType::BF16, DataType::INT32};
    const vector<uint64_t> widths{3, 2, 1};
    constexpr uint64_t outputWidth = 6;

    vector<Tensor> inputsCpu;
    for (uint32_t i = 0; i < 3; ++i) inputsCpu.emplace_back(cpuPlacement, TensorDescriptor(dataTypes[i], {rows, widths[i]}));
    // Every value is exact in each type it passes through.
    auto inputValue = [](uint32_t input, uint32_t row, uint32_t column) -> float {
        if (input == 0) return row * 10.0f + column + 0.25f;
        if (input == 1) return -(row * 10.0f + column) - 0.5f;
        return row * 100.0f - 7.0f;
    };
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < 3; ++c) writeUtilityLayerValue<float>(inputsCpu[0], r * 3 + c, inputValue(0, r, c));
        for (uint32_t c = 0; c < 2; ++c) writeUtilityLayerValue<__nv_bfloat16>(inputsCpu[1], r * 2 + c, inputValue(1, r, c));
        inputsCpu[2].getMemPtr<int32_t>()[r] = static_cast<int32_t>(inputValue(2, r, 0));
    }

    {
        vector<shared_ptr<Layer>> layers;
        for (uint32_t i = 0; i < 3; ++i)
            layers.push_back(make_shared<NetworkInput>(gpuPlacement, dataTypes[i], vector<unsigned long>{rows, widths[i]}));
        auto concatenate = make_shared<Concatenate>(1, 3, DataType::FP32);
        layers.push_back(concatenate);
        for (uint32_t i = 0; i < 3; ++i) layers[i]->connectToNextLayer(concatenate.get(), 0, static_cast<int>(i));
        layers.push_back(make_shared<NoOpLayer>());
        concatenate->connectToNextLayer(layers.back().get());
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));
        layers[layers.size() - 2]->connectToNextLayer(layers.back().get());
        LayerTestHelper::initializeNetwork(layers);

        Tensor outputGpu = layers.back()->getFeatureOutput().value();
        ASSERT_EQ(outputGpu.getDescriptor().getDataType(), DataType::FP32);
        ASSERT_EQ(outputGpu.getDimensions(), (vector<uint64_t>{rows, outputWidth}));
        for (uint32_t i = 0; i < 3; ++i) layers[i]->forward(inputsCpu[i], false);
        Tensor outputCpu(cpuPlacement, outputGpu.getDescriptor());
        Stream stream = layers[0]->getStream();
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        outputCpu.copyFromAsync(outputGpu, stream);
        stream.synchronize();
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < outputWidth; ++c) {
                const uint32_t input = c < 3 ? 0 : c < 5 ? 1 : 2;
                const uint32_t column = c < 3 ? c : c < 5 ? c - 3 : 0;
                EXPECT_EQ(readUtilityLayerValue<float>(outputCpu, r * outputWidth + c), inputValue(input, r, column))
                    << "row " << r << " column " << c;
            }
        }
        LayerTestHelper::tearDownNetwork(layers);
    }

    {
        vector<shared_ptr<Layer>> layers;
        vector<shared_ptr<Layer>> upstream;
        for (uint32_t i = 0; i < 3; ++i) {
            layers.push_back(make_shared<NetworkInput>(gpuPlacement, dataTypes[i], vector<unsigned long>{rows, widths[i]}));
            upstream.push_back(make_shared<NoOpLayer>());
            layers.push_back(upstream.back());
            layers[layers.size() - 2]->connectToNextLayer(upstream.back().get());
        }
        auto concatenate = make_shared<Concatenate>(1, 3, DataType::FP32);
        layers.push_back(concatenate);
        for (uint32_t i = 0; i < 3; ++i) upstream[i]->connectToNextLayer(concatenate.get(), 0, static_cast<int>(i));
        layers.push_back(make_shared<BackpropDescriptorSinkLayer>());
        concatenate->connectToNextLayer(layers.back().get());
        LayerTestHelper::initializeNetwork(layers);

        const vector<std::optional<Tensor>> errorInputs = concatenate->getErrorInputs();
        const vector<std::optional<Tensor>> errorOutputs = concatenate->getErrorOutputs();
        ASSERT_EQ(errorInputs.size(), 1U);
        ASSERT_TRUE(errorInputs[0].has_value());
        ASSERT_EQ(errorInputs[0].value().getDescriptor().getDataType(), DataType::FP32);
        ASSERT_EQ(errorOutputs.size(), 3U);
        ASSERT_TRUE(errorOutputs[0].has_value());
        ASSERT_TRUE(errorOutputs[1].has_value());
        EXPECT_FALSE(errorOutputs[2].has_value()) << "an INT32 input takes no gradient";
        EXPECT_EQ(errorOutputs[0].value().getDescriptor().getDataType(), DataType::FP32);
        EXPECT_EQ(errorOutputs[1].value().getDescriptor().getDataType(), DataType::BF16);

        Tensor errorInputCpu(cpuPlacement, errorInputs[0].value().getDescriptor());
        auto gradient = [](uint32_t row, uint32_t column) { return (row + 1) * (column + 1) * 0.5f; };
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < outputWidth; ++c) writeUtilityLayerValue<float>(errorInputCpu, r * outputWidth + c, gradient(r, c));
        Stream stream = concatenate->getStreams()[0];
        Tensor errorInputGpu = errorInputs[0].value();
        errorInputGpu.copyFromAsync(errorInputCpu, stream);
        concatenate->backward(errorInputs[0], rows);
        Tensor wideGradientCpu(cpuPlacement, errorOutputs[0].value().getDescriptor());
        Tensor halfGradientCpu(cpuPlacement, errorOutputs[1].value().getDescriptor());
        wideGradientCpu.copyFromAsync(errorOutputs[0].value(), stream);
        halfGradientCpu.copyFromAsync(errorOutputs[1].value(), stream);
        stream.synchronize();
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < 3; ++c) EXPECT_EQ(readUtilityLayerValue<float>(wideGradientCpu, r * 3 + c), gradient(r, c));
            for (uint32_t c = 0; c < 2; ++c)
                EXPECT_EQ(readUtilityLayerValue<__nv_bfloat16>(halfGradientCpu, r * 2 + c), gradient(r, 3 + c));
        }
        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Split, PropagatesPartialBatchCardinalityToEveryOutput) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor inputDescriptor(DataType::FP32, {4, 2});
    Tensor inputCpu(cpuPlacement, inputDescriptor);

    auto input = make_shared<NetworkInput>(gpuPlacement, DataType::FP32, vector<unsigned long>{4, 2});
    auto split = make_shared<Split>(1, vector<unsigned long>{1, 1}, vector<DataType>(2, DataType::FP32));
    auto firstCapture = make_shared<BatchCardinalityCaptureLayer>();
    auto secondCapture = make_shared<BatchCardinalityCaptureLayer>();
    auto firstOutput = make_shared<NetworkOutput>(cpuPlacement);
    auto secondOutput = make_shared<NetworkOutput>(cpuPlacement);
    vector<shared_ptr<Layer>> layers{input, split, firstCapture, secondCapture, firstOutput, secondOutput};

    input->connectToNextLayer(split.get());
    split->connectToNextLayer(firstCapture.get(), 0);
    split->connectToNextLayer(secondCapture.get(), 1);
    firstCapture->connectToNextLayer(firstOutput.get());
    secondCapture->connectToNextLayer(secondOutput.get());
    LayerTestHelper::initializeNetwork(layers);

    input->forward(inputCpu, false, 2);
    ASSERT_EQ(firstCapture->observedValidExampleCounts, (vector<uint32_t>{2}));
    ASSERT_EQ(secondCapture->observedValidExampleCounts, (vector<uint32_t>{2}));

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(FiniteCheck, RaggedRejectsFullOffsetsCarrier) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor values(cpuPlacement, TensorDescriptor(DataType::FP32, {6, 2}));
    Tensor fullOffsets(cpuPlacement, TensorDescriptor(DataType::UINT32, {4}));
    Stream valuesStream(cpuPlacement);
    Stream offsetsStream(cpuPlacement);
    NoOpLayer valuesProducer;
    NoOpLayer offsetsProducer;

    FiniteCheck finiteCheck(
        "ragged_check",
        12001,
        11001,
        true,
        false,
        true,
        4,
        true,
        FiniteCheck::RaggedConfiguration{
            .batchSize = 3,
            .maxTotalValues = 6,
            .elementsPerValue = 2,
            .offsetsDataType = DataType::UINT32,
        });

    finiteCheck.connectToPreviousLayer(&valuesProducer, values, valuesStream, false, 0);
    EXPECT_THROW(
        finiteCheck.connectToPreviousLayer(&offsetsProducer, fullOffsets, offsetsStream, false, 1), std::runtime_error);
}

TEST(FiniteCheck, RaggedForwardAndBackwardIgnoreUndefinedInactiveCapacity) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor values(cpuPlacement, TensorDescriptor(DataType::FP32, {6, 2}));
    Tensor activeCount(cpuPlacement, TensorDescriptor(DataType::UINT32, {1}));
    Tensor gradient(cpuPlacement, TensorDescriptor(DataType::FP32, {6, 2}));
    Stream valuesStream(cpuPlacement);
    Stream activeCountStream(cpuPlacement);
    NoOpLayer valuesProducer;
    NoOpLayer activeCountProducer;

    auto finiteCheck = make_shared<FiniteCheck>(
        "ragged_check",
        12001,
        11001,
        true,
        true,
        true,
        4,
        true,
        FiniteCheck::RaggedConfiguration{
            .batchSize = 3,
            .maxTotalValues = 6,
            .elementsPerValue = 2,
            .offsetsDataType = DataType::UINT32,
        });
    finiteCheck->connectToPreviousLayer(&valuesProducer, values, valuesStream, false, 0);
    finiteCheck->connectToPreviousLayer(&activeCountProducer, activeCount, activeCountStream, false, 1);

    // This test constructs the implementation-layer chain manually rather than
    // through LayerTestHelper. FiniteCheck still follows the ordinary backward
    // path after checking the incoming gradient, propagating a null error to
    // valuesProducer because backPropagateError is false. The upstream layer
    // therefore must participate in the normal compile/initialize lifecycle.
    valuesProducer.compile();
    valuesProducer.initialize();
    finiteCheck->compile();
    finiteCheck->initialize();

    activeCount.getMemPtr<uint32_t>()[0] = 3;
    RowPartitionRuntime::publishHostState(activeCount,
                                          RowPartitionDescriptor(3, 6, DataType::UINT32),
                                          activeCount.getTensorId(),
                                          {0, 1, 1, 3});

    auto* packedValues = values.getMemPtr<float>();
    auto* packedGradient = gradient.getMemPtr<float>();
    for (uint64_t i = 0; i < 12; ++i) {
        packedValues[i] = static_cast<float>(i + 1);
        packedGradient[i] = static_cast<float>(i + 1);
    }
    // activeCount[0] == 3 packed values, so flat elements [6, 12) are undefined tail.
    packedValues[6] = std::numeric_limits<float>::quiet_NaN();
    packedValues[9] = std::numeric_limits<float>::infinity();
    packedGradient[7] = std::numeric_limits<float>::quiet_NaN();
    packedGradient[10] = -std::numeric_limits<float>::infinity();

    EXPECT_NO_THROW(finiteCheck->forward(activeCount, false, 3));
    EXPECT_NO_THROW(finiteCheck->forward(values, false, 3));
    EXPECT_NO_THROW(finiteCheck->backward(gradient, 3));

    // Move one NaN into the authoritative active prefix and verify both directions report it.
    packedValues[5] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_NO_THROW(finiteCheck->forward(values, false, 3));
    EXPECT_THROW(finiteCheck->forward(activeCount, false, 3), std::runtime_error);
    packedValues[5] = 6.0f;

    packedGradient[4] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(finiteCheck->backward(gradient, 3), std::runtime_error);

    finiteCheck->cleanup();
    valuesProducer.cleanup();
}

TEST(NetworkOutput, ClassifiesWhetherReadyEventEscapesProcessingStreamBarrier) {
    if (MachineEvaluator::instance().getNumGpus() < 1)
        GTEST_SKIP() << "NetworkOutput stream classification test requires a GPU";

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Tensor sourceGpu(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));

    {
        auto input = make_shared<NetworkInput>(sourceGpu);
        auto output = make_shared<NetworkOutput>(gpuPlacement);
        vector<shared_ptr<Layer>> layers{input, output};
        LayerTestHelper::connectAndInitializeNetwork(layers);
        std::optional<Stream> readyStream = output->getIndependentOutputReadyEventStream();
        EXPECT_FALSE(readyStream.has_value());
        LayerTestHelper::tearDownNetwork(layers);
    }

    {
        auto input = make_shared<NetworkInput>(sourceGpu);
        auto output = make_shared<NetworkOutput>(cpuPlacement);
        vector<shared_ptr<Layer>> layers{input, output};
        LayerTestHelper::connectAndInitializeNetwork(layers);
        std::optional<Stream> readyStream = output->getIndependentOutputReadyEventStream();
        ASSERT_TRUE(readyStream.has_value());
        EXPECT_FALSE(readyStream.value() == output->getStream());
        LayerTestHelper::tearDownNetwork(layers);
    }
}
