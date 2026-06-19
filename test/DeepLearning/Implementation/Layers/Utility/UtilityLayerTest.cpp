#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/Concatenate.h"
#include "DeepLearning/Implementation/Layers/Utility/DeviceCrossing.h"
#include "DeepLearning/Implementation/Layers/Utility/Extract.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"
#include "DeepLearning/Implementation/Layers/Utility/Map.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Layers/Utility/Pad.h"
#include "DeepLearning/Implementation/Layers/Utility/Reshape.h"
#include "DeepLearning/Implementation/Layers/Utility/Split.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Layers/Utility/TypeConversion.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_fp8.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

using std::pair;
using std::set;
using std::vector;

using namespace ThorImplementation;
using namespace std;

inline int randomElement(set<int> &filledSet) {
    assert(!filledSet.empty());
    set<int>::iterator it = filledSet.begin();
    advance(it, rand() % filledSet.size());
    int element = *it;
    filledSet.erase(it);
    return element;
}


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

TEST(Map, MapsCorrectlyToSameNumberOfElements) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 10; ++test) {
        int numDimensions = 2;
        vector<unsigned long> dimensions;
        for (int i = 0; i < numDimensions; ++i)
            dimensions.push_back((rand() % 100) + 1);
        int rows = dimensions[0];
        int cols = dimensions[1];
        int numElements = rows * cols;
        TensorDescriptor mappingDescriptor(DataType::UINT32, dimensions);
        Tensor mappingCpu = Tensor(cpuPlacement, mappingDescriptor);
        Tensor mappingGpu = mappingCpu.clone(gpuPlacement);

        TensorDescriptor sourceDestDescriptor(DataType::FP16, dimensions);
        Tensor sourceCpu(cpuPlacement, sourceDestDescriptor);
        Tensor sourceGpu(gpuPlacement, sourceDestDescriptor);
        Tensor destCpu(cpuPlacement, sourceDestDescriptor);

        set<int> destIndexes;
        for (int i = 0; i < numElements; ++i)
            destIndexes.insert(i);

        unsigned int *mappingMem = (unsigned int *)mappingCpu.getMemPtr();
        half *sourceMem = (half *)sourceCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            mappingMem[i] = randomElement(destIndexes);
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));

        Stream stream = layers.front()->getStream();
        mappingGpu.copyFromAsync(mappingCpu, stream);
        stream.synchronize();

        layers.push_back(make_shared<Map<unsigned int>>(mappingGpu, sourceGpu.getDescriptor().getDimensions()));
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        half *destMem = (half *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)destMem[i], (float)sourceMem[mappingMem[i]]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(Map, MapsCorrectlyToFewerElements) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    vector<unsigned long> sourceDimensions;
    sourceDimensions.push_back(10);
    sourceDimensions.push_back(5);
    sourceDimensions.push_back(5);
    int numSourceElements = 250;

    vector<unsigned long> destDimensions;
    destDimensions.push_back(10);
    destDimensions.push_back(10);
    int numDestElements = 100;

    TensorDescriptor sourceDescriptor(DataType::FP16, sourceDimensions);
    Tensor sourceCpu(cpuPlacement, sourceDescriptor);
    Tensor sourceGpu(gpuPlacement, sourceDescriptor);

    TensorDescriptor mappingDescriptor(DataType::UINT8, destDimensions);
    Tensor mappingCpu = Tensor(cpuPlacement, mappingDescriptor);
    Tensor mappingGpu = mappingCpu.clone(gpuPlacement);

    TensorDescriptor destDescriptor(DataType::FP16, destDimensions);
    Tensor destCpu(cpuPlacement, destDescriptor);

    uint8_t *mappingMem = (uint8_t *)mappingCpu.getMemPtr();
    for (int i = 0; i < numDestElements; ++i) {
        mappingMem[i] = rand() % numSourceElements;
    }

    half *sourceMem = (half *)sourceCpu.getMemPtr();
    for (int i = 0; i < numSourceElements; ++i) {
        sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));

    Stream stream = layers.front()->getStream();
    mappingGpu.copyFromAsync(mappingCpu, stream);
    stream.synchronize();

    layers.push_back(make_shared<Map<uint8_t>>(mappingGpu, sourceGpu.getDescriptor().getDimensions()));
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    LayerTestHelper::connectAndInitializeNetwork(layers);
    Tensor outputGpu = layers.back()->getFeatureOutput().value();

    // Network is runnable here
    layers[0]->forward(sourceCpu, false);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
    destCpu.copyFromAsync(outputGpu, stream);

    cudaStatus = cudaStreamSynchronize(stream.getStream());
    assert(cudaStatus == cudaSuccess);

    half *destMem = (half *)destCpu.getMemPtr();
    for (int i = 0; i < numDestElements; ++i) {
        ASSERT_EQ((float)destMem[i], (float)sourceMem[mappingMem[i]]);
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(Map, MapsCorrectlyToMoreElements) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    vector<unsigned long> sourceDimensions;
    sourceDimensions.push_back(10);
    sourceDimensions.push_back(5);
    int numSourceElements = 50;

    vector<unsigned long> destDimensions;
    destDimensions.push_back(10);
    destDimensions.push_back(10);
    int numDestElements = 100;

    TensorDescriptor sourceDescriptor(DataType::FP16, sourceDimensions);
    Tensor sourceCpu(cpuPlacement, sourceDescriptor);
    Tensor sourceGpu(gpuPlacement, sourceDescriptor);

    TensorDescriptor mappingDescriptor(DataType::UINT64, destDimensions);
    Tensor mappingCpu = Tensor(cpuPlacement, mappingDescriptor);
    Tensor mappingGpu = mappingCpu.clone(gpuPlacement);

    TensorDescriptor destDescriptor(DataType::FP16, destDimensions);
    Tensor destCpu(cpuPlacement, destDescriptor);

    unsigned long *mappingMem = (unsigned long *)mappingCpu.getMemPtr();
    for (int i = 0; i < numDestElements; ++i) {
        mappingMem[i] = rand() % numSourceElements;
    }

    half *sourceMem = (half *)sourceCpu.getMemPtr();
    for (int i = 0; i < numSourceElements; ++i) {
        sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }

    vector<shared_ptr<Layer>> layers;
    layers.push_back(make_shared<NetworkInput>(sourceGpu));

    Stream stream = layers.front()->getStream();
    mappingGpu.copyFromAsync(mappingCpu, stream);
    stream.synchronize();

    layers.push_back(make_shared<Map<unsigned long>>(mappingGpu, sourceGpu.getDescriptor().getDimensions()));
    layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

    LayerTestHelper::connectAndInitializeNetwork(layers);
    Tensor outputGpu = layers.back()->getFeatureOutput().value();

    // Network is runnable here
    layers[0]->forward(sourceCpu, false);
    stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
    destCpu.copyFromAsync(outputGpu, stream);

    cudaStatus = cudaStreamSynchronize(stream.getStream());
    assert(cudaStatus == cudaSuccess);

    half *destMem = (half *)destCpu.getMemPtr();
    for (int i = 0; i < numDestElements; ++i) {
        ASSERT_EQ((float)destMem[i], (float)sourceMem[mappingMem[i]]);
    }

    LayerTestHelper::tearDownNetwork(layers);
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

TEST(TypeConversion, Converts) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    cudaError_t cudaStatus;

    for (int test = 0; test < 10; ++test) {
        int numDimensions = (rand() % 5) + 1;
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            dimensions.push_back((rand() % 5) + 1);
            numElements *= dimensions.back();
        }

        TensorDescriptor sourceDescriptor(DataType::FP32, dimensions);
        TensorDescriptor destDescriptor(DataType::INT64, dimensions);
        Tensor sourceCpu(cpuPlacement, sourceDescriptor);
        Tensor sourceGpu(gpuPlacement, sourceDescriptor);
        Tensor destCpu(cpuPlacement, destDescriptor);

        float *sourceMem = (float *)sourceCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));
        layers.push_back(make_shared<TypeConversion>(DataType::INT32));
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        destCpu.copyFromAsync(outputGpu, stream);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        int64_t *destMem = (int64_t *)destCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ(destMem[i], (int64_t)sourceMem[i]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
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


TEST(TensorFanout, BackwardSumsBF16Errors) {
    expectTensorFanoutSumsBackwardErrorsForDType<__nv_bfloat16>();
}

TEST(TensorFanout, BackwardSumsFp8E4M3Errors) {
    expectTensorFanoutSumsBackwardErrorsForDType<__nv_fp8_e4m3>();
}

TEST(TensorFanout, BackwardSumsFp8E5M2Errors) {
    expectTensorFanoutSumsBackwardErrorsForDType<__nv_fp8_e5m2>();
}

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
        layers.push_back(make_shared<Concatenate>(axis));
        for (unsigned int i = 0; i < layers.size() - 1; ++i)
            layers[i]->connectToNextLayer(layers.back().get());
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

TEST(Split, Splits) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    long axisElementsPerDestArray[10];
    long stridePerSourceDimension[10];
    long stridePerDestDimension[5 * 10];

    for (int test = 0; test < 50; ++test) {
        vector<Tensor> partsCpu;
        vector<Tensor> partsGpu;
        Tensor wholeCpu;
        Tensor wholeGpu;

        int numSplitTensors = (rand() % 5) + 2;
        int numDimensions = (rand() % 6) + 1;

        vector<unsigned long> wholeDimensions;
        int axis = rand() % numDimensions;
        for (int d = 0; d < numDimensions; d++) {
            if (d == axis)
                wholeDimensions.push_back(0);
            else
                wholeDimensions.push_back((rand() % 10) + 1);
        }

        for (int t = 0; t < numSplitTensors; ++t) {
            vector<unsigned long> splitArrayDimensions = wholeDimensions;
            splitArrayDimensions[axis] = (rand() % 5) + 1;
            axisElementsPerDestArray[t] = splitArrayDimensions[axis];
            wholeDimensions[axis] += splitArrayDimensions[axis];

            TensorDescriptor partDescriptor(DataType::FP16, splitArrayDimensions);
            partsCpu.emplace_back(cpuPlacement, partDescriptor);
        }
        TensorDescriptor wholeDescriptor(DataType::FP16, wholeDimensions);
        wholeCpu = Tensor(cpuPlacement, wholeDescriptor);
        wholeGpu = Tensor(gpuPlacement, wholeDescriptor);

        long numElements = wholeCpu.getDescriptor().getTotalNumElements();

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(wholeGpu));
        Stream stream = layers.front()->getStream();

        stridePerSourceDimension[numDimensions - 1] = 1;
        for (int dest = 0; dest < numSplitTensors; dest++)
            stridePerDestDimension[dest * numDimensions + numDimensions - 1] = 1;
        for (int i = numDimensions - 2; i >= 0; --i) {
            stridePerSourceDimension[i] = stridePerSourceDimension[i + 1] * wholeDimensions[i + 1];
            for (int dest = 0; dest < numSplitTensors; ++dest)
                if (i + 1 == axis)
                    stridePerDestDimension[dest * numDimensions + i] =
                        stridePerDestDimension[dest * numDimensions + i + 1] * axisElementsPerDestArray[dest];
                else
                    stridePerDestDimension[dest * numDimensions + i] =
                        stridePerDestDimension[dest * numDimensions + i + 1] * wholeDimensions[i + 1];
        }

        half *mem = (half *)wholeCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            mem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<unsigned long> axisElements;
        for (int i = 0; i < numSplitTensors; ++i)
            axisElements.push_back(axisElementsPerDestArray[i]);
        shared_ptr<Layer> splitLayer = make_shared<Split>(axis, axisElements);
        layers.back()->connectToNextLayer(splitLayer.get());
        layers.push_back(splitLayer);
        vector<shared_ptr<Layer>> outputLayers;
        for (int i = 0; i < numSplitTensors; ++i) {
            shared_ptr<Layer> noOpLayer = make_shared<NoOpLayer>();
            layers.push_back(noOpLayer);
            splitLayer->connectToNextLayer(noOpLayer.get());

            shared_ptr<Layer> networkOutputLayer = make_shared<NetworkOutput>(gpuPlacement);
            layers.push_back(networkOutputLayer);
            noOpLayer->connectToNextLayer(networkOutputLayer.get());
            partsGpu.push_back(networkOutputLayer->getFeatureOutput().value());
            outputLayers.push_back(networkOutputLayer);
        }

        LayerTestHelper::initializeNetwork(layers);

        // Network is runnable here
        layers[0]->forward(wholeCpu, false);

        for (unsigned int i = 0; i < partsGpu.size(); ++i) {
            stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(outputLayers[i])->getOutputReadyEvent());
            partsCpu[i].copyFromAsync(partsGpu[i], stream);
        }
        stream.synchronize();

        half *wholeMem = (half *)wholeCpu.getMemPtr();
        vector<int> destTensorAxisIndexStart;
        destTensorAxisIndexStart.push_back(0);
        for (int i = 1; i < numSplitTensors; ++i)
            destTensorAxisIndexStart.push_back(destTensorAxisIndexStart.back() + axisElementsPerDestArray[i - 1]);

        int sourceIndex[10];
        int destIndex[10];
        for (int sourceFlatIndex = 0; sourceFlatIndex < numElements; ++sourceFlatIndex) {
            computeIndex(sourceFlatIndex, sourceIndex, numDimensions, stridePerSourceDimension);
            int dest = 0;
            for (; sourceIndex[axis] >= destTensorAxisIndexStart[dest] + axisElementsPerDestArray[dest]; ++dest)
                ;
            for (int i = 0; i < numDimensions; ++i) {
                if (i == axis)
                    destIndex[i] = sourceIndex[i] - destTensorAxisIndexStart[dest];
                else
                    destIndex[i] = sourceIndex[i];
            }
            int destFlatIndex = computeFlatIndex(destIndex, stridePerDestDimension + dest * numDimensions, numDimensions);

            half *partsMem = (half *)partsCpu[dest].getMemPtr();
            // printf("sourceFlatIndex %d destFlatIndex%d %d sourceIndex[%d][%d] destIndex%d[%d][%d] source %f dest %f\n",
            //       sourceFlatIndex,
            //       dest,
            //       destFlatIndex,
            //       sourceIndex[0],
            //       sourceIndex[1],
            //       dest,
            //       destIndex[0],
            //       destIndex[1],
            //       (float)wholeMem[sourceFlatIndex],
            //       (float)partsMem[destFlatIndex]);
            ASSERT_EQ((float)wholeMem[sourceFlatIndex], (float)partsMem[destFlatIndex]);
        }

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

    // Network is runnable here
    layers[0]->forward(sourceCpu, false);
    Stream outputStream = networkOutput->getStream();
    destCpu.copyFromAsync(outputGpu, outputStream);
    outputStream.synchronize();

    ASSERT_TRUE(outputGpu.getPlacement() == gpu1Placement);

    float *destMem = (float *)destCpu.getMemPtr();
    for (int i = 0; i < numElements; ++i) {
        ASSERT_EQ(destMem[i], sourceMem[i]);
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(Pad, Pads) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 10; ++test) {
        int numDimensions = (rand() % 5) + 1;

        vector<unsigned long> inputDimensions;
        int numInputElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            inputDimensions.push_back((rand() % 7) + 1);
            numInputElements *= inputDimensions.back();
        }

        TensorDescriptor sourceDescriptor(DataType::FP16, inputDimensions);
        Tensor sourceCpu(cpuPlacement, sourceDescriptor);
        Tensor sourceGpu(gpuPlacement, sourceDescriptor);

        half *sourceMem = (half *)sourceCpu.getMemPtr();
        for (int i = 0; i < numInputElements; ++i) {
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));
        Stream stream = layers.front()->getStream();

        map<unsigned int, pair<unsigned int, unsigned int>> paddingAmount;
        vector<unsigned long> outputDimensions = inputDimensions;
        for (int i = 0; i < numDimensions; ++i) {
            if (rand() % 5 == 0)
                continue;
            pair<unsigned int, unsigned int> dimensionPadding(rand() % 6, rand() % 6);
            if (i < 2)
                dimensionPadding = pair<unsigned int, unsigned int>(rand() % 45, rand() % 45);
            paddingAmount[i] = dimensionPadding;
            outputDimensions[i] += dimensionPadding.first + dimensionPadding.second;
        }
        if (paddingAmount.empty()) {
            int d = rand() % numDimensions;
            pair<unsigned int, unsigned int> dimensionPadding(rand() % 6, rand() % 6);
            paddingAmount[d] = dimensionPadding;
            outputDimensions[d] += dimensionPadding.first + dimensionPadding.second;
        }

        TensorDescriptor destDescriptor(DataType::FP16, outputDimensions);
        Tensor destCpu(cpuPlacement, destDescriptor);

        layers.push_back(make_shared<Pad>(paddingAmount));
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);
        stream.synchronize();

        vector<int> stridePerInputDimension;
        vector<int> stridePerOutputDimension;
        for (int d = 0; d < numDimensions; ++d) {
            stridePerInputDimension.push_back(1);
            stridePerOutputDimension.push_back(1);
        }
        for (int d = numDimensions - 2; d >= 0; --d) {
            stridePerInputDimension[d] = stridePerInputDimension[d + 1] * inputDimensions[d + 1];
            stridePerOutputDimension[d] = stridePerOutputDimension[d + 1] * outputDimensions[d + 1];
        }

        half *destMem = (half *)destCpu.getMemPtr();

        unsigned int sourceFlatIndex = 0;
        for (unsigned int destFlatIndex = 0; destFlatIndex < destDescriptor.getTotalNumElements(); ++destFlatIndex) {
            vector<unsigned long> outputDimensionIndex = destCpu.getDescriptor().getDimensionalIndex(destFlatIndex);

            // Just a sanity check on this test
            if (destFlatIndex == destDescriptor.getTotalNumElements() - 1) {
                for (unsigned int d = 0; d < inputDimensions.size(); ++d)
                    assert(outputDimensionIndex[d] == outputDimensions[d] - 1);
            }

            for (int d = 0; d < numDimensions; ++d) {
                if (paddingAmount.count(d) == 1 && (outputDimensionIndex[d] < paddingAmount[d].first ||
                                                    outputDimensionIndex[d] >= paddingAmount[d].first + inputDimensions[d])) {
                    ASSERT_EQ((float)destMem[destFlatIndex], 0.0f);
                    break;
                } else if (d == numDimensions - 1) {
                    vector<unsigned long> inputDimensionIndex = sourceCpu.getDescriptor().getDimensionalIndex(sourceFlatIndex);

                    // Just a sanity check on this test
                    for (int j = 0; j < numDimensions; ++j)
                        ASSERT_EQ(inputDimensionIndex[j], outputDimensionIndex[j] - paddingAmount[j].first);

                    // Just a sanity check on this test
                    if (sourceFlatIndex == sourceDescriptor.getTotalNumElements() - 1) {
                        inputDimensionIndex = sourceCpu.getDescriptor().getDimensionalIndex(sourceFlatIndex);
                        for (unsigned int j = 0; j < inputDimensions.size(); ++j)
                            ASSERT_EQ(inputDimensionIndex[j], inputDimensions[j] - 1);
                    }

                    ASSERT_EQ((float)destMem[destFlatIndex], (float)sourceMem[sourceFlatIndex]);
                    sourceFlatIndex += 1;
                }
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

unsigned int safeMod(unsigned int a, unsigned int b) {
    if (b == 0)
        return 0;
    else
        return a % b;
}

bool isWithinSpan(vector<unsigned long> sourceDimensionalIndex, vector<pair<unsigned int, unsigned int>> dimensionSpans) {
    for (unsigned int d = 0; d < dimensionSpans.size(); ++d) {
        if (sourceDimensionalIndex[d] < dimensionSpans[d].first || sourceDimensionalIndex[d] > dimensionSpans[d].second)
            return false;
    }
    return true;
}

TEST(Extract, Extracts) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 10; ++test) {
        int numDimensions = (rand() % 5) + 1;

        vector<unsigned long> inputDimensions;
        int numInputElements = 1;
        for (int i = 0; i < numDimensions; ++i) {
            inputDimensions.push_back((rand() % 7) + 1);
            numInputElements *= inputDimensions.back();
        }

        TensorDescriptor sourceDescriptor(DataType::FP16, inputDimensions);
        Tensor sourceCpu(cpuPlacement, sourceDescriptor);
        Tensor sourceGpu(gpuPlacement, sourceDescriptor);

        half *sourceMem = (half *)sourceCpu.getMemPtr();
        for (int i = 0; i < numInputElements; ++i) {
            sourceMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        vector<shared_ptr<Layer>> layers;
        layers.push_back(make_shared<NetworkInput>(sourceGpu));
        Stream stream = layers.front()->getStream();

        vector<pair<unsigned int, unsigned int>> dimensionSpans;
        vector<unsigned long> outputDimensions;
        for (int i = 0; i < numDimensions; ++i) {
            unsigned int start = safeMod(rand(), inputDimensions[i] - 1);
            unsigned int end = safeMod(rand(), inputDimensions[i] - start) + start;
            dimensionSpans.push_back(pair<unsigned int, unsigned int>(start, end));
            outputDimensions.push_back((dimensionSpans[i].second - dimensionSpans[i].first) + 1);
        }

        TensorDescriptor destDescriptor(DataType::FP16, outputDimensions);
        Tensor destCpu(cpuPlacement, destDescriptor);

        layers.push_back(make_shared<Extract>(dimensionSpans));
        layers.push_back(make_shared<NetworkOutput>(gpuPlacement));

        LayerTestHelper::connectAndInitializeNetwork(layers);
        Tensor outputGpu = layers.back()->getFeatureOutput().value();

        // Network is runnable here
        layers[0]->forward(sourceCpu, false);
        stream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());
        destCpu.copyFromAsync(outputGpu, stream);
        stream.synchronize();

        vector<int> stridePerInputDimension;
        vector<int> stridePerOutputDimension;
        for (int d = 0; d < numDimensions; ++d) {
            stridePerInputDimension.push_back(1);
            stridePerOutputDimension.push_back(1);
        }
        for (int d = numDimensions - 2; d >= 0; --d) {
            stridePerInputDimension[d] = stridePerInputDimension[d + 1] * inputDimensions[d + 1];
            stridePerOutputDimension[d] = stridePerOutputDimension[d + 1] * outputDimensions[d + 1];
        }

        half *destMem = (half *)destCpu.getMemPtr();

        unsigned int destFlatIndex = 0;
        for (unsigned int sourceFlatIndex = 0; sourceFlatIndex < sourceDescriptor.getTotalNumElements(); ++sourceFlatIndex) {
            vector<unsigned long> sourceDimensionalIndex = sourceCpu.getDescriptor().getDimensionalIndex(sourceFlatIndex);
            if (!isWithinSpan(sourceDimensionalIndex, dimensionSpans))
                continue;
            ASSERT_EQ((float)destMem[destFlatIndex], (float)sourceMem[sourceFlatIndex]);
            destFlatIndex += 1;
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}
