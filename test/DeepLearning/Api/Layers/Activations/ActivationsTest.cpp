#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Activations/Elu.h"
#include "DeepLearning/Api/Layers/Activations/Exponential.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Selu.h"
#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"
#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "DeepLearning/Api/Layers/Activations/SoftSign.h"
#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(Relu, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Relu::Builder reluBuilder;
    reluBuilder.network(network);
    reluBuilder.featureInput(featureInput);
    shared_ptr<Relu> relu = make_shared<Relu>(*dynamic_cast<Relu *>(reluBuilder.build().get()));

    ASSERT_TRUE(relu->isInitialized());

    Stream stream(0);
    json actual = relu->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "relu"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = relu->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = relu->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = relu->clone();
    Relu *clone = dynamic_cast<Relu *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(relu->getId(), clone->getId());
    ASSERT_GT(relu->getId(), 1u);

    ASSERT_TRUE(*relu == *clone);
    ASSERT_FALSE(*relu != *clone);
    ASSERT_FALSE(*relu > *clone);
    ASSERT_FALSE(*relu < *clone);
}

TEST(Tanh, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Tanh::Builder tanhBuilder;
    tanhBuilder.network(network);
    tanhBuilder.featureInput(featureInput);
    shared_ptr<Tanh> tanh = make_shared<Tanh>(*dynamic_cast<Tanh *>(tanhBuilder.build().get()));

    ASSERT_TRUE(tanh->isInitialized());

    Stream stream(0);
    json actual = tanh->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "tanh"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = tanh->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = tanh->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = tanh->clone();
    Tanh *clone = dynamic_cast<Tanh *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(tanh->getId(), clone->getId());
    ASSERT_GT(tanh->getId(), 1u);

    ASSERT_TRUE(*tanh == *clone);
    ASSERT_FALSE(*tanh != *clone);
    ASSERT_FALSE(*tanh > *clone);
    ASSERT_FALSE(*tanh < *clone);
}

TEST(Exponential, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Exponential::Builder exponentialBuilder;
    exponentialBuilder.network(network);
    exponentialBuilder.featureInput(featureInput);
    shared_ptr<Exponential> exponential = make_shared<Exponential>(*dynamic_cast<Exponential *>(exponentialBuilder.build().get()));

    ASSERT_TRUE(exponential->isInitialized());

    Stream stream(0);
    json actual = exponential->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "exponential"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = exponential->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = exponential->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = exponential->clone();
    Exponential *clone = dynamic_cast<Exponential *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(exponential->getId(), clone->getId());
    ASSERT_GT(exponential->getId(), 1u);

    ASSERT_TRUE(*exponential == *clone);
    ASSERT_FALSE(*exponential != *clone);
    ASSERT_FALSE(*exponential > *clone);
    ASSERT_FALSE(*exponential < *clone);
}

TEST(Gelu, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Gelu::Builder geluBuilder;
    geluBuilder.network(network);
    geluBuilder.featureInput(featureInput);
    shared_ptr<Gelu> gelu = make_shared<Gelu>(*dynamic_cast<Gelu *>(geluBuilder.build().get()));

    ASSERT_TRUE(gelu->isInitialized());

    Stream stream(0);
    json actual = gelu->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "gelu"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = gelu->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = gelu->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = gelu->clone();
    Gelu *clone = dynamic_cast<Gelu *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(gelu->getId(), clone->getId());
    ASSERT_GT(gelu->getId(), 1u);

    ASSERT_TRUE(*gelu == *clone);
    ASSERT_FALSE(*gelu != *clone);
    ASSERT_FALSE(*gelu > *clone);
    ASSERT_FALSE(*gelu < *clone);
}

TEST(HardSigmoid, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    HardSigmoid::Builder hardSigmoidBuilder;
    hardSigmoidBuilder.network(network);
    hardSigmoidBuilder.featureInput(featureInput);
    shared_ptr<HardSigmoid> hardSigmoid = make_shared<HardSigmoid>(*dynamic_cast<HardSigmoid *>(hardSigmoidBuilder.build().get()));

    ASSERT_TRUE(hardSigmoid->isInitialized());

    Stream stream(0);
    json actual = hardSigmoid->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "hard_sigmoid"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = hardSigmoid->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = hardSigmoid->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = hardSigmoid->clone();
    HardSigmoid *clone = dynamic_cast<HardSigmoid *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(hardSigmoid->getId(), clone->getId());
    ASSERT_GT(hardSigmoid->getId(), 1u);

    ASSERT_TRUE(*hardSigmoid == *clone);
    ASSERT_FALSE(*hardSigmoid != *clone);
    ASSERT_FALSE(*hardSigmoid > *clone);
    ASSERT_FALSE(*hardSigmoid < *clone);
}

TEST(Selu, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Selu::Builder seluBuilder;
    seluBuilder.network(network);
    seluBuilder.featureInput(featureInput);
    shared_ptr<Selu> selu = make_shared<Selu>(*dynamic_cast<Selu *>(seluBuilder.build().get()));

    ASSERT_TRUE(selu->isInitialized());

    Stream stream(0);
    json actual = selu->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "selu"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = selu->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = selu->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = selu->clone();
    Selu *clone = dynamic_cast<Selu *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(selu->getId(), clone->getId());
    ASSERT_GT(selu->getId(), 1u);

    ASSERT_TRUE(*selu == *clone);
    ASSERT_FALSE(*selu != *clone);
    ASSERT_FALSE(*selu > *clone);
    ASSERT_FALSE(*selu < *clone);
}

TEST(Sigmoid, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Sigmoid::Builder sigmoidBuilder;
    sigmoidBuilder.network(network);
    sigmoidBuilder.featureInput(featureInput);
    shared_ptr<Sigmoid> sigmoid = make_shared<Sigmoid>(*dynamic_cast<Sigmoid *>(sigmoidBuilder.build().get()));

    ASSERT_TRUE(sigmoid->isInitialized());

    Stream stream(0);
    json actual = sigmoid->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "sigmoid"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = sigmoid->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = sigmoid->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = sigmoid->clone();
    Sigmoid *clone = dynamic_cast<Sigmoid *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(sigmoid->getId(), clone->getId());
    ASSERT_GT(sigmoid->getId(), 1u);

    ASSERT_TRUE(*sigmoid == *clone);
    ASSERT_FALSE(*sigmoid != *clone);
    ASSERT_FALSE(*sigmoid > *clone);
    ASSERT_FALSE(*sigmoid < *clone);
}

TEST(Softmax, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Softmax::Builder softmaxBuilder;
    softmaxBuilder.network(network);
    softmaxBuilder.featureInput(featureInput);
    shared_ptr<Softmax> softmax = make_shared<Softmax>(*dynamic_cast<Softmax *>(softmaxBuilder.build().get()));

    ASSERT_TRUE(softmax->isInitialized());

    Stream stream(0);
    json actual = softmax->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "softmax"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = softmax->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = softmax->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = softmax->clone();
    Softmax *clone = dynamic_cast<Softmax *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(softmax->getId(), clone->getId());
    ASSERT_GT(softmax->getId(), 1u);

    ASSERT_TRUE(*softmax == *clone);
    ASSERT_FALSE(*softmax != *clone);
    ASSERT_FALSE(*softmax > *clone);
    ASSERT_FALSE(*softmax < *clone);
}

TEST(SoftPlus, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    SoftPlus::Builder softPlusBuilder;
    softPlusBuilder.network(network);
    softPlusBuilder.featureInput(featureInput);
    shared_ptr<SoftPlus> softPlus = make_shared<SoftPlus>(*dynamic_cast<SoftPlus *>(softPlusBuilder.build().get()));

    ASSERT_TRUE(softPlus->isInitialized());

    Stream stream(0);
    json actual = softPlus->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "soft_plus"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = softPlus->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = softPlus->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = softPlus->clone();
    SoftPlus *clone = dynamic_cast<SoftPlus *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(softPlus->getId(), clone->getId());
    ASSERT_GT(softPlus->getId(), 1u);

    ASSERT_TRUE(*softPlus == *clone);
    ASSERT_FALSE(*softPlus != *clone);
    ASSERT_FALSE(*softPlus > *clone);
    ASSERT_FALSE(*softPlus < *clone);
}

TEST(SoftSign, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    SoftSign::Builder softSignBuilder;
    softSignBuilder.network(network);
    softSignBuilder.featureInput(featureInput);
    shared_ptr<SoftSign> softSign = make_shared<SoftSign>(*dynamic_cast<SoftSign *>(softSignBuilder.build().get()));

    ASSERT_TRUE(softSign->isInitialized());

    Stream stream(0);
    json actual = softSign->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "soft_sign"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = softSign->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = softSign->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = softSign->clone();
    SoftSign *clone = dynamic_cast<SoftSign *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(softSign->getId(), clone->getId());
    ASSERT_GT(softSign->getId(), 1u);

    ASSERT_TRUE(*softSign == *clone);
    ASSERT_FALSE(*softSign != *clone);
    ASSERT_FALSE(*softSign > *clone);
    ASSERT_FALSE(*softSign < *clone);
}

TEST(Swish, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Swish::Builder swishBuilder;
    swishBuilder.network(network);
    swishBuilder.featureInput(featureInput);
    shared_ptr<Swish> swish = make_shared<Swish>(*dynamic_cast<Swish *>(swishBuilder.build().get()));

    ASSERT_TRUE(swish->isInitialized());

    Stream stream(0);
    json actual = swish->serialize("", stream);
    json expected = {{"version", "1.0.0"}, {"type", "swish"}};
    ASSERT_EQ(expected, actual);

    Optional<Tensor> actualInput = swish->getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = swish->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = swish->clone();
    Swish *clone = dynamic_cast<Swish *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(swish->getId(), clone->getId());
    ASSERT_GT(swish->getId(), 1u);

    ASSERT_TRUE(*swish == *clone);
    ASSERT_FALSE(*swish != *clone);
    ASSERT_FALSE(*swish > *clone);
    ASSERT_FALSE(*swish < *clone);
}
