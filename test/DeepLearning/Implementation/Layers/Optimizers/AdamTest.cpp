#include "Thor.h"
#include "gtest/gtest.h"

using namespace ThorImplementation;
using namespace std;

// Test the Adam constructor
TEST(AdamTest, Constructor) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, {128, 20});

    Tensor exampleInputTensor(gpuPlacement, descriptor);
    shared_ptr<NetworkInput> networkInput = make_shared<NetworkInput>(exampleInputTensor);
    shared_ptr<FullyConnected> fc0 = make_shared<FullyConnected>(10, true);
    shared_ptr<FullyConnected> fc1 = make_shared<FullyConnected>(10, true);
    networkInput->connectToNextLayer(fc0.get());
    fc0->connectToNextLayer(fc1.get());

    Adam adam(fc0, 0.1f, 0.9f, 0.999f, 1e-8f, fc0->getErrorInputs()[0], Optional<Tensor>::empty());
    EXPECT_EQ(adam.getAlpha(), 0.1f);
    EXPECT_EQ(adam.getBeta1(), 0.9f);
    EXPECT_EQ(adam.getBeta2(), 0.999f);
    EXPECT_EQ(adam.getEpsilon(), 1e-8f);
}
/*
// Test the Adam::initialize function
TEST(AdamTest, Initialize) {
    // Prepare inputs
    std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = std::make_shared<TrainableWeightsBiasesLayer>();
    Adam adam(trainableLayer, 0.1, 0.9, 0.999, 1e-8);
    // Call the function to be tested
    adam.initialize();
    // Check the results
    EXPECT_EQ(adam.getT(), 0.0);
}

// Test the Adam::computeWeightsUpdate function
TEST(AdamTest, ComputeWeightsUpdate) {
    // This test is complex and would require mocking the trainableLayer and the CUDA calls
    // This is beyond the scope of this example but should be included in a complete test suite
}

// Test the Adam::updateWeights function
TEST(AdamTest, UpdateWeights) {
    // This test is complex and would require mocking the CUDA calls
    // This is beyond the scope of this example but should be included in a complete test suite
}

// Test the Adam::setAlpha function
TEST(AdamTest, SetAlpha) {
    std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = std::make_shared<TrainableWeightsBiasesLayer>();
    Adam adam(trainableLayer, 0.1, 0.9, 0.999, 1e-8);
    adam.setAlpha(0.2);
    EXPECT_EQ(adam.getAlpha(), 0.2);
}

// Test the Adam::setBeta1 function
TEST(AdamTest, SetBeta1) {
    std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = std::make_shared<TrainableWeightsBiasesLayer>();
    Adam adam(trainableLayer, 0.1, 0.9, 0.999, 1e-8);
    adam.setBeta1(0.8);
    EXPECT_EQ(adam.getBeta1(), 0.8);
}

// Test the Adam::setBeta2 function
TEST(AdamTest, SetBeta2) {
    std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = std::make_shared<TrainableWeightsBiasesLayer>();
    Adam adam(trainableLayer, 0.1, 0.9, 0.999, 1e-8);
    adam.setBeta2(0.998);
    EXPECT_EQ(adam.getBeta2(), 0.998);
}

// Test the Adam::setEpsilon function
TEST(AdamTest, SetEpsilon) {
    std::shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = std::make_shared<TrainableWeightsBiasesLayer>();
    Adam adam(trainableLayer, 0.1, 0.9, 0.999, 1e-8);
    adam.setEpsilon(1e-7);
    EXPECT_EQ(adam.getEpsilon(), 1e-7);
}
 */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}