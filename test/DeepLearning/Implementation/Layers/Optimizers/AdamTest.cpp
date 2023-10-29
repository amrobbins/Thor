#include "Thor.h"
#include "gtest/gtest.h"

using namespace ThorImplementation;
using namespace std;

#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

shared_ptr<TrainableWeightsBiasesLayer> constructTrainableLayer(bool hasBias = true) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, {(uint64_t)(rand() % 80) + 1, (uint64_t)(rand() % 150) + 1});

    Tensor exampleInputTensor(gpuPlacement, descriptor);
    shared_ptr<NetworkInput> networkInput = make_shared<NetworkInput>(exampleInputTensor);
    shared_ptr<FullyConnected> fc0 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, hasBias);
    shared_ptr<FullyConnected> fc1 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, true);
    networkInput->connectToNextLayer(fc0.get());
    fc0->connectToNextLayer(fc1.get());
    return fc0;
}

shared_ptr<TrainableWeightsBiasesLayer> constructTrainableLayerWithMultipleConnections(uint32_t numConnections, bool hasBias = true) {
    assert(numConnections > 0);

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, {(uint64_t)(rand() % 80) + 1, (uint64_t)(rand() % 150) + 1});

    Tensor exampleInputTensor(gpuPlacement, descriptor);
    shared_ptr<NetworkInput> networkInput = make_shared<NetworkInput>(exampleInputTensor);
    shared_ptr<FullyConnected> fc0 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, hasBias);
    networkInput->connectToNextLayer(fc0.get());

    for (uint32_t connection = 0; connection < numConnections; ++connection) {
        shared_ptr<NetworkInput> networkInputOther = make_shared<NetworkInput>(exampleInputTensor);
        networkInputOther->connectToNextLayer(fc0.get());
        shared_ptr<FullyConnected> fcOther = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, true);
        fc0->connectToNextLayer(fcOther.get());
    }
    return fc0;
}

template <typename T>
void computeBiasesGradientCpu(vector<T> &errorIn, vector<T> &biasesGradient, uint32_t batchSize, uint32_t exampleSize, bool accumulate) {
    if (accumulate)
        ASSERT_EQ(biasesGradient.size(), exampleSize);
    else
        biasesGradient = vector<T>(exampleSize, 0.0f);

    for (uint32_t i = 0; i < batchSize; ++i) {
        for (uint32_t j = 0; j < exampleSize; ++j) {
            T gradientComponent = errorIn[i * exampleSize + j];
            if (accumulate || i != 0)
                biasesGradient[j] += gradientComponent;
            else
                biasesGradient[j] = gradientComponent;
        }
    }
}

// Test the Adam constructor
TEST(AdamTest, Constructor) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
    EXPECT_EQ(adam.getAlpha(), 0.1f);
    EXPECT_EQ(adam.getBeta1(), 0.9f);
    EXPECT_EQ(adam.getBeta2(), 0.999f);
    EXPECT_EQ(adam.getEpsilon(), 1e-8f);
}

// Test the Adam::initialize function
TEST(AdamTest, Initialize) {
    // Prepare inputs
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
    // Call the function to be tested
    adam.initialize();
    // Check the results
    EXPECT_EQ(adam.getT(), 0.0f);
    // Also cover setT
    adam.setT(5.0f);
    EXPECT_EQ(adam.getT(), 5.0f);
}

// Test the Adam::setAlpha function
TEST(AdamTest, SetAlpha) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
    adam.setAlpha(0.2f);
    EXPECT_EQ(adam.getAlpha(), 0.2f);
}

// Test the Adam::setBeta1 function
TEST(AdamTest, SetBeta1) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
    adam.setBeta1(0.8f);
    EXPECT_EQ(adam.getBeta1(), 0.8f);
}

// Test the Adam::setBeta2 function
TEST(AdamTest, SetBeta2) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
    adam.setBeta2(0.998f);
    EXPECT_EQ(adam.getBeta2(), 0.998f);
}

// Test the Adam::setEpsilon function
TEST(AdamTest, SetEpsilon) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
    adam.setEpsilon(1e-7f);
    EXPECT_EQ(adam.getEpsilon(), 1e-7f);
}

TEST(AdamTest, updateHyperParameters) {
    // FIXME: Implement
}
TEST(AdamTest, getAllHyperParameters) {
    // FIXME: Implement
}

// Test the Adam::computeWeightsUpdate and Adam::updateWeights functions
TEST(AdamTest, UpdateWeightsFP16) {
    srand(time(nullptr));

    half alpha = 0.02f;
    half beta1 = 0.82f;
    half beta2 = 0.933f;
    half epsilon = 1e-5f;
    float t = 0.0f;
    bool hasBias = rand() % 4 ? true : false;
    shared_ptr<FullyConnected> fc = dynamic_pointer_cast<FullyConnected>(constructTrainableLayer(hasBias));
    ASSERT_NE(fc, nullptr);
    shared_ptr<Adam> adam = make_shared<Adam>(fc, alpha, beta1, beta2, epsilon, fc->getErrorInputs()[0], Optional<Tensor>::empty());
    fc->compile();
    fc->setOptimizer(dynamic_pointer_cast<Optimizer>(adam));
    adam->initialize();

    uint32_t batchSize = fc->getErrorInputs()[0].get().getDimensions()[0];
    uint32_t exampleSize = fc->getErrorInputs()[0].get().getTotalNumElements() / batchSize;

    Stream dataStream = fc->getStreams()[0];
    Stream gradientUpdateStream = fc->getOptimizer().get()->getGradientUpdateStream();

    vector<half> weightsGradientCpu;
    vector<half> biasesGradientCpu;
    vector<half> mCpu;
    vector<half> vCpu;
    vector<half> weightsUpdateCpu;
    vector<half> mBiasesCpu;
    vector<half> vBiasesCpu;
    vector<half> biasesUpdateCpu;

    Tensor weightsGradientGpu;
    Optional<Tensor> maybeBiasesGradientGpu;

    weightsGradientGpu = adam->getWeightsGradient();
    maybeBiasesGradientGpu = adam->getBiasesGradient();
    ASSERT_EQ(maybeBiasesGradientGpu.isPresent(), hasBias);
    mCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    vCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    if (maybeBiasesGradientGpu.isPresent()) {
        mBiasesCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
        vBiasesCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
    }

    uint32_t numWeights = weightsGradientGpu.getTotalNumElements();
    uint32_t numBiases = 0;
    if (maybeBiasesGradientGpu.isPresent())
        numBiases = maybeBiasesGradientGpu.get().getTotalNumElements();

    weightsGradientCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    weightsUpdateCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    if (maybeBiasesGradientGpu.isPresent()) {
        biasesGradientCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
        biasesUpdateCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
    }

    t = 1;

    for (uint32_t i = 0; i < 3; ++i) {
        // populate featureInput and errorInput so that gradient can be computed
        vector<half> featureInputVector;
        Tensor featureInputGpu = fc->getFeatureInputs()[0];
        for (uint32_t j = 0; j < featureInputGpu.getTotalNumElements(); ++j) {
            featureInputVector.push_back(2.0f / ((rand() % 100) + 1.0f));
        }
        featureInputGpu.setValues(featureInputVector, dataStream);

        vector<half> errorInputVector;
        Tensor errorInputGpu = fc->getErrorInputs()[0];
        for (uint32_t i = 0; i < errorInputGpu.getTotalNumElements(); ++i) {
            errorInputVector.push_back(2.0f / ((rand() % 100) + 1.0f));
        }
        errorInputGpu.setValues(errorInputVector, dataStream);

        bool accumulateValues = i == 0 ? false : true;

        adam->computeWeightsUpdate(featureInputGpu, errorInputGpu, accumulateValues);

        // Cpu computation
        // compute gradient
        matrixMultiplyCpuHalf(featureInputVector.data(),
                              errorInputVector.data(),
                              weightsGradientCpu.data(),
                              featureInputGpu.getDimensions()[0],
                              featureInputGpu.getDimensions()[1],
                              errorInputGpu.getDimensions()[0],
                              errorInputGpu.getDimensions()[1],
                              featureInputGpu.getDimensions()[1],
                              errorInputGpu.getDimensions()[1],
                              weightsGradientGpu.getDimensions()[1],
                              true,
                              false,
                              accumulateValues,
                              false);

        // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
        vector<half> weightsGradientGpuVector;
        weightsGradientGpu.loadValuesIntoVector(weightsGradientGpuVector, gradientUpdateStream);
        for (uint32_t j; j < numWeights; ++j) {
            // printf("%f %f\n", (float)weightsGradientCpu[j], (float)weightsGradientGpuVector[j]);
            ASSERT_LT(abs((float)(weightsGradientCpu[j] - weightsGradientGpuVector[j])),
                      max(0.00001, abs((double)weightsGradientCpu[j] * .01)));
        }
        weightsGradientGpu.loadValuesIntoVector(weightsGradientCpu, gradientUpdateStream);

        if (hasBias) {
            // compute the biases gradient
            computeBiasesGradientCpu(errorInputVector, biasesGradientCpu, batchSize, exampleSize, accumulateValues);

            // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
            vector<half> biasesGradientGpuVector;
            maybeBiasesGradientGpu.get().loadValuesIntoVector(biasesGradientGpuVector, gradientUpdateStream);
            for (uint32_t j; j < numBiases; ++j) {
                // printf("%f %f\n", (float)biasesGradientCpu[j], (float)biasesGradientGpuVector[j]);
                ASSERT_LT(abs((float)(biasesGradientCpu[j] - biasesGradientGpuVector[j])),
                          max(0.00001, abs((double)biasesGradientCpu[j] * .01)));
            }
            maybeBiasesGradientGpu.get().loadValuesIntoVector(biasesGradientCpu, gradientUpdateStream);
        }

        // update m, v
        for (uint32_t j = 0; j < numWeights; ++j) {
            // if (j == 0)
            //     printf("m calc %f = %f * %f + (1.0f - %f) * %f\n", (float)(beta1 * mCpu[j] + ((half)1.0f - beta1) *
            //     weightsGradientCpu[j]), (float)beta1, (float)mCpu[j], (float)beta1, (float)weightsGradientCpu[j]);
            mCpu[j] = beta1 * mCpu[j] + ((half)1.0f - beta1) * weightsGradientCpu[j];
            // if (j == 0)
            //     printf("v calc = %f = %f * %f + (1.0 - %f) * (%f * %f)\n", (float)(beta2 * vCpu[j] + ((half)1.0f - beta2) *
            //     (weightsGradientCpu[j] * weightsGradientCpu[j])),
            //            (float)beta2, (float)vCpu[j], (float)((half)1.0f - beta2), (float)weightsGradientCpu[j],
            //            (float)weightsGradientCpu[j]);
            vCpu[j] = beta2 * vCpu[j] + ((half)1.0f - beta2) * (weightsGradientCpu[j] * weightsGradientCpu[j]);
        }
        for (uint32_t j = 0; j < numBiases; ++j) {
            mBiasesCpu[j] = beta1 * mBiasesCpu[j] + ((half)1.0f - beta1) * biasesGradientCpu[j];
            vBiasesCpu[j] = beta2 * vBiasesCpu[j] + ((half)1.0f - beta2) * (biasesGradientCpu[j] * biasesGradientCpu[j]);
        }
    }

    gradientUpdateStream.synchronize();
    double thresh;

    // Ensure t is 1.0f as expected -> 3 computeWeightsUpdate(...) occurred but only the first had accumulateValues = 0
    ASSERT_EQ(adam->getT(), 1.0f);

    // Get m ensure it is correct, then load actual values to maintain test accuracy downstream
    vector<half> mGpuVector;
    adam->getM(mGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        thresh = max(0.00001, abs((double)mCpu[j] * .01));
        if (!(abs((double)(mCpu[j] - mGpuVector[j])) < thresh))
            printf("mCpu[%d] %f mGpu[%d] %f\n", j, (float)mCpu[j], j, (float)mGpuVector[j]);
        ASSERT_LT(abs((double)(mCpu[j] - mGpuVector[j])), thresh);
    }
    mCpu = mGpuVector;

    // Get v ensure it is correct, then load actual values to maintain test accuracy downstream
    vector<half> vGpuVector;
    adam->getV(vGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        thresh = max(0.00001, abs((double)vCpu[j] * .01));
        // printf("v cpu %f gpu %f\n", (float)vCpu[j], (float)vGpuVector[j]);
        ASSERT_LT(abs((double)(vCpu[j] - vGpuVector[j])), thresh);
    }
    vCpu = vGpuVector;

    if (hasBias) {
        // Get mBiases ensure it is correct, then load actual values to maintain test accuracy downstream
        vector<half> mBiasesGpuVector;
        adam->getMBias(mBiasesGpuVector);
        for (uint32_t j = 0; j < numBiases; ++j) {
            // printf("%f %f\n", (float)mBiasesCpu[j], (float)mBiasesGpuVector[j]);
            thresh = max(0.00001, abs((double)mBiasesCpu[j] * .01));
            ASSERT_LT(abs((double)(mBiasesCpu[j] - mBiasesGpuVector[j])), thresh);
        }
        mBiasesCpu = mBiasesGpuVector;

        // Get vBiases ensure it is correct, then load actual values to maintain test accuracy downstream
        vector<half> vBiasesGpuVector;
        adam->getVBias(vBiasesGpuVector);
        for (uint32_t j = 0; j < numBiases; ++j) {
            thresh = max(0.00001, abs((double)vBiasesCpu[j] * .01));
            ASSERT_LT(abs((double)(vBiasesCpu[j] - vBiasesGpuVector[j])), thresh);
        }
        vBiasesCpu = vBiasesGpuVector;
    }

    // Compute weights and biases update values
    half alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
    // printf("alphaT = %f = %f * %f / %f, beta1 %f beta2 %f t %f\n", (float)alphaT, (float)alpha, sqrtf(1.0f - powf(beta2, t)), (1.0f -
    // powf(beta1, t)), (float)beta1, (float)beta2, t);
    for (uint32_t j = 0; j < numWeights; ++j) {
        weightsUpdateCpu[j] = (-alphaT * mCpu[j]) / ((half)sqrtf(vCpu[j]) + epsilon);
        // if (j == 0)
        //     printf("CPU weightUpdate = %f / %f = %f      alphaT %f, m %f\n", float(-alphaT * mCpu[j]), float(((half)sqrtf(vCpu[j]) +
        //     epsilon)), (float)weightsUpdateCpu[j], (float)-alphaT, float(mCpu[j]));
    }
    for (uint32_t j = 0; j < numBiases; ++j)
        biasesUpdateCpu[j] = (-alphaT * mBiasesCpu[j]) / ((half)sqrtf(vBiasesCpu[j]) + epsilon);

    // Get weightsUpdate ensure it is correct
    vector<half> weightsUpdateGpuVector;
    adam->getWeightsUpdate(weightsUpdateGpuVector);
    vector<half> biasesUpdateGpuVector;
    adam->getBiasesUpdate(biasesUpdateGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        // printf("[%d] cpu: %f gpu %f      -alphaT %f mCpu %f sqrt(vCpu) %f epsilon %f\n", j, (float)weightsUpdateCpu[j],
        // (float)weightsUpdateGpuVector[j], (float)-alphaT, (float)mCpu[j], sqrtf(vCpu[j]), (float)epsilon);
        ASSERT_LT(abs((double)(weightsUpdateCpu[j] - weightsUpdateGpuVector[j])), 0.0002);
    }
    for (uint32_t j = 0; j < numBiases; ++j) {
        ASSERT_LT(abs((double)(biasesUpdateCpu[j] - biasesUpdateGpuVector[j])), 0.0002);
    }

    // set the weights and then call updateWeights.
    Tensor weightsGpu = fc->getWeights();
    Optional<Tensor> biasesGpu = fc->getBiases();
    vector<half> weightsCpu(weightsGpu.getTotalNumElements(), 0.0f);
    vector<half> biasesCpu;
    for (uint32_t i = 0; i < weightsGpu.getTotalNumElements(); ++i)
        weightsCpu[i] = (rand() % 50) / ((rand() % 50) + 1);
    weightsGpu.setValues(weightsCpu, gradientUpdateStream);
    if (hasBias) {
        biasesCpu = vector<half>(biasesGpu.get().getTotalNumElements(), 0.0f);
        for (uint32_t i = 0; i < biasesGpu.get().getTotalNumElements(); ++i)
            biasesCpu[i] = (rand() % 50) / ((rand() % 50) + 1);
        biasesGpu.get().setValues(biasesCpu, gradientUpdateStream);
    }
    // batchSize is only used for math, it doesn't need to match the tensor dimension in this test.
    adam->updateWeights(weightsGpu, biasesGpu, batchSize);
    gradientUpdateStream.synchronize();

    // Check that the weights have been updated to the proper values
    vector<half> weightsGpuVector;
    fc->getWeights().loadValuesIntoVector(weightsGpuVector, gradientUpdateStream);
    for (uint32_t j = 0; j < numWeights; ++j) {
        half updatedWeight = weightsCpu[j] + (half)((float)weightsUpdateCpu[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
        // printf("%f %f\n", (float)updatedWeight, (float)weightsGpuVector[j]);
        thresh = max(0.00001, abs((double)updatedWeight * .001));
        ASSERT_LT(abs((double)(updatedWeight - weightsGpuVector[j])), thresh);
    }
}

// Test 5 batches to ensure t is updated properly
TEST(AdamTest, FullyConnectedBackwardWithAdam5BatchesFP16) {
    srand(time(nullptr));

    half alpha = 0.02f;
    half beta1 = 0.82f;
    half beta2 = 0.933f;
    half epsilon = 1e-5f;
    bool hasBias = rand() % 4 ? true : false;
    shared_ptr<FullyConnected> fc = dynamic_pointer_cast<FullyConnected>(constructTrainableLayer(hasBias));
    ASSERT_NE(fc, nullptr);
    shared_ptr<Adam> adam = make_shared<Adam>(fc, alpha, beta1, beta2, epsilon, fc->getErrorInputs()[0], Optional<Tensor>::empty());
    fc->compile();
    fc->setOptimizer(dynamic_pointer_cast<Optimizer>(adam));
    adam->initialize();

    uint32_t batchSize = fc->getErrorInputs()[0].get().getDimensions()[0];
    uint32_t exampleSize = fc->getErrorInputs()[0].get().getTotalNumElements() / batchSize;

    Stream dataStream = fc->getStreams()[0];
    Stream gradientUpdateStream = fc->getOptimizer().get()->getGradientUpdateStream();

    vector<half> weightsGradientCpu;
    vector<half> biasesGradientCpu;
    vector<half> mCpu;
    vector<half> vCpu;
    vector<half> weightsUpdateCpu;
    vector<half> mBiasesCpu;
    vector<half> vBiasesCpu;
    vector<half> biasesUpdateCpu;

    Tensor weightsGradientGpu;
    Optional<Tensor> maybeBiasesGradientGpu;

    weightsGradientGpu = adam->getWeightsGradient();
    maybeBiasesGradientGpu = adam->getBiasesGradient();
    ASSERT_EQ(maybeBiasesGradientGpu.isPresent(), hasBias);
    mCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    vCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    if (maybeBiasesGradientGpu.isPresent()) {
        mBiasesCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
        vBiasesCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
    }

    uint32_t numWeights = weightsGradientGpu.getTotalNumElements();
    uint32_t numBiases = 0;
    if (maybeBiasesGradientGpu.isPresent())
        numBiases = maybeBiasesGradientGpu.get().getTotalNumElements();

    weightsGradientCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    weightsUpdateCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    if (maybeBiasesGradientGpu.isPresent()) {
        biasesGradientCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
        biasesUpdateCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
    }

    float t = 0.0f;
    for (uint32_t batch = 0; batch < 5; ++batch) {
        t += 1.0f;
        for (uint32_t i = 0; i < 3; ++i) {
            // populate featureInput and errorInput so that gradient can be computed
            vector<half> featureInputVector;
            Tensor featureInputGpu = fc->getFeatureInputs()[0];
            for (uint32_t j = 0; j < featureInputGpu.getTotalNumElements(); ++j) {
                featureInputVector.push_back(2.0f / ((rand() % 100) + 1.0f));
            }
            featureInputGpu.setValues(featureInputVector, dataStream);

            vector<half> errorInputVector;
            Tensor errorInputGpu = fc->getErrorInputs()[0];
            for (uint32_t i = 0; i < errorInputGpu.getTotalNumElements(); ++i) {
                errorInputVector.push_back(2.0f / ((rand() % 100) + 1.0f));
            }
            errorInputGpu.setValues(errorInputVector, dataStream);

            bool accumulateValues = i == 0 ? false : true;

            adam->computeWeightsUpdate(featureInputGpu, errorInputGpu, accumulateValues);

            // Cpu computation
            // compute gradient
            matrixMultiplyCpuHalf(featureInputVector.data(),
                                  errorInputVector.data(),
                                  weightsGradientCpu.data(),
                                  featureInputGpu.getDimensions()[0],
                                  featureInputGpu.getDimensions()[1],
                                  errorInputGpu.getDimensions()[0],
                                  errorInputGpu.getDimensions()[1],
                                  featureInputGpu.getDimensions()[1],
                                  errorInputGpu.getDimensions()[1],
                                  weightsGradientGpu.getDimensions()[1],
                                  true,
                                  false,
                                  accumulateValues,
                                  false);

            // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
            vector<half> weightsGradientGpuVector;
            weightsGradientGpu.loadValuesIntoVector(weightsGradientGpuVector, gradientUpdateStream);
            for (uint32_t j; j < numWeights; ++j) {
                // printf("%f %f\n", (float)weightsGradientCpu[j], (float)weightsGradientGpuVector[j]);
                ASSERT_LT(abs((float)(weightsGradientCpu[j] - weightsGradientGpuVector[j])),
                          max(0.00001, abs((double)weightsGradientCpu[j] * .01)));
            }
            weightsGradientGpu.loadValuesIntoVector(weightsGradientCpu, gradientUpdateStream);

            if (hasBias) {
                // compute the biases gradient
                computeBiasesGradientCpu(errorInputVector, biasesGradientCpu, batchSize, exampleSize, accumulateValues);

                // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
                vector<half> biasesGradientGpuVector;
                maybeBiasesGradientGpu.get().loadValuesIntoVector(biasesGradientGpuVector, gradientUpdateStream);
                for (uint32_t j; j < numBiases; ++j) {
                    // printf("%f %f\n", (float)biasesGradientCpu[j], (float)biasesGradientGpuVector[j]);
                    ASSERT_LT(abs((float)(biasesGradientCpu[j] - biasesGradientGpuVector[j])),
                              max(0.00001, abs((double)biasesGradientCpu[j] * .01)));
                }
                maybeBiasesGradientGpu.get().loadValuesIntoVector(biasesGradientCpu, gradientUpdateStream);
            }

            // update m, v
            for (uint32_t j = 0; j < numWeights; ++j) {
                // if (j == 0)
                //     printf("m calc %f = %f * %f + (1.0f - %f) * %f\n", (float)(beta1 * mCpu[j] + ((half)1.0f - beta1) *
                //     weightsGradientCpu[j]), (float)beta1, (float)mCpu[j], (float)beta1, (float)weightsGradientCpu[j]);
                mCpu[j] = beta1 * mCpu[j] + ((half)1.0f - beta1) * weightsGradientCpu[j];
                // if (j == 0)
                //     printf("v calc = %f = %f * %f + (1.0 - %f) * (%f * %f)\n", (float)(beta2 * vCpu[j] + ((half)1.0f - beta2) *
                //     (weightsGradientCpu[j] * weightsGradientCpu[j])),
                //            (float)beta2, (float)vCpu[j], (float)((half)1.0f - beta2), (float)weightsGradientCpu[j],
                //            (float)weightsGradientCpu[j]);
                vCpu[j] = beta2 * vCpu[j] + ((half)1.0f - beta2) * (weightsGradientCpu[j] * weightsGradientCpu[j]);
            }
            for (uint32_t j = 0; j < numBiases; ++j) {
                mBiasesCpu[j] = beta1 * mBiasesCpu[j] + ((half)1.0f - beta1) * biasesGradientCpu[j];
                vBiasesCpu[j] = beta2 * vBiasesCpu[j] + ((half)1.0f - beta2) * (biasesGradientCpu[j] * biasesGradientCpu[j]);
            }
        }
    }

    gradientUpdateStream.synchronize();
    double thresh;

    // Ensure t is 1.0f as expected -> 3 computeWeightsUpdate(...) occurred but only the first had accumulateValues = 0
    ASSERT_EQ(adam->getT(), 5.0f);

    // Get m ensure it is correct, then load actual values to maintain test accuracy downstream
    vector<half> mGpuVector;
    adam->getM(mGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        thresh = max(0.00001, abs((double)mCpu[j] * .01));
        if (!(abs((double)(mCpu[j] - mGpuVector[j])) < thresh))
            printf("mCpu[%d] %f mGpu[%d] %f\n", j, (float)mCpu[j], j, (float)mGpuVector[j]);
        ASSERT_LT(abs((double)(mCpu[j] - mGpuVector[j])), thresh);
    }
    mCpu = mGpuVector;

    // Get v ensure it is correct, then load actual values to maintain test accuracy downstream
    vector<half> vGpuVector;
    adam->getV(vGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        thresh = max(0.00001, abs((double)vCpu[j] * .01));
        // printf("v cpu %f gpu %f\n", (float)vCpu[j], (float)vGpuVector[j]);
        ASSERT_LT(abs((double)(vCpu[j] - vGpuVector[j])), thresh);
    }
    vCpu = vGpuVector;

    if (hasBias) {
        // Get mBiases ensure it is correct, then load actual values to maintain test accuracy downstream
        vector<half> mBiasesGpuVector;
        adam->getMBias(mBiasesGpuVector);
        for (uint32_t j = 0; j < numBiases; ++j) {
            // printf("%f %f\n", (float)mBiasesCpu[j], (float)mBiasesGpuVector[j]);
            thresh = max(0.00001, abs((double)mBiasesCpu[j] * .01));
            ASSERT_LT(abs((double)(mBiasesCpu[j] - mBiasesGpuVector[j])), thresh);
        }
        mBiasesCpu = mBiasesGpuVector;

        // Get vBiases ensure it is correct, then load actual values to maintain test accuracy downstream
        vector<half> vBiasesGpuVector;
        adam->getVBias(vBiasesGpuVector);
        for (uint32_t j = 0; j < numBiases; ++j) {
            thresh = max(0.00001, abs((double)vBiasesCpu[j] * .01));
            ASSERT_LT(abs((double)(vBiasesCpu[j] - vBiasesGpuVector[j])), thresh);
        }
        vBiasesCpu = vBiasesGpuVector;
    }

    // Compute weights and biases update values
    half alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
    for (uint32_t j = 0; j < numWeights; ++j) {
        weightsUpdateCpu[j] = (-alphaT * mCpu[j]) / ((half)sqrtf(vCpu[j]) + epsilon);
    }
    for (uint32_t j = 0; j < numBiases; ++j)
        biasesUpdateCpu[j] = (-alphaT * mBiasesCpu[j]) / ((half)sqrtf(vBiasesCpu[j]) + epsilon);

    // Get weightsUpdate ensure it is correct
    vector<half> weightsUpdateGpuVector;
    adam->getWeightsUpdate(weightsUpdateGpuVector);
    vector<half> biasesUpdateGpuVector;
    adam->getBiasesUpdate(biasesUpdateGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        // printf("[%d] cpu: %f gpu %f      -alphaT %f mCpu %f sqrt(vCpu) %f epsilon %f\n", j, (float)weightsUpdateCpu[j],
        // (float)weightsUpdateGpuVector[j], (float)-alphaT, (float)mCpu[j], sqrtf(vCpu[j]), (float)epsilon);
        ASSERT_LT(abs((double)(weightsUpdateCpu[j] - weightsUpdateGpuVector[j])), 0.0002);
    }
    for (uint32_t j = 0; j < numBiases; ++j) {
        ASSERT_LT(abs((double)(biasesUpdateCpu[j] - biasesUpdateGpuVector[j])), 0.0002);
    }

    // set the weights and then call updateWeights.
    Tensor weightsGpu = fc->getWeights();
    Optional<Tensor> biasesGpu = fc->getBiases();
    vector<half> weightsCpu(weightsGpu.getTotalNumElements(), 0.0f);
    vector<half> biasesCpu;
    for (uint32_t i = 0; i < weightsGpu.getTotalNumElements(); ++i)
        weightsCpu[i] = (rand() % 50) / ((rand() % 50) + 1);
    weightsGpu.setValues(weightsCpu, gradientUpdateStream);
    if (hasBias) {
        biasesCpu = vector<half>(biasesGpu.get().getTotalNumElements(), 0.0f);
        for (uint32_t i = 0; i < biasesGpu.get().getTotalNumElements(); ++i)
            biasesCpu[i] = (rand() % 50) / ((rand() % 50) + 1);
        biasesGpu.get().setValues(biasesCpu, gradientUpdateStream);
    }
    // batchSize is only used for math, it doesn't need to match the tensor dimension in this test.
    adam->updateWeights(weightsGpu, biasesGpu, batchSize);
    gradientUpdateStream.synchronize();

    // Check that the weights have been updated to the proper values
    vector<half> weightsGpuVector;
    fc->getWeights().loadValuesIntoVector(weightsGpuVector, gradientUpdateStream);
    for (uint32_t j = 0; j < numWeights; ++j) {
        half updatedWeight = weightsCpu[j] + (half)((float)weightsUpdateCpu[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
        // printf("%f %f\n", (float)updatedWeight, (float)weightsGpuVector[j]);
        thresh = max(0.00001, abs((double)updatedWeight * .001));
        ASSERT_LT(abs((double)(updatedWeight - weightsGpuVector[j])), thresh);
    }
}

/*
// 1. Fill feature in random
// 2. fill errorIn random
// 3. fill weights in random
// 4. call forward
// 5. Verify featureOutput
// 6. call backward
// 7. Check weights values
// 8. call forward
// 9. Verify featureOutput
// 10. fill errorIn random
// 11. call backward
// 12. Check weights values
// 13. Call forward for a third time, this time in inference mode
// 14. Verify featureOutput
TEST(AdamTest, FullForwardBackewordFunctionalityFP16) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    uint32_t batchSize = (rand() % 300) + 1;
    uint32_t numInputFeatures = (rand() % 300) + 1;
    uint32_t numOutputFeatures = (rand() % 300) + 1;
    bool hasBias = rand() % 2;

    half alpha = 0.02f;
    half beta1 = 0.82f;
    half beta2 = 0.933f;
    half epsilon = 1e-5f;
    shared_ptr<FullyConnected> fc = dynamic_pointer_cast<FullyConnected>(constructTrainableLayer(hasBias));
    ASSERT_NE(fc, nullptr);
    shared_ptr<Adam> adam = make_shared<Adam>(fc, alpha, beta1, beta2, epsilon, fc->getErrorInputs()[0], Optional<Tensor>::empty());

    Tensor networkFeatureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));

    float initialLearningRate = 10000.0 / (1 + rand() % 33);
    float decay = ((rand() % 5) / 10.0f);
    float momentum = (1.0f + (rand() % 100)) / 100.0f;
    bool useNesterovMomentum = true;
    uint64_t epoch = rand() % 10;

    vector<shared_ptr<Layer>> layers;

    layers.push_back(
        make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, networkFeatureIn.getDescriptor().getDimensions()));
    layers.push_back(make_shared<NoOpLayer>());
    shared_ptr<FullyConnected> fullyConnectedLayer = make_shared<FullyConnected>(numOutputFeatures, hasBias);
    layers.push_back(fullyConnectedLayer);
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

    Stream dataStream = layers.front()->getStream();

    LayerTestHelper::connectAndInitializeNetwork(layers);

    Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs());
    Tensor featureOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureOutputs());
    Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
    Tensor errorOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
    Tensor weights = fullyConnectedLayer->getWeights();
    featureInput.fillRandom(-2.0, 2.0, dataStream);
    errorInput.fillRandom(-2.0, 2.0, dataStream);
    weights.fillRandom(-2.0, 2.0, dataStream);
    dataStream.synchronize();
    shared_ptr<Adam> adma =
        make_shared<Adam>(fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, errorInput, errorOutput);
    fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(adam));
    Tensor weightsGradient = adam->getWeightsGradient();

    ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
    ASSERT_EQ(sgd->getDecay(), decay);
    ASSERT_EQ(sgd->getMomentum(), momentum);
    ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesterovMomentum);

    Tensor featureInput_h = featureInput.clone(cpuPlacement);
    Tensor featureOutput_h = featureOutput.clone(cpuPlacement);
    Tensor featureOutputGpu_h = featureOutput_h.clone();
    Tensor errorInput_h = errorInput.clone(cpuPlacement);
    Tensor errorOutput_h = errorOutput.clone(cpuPlacement);
    Tensor errorOutputGpu_h = errorOutput_h.clone();
    Tensor weights_h = weights.clone(cpuPlacement);
    Tensor projectedWeights_h = weights_h.clone();
    Tensor projectedWeightsGpu_h = weights_h.clone();
    Tensor weightsGpu_h = weights_h.clone();
    Tensor weightsGradient_h = weightsGradient.clone(cpuPlacement);
    Tensor weightsGradientGpu_h = weightsGradient_h.clone();
    half *featureInMem_h = featureInput_h.getMemPtr<half>();
    half *featureOutMem_h = featureOutput_h.getMemPtr<half>();
    half *featureOutGpuMem_h = featureOutputGpu_h.getMemPtr<half>();
    half *errorInMem_h = errorInput_h.getMemPtr<half>();
    half *errorOutMem_h = errorOutput_h.getMemPtr<half>();
    half *errorOutGpuMem_h = errorOutputGpu_h.getMemPtr<half>();
    half *weightsMem_h = weights_h.getMemPtr<half>();
    half *weightsGpuMem_h = weightsGpu_h.getMemPtr<half>();
    half *projectedWeightsMem_h = projectedWeights_h.getMemPtr<half>();
    half *projectedWeightsGpuMem_h = projectedWeightsGpu_h.getMemPtr<half>();
    half *weightsGradientMem_h = weightsGradient_h.getMemPtr<half>();
    half *weightsGradientGpuMem_h = weightsGradientGpu_h.getMemPtr<half>();
    featureInput_h.copyFromAsync(featureInput, dataStream);
    errorInput_h.copyFromAsync(errorInput, dataStream);
    weights_h.copyFromAsync(weights, dataStream);

    Optional<Tensor> biases = fullyConnectedLayer->getBiases();
    Tensor biases_h;
    Tensor biasesGpu_h;
    half *biasesMem_h;
    half *biasesGpuMem_h;
    Optional<Tensor> biasesGradient = sgd->getBiasesGradient();
    Tensor biasesGradient_h;
    Tensor biasesGradientGpu_h;
    half *biasesGradientMem_h;
    half *biasesGradientGpuMem_h;
    Tensor projectedBiases_h;
    Tensor projectedBiasesGpu_h;
    half *projectedBiasesMem_h;
    half *projectedBiasesGpuMem_h;
    if (hasBias) {
        ASSERT_TRUE(biases.isPresent());
        ASSERT_TRUE(biasesGradient.isPresent());
        biases.get().fillRandom(-2.0, 2.0, dataStream);
        biases_h = biases.get().clone(cpuPlacement);
        biases_h.copyFromAsync(biases, dataStream);
        biasesGpu_h = biases_h.clone();
        biasesMem_h = biases_h.getMemPtr<half>();
        biasesGpuMem_h = biasesGpu_h.getMemPtr<half>();
        biasesGradient_h = biasesGradient.get().clone(cpuPlacement);
        biasesGradientGpu_h = biasesGradient_h.clone();
        biasesGradientMem_h = biasesGradient_h.getMemPtr<half>();
        biasesGradientGpuMem_h = biasesGradientGpu_h.getMemPtr<half>();
        projectedBiases_h = biases_h.clone();
        projectedBiasesGpu_h = biases_h.clone();
        projectedBiasesMem_h = projectedBiases_h.getMemPtr<half>();
        projectedBiasesGpuMem_h = projectedBiasesGpu_h.getMemPtr<half>();
    }

    Tensor weightsUpdate_h = weights_h.clone();
    half *weightsUpdateMem_h = weightsUpdate_h.getMemPtr<half>();
    weightsUpdate_h.clearAsync(dataStream);
    Tensor biasesUpdate_h;
    half *biasesUpdateMem_h;
    if (hasBias) {
        biasesUpdate_h = biases_h.clone();
        biasesUpdateMem_h = biasesUpdate_h.getMemPtr<half>();
        biasesUpdate_h.clearAsync(dataStream);
    }
    dataStream.synchronize();

    dataStream.synchronize();
    Stream gradientUpdateStream = sgd->getGradientUpdateStream();

    uint32_t batchesPerEpoch = 1 + rand() % 10000;
    assert(momentum > 0.0f);
    assert(useNesterovMomentum == true);
    float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, epoch);
    sgd->updateHyperParameters(epoch, rand() % batchesPerEpoch, batchesPerEpoch);

    // Call forward
    fullyConnectedLayer->forward(featureInput, false);

    // Verify featureOutput
    featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
    dataStream.synchronize();
    matrixMultiplyCpuHalf(featureInMem_h,
                          weightsMem_h,
                          featureOutMem_h,
                          batchSize,
                          numInputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          false,
                          false,
                          false,
                          false);
    verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.02, 0.03f);

    // Call backward
    fullyConnectedLayer->backward(errorInput);

    // Verify errorOutput
    errorOutputGpu_h.copyFromAsync(errorOutput, dataStream);
    dataStream.synchronize();
    matrixMultiplyCpuHalf(errorInMem_h,
                          weightsMem_h,
                          errorOutMem_h,
                          batchSize,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          numInputFeatures,
                          false,
                          true,
                          false,
                          false);
    verifyMatricesMatch(errorOutMem_h, errorOutGpuMem_h, batchSize, numInputFeatures, false, 0.4f, 0.03f);

    // Verify weights
    gradientUpdateStream.synchronize();
    matrixMultiplyCpuHalf(featureInMem_h,
                          errorInMem_h,
                          weightsGradientMem_h,
                          batchSize,
                          numInputFeatures,
                          batchSize,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          true,
                          false,
                          false,
                          false);
    weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
    gradientUpdateStream.synchronize();
    verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);

    if (hasBias) {
        reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
        biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
        gradientUpdateStream.synchronize();
        verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.02f);
    }

    weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
    projectedWeightsGpu_h.copyFromAsync(sgd->getProjectedWeights(), gradientUpdateStream);
    gradientUpdateStream.synchronize();

    // printf("%f * %f - (%f * %f) / %d\n",
    //        (float)weightsUpdateMem_h[0],
    //        momentum,
    //        currentLearningRate,
    //        (float)weightsGradientGpuMem_h[0],
    //        batchSize);
    for (uint32_t row = 0; row < numInputFeatures; ++row) {
        for (uint32_t col = 0; col < numOutputFeatures; ++col) {
            uint32_t index = row * numOutputFeatures + col;
            weightsUpdateMem_h[index] =
                (float)weightsUpdateMem_h[index] * momentum -
                (((float)weightsGradientGpuMem_h[index] * currentLearningRate) / batchSize) / Loss::getLossScalingFactor();
            weightsMem_h[index] += (float)weightsUpdateMem_h[index];
            projectedWeightsMem_h[index] = weightsMem_h[index] + (half)momentum * weightsUpdateMem_h[index];
        }
    }
    verifyMatricesMatch(weightsMem_h, weightsGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
    verifyMatricesMatch(projectedWeightsMem_h, projectedWeightsGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
    // Don't let error increase as test gets longer:
    weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
    projectedWeights_h.copyFromAsync(projectedWeightsGpu_h, gradientUpdateStream);

    if (hasBias) {
        biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
        projectedBiasesGpu_h.copyFromAsync(sgd->getProjectedBiases(), gradientUpdateStream);
        gradientUpdateStream.synchronize();

        for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
            biasesUpdateMem_h[outputFeature] =
                (float)biasesUpdateMem_h[outputFeature] * momentum -
                (((float)biasesGradientGpuMem_h[outputFeature] * currentLearningRate) / batchSize) / Loss::getLossScalingFactor();
            biasesMem_h[outputFeature] += (float)biasesUpdateMem_h[outputFeature];
            projectedBiasesMem_h[outputFeature] = biasesMem_h[outputFeature] + (half)momentum * biasesUpdateMem_h[outputFeature];
        }
        verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
        verifyMatricesMatch(projectedBiasesMem_h, projectedBiasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
        // Don't let error increase as test gets longer:
        biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
        projectedBiases_h.copyFromAsync(projectedBiasesGpu_h, gradientUpdateStream);
    }

    // Re-randomize featureInput and errorInput
    featureInput.fillRandom(-2.0, 2.0, dataStream);
    errorInput.fillRandom(-2.0, 2.0, dataStream);
    featureInput_h.copyFromAsync(featureInput, dataStream);
    errorInput_h.copyFromAsync(errorInput, dataStream);
    dataStream.synchronize();

    // Call forward for second time, now there is a previous weights update
    fullyConnectedLayer->forward(featureInput, false);

    // Verify featureOutput
    featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
    dataStream.synchronize();
    matrixMultiplyCpuHalf(featureInMem_h,
                          projectedWeightsMem_h,
                          featureOutMem_h,
                          batchSize,
                          numInputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          false,
                          false,
                          false,
                          false);
    verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.02, 0.03f);

    // Call backward
    fullyConnectedLayer->backward(errorInput);

    // Verify errorOutput
    errorOutputGpu_h.copyFromAsync(errorOutput, dataStream);
    dataStream.synchronize();
    matrixMultiplyCpuHalf(errorInMem_h,
                          weightsMem_h,
                          errorOutMem_h,
                          batchSize,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          numInputFeatures,
                          false,
                          true,
                          false,
                          false);
    verifyMatricesMatch(errorOutMem_h, errorOutGpuMem_h, batchSize, numInputFeatures, false, 0.4f, 0.03f);

    // Verify weights
    gradientUpdateStream.synchronize();
    matrixMultiplyCpuHalf(featureInMem_h,
                          errorInMem_h,
                          weightsGradientMem_h,
                          batchSize,
                          numInputFeatures,
                          batchSize,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          true,
                          false,
                          false,
                          false);
    weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
    gradientUpdateStream.synchronize();
    verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);

    if (hasBias) {
        reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
        biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
        gradientUpdateStream.synchronize();
        verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
    }

    weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
    projectedWeightsGpu_h.copyFromAsync(sgd->getProjectedWeights(), gradientUpdateStream);
    gradientUpdateStream.synchronize();

    // printf("%f * %f - (%f * %f) / %d\n",
    //        (float)weightsUpdateMem_h[0],
    //        momentum,
    //        currentLearningRate,
    //        (float)weightsGradientGpuMem_h[0],
    //        batchSize);
    for (uint32_t row = 0; row < numInputFeatures; ++row) {
        for (uint32_t col = 0; col < numOutputFeatures; ++col) {
            uint32_t index = row * numOutputFeatures + col;
            weightsUpdateMem_h[index] =
                (float)weightsUpdateMem_h[index] * momentum -
                (((float)weightsGradientGpuMem_h[index] * currentLearningRate) / batchSize) / Loss::getLossScalingFactor();
            weightsMem_h[index] += (float)weightsUpdateMem_h[index];
            projectedWeightsMem_h[index] = weightsMem_h[index] + (half)momentum * weightsUpdateMem_h[index];
        }
    }
    verifyMatricesMatch(weightsMem_h, weightsGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
    verifyMatricesMatch(projectedWeightsMem_h, projectedWeightsGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
    // Don't let error increase as test gets longer:
    weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
    projectedWeights_h.copyFromAsync(projectedWeightsGpu_h, gradientUpdateStream);

    if (hasBias) {
        biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
        projectedBiasesGpu_h.copyFromAsync(sgd->getProjectedBiases(), gradientUpdateStream);
        gradientUpdateStream.synchronize();

        for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
            biasesUpdateMem_h[outputFeature] =
                (float)biasesUpdateMem_h[outputFeature] * momentum -
                (((float)biasesGradientGpuMem_h[outputFeature] * currentLearningRate) / batchSize) / Loss::getLossScalingFactor();
            biasesMem_h[outputFeature] += (float)biasesUpdateMem_h[outputFeature];
            projectedBiasesMem_h[outputFeature] = biasesMem_h[outputFeature] + (half)momentum * biasesUpdateMem_h[outputFeature];
        }
        verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
        verifyMatricesMatch(projectedBiasesMem_h, projectedBiasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
        // Don't let error increase as test gets longer:
        biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
        projectedBiases_h.copyFromAsync(projectedBiasesGpu_h, gradientUpdateStream);
    }

    // Call forward for a third time, this time in inference mode, ensure that projected weights are not used to create featureOutput
    fullyConnectedLayer->forward(featureInput, true);

    // Verify featureOutput
    featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
    dataStream.synchronize();
    matrixMultiplyCpuHalf(featureInMem_h,
                          weightsMem_h,
                          featureOutMem_h,
                          batchSize,
                          numInputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          false,
                          false,
                          false,
                          false);
    verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.02, 0.03f);
}
*/
/*
TEST(AdamTest, multipleStamps) {
    srand(time(nullptr));
    half alpha = 0.02f;
    half beta1 = 0.82f;
    half beta2 = 0.933f;
    half epsilon = 1e-5f;
    float t = 0.0f;
    bool hasBias = rand() % 4 ? true : false;
    uint32_t numConnections = 1 + (rand() % 4);
    shared_ptr<FullyConnected> fc = dynamic_pointer_cast<FullyConnected>(constructTrainableLayerWithMultipleConnections(numConnections,
hasBias)); ASSERT_NE(fc, nullptr); shared_ptr<Adam> adam = make_shared<Adam>(fc, alpha, beta1, beta2, epsilon, fc->getErrorInputs()[0],
Optional<Tensor>::empty()); fc->compile(); fc->setOptimizer(dynamic_pointer_cast<Optimizer>(adam)); adam->initialize();

    uint32_t batchSize = fc->getErrorInputs()[0].get().getDimensions()[0];
    uint32_t exampleSize = fc->getErrorInputs()[0].get().getTotalNumElements() / batchSize;

    Stream dataStream = fc->getStreams()[0];
    Stream gradientUpdateStream = fc->getOptimizer().get()->getGradientUpdateStream();

    vector<half> weightsGradientCpu;
    vector<half> biasesGradientCpu;
    vector<half> mCpu;
    vector<half> vCpu;
    vector<half> weightsUpdateCpu;
    vector<half> mBiasesCpu;
    vector<half> vBiasesCpu;
    vector<half> biasesUpdateCpu;

    Tensor weightsGradientGpu;
    Optional<Tensor> maybeBiasesGradientGpu;

    weightsGradientGpu = adam->getWeightsGradient();
    maybeBiasesGradientGpu = adam->getBiasesGradient();
    ASSERT_EQ(maybeBiasesGradientGpu.isPresent(), hasBias);
    mCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    vCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    if (maybeBiasesGradientGpu.isPresent()) {
        mBiasesCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
        vBiasesCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
    }

    uint32_t numWeights = weightsGradientGpu.getTotalNumElements();
    uint32_t numBiases = 0;
    if (maybeBiasesGradientGpu.isPresent())
        numBiases = maybeBiasesGradientGpu.get().getTotalNumElements();

    weightsGradientCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    weightsUpdateCpu = vector<half>(weightsGradientGpu.getTotalNumElements(), 0.0f);
    if (maybeBiasesGradientGpu.isPresent()) {
        biasesGradientCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
        biasesUpdateCpu = vector<half>(maybeBiasesGradientGpu.get().getTotalNumElements(), 0.0f);
    }

    for(uint32_t i = 0; i < 3; ++i) {
        t += 1;

        vector<uint32_t> connectionNumbers;
        for(uint32_t j = 0; j < numConnections; ++j)
            connectionNumbers.push_back(j);
        random_device rd;
        mt19937 gen(rd());
        shuffle(connectionNumbers.begin(), connectionNumbers.end(), gen);

        for(uint32_t connectionIndex = 0; connectionIndex < numConnections; ++connectionIndex) {
            uint32_t connectionNumber = connectionNumbers[connectionIndex];
        }
        // populate featureInput and errorInput so that gradient can be computed
        vector<half> featureInputVector;
        Tensor featureInputGpu = fc->getFeatureInputs()[0];
        for (uint32_t j = 0; j < featureInputGpu.getTotalNumElements(); ++j) {
            featureInputVector.push_back(2.0f / ((rand() % 100) + 1.0f));
        }
        featureInputGpu.setValues(featureInputVector, dataStream);

        vector<half> errorInputVector;
        Tensor errorInputGpu = fc->getErrorInputs()[0];
        for (uint32_t j = 0; j < errorInputGpu.getTotalNumElements(); ++j) {
            errorInputVector.push_back(2.0f / ((rand() % 100) + 1.0f));
        }
        errorInputGpu.setValues(errorInputVector, dataStream);

        bool accumulateValues = i == 0 ? false : true;

        adam->computeWeightsUpdate(featureInputGpu, errorInputGpu, accumulateValues);

        // Cpu computation
        // compute gradient
        matrixMultiplyCpuHalf(featureInputVector.data(),
                              errorInputVector.data(),
                              weightsGradientCpu.data(),
                              featureInputGpu.getDimensions()[0],
                              featureInputGpu.getDimensions()[1],
                              errorInputGpu.getDimensions()[0],
                              errorInputGpu.getDimensions()[1],
                              featureInputGpu.getDimensions()[1],
                              errorInputGpu.getDimensions()[1],
                              weightsGradientGpu.getDimensions()[1],
                              true,
                              false,
                              accumulateValues,
                              false);

        // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
        vector<half> weightsGradientGpuVector;
        weightsGradientGpu.loadValuesIntoVector(weightsGradientGpuVector, gradientUpdateStream);
        for (uint32_t j; j < numWeights; ++j) {
            //printf("%f %f\n", (float)weightsGradientCpu[j], (float)weightsGradientGpuVector[j]);
            ASSERT_LT(abs((float)(weightsGradientCpu[j] - weightsGradientGpuVector[j])), max(0.00001, abs((double)weightsGradientCpu[j] *
.01)));
        }
        weightsGradientGpu.loadValuesIntoVector(weightsGradientCpu, gradientUpdateStream);

        if (hasBias) {
            // compute the biases gradient
            computeBiasesGradientCpu(errorInputVector, biasesGradientCpu, batchSize, exampleSize, accumulateValues);

            // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
            vector<half> biasesGradientGpuVector;
            maybeBiasesGradientGpu.get().loadValuesIntoVector(biasesGradientGpuVector, gradientUpdateStream);
            for (uint32_t j; j < numBiases; ++j) {
                //printf("%f %f\n", (float)biasesGradientCpu[j], (float)biasesGradientGpuVector[j]);
                ASSERT_LT(abs((float)(biasesGradientCpu[j] - biasesGradientGpuVector[j])), max(0.00001, abs((double)biasesGradientCpu[j] *
.01)));
            }
            maybeBiasesGradientGpu.get().loadValuesIntoVector(biasesGradientCpu, gradientUpdateStream);
        }

        // update m, v
        for (uint32_t j = 0; j < numWeights; ++j) {
            //if (j == 0)
            //    printf("m calc %f = %f * %f + (1.0f - %f) * %f\n", (float)(beta1 * mCpu[j] + ((half)1.0f - beta1) *
weightsGradientCpu[j]), (float)beta1, (float)mCpu[j], (float)beta1, (float)weightsGradientCpu[j]); mCpu[j] = beta1 * mCpu[j] + ((half)1.0f -
beta1) * weightsGradientCpu[j];
            //if (j == 0)
            //    printf("v calc = %f = %f * %f + (1.0 - %f) * (%f * %f)\n", (float)(beta2 * vCpu[j] + ((half)1.0f - beta2) *
(weightsGradientCpu[j] * weightsGradientCpu[j])),
            //           (float)beta2, (float)vCpu[j], (float)((half)1.0f - beta2), (float)weightsGradientCpu[j],
(float)weightsGradientCpu[j]); vCpu[j] = beta2 * vCpu[j] + ((half)1.0f - beta2) * (weightsGradientCpu[j] * weightsGradientCpu[j]);
        }
        for (uint32_t j = 0; j < numBiases; ++j) {
            mBiasesCpu[j] = beta1 * mBiasesCpu[j] + ((half)1.0f - beta1) * biasesGradientCpu[j];
            vBiasesCpu[j] = beta2 * vBiasesCpu[j] + ((half)1.0f - beta2) * (biasesGradientCpu[j] * biasesGradientCpu[j]);
        }
    }

    gradientUpdateStream.synchronize();
    double thresh;

    // Ensure t is 3.0f as expected
    ASSERT_EQ(adam->getT(), 3.0f);

    // Get m ensure it is correct, then load actual values to maintain test accuracy downstream
    vector<half> mGpuVector;
    adam->getM(mGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        thresh = max(0.00001, abs((double)mCpu[j] * .01));
        ASSERT_LT(abs((double)(mCpu[j] - mGpuVector[j])), thresh);
    }
    mCpu = mGpuVector;

    // Get v ensure it is correct, then load actual values to maintain test accuracy downstream
    vector<half> vGpuVector;
    adam->getV(vGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        thresh = max(0.00001, abs((double)vCpu[j] * .01));
        //printf("v cpu %f gpu %f\n", (float)vCpu[j], (float)vGpuVector[j]);
        ASSERT_LT(abs((double)(vCpu[j] - vGpuVector[j])), thresh);
    }
    vCpu = vGpuVector;

    if (hasBias) {
        // Get mBiases ensure it is correct, then load actual values to maintain test accuracy downstream
        vector<half> mBiasesGpuVector;
        adam->getMBias(mBiasesGpuVector);
        for (uint32_t j = 0; j < numBiases; ++j) {
            //printf("%f %f\n", (float)mBiasesCpu[j], (float)mBiasesGpuVector[j]);
            thresh = max(0.00001, abs((double)mBiasesCpu[j] * .01));
            ASSERT_LT(abs((double)(mBiasesCpu[j] - mBiasesGpuVector[j])), thresh);
        }
        mBiasesCpu = mBiasesGpuVector;

        // Get vBiases ensure it is correct, then load actual values to maintain test accuracy downstream
        vector<half> vBiasesGpuVector;
        adam->getVBias(vBiasesGpuVector);
        for (uint32_t j = 0; j < numBiases; ++j) {
            thresh = max(0.00001, abs((double)vBiasesCpu[j] * .01));
            ASSERT_LT(abs((double)(vBiasesCpu[j] - vBiasesGpuVector[j])), thresh);
        }
        vBiasesCpu = vBiasesGpuVector;
    }

    // Compute weights and biases update values
    half alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
    //printf("alphaT = %f = %f * %f / %f, beta1 %f beta2 %f t %f\n", (float)alphaT, (float)alpha, sqrtf(1.0f - powf(beta2, t)), (1.0f -
powf(beta1, t)), (float)beta1, (float)beta2, t); for (uint32_t j = 0; j < numWeights; ++j) { weightsUpdateCpu[j] = (-alphaT * mCpu[j]) /
((half)sqrtf(vCpu[j]) + epsilon);
        //if (j == 0)
        //    printf("%f = (%f * %f) / (sqrt(%f) + %f)\n", (float)weightsUpdateCpu[j], (float)-alphaT, (float)mCpu[j], (float)vCpu[j],
(float)epsilon);
    }
    for (uint32_t j = 0; j < numBiases; ++j)
        biasesUpdateCpu[j] = (-alphaT * mBiasesCpu[j]) / ((half)sqrtf(vBiasesCpu[j]) + epsilon);

    // Get weightsUpdate ensure it is correct
    vector<half> weightsUpdateGpuVector;
    adam->getWeightsUpdate(weightsUpdateGpuVector);
    vector<half> biasesUpdateGpuVector;
    adam->getBiasesUpdate(biasesUpdateGpuVector);
    for (uint32_t j = 0; j < numWeights; ++j) {
        //printf("[%d] cpu: %f gpu %f      -alphaT %f mCpu %f sqrt(vCpu) %f epsilon %f\n", j, (float)weightsUpdateCpu[j],
(float)weightsUpdateGpuVector[j], (float)-alphaT, (float)mCpu[j], sqrtf(vCpu[j]), (float)epsilon);
        ASSERT_LT(abs((double)(weightsUpdateCpu[j] - weightsUpdateGpuVector[j])), 0.0001);
    }
    for (uint32_t j = 0; j < numBiases; ++j) {
        ASSERT_LT(abs((double)(biasesUpdateCpu[j] - biasesUpdateGpuVector[j])), 0.0001);
    }

    // set the weights and then call updateWeights.
    Tensor weightsGpu = fc->getWeights();
    Optional<Tensor> biasesGpu = fc->getBiases();
    vector<half> weightsCpu(weightsGpu.getTotalNumElements(), 0.0f);
    vector<half> biasesCpu;
    for (uint32_t i = 0; i < weightsGpu.getTotalNumElements(); ++i)
        weightsCpu[i] = (rand() % 50) / ((rand() % 50) + 1);
    weightsGpu.setValues(weightsCpu, gradientUpdateStream);
    if (hasBias) {
        biasesCpu = vector<half>(biasesGpu.get().getTotalNumElements(), 0.0f);
        for (uint32_t i = 0; i < biasesGpu.get().getTotalNumElements(); ++i)
            biasesCpu[i] = (rand() % 50) / ((rand() % 50) + 1);
        biasesGpu.get().setValues(biasesCpu, gradientUpdateStream);
    }
    // batchSize is only used for math, it doesn't need to match the tensor dimension in this test.
    adam->updateWeights(weightsGpu, biasesGpu, batchSize);
    gradientUpdateStream.synchronize();

    // Check that the weights have been updated to the proper values
    vector<half> weightsGpuVector;
    fc->getWeights().loadValuesIntoVector(weightsGpuVector, gradientUpdateStream);
    for (uint32_t j = 0; j < numWeights; ++j) {
        half updatedWeight = weightsCpu[j] + (half)((float)weightsUpdateCpu[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
        //printf("%f %f\n", (float)updatedWeight, (float)weightsGpuVector[j]);
        thresh = max(0.00001, abs((double)updatedWeight * .001));
        ASSERT_LT(abs((double)(updatedWeight - weightsGpuVector[j])), thresh);
    }
}
*/

/* FIXME: when FC supports FP32 test FP32 adam:
TEST(AdamTest, updateWeightsFP32) {
    shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
    Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
}
*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}