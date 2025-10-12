//#include "Thor.h"
//#include "gtest/gtest.h"
//
//#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
//#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
//#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"
//
// using namespace ThorImplementation;
// using namespace std;
//
//#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"
//
// void verifyMatricesMatch(
//    half *expected, half *actual, uint32_t rows, uint32_t cols, bool print = false, float staticThresh = 0.1, float dynamicThresh = 0.004)
//    { for (uint32_t row = 0; row < rows; ++row) {
//        for (uint32_t col = 0; col < cols; ++col) {
//            float expectedValue = expected[row * cols + col];
//            float actualValue = actual[row * cols + col];
//            float diff = abs(expectedValue - actualValue);
//            float scaledThresh = max(staticThresh, fabsf(expectedValue * dynamicThresh));
//            if (print || diff > scaledThresh) {
//                printf("[%d,%d] GPU %f vs %f CPU\n", row, col, actualValue, expectedValue);
//            }
//            fflush(stdout);
//            assert(diff <= scaledThresh);
//            ASSERT_LE(diff, scaledThresh);
//        }
//    }
//}
//
// void reduceBatch(half *original, half *reduced, uint32_t batchSize, uint32_t featureOutSize, bool accumulate) {
//    for (uint32_t b = 0; b < batchSize; ++b) {
//        for (uint32_t o = 0; o < featureOutSize; ++o) {
//            if (!accumulate && b == 0)
//                reduced[o] = original[b * featureOutSize + o];
//            else
//                reduced[o] += original[b * featureOutSize + o];
//        }
//    }
//}
//
// shared_ptr<TrainableWeightsBiasesLayer> constructTrainableLayer(bool hasBias = true) {
//    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
//    TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, {(uint64_t)(rand() % 80) + 1, (uint64_t)(rand() % 150) + 1});
//
//    Tensor exampleInputTensor(gpuPlacement, descriptor);
//    shared_ptr<NetworkInput> networkInput = make_shared<NetworkInput>(exampleInputTensor);
//    shared_ptr<FullyConnected> fc0 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, hasBias);
//    shared_ptr<FullyConnected> fc1 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, true);
//    networkInput->connectToNextLayer(fc0.get());
//    fc0->connectToNextLayer(fc1.get());
//    return fc0;
//}
//
// shared_ptr<TrainableWeightsBiasesLayer> constructTrainableLayerWithMultipleConnections(uint32_t numConnections, bool hasBias = true) {
//    assert(numConnections > 0);
//
//    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
//    TensorDescriptor descriptor(TensorDescriptor::DataType::FP16, {(uint64_t)(rand() % 80) + 1, (uint64_t)(rand() % 150) + 1});
//
//    Tensor exampleInputTensor(gpuPlacement, descriptor);
//    shared_ptr<NetworkInput> networkInput = make_shared<NetworkInput>(exampleInputTensor);
//    shared_ptr<FullyConnected> fc0 = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, hasBias);
//    networkInput->connectToNextLayer(fc0.get());
//
//    for (uint32_t connection = 0; connection < numConnections; ++connection) {
//        shared_ptr<NetworkInput> networkInputOther = make_shared<NetworkInput>(exampleInputTensor);
//        networkInputOther->connectToNextLayer(fc0.get());
//        shared_ptr<FullyConnected> fcOther = make_shared<FullyConnected>((uint64_t)(rand() % 300) + 1, true);
//        fc0->connectToNextLayer(fcOther.get());
//    }
//    return fc0;
//}
//
// template <typename T>
// void computeBiasesGradientCpu(T *errorIn, T *biasesGradient, uint32_t batchSize, uint32_t exampleSize, bool accumulate) {
//    if (!accumulate) {
//        for (uint32_t i = 0; i < exampleSize; ++i)
//            biasesGradient[i] = T(0.0f);
//    }
//
//    for (uint32_t i = 0; i < batchSize; ++i) {
//        for (uint32_t j = 0; j < exampleSize; ++j) {
//            T gradientComponent = errorIn[i * exampleSize + j];
//            if (accumulate || i != 0)
//                biasesGradient[j] += gradientComponent;
//            else
//                biasesGradient[j] = gradientComponent;
//        }
//    }
//}
//
//// Test the Adam constructor
// TEST(AdamTest, Constructor) {
//     shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
//     Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
//     EXPECT_EQ(adam.getAlpha(), 0.1f);
//     EXPECT_EQ(adam.getBeta1(), 0.9f);
//     EXPECT_EQ(adam.getBeta2(), 0.999f);
//     EXPECT_EQ(adam.getEpsilon(), 1e-8f);
// }
//
//// Test the Adam::initialize function
// TEST(AdamTest, Initialize) {
//     // Prepare inputs
//     shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
//     Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
//     // Call the function to be tested
//     adam.initialize();
//     // Check the results
//     EXPECT_EQ(adam.getT(), 0.0f);
//     // Also cover setT
//     adam.setT(5.0f);
//     EXPECT_EQ(adam.getT(), 5.0f);
// }
//
//// Test the Adam::setAlpha function
// TEST(AdamTest, SetAlpha) {
//     shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
//     Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
//     adam.setAlpha(0.2f);
//     EXPECT_EQ(adam.getAlpha(), 0.2f);
// }
//
//// Test the Adam::setBeta1 function
// TEST(AdamTest, SetBeta1) {
//     shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
//     Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
//     adam.setBeta1(0.8f);
//     EXPECT_EQ(adam.getBeta1(), 0.8f);
// }
//
//// Test the Adam::setBeta2 function
// TEST(AdamTest, SetBeta2) {
//     shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
//     Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
//     adam.setBeta2(0.998f);
//     EXPECT_EQ(adam.getBeta2(), 0.998f);
// }
//
//// Test the Adam::setEpsilon function
// TEST(AdamTest, SetEpsilon) {
//     shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
//     Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
//     adam.setEpsilon(1e-7f);
//     EXPECT_EQ(adam.getEpsilon(), 1e-7f);
// }
//
// TEST(AdamTest, updateHyperParameters) {
//     // FIXME: Implement
// }
// TEST(AdamTest, getAllHyperParameters) {
//     // FIXME: Implement
// }
//
//// Test the Adam::computeWeightsUpdate and Adam::updateWeights functions
// TEST(AdamTest, UpdateWeightsFP16) {
//     srand(time(nullptr));
//
//     TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
//     TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
//
//     half alpha = 0.02f;
//     half beta1 = 0.82f;
//     half beta2 = 0.933f;
//     half epsilon = 1e-5f;
//     float t = 0.0f;
//     bool hasBias = rand() % 4 ? true : false;
//     shared_ptr<FullyConnected> fc = dynamic_pointer_cast<FullyConnected>(constructTrainableLayer(hasBias));
//     ASSERT_NE(fc, nullptr);
//     shared_ptr<Adam> adam = make_shared<Adam>(fc, alpha, beta1, beta2, epsilon, fc->getErrorInputs()[0], Optional<Tensor>::empty());
//     fc->compile();
//     fc->setOptimizer(dynamic_pointer_cast<Optimizer>(adam));
//     adam->initialize();
//
//     uint32_t batchSize = fc->getErrorInputs()[0].get().getDimensions()[0];
//     uint32_t exampleSize = fc->getErrorInputs()[0].get().getTotalNumElements() / batchSize;
//
//     Stream dataStream = fc->getStreams()[0];
//     Stream gradientUpdateStream = fc->getOptimizer().get()->getGradientUpdateStream();
//
//     Tensor weights = fc->getWeights();
//     Tensor weights_h = weights.clone(cpuPlacement);
//     half *weightsMem_h = weights_h.getMemPtr<half>();
//     Tensor weightsGpu_h = weights_h.clone(cpuPlacement);
//     half *weightsGpuMem_h = weightsGpu_h.getMemPtr<half>();
//     Optional<Tensor> biases = fc->getBiases();
//     Tensor biases_h;
//     half *biasesMem_h;
//     Tensor biasesGpu_h;
//     half *biasesGpuMem_h;
//     ASSERT_EQ(hasBias, biases.isPresent());
//     if (biases.isPresent()) {
//         biases_h = biases.get().clone(cpuPlacement);
//         biasesMem_h = biases_h.getMemPtr<half>();
//         biasesGpu_h = biases_h.clone();
//         biasesGpuMem_h = biasesGpu_h.getMemPtr<half>();
//     }
//     Tensor weightsGradient = adam->getWeightsGradient();
//     Tensor weightsGradient_h = weightsGradient.clone(cpuPlacement);
//     half *weightsGradientMem_h = weightsGradient_h.getMemPtr<half>();
//     Tensor weightsGradientGpu_h = weightsGradient_h.clone(cpuPlacement);
//     half *weightsGradientGpuMem_h = weightsGradientGpu_h.getMemPtr<half>();
//     Optional<Tensor> biasesGradient = adam->getBiasesGradient();
//     Tensor biasesGradient_h;
//     half *biasesGradientMem_h;
//     Tensor biasesGradientGpu_h;
//     half *biasesGradientGpuMem_h;
//     ASSERT_EQ(hasBias, biasesGradient.isPresent());
//     if (biasesGradient.isPresent()) {
//         biasesGradient_h = biasesGradient.get().clone(cpuPlacement);
//         biasesGradientMem_h = biasesGradient_h.getMemPtr<half>();
//         biasesGradientGpu_h = biasesGradient_h.clone();
//         biasesGradientGpuMem_h = biasesGradientGpu_h.getMemPtr<half>();
//     }
//     Tensor m = adam->getM();
//     Tensor m_h = m.clone(cpuPlacement);
//     m_h.fillZero(dataStream);
//     half *mMem_h = m_h.getMemPtr<half>();
//     Tensor mGpu_h = m_h.clone();
//     half *mGpuMem_h = mGpu_h.getMemPtr<half>();
//     Tensor v = adam->getV();
//     Tensor v_h = v.clone(cpuPlacement);
//     v_h.fillZero(dataStream);
//     half *vMem_h = v_h.getMemPtr<half>();
//     Tensor vGpu_h = v_h.clone();
//     half *vGpuMem_h = vGpu_h.getMemPtr<half>();
//     Tensor weightsUpdate = adam->getWeightsUpdate();
//     Tensor weightsUpdate_h = weightsUpdate.clone(cpuPlacement);
//     half *weightsUpdateMem_h = weightsUpdate_h.getMemPtr<half>();
//     Tensor weightsUpdateGpu_h = weightsUpdate_h.clone();
//     half *weightsUpdateGpuMem_h = weightsUpdateGpu_h.getMemPtr<half>();
//     Optional<Tensor> mBiases = adam->getMBias();
//     Tensor mBiases_h;
//     half *mBiasesMem_h;
//     Tensor mBiasesGpu_h;
//     half *mBiasesGpuMem_h;
//     ASSERT_EQ(mBiases.isPresent(), biasesGradient.isPresent());
//     if (mBiases.isPresent()) {
//         mBiases_h = mBiases.get().clone(cpuPlacement);
//         mBiasesMem_h = mBiases_h.getMemPtr<half>();
//         mBiasesGpu_h = mBiases_h.clone();
//         mBiasesGpuMem_h = mBiasesGpu_h.getMemPtr<half>();
//     }
//     Optional<Tensor> vBiases = adam->getVBias();
//     Tensor vBiases_h;
//     half *vBiasesMem_h;
//     Tensor vBiasesGpu_h;
//     half *vBiasesGpuMem_h;
//     ASSERT_EQ(vBiases.isPresent(), mBiases.isPresent());
//     if (vBiases.isPresent()) {
//         vBiases_h = vBiases.get().clone(cpuPlacement);
//         vBiasesMem_h = vBiases_h.getMemPtr<half>();
//         vBiasesGpu_h = vBiases_h.clone();
//         vBiasesGpuMem_h = vBiasesGpu_h.getMemPtr<half>();
//     }
//     Optional<Tensor> biasesUpdate = adam->getBiasesUpdate();
//     Tensor biasesUpdate_h;
//     half *biasesUpdateMem_h;
//     Tensor biasesUpdateGpu_h;
//     half *biasesUpdateGpuMem_h;
//     assert(biasesUpdate.isPresent() == vBiases.isPresent());
//     if (biasesUpdate.isPresent()) {
//         biasesUpdate_h = biasesUpdate.get().clone(cpuPlacement);
//         biasesUpdateMem_h = biasesUpdate_h.getMemPtr<half>();
//         biasesUpdateGpu_h = biasesUpdate_h.clone();
//         biasesUpdateGpuMem_h = biasesUpdateGpu_h.getMemPtr<half>();
//     }
//     dataStream.synchronize();
//
//     uint32_t numWeights = weightsGradient.getTotalNumElements();
//     uint32_t numBiases = 0;
//     if (biasesGradient.isPresent())
//         numBiases = biasesGradient.get().getTotalNumElements();
//
//     t = 1;
//
//     Tensor featureInput = fc->getFeatureInputs()[0];
//     Tensor featureInput_h = featureInput.clone(cpuPlacement);
//     half *featureInputMem_h = featureInput_h.getMemPtr<half>();
//     Tensor errorInput = fc->getErrorInputs()[0];
//     Tensor errorInput_h = errorInput.clone(cpuPlacement);
//     half *errorInputMem_h = errorInput_h.getMemPtr<half>();
//     for (uint32_t i = 0; i < 3; ++i) {
//         // populate featureInput and errorInput so that gradient can be computed
//         for (uint32_t j = 0; j < featureInput.getTotalNumElements(); ++j) {
//             featureInputMem_h[j] = 2.0f / ((rand() % 100) + 1.0f);
//         }
//         featureInput.copyFromAsync(featureInput_h, dataStream);
//
//         for (uint32_t j = 0; j < errorInput.getTotalNumElements(); ++j) {
//             errorInputMem_h[j] = 2.0f / ((rand() % 100) + 1.0f);
//         }
//         errorInput.copyFromAsync(errorInput_h, dataStream);
//
//         bool accumulateValues = i == 0 ? false : true;
//
//         adam->computeWeightsUpdate(featureInput, errorInput, accumulateValues);
//
//         // Cpu computation
//         // compute gradient
//         matrixMultiplyCpuHalf(featureInputMem_h,
//                               errorInputMem_h,
//                               weightsGradientMem_h,
//                               featureInput.getDimensions()[0],
//                               featureInput.getDimensions()[1],
//                               errorInput.getDimensions()[0],
//                               errorInput.getDimensions()[1],
//                               featureInput.getDimensions()[1],
//                               errorInput.getDimensions()[1],
//                               weightsGradient.getDimensions()[1],
//                               true,
//                               false,
//                               accumulateValues,
//                               false);
//
//         // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
//         weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j; j < numWeights; ++j) {
//             // printf("%f %f\n", (float)weightsGradientMem_h[j], (float)weightsGradientGpuMem_h[j]);
//             ASSERT_LT(abs((float)(weightsGradientMem_h[j] - weightsGradientGpuMem_h[j])),
//                       max(0.00001, abs((double)weightsGradientMem_h[j] * .01)));
//         }
//         weightsGradient_h.copyFromAsync(weightsGradientGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//
//         if (hasBias) {
//             // compute the biases gradient
//             computeBiasesGradientCpu(errorInputMem_h, biasesGradientMem_h, batchSize, exampleSize, accumulateValues);
//
//             // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
//             biasesGradientGpu_h.copyFromAsync(biasesGradient_h, gradientUpdateStream);
//             gradientUpdateStream.synchronize();
//             for (uint32_t j; j < numBiases; ++j) {
//                 // printf("%f %f\n", (float)biasesGradientMem_h[j], (float)biasesGradientGpuVector[j]);
//                 ASSERT_LT(abs((float)(biasesGradientMem_h[j] - biasesGradientGpuMem_h[j])),
//                           max(0.00001, abs((double)biasesGradientMem_h[j] * .01)));
//             }
//             biasesGradient_h.copyFromAsync(biasesGradientGpu_h, gradientUpdateStream);
//             gradientUpdateStream.synchronize();
//         }
//
//         // update m, v
//         for (uint32_t j = 0; j < numWeights; ++j) {
//             // if (j == 0)
//             //     printf("m calc %f = %f * %f + (1.0f - %f) * %f\n", (float)(beta1 * mCpu[j] + ((half)1.0f - beta1) *
//             //     weightsGradientMem_h[j]), (float)beta1, (float)mCpu[j], (float)beta1, (float)weightsGradientMem_h[j]);
//             mMem_h[j] = beta1 * mMem_h[j] + ((half)1.0f - beta1) * weightsGradientMem_h[j];
//             // if (j == 0)
//             //     printf("v calc = %f = %f * %f + (1.0 - %f) * (%f * %f)\n", (float)(beta2 * vMem_h[j] + ((half)1.0f - beta2) *
//             //     (weightsGradientMem_h[j] * weightsGradientMem_h[j])),
//             //            (float)beta2, (float)vMem_h[j], (float)((half)1.0f - beta2), (float)weightsGradientMem_h[j],
//             //            (float)weightsGradientMem_h[j]);
//             vMem_h[j] = beta2 * vMem_h[j] + ((half)1.0f - beta2) * (weightsGradientMem_h[j] * weightsGradientMem_h[j]);
//         }
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             mBiasesMem_h[j] = beta1 * mBiasesMem_h[j] + ((half)1.0f - beta1) * biasesGradientMem_h[j];
//             vBiasesMem_h[j] = beta2 * vBiasesMem_h[j] + ((half)1.0f - beta2) * (biasesGradientMem_h[j] * biasesGradientMem_h[j]);
//         }
//     }
//
//     gradientUpdateStream.synchronize();
//     double thresh;
//
//     // Ensure t is 1.0f as expected -> 3 computeWeightsUpdate(...) occurred but only the first had accumulateValues = 0
//     ASSERT_EQ(adam->getT(), 1.0f);
//
//     // Get m ensure it is correct, then load actual values to maintain test accuracy downstream
//     mGpu_h.copyFromAsync(m, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         thresh = max(0.00001, abs((double)mMem_h[j] * .01));
//         if (!(abs((double)(mMem_h[j] - mGpuMem_h[j])) < thresh))
//             printf("mMem_h[%d] %f mGpu[%d] %f\n", j, (float)mMem_h[j], j, (float)mGpuMem_h[j]);
//         ASSERT_LT(abs((double)(mMem_h[j] - mGpuMem_h[j])), thresh);
//     }
//     m_h.copyFromAsync(mGpu_h, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     // Get v ensure it is correct, then load actual values to maintain test accuracy downstream
//     vGpu_h.copyFromAsync(v, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         thresh = max(0.00001, abs((double)vMem_h[j] * .01));
//         // printf("v cpu %f gpu %f\n", (float)vMem_h[j], (float)vGpuVector[j]);
//         ASSERT_LT(abs((double)(vMem_h[j] - vGpuMem_h[j])), thresh);
//     }
//     v_h.copyFromAsync(vGpu_h, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     if (hasBias) {
//         // Get mBiases ensure it is correct, then load actual values to maintain test accuracy downstream
//         mBiasesGpu_h.copyFromAsync(mBiases, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             // printf("%f %f\n", (float)mBiasesMem_h[j], (float)mBiasesGpuMem_h[j]);
//             thresh = max(0.03, abs((double)mBiasesMem_h[j] * .02));
//             ASSERT_LT(abs((double)(mBiasesMem_h[j] - mBiasesGpuMem_h[j])), thresh);
//         }
//         mBiases_h.copyFromAsync(mBiasesGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//
//         // Get vBiases ensure it is correct, then load actual values to maintain test accuracy downstream
//         vBiasesGpu_h.copyFromAsync(vBiases, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             thresh = max(0.03, abs((double)vBiasesMem_h[j] * .02));
//             ASSERT_LT(abs((double)(vBiasesMem_h[j] - vBiasesGpuMem_h[j])), thresh);
//         }
//         vBiases_h.copyFromAsync(vBiasesGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//     }
//
//     // Compute weights and biases update values
//     half alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
//     // printf("alphaT = %f = %f * %f / %f, beta1 %f beta2 %f t %f\n", (float)alphaT, (float)alpha, sqrtf(1.0f - powf(beta2, t)), (1.0f -
//     // powf(beta1, t)), (float)beta1, (float)beta2, t);
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         weightsUpdateMem_h[j] = (-alphaT * mMem_h[j]) / ((half)sqrtf(vMem_h[j]) + epsilon);
//         // if (j == 0)
//         //     printf("CPU weightUpdate = %f / %f = %f      alphaT %f, m %f\n", float(-alphaT * mMem_h[j]), float(((half)sqrtf(vMem_h[j])
//         +
//         //     epsilon)), (float)weightsUpdateMem_h[j], (float)-alphaT, float(mMem_h[j]));
//     }
//     for (uint32_t j = 0; j < numBiases; ++j)
//         biasesUpdateMem_h[j] = (-alphaT * mBiasesMem_h[j]) / ((half)sqrtf(vBiasesMem_h[j]) + epsilon);
//
//     // Get weightsUpdate ensure it is correct
//     weightsUpdateGpu_h.copyFromAsync(weightsUpdate, gradientUpdateStream);
//     if (numBiases > 0)
//         biasesUpdateGpu_h.copyFromAsync(biasesUpdate, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         // printf("[%d] cpu: %f gpu %f      -alphaT %f mMem_h %f sqrt(vMem_h) %f epsilon %f\n", j, (float)weightsUpdateMem_h[j],
//         // (float)weightsUpdateGpuMem_h[j], (float)-alphaT, (float)mMem_h[j], sqrtf(vMem_h[j]), (float)epsilon);
//         ASSERT_LT(abs((double)(weightsUpdateMem_h[j] - weightsUpdateGpuMem_h[j])), 0.0002);
//     }
//     for (uint32_t j = 0; j < numBiases; ++j) {
//         ASSERT_LT(abs((double)(biasesUpdateMem_h[j] - biasesUpdateGpuMem_h[j])), 0.0002);
//     }
//
//     // set the weights and then call updateWeights.
//     weights.fillRandom(-2, 2, gradientUpdateStream);
//     weights_h.copyFromAsync(weights, gradientUpdateStream);
//     if (biases.isPresent()) {
//         biases.get().fillRandom(-2, 2, gradientUpdateStream);
//         biases_h.copyFromAsync(biases, gradientUpdateStream);
//         biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
//     }
//     // batchSize is only used for math, it doesn't need to match the tensor dimension in this test.
//     adam->updateWeights(weights, biases, batchSize);
//     weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     // Check that the weights have been updated to the proper values
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         half updatedWeight =
//             weightsMem_h[j] + (half)((float)weightsUpdateMem_h[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
//         // printf("%f %f\n", (float)updatedWeight, (float)weightsGpuVector[j]);
//         thresh = max(0.001, abs((double)updatedWeight * .01));
//         ASSERT_LT(abs((double)(updatedWeight - weightsGpuMem_h[j])), thresh);
//     }
//
//     if (biases.isPresent()) {
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             half updatedBias =
//                 biasesMem_h[j] + (half)((float)biasesUpdateMem_h[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
//             // printf("%f %f\n", (float)updatedBias, (float)biasesGpuMem_h[j]);
//             thresh = max(0.001, abs((double)updatedBias * .01));
//             ASSERT_LT(abs((double)(updatedBias - biasesGpuMem_h[j])), thresh);
//         }
//     }
// }
//
//// Test 5 batches to ensure t is updated properly
// TEST(AdamTest, FullyConnectedBackwardWithAdam5BatchesFP16) {
//     srand(time(nullptr));
//
//     TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
//     TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
//
//     half alpha = 0.02f;
//     half beta1 = 0.82f;
//     half beta2 = 0.933f;
//     half epsilon = 1e-5f;
//     bool hasBias = rand() % 4 ? true : false;
//     shared_ptr<FullyConnected> fc = dynamic_pointer_cast<FullyConnected>(constructTrainableLayer(hasBias));
//     ASSERT_NE(fc, nullptr);
//     shared_ptr<Adam> adam = make_shared<Adam>(fc, alpha, beta1, beta2, epsilon, fc->getErrorInputs()[0], Optional<Tensor>::empty());
//     fc->compile();
//     fc->setOptimizer(dynamic_pointer_cast<Optimizer>(adam));
//     adam->initialize();
//
//     uint32_t batchSize = fc->getErrorInputs()[0].get().getDimensions()[0];
//     uint32_t exampleSize = fc->getErrorInputs()[0].get().getTotalNumElements() / batchSize;
//
//     Stream dataStream = fc->getStreams()[0];
//     Stream gradientUpdateStream = fc->getOptimizer().get()->getGradientUpdateStream();
//
//     Tensor weights = fc->getWeights();
//     Tensor weights_h = weights.clone(cpuPlacement);
//     half *weightsMem_h = weights_h.getMemPtr<half>();
//     Tensor weightsGpu_h = weights_h.clone(cpuPlacement);
//     half *weightsGpuMem_h = weightsGpu_h.getMemPtr<half>();
//     Optional<Tensor> biases = fc->getBiases();
//     Tensor biases_h;
//     half *biasesMem_h;
//     Tensor biasesGpu_h;
//     half *biasesGpuMem_h;
//     ASSERT_EQ(hasBias, biases.isPresent());
//     if (biases.isPresent()) {
//         biases_h = biases.get().clone(cpuPlacement);
//         biasesMem_h = biases_h.getMemPtr<half>();
//         biasesGpu_h = biases_h.clone();
//         biasesGpuMem_h = biasesGpu_h.getMemPtr<half>();
//     }
//     Tensor weightsGradient = adam->getWeightsGradient();
//     Tensor weightsGradient_h = weightsGradient.clone(cpuPlacement);
//     half *weightsGradientMem_h = weightsGradient_h.getMemPtr<half>();
//     Tensor weightsGradientGpu_h = weightsGradient_h.clone(cpuPlacement);
//     half *weightsGradientGpuMem_h = weightsGradientGpu_h.getMemPtr<half>();
//     Optional<Tensor> biasesGradient = adam->getBiasesGradient();
//     Tensor biasesGradient_h;
//     half *biasesGradientMem_h;
//     Tensor biasesGradientGpu_h;
//     half *biasesGradientGpuMem_h;
//     ASSERT_EQ(hasBias, biasesGradient.isPresent());
//     if (biasesGradient.isPresent()) {
//         biasesGradient_h = biasesGradient.get().clone(cpuPlacement);
//         biasesGradientMem_h = biasesGradient_h.getMemPtr<half>();
//         biasesGradientGpu_h = biasesGradient_h.clone();
//         biasesGradientGpuMem_h = biasesGradientGpu_h.getMemPtr<half>();
//     }
//     Tensor m = adam->getM();
//     Tensor m_h = m.clone(cpuPlacement);
//     m_h.fillZero(dataStream);
//     half *mMem_h = m_h.getMemPtr<half>();
//     Tensor mGpu_h = m_h.clone();
//     half *mGpuMem_h = mGpu_h.getMemPtr<half>();
//     Tensor v = adam->getV();
//     Tensor v_h = v.clone(cpuPlacement);
//     v_h.fillZero(dataStream);
//     half *vMem_h = v_h.getMemPtr<half>();
//     Tensor vGpu_h = v_h.clone();
//     half *vGpuMem_h = vGpu_h.getMemPtr<half>();
//     Tensor weightsUpdate = adam->getWeightsUpdate();
//     Tensor weightsUpdate_h = weightsUpdate.clone(cpuPlacement);
//     half *weightsUpdateMem_h = weightsUpdate_h.getMemPtr<half>();
//     Tensor weightsUpdateGpu_h = weightsUpdate_h.clone();
//     half *weightsUpdateGpuMem_h = weightsUpdateGpu_h.getMemPtr<half>();
//     Optional<Tensor> mBiases = adam->getMBias();
//     Tensor mBiases_h;
//     half *mBiasesMem_h;
//     Tensor mBiasesGpu_h;
//     half *mBiasesGpuMem_h;
//     ASSERT_EQ(mBiases.isPresent(), biasesGradient.isPresent());
//     if (mBiases.isPresent()) {
//         mBiases_h = mBiases.get().clone(cpuPlacement);
//         mBiasesMem_h = mBiases_h.getMemPtr<half>();
//         mBiasesGpu_h = mBiases_h.clone();
//         mBiasesGpuMem_h = mBiasesGpu_h.getMemPtr<half>();
//     }
//     Optional<Tensor> vBiases = adam->getVBias();
//     Tensor vBiases_h;
//     half *vBiasesMem_h;
//     Tensor vBiasesGpu_h;
//     half *vBiasesGpuMem_h;
//     ASSERT_EQ(vBiases.isPresent(), mBiases.isPresent());
//     if (vBiases.isPresent()) {
//         vBiases_h = vBiases.get().clone(cpuPlacement);
//         vBiasesMem_h = vBiases_h.getMemPtr<half>();
//         vBiasesGpu_h = vBiases_h.clone();
//         vBiasesGpuMem_h = vBiasesGpu_h.getMemPtr<half>();
//     }
//     Optional<Tensor> biasesUpdate = adam->getBiasesUpdate();
//     Tensor biasesUpdate_h;
//     half *biasesUpdateMem_h;
//     Tensor biasesUpdateGpu_h;
//     half *biasesUpdateGpuMem_h;
//     assert(biasesUpdate.isPresent() == vBiases.isPresent());
//     if (biasesUpdate.isPresent()) {
//         biasesUpdate_h = biasesUpdate.get().clone(cpuPlacement);
//         biasesUpdateMem_h = biasesUpdate_h.getMemPtr<half>();
//         biasesUpdateGpu_h = biasesUpdate_h.clone();
//         biasesUpdateGpuMem_h = biasesUpdateGpu_h.getMemPtr<half>();
//     }
//     dataStream.synchronize();
//
//     uint32_t numWeights = weightsGradient.getTotalNumElements();
//     uint32_t numBiases = 0;
//     if (biasesGradient.isPresent())
//         numBiases = biasesGradient.get().getTotalNumElements();
//
//     Tensor featureInput = fc->getFeatureInputs()[0];
//     Tensor featureInput_h = featureInput.clone(cpuPlacement);
//     half *featureInputMem_h = featureInput_h.getMemPtr<half>();
//     Tensor errorInput = fc->getErrorInputs()[0];
//     Tensor errorInput_h = errorInput.clone(cpuPlacement);
//     half *errorInputMem_h = errorInput_h.getMemPtr<half>();
//
//     float t = 0.0f;
//     for (uint32_t batch = 0; batch < 5; ++batch) {
//         t += 1.0f;
//         for (uint32_t i = 0; i < 3; ++i) {
//             // populate featureInput and errorInput so that gradient can be computed
//             for (uint32_t j = 0; j < featureInput.getTotalNumElements(); ++j) {
//                 featureInputMem_h[j] = 2.0f / ((rand() % 100) + 1.0f);
//             }
//             featureInput.copyFromAsync(featureInput_h, dataStream);
//
//             for (uint32_t j = 0; j < errorInput.getTotalNumElements(); ++j) {
//                 errorInputMem_h[j] = 2.0f / ((rand() % 100) + 1.0f);
//             }
//             errorInput.copyFromAsync(errorInput_h, dataStream);
//
//             bool accumulateValues = i == 0 ? false : true;
//
//             adam->computeWeightsUpdate(featureInput, errorInput, accumulateValues);
//
//             // Cpu computation
//             // compute gradient
//             matrixMultiplyCpuHalf(featureInputMem_h,
//                                   errorInputMem_h,
//                                   weightsGradientMem_h,
//                                   featureInput.getDimensions()[0],
//                                   featureInput.getDimensions()[1],
//                                   errorInput.getDimensions()[0],
//                                   errorInput.getDimensions()[1],
//                                   featureInput.getDimensions()[1],
//                                   errorInput.getDimensions()[1],
//                                   weightsGradient.getDimensions()[1],
//                                   true,
//                                   false,
//                                   accumulateValues,
//                                   false);
//
//             // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
//             weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
//             gradientUpdateStream.synchronize();
//             for (uint32_t j; j < numWeights; ++j) {
//                 // printf("%f %f\n", (float)weightsGradientMem_h[j], (float)weightsGradientGpuMem_h[j]);
//                 ASSERT_LT(abs((float)(weightsGradientMem_h[j] - weightsGradientGpuMem_h[j])),
//                           max(0.00001, abs((double)weightsGradientMem_h[j] * .01)));
//             }
//             weightsGradient_h.copyFromAsync(weightsGradientGpu_h, gradientUpdateStream);
//             gradientUpdateStream.synchronize();
//
//             if (hasBias) {
//                 // compute the biases gradient
//                 computeBiasesGradientCpu(errorInputMem_h, biasesGradientMem_h, batchSize, exampleSize, accumulateValues);
//
//                 // Get the gradient from FC, ensure it is correct, then copy it for use on cpu to improve testing accuracy
//                 biasesGradientGpu_h.copyFromAsync(biasesGradient_h, gradientUpdateStream);
//                 gradientUpdateStream.synchronize();
//                 for (uint32_t j; j < numBiases; ++j) {
//                     // printf("%f %f\n", (float)biasesGradientMem_h[j], (float)biasesGradientGpuVector[j]);
//                     ASSERT_LT(abs((float)(biasesGradientMem_h[j] - biasesGradientGpuMem_h[j])),
//                               max(0.00001, abs((double)biasesGradientMem_h[j] * .01)));
//                 }
//                 biasesGradient_h.copyFromAsync(biasesGradientGpu_h, gradientUpdateStream);
//                 gradientUpdateStream.synchronize();
//             }
//
//             // update m, v
//             for (uint32_t j = 0; j < numWeights; ++j) {
//                 // if (j == 0)
//                 //     printf("m calc %f = %f * %f + (1.0f - %f) * %f\n", (float)(beta1 * mCpu[j] + ((half)1.0f - beta1) *
//                 //     weightsGradientMem_h[j]), (float)beta1, (float)mCpu[j], (float)beta1, (float)weightsGradientMem_h[j]);
//                 mMem_h[j] = beta1 * mMem_h[j] + ((half)1.0f - beta1) * weightsGradientMem_h[j];
//                 // if (j == 0)
//                 //     printf("v calc = %f = %f * %f + (1.0 - %f) * (%f * %f)\n", (float)(beta2 * vMem_h[j] + ((half)1.0f - beta2) *
//                 //     (weightsGradientMem_h[j] * weightsGradientMem_h[j])),
//                 //            (float)beta2, (float)vMem_h[j], (float)((half)1.0f - beta2), (float)weightsGradientMem_h[j],
//                 //            (float)weightsGradientMem_h[j]);
//                 vMem_h[j] = beta2 * vMem_h[j] + ((half)1.0f - beta2) * (weightsGradientMem_h[j] * weightsGradientMem_h[j]);
//             }
//             for (uint32_t j = 0; j < numBiases; ++j) {
//                 mBiasesMem_h[j] = beta1 * mBiasesMem_h[j] + ((half)1.0f - beta1) * biasesGradientMem_h[j];
//                 vBiasesMem_h[j] = beta2 * vBiasesMem_h[j] + ((half)1.0f - beta2) * (biasesGradientMem_h[j] * biasesGradientMem_h[j]);
//             }
//         }
//     }
//
//     gradientUpdateStream.synchronize();
//     double thresh;
//
//     // Ensure t is 1.0f as expected -> 3 computeWeightsUpdate(...) occurred but only the first had accumulateValues = 0
//     ASSERT_EQ(adam->getT(), 5.0f);
//
//     // Get m ensure it is correct, then load actual values to maintain test accuracy downstream
//     mGpu_h.copyFromAsync(m, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         thresh = max(0.00001, abs((double)mMem_h[j] * .01));
//         if (!(abs((double)(mMem_h[j] - mGpuMem_h[j])) < thresh))
//             printf("mMem_h[%d] %f mGpu[%d] %f\n", j, (float)mMem_h[j], j, (float)mGpuMem_h[j]);
//         ASSERT_LT(abs((double)(mMem_h[j] - mGpuMem_h[j])), thresh);
//     }
//     m_h.copyFromAsync(mGpu_h, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     // Get v ensure it is correct, then load actual values to maintain test accuracy downstream
//     vGpu_h.copyFromAsync(v, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         thresh = max(0.00001, abs((double)vMem_h[j] * .01));
//         // printf("v cpu %f gpu %f\n", (float)vMem_h[j], (float)vGpuVector[j]);
//         ASSERT_LT(abs((double)(vMem_h[j] - vGpuMem_h[j])), thresh);
//     }
//     v_h.copyFromAsync(vGpu_h, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     if (hasBias) {
//         // Get mBiases ensure it is correct, then load actual values to maintain test accuracy downstream
//         mBiasesGpu_h.copyFromAsync(mBiases, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             // printf("%f %f\n", (float)mBiasesMem_h[j], (float)mBiasesGpuMem_h[j]);
//             thresh = max(0.05, abs((double)mBiasesMem_h[j] * .02));
//             ASSERT_LT(abs((double)(mBiasesMem_h[j] - mBiasesGpuMem_h[j])), thresh);
//         }
//         mBiases_h.copyFromAsync(mBiasesGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//
//         // Get vBiases ensure it is correct, then load actual values to maintain test accuracy downstream
//         vBiasesGpu_h.copyFromAsync(vBiases, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             thresh = max(0.05, abs((double)vBiasesMem_h[j] * .02));
//             ASSERT_LT(abs((double)(vBiasesMem_h[j] - vBiasesGpuMem_h[j])), thresh);
//         }
//         vBiases_h.copyFromAsync(vBiasesGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//     }
//
//     // Compute weights and biases update values
//     half alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
//     // printf("alphaT = %f = %f * %f / %f, beta1 %f beta2 %f t %f\n", (float)alphaT, (float)alpha, sqrtf(1.0f - powf(beta2, t)), (1.0f -
//     // powf(beta1, t)), (float)beta1, (float)beta2, t);
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         weightsUpdateMem_h[j] = (-alphaT * mMem_h[j]) / ((half)sqrtf(vMem_h[j]) + epsilon);
//         // if (j == 0)
//         //     printf("CPU weightUpdate = %f / %f = %f      alphaT %f, m %f\n", float(-alphaT * mMem_h[j]), float(((half)sqrtf(vMem_h[j])
//         +
//         //     epsilon)), (float)weightsUpdateMem_h[j], (float)-alphaT, float(mMem_h[j]));
//     }
//     for (uint32_t j = 0; j < numBiases; ++j)
//         biasesUpdateMem_h[j] = (-alphaT * mBiasesMem_h[j]) / ((half)sqrtf(vBiasesMem_h[j]) + epsilon);
//
//     // Get weightsUpdate ensure it is correct
//     weightsUpdateGpu_h.copyFromAsync(weightsUpdate, gradientUpdateStream);
//     if (numBiases > 0)
//         biasesUpdateGpu_h.copyFromAsync(biasesUpdate, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         // printf("[%d] cpu: %f gpu %f      -alphaT %f mMem_h %f sqrt(vMem_h) %f epsilon %f\n", j, (float)weightsUpdateMem_h[j],
//         // (float)weightsUpdateGpuMem_h[j], (float)-alphaT, (float)mMem_h[j], sqrtf(vMem_h[j]), (float)epsilon);
//         ASSERT_LT(abs((double)(weightsUpdateMem_h[j] - weightsUpdateGpuMem_h[j])), 0.0002);
//     }
//     for (uint32_t j = 0; j < numBiases; ++j) {
//         ASSERT_LT(abs((double)(biasesUpdateMem_h[j] - biasesUpdateGpuMem_h[j])), 0.0002);
//     }
//
//     // set the weights and then call updateWeights.
//     weights.fillRandom(-2, 2, gradientUpdateStream);
//     weights_h.copyFromAsync(weights, gradientUpdateStream);
//     if (biases.isPresent()) {
//         biases.get().fillRandom(-2, 2, gradientUpdateStream);
//         biases_h.copyFromAsync(biases, gradientUpdateStream);
//         biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
//     }
//     // batchSize is only used for math, it doesn't need to match the tensor dimension in this test.
//     adam->updateWeights(weights, biases, batchSize);
//     weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     // Check that the weights have been updated to the proper values
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         half updatedWeight =
//             weightsMem_h[j] + (half)((float)weightsUpdateMem_h[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
//         // printf("%f %f\n", (float)updatedWeight, (float)weightsGpuVector[j]);
//         thresh = max(0.001, abs((double)updatedWeight * .01));
//         ASSERT_LT(abs((double)(updatedWeight - weightsGpuMem_h[j])), thresh);
//     }
//
//     if (biases.isPresent()) {
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             half updatedBias =
//                 biasesMem_h[j] + (half)((float)biasesUpdateMem_h[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
//             // printf("%f %f\n", (float)updatedBias, (float)biasesGpuMem_h[j]);
//             thresh = max(0.001, abs((double)updatedBias * .01));
//             ASSERT_LT(abs((double)(updatedBias - biasesGpuMem_h[j])), thresh);
//         }
//     }
// }
//
//// 1. Fill feature in random
//// 2. fill errorIn random
//// 3. fill weights in random
//// 4. call forward
//// 5. Verify featureOutput
//// 6. call backward
//// 7. Check weights values
//// 8. fill featureIn random
//// 9. call forward
//// 10. Verify featureOutput
//// 11. fill errorIn random
//// 12. call backward
//// 13. Check weights values
//// 14. Call forward for a third time, this time in inference mode
//// 15. Verify featureOutput
// TEST(AdamTest, FullForwardBackwordFunctionalityFP16) {
//     srand(time(nullptr));
//
//     TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
//     TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
//
//     uint32_t batchSize = (rand() % 300) + 1;
//     uint32_t numInputFeatures = (rand() % 300) + 1;
//     uint32_t numOutputFeatures = (rand() % 300) + 1;
//     bool hasBias = rand() % 2;
//
//     Tensor networkFeatureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));
//
//     half alpha = 0.02f;
//     half beta1 = 0.82f;
//     half beta2 = 0.933f;
//     half epsilon = 1e-5f;
//     uint64_t epoch = rand() % 10;
//
//     vector<shared_ptr<Layer>> layers;
//
//     layers.push_back(
//         make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, networkFeatureIn.getDescriptor().getDimensions()));
//     layers.push_back(make_shared<NoOpLayer>());
//     shared_ptr<FullyConnected> fullyConnectedLayer = make_shared<FullyConnected>(numOutputFeatures, hasBias);
//     layers.push_back(fullyConnectedLayer);
//     layers.push_back(make_shared<NoOpLayer>());
//     layers.push_back(make_shared<NetworkOutput>(cpuPlacement));
//
//     Stream dataStream = layers.front()->getStream();
//
//     LayerTestHelper::connectAndInitializeNetwork(layers);
//
//     Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs());
//     Tensor featureOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureOutputs());
//     Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
//     Tensor errorOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
//     Tensor weights = fullyConnectedLayer->getWeights();
//     featureInput.fillRandom(-2.0, 2.0, dataStream);
//     errorInput.fillRandom(-2.0, 2.0, dataStream);
//     weights.fillRandom(-2.0, 2.0, dataStream);
//     dataStream.synchronize();
//     shared_ptr<Adam> adam = make_shared<Adam>(fullyConnectedLayer,
//                                               alpha,
//                                               beta1,
//                                               beta2,
//                                               epsilon,
//                                               fullyConnectedLayer->getErrorInputs()[0],
//                                               fullyConnectedLayer->getErrorOutputs()[0]);
//     adam->initialize();
//     fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(adam));
//
//     Tensor featureInput_h = featureInput.clone(cpuPlacement);
//     Tensor featureOutput_h = featureOutput.clone(cpuPlacement);
//     Tensor featureOutputGpu_h = featureOutput_h.clone();
//     Tensor errorInput_h = errorInput.clone(cpuPlacement);
//     Tensor errorOutput_h = errorOutput.clone(cpuPlacement);
//     Tensor errorOutputGpu_h = errorOutput_h.clone();
//     Tensor weights_h = weights.clone(cpuPlacement);
//     Tensor weightsGpu_h = weights_h.clone();
//     Tensor weightsGradient = adam->getWeightsGradient();
//     Tensor weightsGradient_h = weightsGradient.clone(cpuPlacement);
//     Tensor weightsGradientGpu_h = weightsGradient_h.clone();
//     half *featureInMem_h = featureInput_h.getMemPtr<half>();
//     half *featureOutMem_h = featureOutput_h.getMemPtr<half>();
//     half *featureOutGpuMem_h = featureOutputGpu_h.getMemPtr<half>();
//     half *errorInMem_h = errorInput_h.getMemPtr<half>();
//     half *errorOutMem_h = errorOutput_h.getMemPtr<half>();
//     half *errorOutGpuMem_h = errorOutputGpu_h.getMemPtr<half>();
//     half *weightsMem_h = weights_h.getMemPtr<half>();
//     half *weightsGpuMem_h = weightsGpu_h.getMemPtr<half>();
//     half *weightsGradientMem_h = weightsGradient_h.getMemPtr<half>();
//     half *weightsGradientGpuMem_h = weightsGradientGpu_h.getMemPtr<half>();
//     featureInput_h.copyFromAsync(featureInput, dataStream);
//     errorInput_h.copyFromAsync(errorInput, dataStream);
//     weights_h.copyFromAsync(weights, dataStream);
//
//     Optional<Tensor> biases = fullyConnectedLayer->getBiases();
//     Tensor biases_h;
//     Tensor biasesGpu_h;
//     half *biasesMem_h;
//     half *biasesGpuMem_h;
//     Optional<Tensor> biasesGradient = adam->getBiasesGradient();
//     Tensor biasesGradient_h;
//     Tensor biasesGradientGpu_h;
//     half *biasesGradientMem_h;
//     half *biasesGradientGpuMem_h;
//     Tensor projectedBiases_h;
//     if (hasBias) {
//         ASSERT_TRUE(biases.isPresent());
//         ASSERT_TRUE(biasesGradient.isPresent());
//         biases.get().fillRandom(-2.0, 2.0, dataStream);
//         biases_h = biases.get().clone(cpuPlacement);
//         biases_h.copyFromAsync(biases, dataStream);
//         biasesGpu_h = biases_h.clone();
//         biasesMem_h = biases_h.getMemPtr<half>();
//         biasesGpuMem_h = biasesGpu_h.getMemPtr<half>();
//         biasesGradient_h = biasesGradient.get().clone(cpuPlacement);
//         biasesGradientGpu_h = biasesGradient_h.clone();
//         biasesGradientMem_h = biasesGradient_h.getMemPtr<half>();
//         biasesGradientGpuMem_h = biasesGradientGpu_h.getMemPtr<half>();
//     }
//
//     Tensor weightsUpdate = adam->getWeightsUpdate();
//     Tensor weightsUpdate_h = weightsUpdate.clone(cpuPlacement);
//     half *weightsUpdateMem_h = weightsUpdate_h.getMemPtr<half>();
//     Tensor weightsUpdateGpu_h = weightsUpdate_h.clone();
//     half *weightsUpdateGpuMem_h = weightsUpdateGpu_h.getMemPtr<half>();
//     Optional<Tensor> biasesUpdate = adam->getBiasesUpdate();
//     Tensor biasesUpdate_h;
//     half *biasesUpdateMem_h;
//     Tensor biasesUpdateGpu_h;
//     half *biasesUpdateGpuMem_h;
//     ASSERT_EQ(biasesUpdate.isPresent(), biases.isPresent());
//     if (biasesUpdate.isPresent()) {
//         biasesUpdate_h = biasesUpdate.get().clone(cpuPlacement);
//         biasesUpdateMem_h = biasesUpdate_h.getMemPtr<half>();
//         biasesUpdateGpu_h = biasesUpdate_h.clone();
//         biasesUpdateGpuMem_h = biasesUpdateGpu_h.getMemPtr<half>();
//     }
//
//     Tensor m = adam->getM();
//     Tensor m_h = m.clone(cpuPlacement);
//     m_h.fillZero(dataStream);
//     half *mMem_h = m_h.getMemPtr<half>();
//     Tensor mGpu_h = m_h.clone();
//     half *mGpuMem_h = mGpu_h.getMemPtr<half>();
//     Tensor v = adam->getV();
//     Tensor v_h = v.clone(cpuPlacement);
//     v_h.fillZero(dataStream);
//     half *vMem_h = v_h.getMemPtr<half>();
//     Tensor vGpu_h = v_h.clone();
//     half *vGpuMem_h = vGpu_h.getMemPtr<half>();
//
//     Optional<Tensor> mBiases = adam->getMBias();
//     Tensor mBiases_h;
//     half *mBiasesMem_h;
//     Tensor mBiasesGpu_h;
//     half *mBiasesGpuMem_h;
//     ASSERT_EQ(mBiases.isPresent(), biasesGradient.isPresent());
//     if (mBiases.isPresent()) {
//         mBiases_h = mBiases.get().clone(cpuPlacement);
//         mBiasesMem_h = mBiases_h.getMemPtr<half>();
//         mBiasesGpu_h = mBiases_h.clone();
//         mBiasesGpuMem_h = mBiasesGpu_h.getMemPtr<half>();
//     }
//     Optional<Tensor> vBiases = adam->getVBias();
//     Tensor vBiases_h;
//     half *vBiasesMem_h;
//     Tensor vBiasesGpu_h;
//     half *vBiasesGpuMem_h;
//     ASSERT_EQ(vBiases.isPresent(), mBiases.isPresent());
//     if (vBiases.isPresent()) {
//         vBiases_h = vBiases.get().clone(cpuPlacement);
//         vBiasesMem_h = vBiases_h.getMemPtr<half>();
//         vBiasesGpu_h = vBiases_h.clone();
//         vBiasesGpuMem_h = vBiasesGpu_h.getMemPtr<half>();
//     }
//
//     dataStream.synchronize();
//     Stream gradientUpdateStream = adam->getGradientUpdateStream();
//
//     uint32_t numWeights = weightsGradient.getTotalNumElements();
//     uint32_t numBiases = 0;
//
//     uint32_t batchesPerEpoch = 1 + rand() % 10000;
//     float t = 0.0f;
//     unordered_map<string, float> hyperparameters = adam->updateHyperParameters(epoch, rand() % batchesPerEpoch, batchesPerEpoch);
//     ASSERT_EQ(hyperparameters["t"], t);
//
//     // Call forward
//     fullyConnectedLayer->forward(featureInput, false);
//     featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
//     dataStream.synchronize();
//
//     // Verify featureOutput
//     matrixMultiplyCpuHalf(featureInMem_h,
//                           weightsMem_h,
//                           featureOutMem_h,
//                           batchSize,
//                           numInputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           false,
//                           false,
//                           false,
//                           false);
//     // printf("Num input features %d\n", numInputFeatures);
//     verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.02, 0.03f);
//
//     // Call backward
//     fullyConnectedLayer->backward(errorInput);
//     t += 1;
//
//     // Verify errorOutput
//     errorOutputGpu_h.copyFromAsync(errorOutput, dataStream);
//     dataStream.synchronize();
//     matrixMultiplyCpuHalf(errorInMem_h,
//                           weightsMem_h,
//                           errorOutMem_h,
//                           batchSize,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           false,
//                           true,
//                           false,
//                           false);
//     verifyMatricesMatch(errorOutMem_h, errorOutGpuMem_h, batchSize, numInputFeatures, false, 0.4f, 0.03f);
//
//     // Verify weights
//     gradientUpdateStream.synchronize();
//     matrixMultiplyCpuHalf(featureInMem_h,
//                           errorInMem_h,
//                           weightsGradientMem_h,
//                           batchSize,
//                           numInputFeatures,
//                           batchSize,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           true,
//                           false,
//                           false,
//                           false);
//     weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
//
//     if (hasBias) {
//         reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
//         biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.02f);
//     }
//
//     weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     double thresh;
//
//     ASSERT_EQ(adam->getT(), 1.0f);
//
//     // update m, v
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         // if (j == 0)
//         //     printf("m calc %f = %f * %f + (1.0f - %f) * %f\n", (float)(beta1 * mCpu[j] + ((half)1.0f - beta1) *
//         //     weightsGradientMem_h[j]), (float)beta1, (float)mCpu[j], (float)beta1, (float)weightsGradientMem_h[j]);
//         mMem_h[j] = beta1 * mMem_h[j] + ((half)1.0f - beta1) * weightsGradientMem_h[j];
//         // if (j == 0)
//         //     printf("v calc = %f = %f * %f + (1.0 - %f) * (%f * %f)\n", (float)(beta2 * vMem_h[j] + ((half)1.0f - beta2) *
//         //     (weightsGradientMem_h[j] * weightsGradientMem_h[j])),
//         //            (float)beta2, (float)vMem_h[j], (float)((half)1.0f - beta2), (float)weightsGradientMem_h[j],
//         //            (float)weightsGradientMem_h[j]);
//         vMem_h[j] = beta2 * vMem_h[j] + ((half)1.0f - beta2) * (weightsGradientMem_h[j] * weightsGradientMem_h[j]);
//     }
//     for (uint32_t j = 0; j < numBiases; ++j) {
//         mBiasesMem_h[j] = beta1 * mBiasesMem_h[j] + ((half)1.0f - beta1) * biasesGradientMem_h[j];
//         vBiasesMem_h[j] = beta2 * vBiasesMem_h[j] + ((half)1.0f - beta2) * (biasesGradientMem_h[j] * biasesGradientMem_h[j]);
//     }
//
//     // Get m ensure it is correct, then load actual values to maintain test accuracy downstream
//     mGpu_h.copyFromAsync(m, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         thresh = max(0.005, abs((double)mMem_h[j] * .05));
//         if (!(abs((double)(mMem_h[j] - mGpuMem_h[j])) < thresh))
//             printf("mMem_h[%d] %f mGpu[%d] %f\n", j, (float)mMem_h[j], j, (float)mGpuMem_h[j]);
//         ASSERT_LT(abs((double)(mMem_h[j] - mGpuMem_h[j])), thresh);
//     }
//     m_h.copyFromAsync(mGpu_h, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     // Get v ensure it is correct, then load actual values to maintain test accuracy downstream
//     vGpu_h.copyFromAsync(v, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         thresh = max(0.005, abs((double)vMem_h[j] * .05));
//         // printf("v cpu %f gpu %f\n", (float)vMem_h[j], (float)vGpuVector[j]);
//         ASSERT_LT(abs((double)(vMem_h[j] - vGpuMem_h[j])), thresh);
//     }
//     v_h.copyFromAsync(vGpu_h, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     if (hasBias) {
//         // Get mBiases ensure it is correct, then load actual values to maintain test accuracy downstream
//         mBiasesGpu_h.copyFromAsync(mBiases, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             // printf("%f %f\n", (float)mBiasesMem_h[j], (float)mBiasesGpuMem_h[j]);
//             thresh = max(0.03, abs((double)mBiasesMem_h[j] * .02));
//             ASSERT_LT(abs((double)(mBiasesMem_h[j] - mBiasesGpuMem_h[j])), thresh);
//         }
//         mBiases_h.copyFromAsync(mBiasesGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//
//         // Get vBiases ensure it is correct, then load actual values to maintain test accuracy downstream
//         vBiasesGpu_h.copyFromAsync(vBiases, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             thresh = max(0.03, abs((double)vBiasesMem_h[j] * .02));
//             ASSERT_LT(abs((double)(vBiasesMem_h[j] - vBiasesGpuMem_h[j])), thresh);
//         }
//         vBiases_h.copyFromAsync(vBiasesGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//     }
//
//     // Compute weights and biases update values
//     half alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
//     // printf("alphaT = %f = %f * %f / %f, beta1 %f beta2 %f t %f\n",
//     //        (float)alphaT,
//     //        (float)alpha,
//     //        sqrtf(1.0f - powf(beta2, t)),
//     //       (1.0f - powf(beta1, t)),
//     //        (float)beta1,
//     //        (float)beta2,
//     //        t);
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         weightsUpdateMem_h[j] = (-alphaT * mMem_h[j]) / ((half)sqrtf(vMem_h[j]) + epsilon);
//         // if (j == 0)
//         //     printf("CPU weightUpdate = %f / %f = %f      alphaT %f, m %f\n",
//         //            float(-alphaT * mMem_h[j]),
//         //            float(((half)sqrtf(vMem_h[j]) + epsilon)),
//         //            (float)weightsUpdateMem_h[j],
//         //            (float)-alphaT,
//         //            float(mMem_h[j]));
//     }
//     for (uint32_t j = 0; j < numBiases; ++j)
//         biasesUpdateMem_h[j] = (-alphaT * mBiasesMem_h[j]) / ((half)sqrtf(vBiasesMem_h[j]) + epsilon);
//
//     // Get weightsUpdate ensure it is correct
//     weightsUpdateGpu_h.copyFromAsync(weightsUpdate, gradientUpdateStream);
//     if (numBiases > 0)
//         biasesUpdateGpu_h.copyFromAsync(biasesUpdate, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         // printf("[%d] cpu: %f gpu %f      -alphaT %f mMem_h %f sqrt(vMem_h) %f epsilon %f\n", j, (float)weightsUpdateMem_h[j],
//         // (float)weightsUpdateGpuMem_h[j], (float)-alphaT, (float)mMem_h[j], sqrtf(vMem_h[j]), (float)epsilon);
//         ASSERT_LT(abs((double)(weightsUpdateMem_h[j] - weightsUpdateGpuMem_h[j])), 0.0002);
//     }
//     for (uint32_t j = 0; j < numBiases; ++j) {
//         ASSERT_LT(abs((double)(biasesUpdateMem_h[j] - biasesUpdateGpuMem_h[j])), 0.0002);
//     }
//
//     // update weights
//     // batchSize is only used for math, it doesn't need to match the tensor dimension in this test.
//     adam->updateWeights(weights, biases, batchSize);
//     weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     // Check that the weights have been updated to the proper values
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         half updatedWeight =
//             weightsMem_h[j] + (half)((float)weightsUpdateMem_h[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
//         // printf("%f %f\n", (float)updatedWeight, (float)weightsGpuVector[j]);
//         thresh = max(0.001, abs((double)updatedWeight * .01));
//         ASSERT_LT(abs((double)(updatedWeight - weightsGpuMem_h[j])), thresh);
//     }
//     weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
//
//     if (biases.isPresent()) {
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             half updatedBias =
//                 biasesMem_h[j] + (half)((float)biasesUpdateMem_h[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
//             // printf("%f %f\n", (float)updatedBias, (float)biasesGpuMem_h[j]);
//             thresh = max(0.001, abs((double)updatedBias * .01));
//             ASSERT_LT(abs((double)(updatedBias - biasesGpuMem_h[j])), thresh);
//         }
//         biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
//     }
//     gradientUpdateStream.synchronize();
//
//     ////////////////////////////////////////////
//
//     // Re-randomize the inputs
//     featureInput.fillRandom(-2.0, 2.0, dataStream);
//     featureInput_h.copyFromAsync(featureInput, dataStream);
//     errorInput.fillRandom(-2.0, 2.0, dataStream);
//     errorInput_h.copyFromAsync(errorInput, dataStream);
//
//     // Call forward
//     fullyConnectedLayer->forward(featureInput, false);
//     featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
//     dataStream.synchronize();
//
//     // Verify featureOutput
//     matrixMultiplyCpuHalf(featureInMem_h,
//                           weightsMem_h,
//                           featureOutMem_h,
//                           batchSize,
//                           numInputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           false,
//                           false,
//                           false,
//                           false);
//     // printf("Num input features %d\n", numInputFeatures);
//     verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.02, 0.03f);
//
//     // Call backward
//     fullyConnectedLayer->backward(errorInput);
//     t += 1;
//
//     // Verify errorOutput
//     errorOutputGpu_h.copyFromAsync(errorOutput, dataStream);
//     dataStream.synchronize();
//     matrixMultiplyCpuHalf(errorInMem_h,
//                           weightsMem_h,
//                           errorOutMem_h,
//                           batchSize,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           false,
//                           true,
//                           false,
//                           false);
//     verifyMatricesMatch(errorOutMem_h, errorOutGpuMem_h, batchSize, numInputFeatures, false, 0.4f, 0.03f);
//
//     // Verify weights
//     gradientUpdateStream.synchronize();
//     matrixMultiplyCpuHalf(featureInMem_h,
//                           errorInMem_h,
//                           weightsGradientMem_h,
//                           batchSize,
//                           numInputFeatures,
//                           batchSize,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           true,
//                           false,
//                           false,
//                           false);
//     weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
//
//     if (hasBias) {
//         reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
//         biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.02f);
//     }
//
//     weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     ASSERT_EQ(adam->getT(), 2.0f);
//
//     // update m, v
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         // if (j == 0)
//         //     printf("m calc %f = %f * %f + (1.0f - %f) * %f\n", (float)(beta1 * mCpu[j] + ((half)1.0f - beta1) *
//         //     weightsGradientMem_h[j]), (float)beta1, (float)mCpu[j], (float)beta1, (float)weightsGradientMem_h[j]);
//         mMem_h[j] = beta1 * mMem_h[j] + ((half)1.0f - beta1) * weightsGradientMem_h[j];
//         // if (j == 0)
//         //     printf("v calc = %f = %f * %f + (1.0 - %f) * (%f * %f)\n", (float)(beta2 * vMem_h[j] + ((half)1.0f - beta2) *
//         //     (weightsGradientMem_h[j] * weightsGradientMem_h[j])),
//         //            (float)beta2, (float)vMem_h[j], (float)((half)1.0f - beta2), (float)weightsGradientMem_h[j],
//         //            (float)weightsGradientMem_h[j]);
//         vMem_h[j] = beta2 * vMem_h[j] + ((half)1.0f - beta2) * (weightsGradientMem_h[j] * weightsGradientMem_h[j]);
//     }
//     for (uint32_t j = 0; j < numBiases; ++j) {
//         mBiasesMem_h[j] = beta1 * mBiasesMem_h[j] + ((half)1.0f - beta1) * biasesGradientMem_h[j];
//         vBiasesMem_h[j] = beta2 * vBiasesMem_h[j] + ((half)1.0f - beta2) * (biasesGradientMem_h[j] * biasesGradientMem_h[j]);
//     }
//
//     // Get m ensure it is correct, then load actual values to maintain test accuracy downstream
//     mGpu_h.copyFromAsync(m, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         thresh = max(0.005, abs((double)mMem_h[j] * .05));
//         if (!(abs((double)(mMem_h[j] - mGpuMem_h[j])) < thresh))
//             printf("mMem_h[%d] %f mGpu[%d] %f\n", j, (float)mMem_h[j], j, (float)mGpuMem_h[j]);
//         ASSERT_LT(abs((double)(mMem_h[j] - mGpuMem_h[j])), thresh);
//     }
//     m_h.copyFromAsync(mGpu_h, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     // Get v ensure it is correct, then load actual values to maintain test accuracy downstream
//     vGpu_h.copyFromAsync(v, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         thresh = max(0.005, abs((double)vMem_h[j] * .05));
//         // printf("v cpu %f gpu %f\n", (float)vMem_h[j], (float)vGpuVector[j]);
//         ASSERT_LT(abs((double)(vMem_h[j] - vGpuMem_h[j])), thresh);
//     }
//     v_h.copyFromAsync(vGpu_h, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     if (hasBias) {
//         // Get mBiases ensure it is correct, then load actual values to maintain test accuracy downstream
//         mBiasesGpu_h.copyFromAsync(mBiases, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             // printf("%f %f\n", (float)mBiasesMem_h[j], (float)mBiasesGpuMem_h[j]);
//             thresh = max(0.03, abs((double)mBiasesMem_h[j] * .02));
//             ASSERT_LT(abs((double)(mBiasesMem_h[j] - mBiasesGpuMem_h[j])), thresh);
//         }
//         mBiases_h.copyFromAsync(mBiasesGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//
//         // Get vBiases ensure it is correct, then load actual values to maintain test accuracy downstream
//         vBiasesGpu_h.copyFromAsync(vBiases, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             thresh = max(0.03, abs((double)vBiasesMem_h[j] * .02));
//             ASSERT_LT(abs((double)(vBiasesMem_h[j] - vBiasesGpuMem_h[j])), thresh);
//         }
//         vBiases_h.copyFromAsync(vBiasesGpu_h, gradientUpdateStream);
//         gradientUpdateStream.synchronize();
//     }
//
//     // Compute weights and biases update values
//     alphaT = alpha * (half)sqrtf(1.0f - powf(beta2, t)) / (half)(1.0f - powf(beta1, t));
//     printf("alphaT = %f = %f * %f / %f, beta1 %f beta2 %f t %f\n",
//            (float)alphaT,
//            (float)alpha,
//            sqrtf(1.0f - powf(beta2, t)),
//            (1.0f - powf(beta1, t)),
//            (float)beta1,
//            (float)beta2,
//            t);
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         weightsUpdateMem_h[j] = (-alphaT * mMem_h[j]) / ((half)sqrtf(vMem_h[j]) + epsilon);
//         if (j == 0)
//             printf("CPU weightUpdate = %f / %f = %f      alphaT %f, m %f\n",
//                    float(-alphaT * mMem_h[j]),
//                    float(((half)sqrtf(vMem_h[j]) + epsilon)),
//                    (float)weightsUpdateMem_h[j],
//                    (float)-alphaT,
//                    float(mMem_h[j]));
//     }
//     for (uint32_t j = 0; j < numBiases; ++j)
//         biasesUpdateMem_h[j] = (-alphaT * mBiasesMem_h[j]) / ((half)sqrtf(vBiasesMem_h[j]) + epsilon);
//
//     // Get weightsUpdate ensure it is correct
//     weightsUpdateGpu_h.copyFromAsync(weightsUpdate, gradientUpdateStream);
//     if (numBiases > 0)
//         biasesUpdateGpu_h.copyFromAsync(biasesUpdate, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         // printf("[%d] cpu: %f gpu %f      -alphaT %f mMem_h %f sqrt(vMem_h) %f epsilon %f\n", j, (float)weightsUpdateMem_h[j],
//         // (float)weightsUpdateGpuMem_h[j], (float)-alphaT, (float)mMem_h[j], sqrtf(vMem_h[j]), (float)epsilon);
//         ASSERT_LT(abs((double)(weightsUpdateMem_h[j] - weightsUpdateGpuMem_h[j])), 0.0002);
//     }
//     for (uint32_t j = 0; j < numBiases; ++j) {
//         ASSERT_LT(abs((double)(biasesUpdateMem_h[j] - biasesUpdateGpuMem_h[j])), 0.0002);
//     }
//
//     // update weights
//     // batchSize is only used for math, it doesn't need to match the tensor dimension in this test.
//     adam->updateWeights(weights, biases, batchSize);
//     weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
//     gradientUpdateStream.synchronize();
//
//     // Check that the weights have been updated to the proper values
//     for (uint32_t j = 0; j < numWeights; ++j) {
//         half updatedWeight =
//             weightsMem_h[j] + (half)((float)weightsUpdateMem_h[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
//         // printf("%f %f\n", (float)updatedWeight, (float)weightsGpuVector[j]);
//         thresh = max(0.001, abs((double)updatedWeight * .01));
//         ASSERT_LT(abs((double)(updatedWeight - weightsGpuMem_h[j])), thresh);
//     }
//     weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
//
//     if (biases.isPresent()) {
//         for (uint32_t j = 0; j < numBiases; ++j) {
//             half updatedBias =
//                 biasesMem_h[j] + (half)((float)biasesUpdateMem_h[j] / (float)(Loss::getLossScalingFactor() * (float)batchSize));
//             // printf("%f %f\n", (float)updatedBias, (float)biasesGpuMem_h[j]);
//             thresh = max(0.001, abs((double)updatedBias * .01));
//             ASSERT_LT(abs((double)(updatedBias - biasesGpuMem_h[j])), thresh);
//         }
//         biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
//     }
//     gradientUpdateStream.synchronize();
//
//     // Re-randomize the inputs
//     featureInput.fillRandom(-2.0, 2.0, dataStream);
//     featureInput_h.copyFromAsync(featureInput, dataStream);
//     errorInput.fillRandom(-2.0, 2.0, dataStream);
//     errorInput_h.copyFromAsync(errorInput, dataStream);
//
//     // Call forward, this time in inference mode
//     fullyConnectedLayer->forward(featureInput, true);
//     featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
//     dataStream.synchronize();
//
//     // Verify featureOutput
//     matrixMultiplyCpuHalf(featureInMem_h,
//                           weightsMem_h,
//                           featureOutMem_h,
//                           batchSize,
//                           numInputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numInputFeatures,
//                           numOutputFeatures,
//                           numOutputFeatures,
//                           false,
//                           false,
//                           false,
//                           false);
//     // printf("Num input features %d\n", numInputFeatures);
//     verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.02, 0.03f);
//
//     ///////////////////////////////////////////////////////////
// }
//
///* FIXME: when FC supports FP32 test FP32 adam:
// TEST(AdamTest, updateWeightsFP32) {
//     shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = constructTrainableLayer();
//     Adam adam(trainableLayer, 0.1f, 0.9f, 0.999f, 1e-8f, trainableLayer->getErrorInputs()[0], Optional<Tensor>::empty());
// }
//*/
//
