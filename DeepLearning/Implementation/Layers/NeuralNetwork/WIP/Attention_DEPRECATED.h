#include <optional>
// #pragma once
//
// #include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"
// #include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
// #include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"
// #include "Utilities/TensorOperations/GpuAttention/AttentionDescriptor.h"
//
// #include <memory>
//
//  namespace ThorImplementation {
//
//  class Attention : public TrainableWeightsBiasesLayer {
//    public:
//
//
//     virtual ~Attention() {}
//
//     Attention(const uint32_t numHeads,
//               const uint32_t numOutputFeatures,
//               const bool hasBias,
//               const int32_t qSize,
//               const int32_t kSize,
//               const int32_t vSize,
//               const int32_t qProjSize,
//               const int32_t kProjSize,
//               const int32_t vProjSize,
//               const int32_t oProjSize,
//               const int32_t qoMaxSeqLength,
//               const int32_t kvMaxSeqLength,
//               const int32_t maxBatchSize,
//               const int32_t maxBeamSize,
//               const AttentionDescriptor::QUERY_MAPPING queryMapping = AttentionDescriptor::QUERY_MAPPING::ONE_TO_ONE,
//               const float attentionDropOutProbability = 0.0f,
//               const float postAttentionDropOutProbability = 0.0f,
//               const int64_t stampedId = -1)
//         : TrainableWeightsBiasesLayer(hasBias, stampedId),
//           numHeads(numHeads),
//           numOutputFeatures(numOutputFeatures),
//           hasBias(hasBias),
//           qSize(qSize),
//           kSize(kSize),
//           vSize(vSize),
//           qProjSize(qProjSize),
//           kProjSize(kProjSize),
//           vProjSize(vProjSize),
//           oProjSize(oProjSize),
//           qoMaxSeqLength(qoMaxSeqLength),
//           kvMaxSeqLength(kvMaxSeqLength),
//           maxBatchSize(maxBatchSize),
//           maxBeamSize(maxBeamSize),
//           queryMapping(queryMapping),
//           attentionDropOutProbability(attentionDropOutProbability),
//           postAttentionDropOutProbability(postAttentionDropOutProbability) {}
//
//     Attention(SharedWeightsPackage sharedWeightsPackage,
//               const uint32_t numHeads,
//               const bool hasBias,
//               const int32_t qSize,
//               const int32_t kSize,
//               const int32_t vSize,
//               const int32_t qProjSize,
//               const int32_t kProjSize,
//               const int32_t vProjSize,
//               const int32_t oProjSize,
//               const int32_t qoMaxSeqLength,
//               const int32_t kvMaxSeqLength,
//               const int32_t maxBatchSize,
//               const int32_t maxBeamSize,
//               const AttentionDescriptor::QUERY_MAPPING queryMapping = AttentionDescriptor::QUERY_MAPPING::ONE_TO_ONE,
//               const float attentionDropOutProbability = 0.0f,
//               const float postAttentionDropOutProbability = 0.0f,
//               int64_t stampedId = -1)
//         : TrainableWeightsBiasesLayer(sharedWeightsPackage, stampedId),
//           numHeads(numHeads),
//           numOutputFeatures(sharedWeightsPackage.weights.getDescriptor().getDimensions()[1],
//           hasBias(hasBias),
//           qSize(qSize),
//           kSize(kSize),
//           vSize(vSize),
//           qProjSize(qProjSize),
//           kProjSize(kProjSize),
//           vProjSize(vProjSize),
//           oProjSize(oProjSize),
//           qoMaxSeqLength(qoMaxSeqLength),
//           kvMaxSeqLength(kvMaxSeqLength),
//           maxBatchSize(maxBatchSize),
//           maxBeamSize(maxBeamSize),
//           queryMapping(queryMapping),
//           attentionDropOutProbability(attentionDropOutProbability),
//           postAttentionDropOutProbability(postAttentionDropOutProbability) {}
//
//     virtual std::optional<Tensor> createFeatureOutputTensor() {
//         THOR_THROW_IF_FALSE(!featureInputs.empty());
//         THOR_THROW_IF_FALSE(featureInputs.back().has_value());
//
//         return Tensor(featureInputs.back().value().getPlacement(),
//                       TensorDescriptor(TensorDescriptor::DataType::FP16,
//                                        {featureInputs[0].value().getDescriptor().getDimensions()[0], numOutputFeatures}));
//     }
//
//     virtual void createWeightsIfNecessary() {
//         if (!usingSharedWeights && !weights.isInitialized()) {
//             std::vector<unsigned long> weightsDimensions;
//             std::optional<Tensor> maybeAFeatureInput = getFirstPresentTensor(featureInputs);
//             THOR_THROW_IF_FALSE(maybeAFeatureInput.has_value());
//             Tensor aFeatureInput = maybeAFeatureInput.value();
//             THOR_THROW_IF_FALSE(aFeatureInput.getDimensions().size() == 2);
//             weightsDimensions.push_back(aFeatureInput.getDescriptor().getDimensions()[1]);
//             weightsDimensions.push_back(numOutputFeatures);
//             TensorDescriptor weightsDescriptor = TensorDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
//             weights = Tensor(aFeatureInput.getPlacement(), weightsDescriptor);
//             if (hasBias) {
//                 biases = Tensor(aFeatureInput.getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP16, {numOutputFeatures}));
//             }
//         }
//     }
//
//     virtual void compile() {
//         int gpuNum;
//         THOR_THROW_IF_FALSE(!featureInputs.empty());
//         THOR_THROW_IF_FALSE(featureInputs[0].has_value());
//         THOR_THROW_IF_FALSE(featureInputs[0].value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
//         THOR_THROW_IF_FALSE(!streams.empty());
//         gpuNum = featureInputs[0].value().getPlacement().getDeviceNum();
//         ScopedGpu scopedGpu(gpuNum);
//
//         cudnnStatus_t cudnnStatus;
//         batchSize = featureInputs[0].value().getDescriptor().getDimensions()[0];
//         numInputFeatures = featureInputs[0].value().getDescriptor().getDimensions()[1];
//
//         attentionDescriptor = AttentionDescriptor(
//             numHeads,
//             hasBias,
//             qSize,
//             kSize,
//             vSize,
//             qProjSize,
//             kProjSize,
//             vProjSize,
//             oProjSize,
//             qoMaxSeqLength,
//             kvMaxSeqLength,
//             maxBatchSize,
//             maxBeamSize,
//             queryMapping,
//             attentionDropOutProbability,
//             postAttentionDropOutProbability,
//             featureInputs[0].value().getPlacement(),
//             streams[0].getCudnnHandle());
//
//
//         if (!isInferenceOnly()) {
//
//         }
//
//         if (hasBias) {
//         }
//     }
//
//     virtual void setOptimizer(std::optional<std::shared_ptr<Optimizer>> optimizer) {
//         TrainableWeightsBiasesLayer::setOptimizer(optimizer);
//
//         if (!isInferenceOnly()) {
//             THOR_THROW_IF_FALSE(optimizer.has_value());
//             std::optional<Tensor> anErrorInput = getFirstPresentTensor(errorInputs);
//             THOR_THROW_IF_FALSE(anErrorInput.has_value());
//             biasBatchReduce = std::unique_ptr<BatchReduce>(new BatchReduce(batchSize,
//                                                                            batchSize,
//                                                                            anErrorInput.value().getDimensions()[1],
//                                                                            true,
//                                                                            false,
//                                                                            ThorImplementation::TensorDescriptor::DataType::FP16,
//                                                                            ThorImplementation::TensorDescriptor::DataType::FP16,
//                                                                            optimizer.value()->getGradientUpdateStream(),
//                                                                            false));
//         }
//     }
//
//     virtual void infer(std::optional<Tensor> inputTensor,
//                        std::optional<Tensor> outputTensor,
//                        Stream stream,
//                        unsigned int connectionNumber,
//                        Tensor weights,
//                        std::optional<Tensor> biases) {
//         THOR_THROW_IF_FALSE(inputTensor.has_value());
//         THOR_THROW_IF_FALSE(outputTensor.has_value());
//         THOR_THROW_IF_FALSE(inputTensor.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
//
//         CublasMatrixMultiply::instance().multiply(inputTensor,
//                                                   weights,
//                                                   outputTensor,
//                                                   workspaceForward,
//                                                   batchSize,
//                                                   numInputFeatures,
//                                                   numInputFeatures,
//                                                   numOutputFeatures,
//                                                   false,
//                                                   false,
//                                                   false,
//                                                   false,
//                                                   TensorDescriptor::DataType::FP16,
//                                                   stream);
//
//         if (hasBias) {
//             THOR_THROW_IF_FALSE(biases.has_value());
//             THOR_THROW_IF_FALSE(cudnnBiasDescriptor.has_value());
//             THOR_THROW_IF_FALSE(cudnnFeatureOutputDescriptor.has_value());
//
//             cudnnAddTensor(stream.getCudnnHandle(),
//                            &ALPHA_NO_SCALE,
//                            cudnnBiasDescriptor.value(),
//                            biases.value().getMemPtr(),
//                            &BETA_ACCUMULATE,
//                            cudnnFeatureOutputDescriptor.value(),
//                            outputTensor.value().getMemPtr());
//         }
//     }
//
//     // Note: backward() syncs gradient stream with data stream prior to calling this to ensure error in is ready at end of gradient
//     stream virtual void backProp(std::optional<Tensor> dataIn,
//                           std::optional<Tensor> errorIn,
//                           std::optional<Tensor> errorOut,
//                           Stream dataStream,
//                           unsigned int connectionNumber,
//                           bool accumulateGradient) {
//         THOR_THROW_IF_FALSE(errorIn.has_value());
//         THOR_THROW_IF_FALSE(errorIn.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
//
//         if (errorOut.has_value()) {
//             THOR_THROW_IF_FALSE(dataStream.isInitialized());
//
//             CublasMatrixMultiply::instance().multiply(errorIn,
//                                                       weights,
//                                                       errorOut,
//                                                       workspaceBackwardData,
//                                                       batchSize,
//                                                       numOutputFeatures,
//                                                       numInputFeatures,
//                                                       numOutputFeatures,
//                                                       false,
//                                                       true,
//                                                       false,
//                                                       false,
//                                                       TensorDescriptor::DataType::FP16,
//                                                       dataStream);
//         }
//
//         if (!isInferenceOnly()) {
//             THOR_THROW_IF_FALSE(optimizer.has_value());
//
//             // backward() syncs gradient stream with data stream prior to calling this to ensure error in is ready at end of gradient
//             stream optimizer.value()->computeWeightsUpdate(dataIn, errorIn, accumulateGradient);
//
//             // weights update cannot be applied to weights until errorOut has been computed since weights are part of that computation
//             // so to enforce this gradientUpdateStream says that gradient is not ready to be applied until both errorOut and gradient are
//             // computed
//             optimizer.value()->getGradientUpdateStream().waitEvent(dataStream.putEvent());
//             // Now at the end of gradientUpdateStream errorOut and gradients are ready from the updates for this connection.
//
//             // Upon processing the last connection, schedule the upate to the weights memory.
//             if (stillWaitingForErrorInputTensors.empty()) {
//                 optimizer.value()->updateWeights(weights, biases, batchSize);
//             }
//
//             // weights will be updated at the current end of the gradientUpdateStream
//             // so Forward() must wait until gradientUpdateStream is finished.
//             // This is accomplished in TrainableWeightsBiasesLayer::forward().
//         }
//     }
//
//     // Compute the weights gradient for the specified connection number, accumulate as necessary.
//     // This computation runs on optimizer.gradientUpdateStream.
//     virtual void computeWeightsGradient(std::optional<Tensor> weightsGradient,
//                                         std::optional<Tensor> biasesGradient,
//                                         std::optional<Tensor> featureIn,
//                                         std::optional<Tensor> errorIn,
//                                         Stream gradientUpdateStream,
//                                         bool accumulateGradient) {
//         // Ensure all memory properly allocated
//         THOR_THROW_IF_FALSE(weightsGradient.has_value());
//         THOR_THROW_IF_FALSE(weightsGradient.value().getDescriptor() == weights.getDescriptor());
//         THOR_THROW_IF_FALSE(weightsGradient.value().getPlacement() == weights.getPlacement());
//         THOR_THROW_IF_FALSE(weightsGradient.value().getMemPtr() != weights.getMemPtr());
//         if (hasBias) {
//             THOR_THROW_IF_FALSE(biasesGradient.has_value());
//             THOR_THROW_IF_FALSE(biases.has_value());
//             THOR_THROW_IF_FALSE(biasesGradient.value().getDescriptor() == biasesGradient.value().getDescriptor());
//             THOR_THROW_IF_FALSE(biasesGradient.value().getMemPtr() != biases.value().getMemPtr());
//             THOR_THROW_IF_FALSE(biasesGradient.value().getPlacement() == biases.value().getPlacement());
//         } else {
//             THOR_THROW_IF_FALSE(!biasesGradient.has_value());
//         }
//
//         if (!errorIn.has_value())
//             return;
//         THOR_THROW_IF_FALSE(featureIn.has_value());
//
//         CublasMatrixMultiply::instance().multiply(featureIn,
//                                                   errorIn,
//                                                   weightsGradient,
//                                                   workspaceBackwardWeights,
//                                                   batchSize,
//                                                   numInputFeatures,
//                                                   batchSize,
//                                                   numOutputFeatures,
//                                                   true,
//                                                   false,
//                                                   accumulateGradient,
//                                                   false,
//                                                   TensorDescriptor::DataType::FP16,
//                                                   gradientUpdateStream);
//
//         if (hasBias) {
//             biasBatchReduce->reduce(errorIn, biasesGradient, accumulateGradient);
//         }
//     }
//
//     uint64_t flopsPerConnectionPerExample() {
//         std::optional<Tensor> anyFeatureInput = getFirstPresentTensor(featureInputs);
//         std::optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
//         THOR_THROW_IF_FALSE(anyFeatureInput.has_value());
//         THOR_THROW_IF_FALSE(anyFeatureOutput.has_value());
//         uint64_t flops = 2 * numInputFeatures * numOutputFeatures - numOutputFeatures;
//         if (hasBias)
//             flops += numOutputFeatures;
//         return flops;
//     }
//
//     uint64_t flopsPerGradientUpdatePerExample() {
//         std::optional<Tensor> anyFeatureInput = getFirstPresentTensor(featureInputs);
//         std::optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
//         THOR_THROW_IF_FALSE(anyFeatureInput.has_value());
//         THOR_THROW_IF_FALSE(anyFeatureOutput.has_value());
//         uint64_t flops = numInputFeatures * numOutputFeatures;
//         if (hasBias)
//             flops += numOutputFeatures;
//         return flops;
//     }
//
//     virtual uint64_t floatingPointOperationsPerExampleForward() {
//         uint32_t connectionMultiplier = 0;
//         for (uint32_t i = 0; i < featureInputs.size(); ++i) {
//             if (featureInputs[i].has_value())
//                 connectionMultiplier += 1;
//         }
//
//         return connectionMultiplier * flopsPerConnectionPerExample();
//     }
//
//     virtual uint64_t floatingPointOperationsPerExampleBackward() {
//         if (!isInferenceOnly())
//             return 0;
//
//         uint32_t connectionMultiplier = 0;
//         uint32_t sums = 0;
//         for (uint32_t i = 0; i < errorInputs.size(); ++i) {
//             if (errorInputs[i].has_value()) {
//                 if (connectionMultiplier == 0)
//                     connectionMultiplier += 1;
//                 else
//                     sums += 1;
//             }
//         }
//         for (uint32_t i = 0; i < errorOutputs.size(); ++i) {
//             if (errorOutputs[i].has_value())
//                 connectionMultiplier += 1;
//         }
//
//         std::optional<Tensor> anyErrorInput = getFirstPresentTensor(errorInputs);
//         THOR_THROW_IF_FALSE(anyErrorInput.has_value());
//
//         return connectionMultiplier * flopsPerConnectionPerExample() +
//                (sums * anyErrorInput.value().getDescriptor().getTotalNumElements()) / batchSize + flopsPerGradientUpdatePerExample();
//     }
//
//    private:
//     static const float ALPHA_NO_SCALE;
//     static const float BETA_CLEAR;
//     static const float BETA_ACCUMULATE;
//
//     const uint32_t numHeads;
//     const bool hasBias;
//     const int32_t qSize;
//     const int32_t kSize;
//     const int32_t vSize;
//     const int32_t qProjSize;
//     const int32_t kProjSize;
//     const int32_t vProjSize;
//     const int32_t oProjSize;
//     const int32_t qoMaxSeqLength;
//     const int32_t kvMaxSeqLength;
//     const int32_t maxBatchSize;
//     const int32_t maxBeamSize;
//     const AttentionDescriptor::QUERY_MAPPING queryMapping;
//     const float attentionDropOutProbability;
//     const float postAttentionDropOutProbability;
//
//     std::optional<AttentionDescriptor> attentionDescriptor;
//
//     void createBiasesCudnnTensorDescriptor() {
//         cudnnStatus_t cudnnStatus;
//         cudnnTensorDescriptor_t descriptor;
//
//         cudnnStatus = cudnnCreateTensorDescriptor(&descriptor);
//         THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
//         std::optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
//         THOR_THROW_IF_FALSE(anyFeatureOutput.has_value());
//         uint32_t numOutputFeatures = anyFeatureOutput.value().getDescriptor().getDimensions()[1];
//         cudnnStatus = cudnnSetTensor4dDescriptor(descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, numOutputFeatures, 1, 1);
//         THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
//
//         cudnnBiasDescriptor = descriptor;
//     }
//
//     void createFeatureOutputCudnnTensorDescriptor() {
//         cudnnStatus_t cudnnStatus;
//         cudnnTensorDescriptor_t descriptor;
//
//         cudnnStatus = cudnnCreateTensorDescriptor(&descriptor);
//         THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
//         std::optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
//         THOR_THROW_IF_FALSE(anyFeatureOutput.has_value());
//         std::vector<uint64_t> dimensions = anyFeatureOutput.value().getDescriptor().getDimensions();
//         cudnnStatus = cudnnSetTensor4dDescriptor(descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, dimensions[0], dimensions[1], 1, 1);
//         THOR_THROW_IF_FALSE(cudnnStatus == CUDNN_STATUS_SUCCESS);
//
//         cudnnFeatureOutputDescriptor = descriptor;
//     }
//
//     uint32_t numInputFeatures;
//     const uint32_t numOutputFeatures;
//     uint32_t batchSize;
//
//
//     std::optional<cudnnTensorDescriptor_t> cudnnBiasDescriptor;
//     std::optional<cudnnTensorDescriptor_t> cudnnFeatureOutputDescriptor;
//
//     std::optional<Tensor> workspaceForward;
//     std::optional<Tensor> workspaceBackwardData;
//     std::optional<Tensor> workspaceBackwardWeights;
//
//     std::unique_ptr<BatchReduce> biasBatchReduce;
// };
//
// }  // namespace ThorImplementation