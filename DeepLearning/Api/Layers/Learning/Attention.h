/* WIP
#pragma once

#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"

#include <memory>

namespace ThorImplementation {

class Attention : public TrainableWeightsBiasesLayer {
   public:
    enum class QUERY_MAPPING { ONE_TO_ONE = 57, ALL_TO_ONE = 58 };

    virtual ~Attention() {}

    Attention(const uint32_t numHeads,
              const bool hasBias,
              const int32_t qSize,
              const int32_t kSize,
              const int32_t vSize,
              const int32_t qProjSize,
              const int32_t kProjSize,
              const int32_t vProjSize,
              const int32_t oProjSize,
              const int32_t qoMaxSeqLength,
              const int32_t kvMaxSeqLength,
              const int32_t maxBatchSize,
              const int32_t maxBeamSize,
              const QUERY_MAPPING queryMapping = QUERY_MAPPING::ONE_TO_ONE,
              const bool useAttentionDropOut = false,
              const float attentionDropOutProbability = 0.0f,
              const bool usePostAttentionDropOut = false,
              const float postAttentionDropOutProbability = 0.0f,
              const int64_t stampedId = -1)
        : TrainableWeightsBiasesLayer(hasBias, stampedId), numOutputFeatures(numOutputFeatures) {
        this->numHeads = numHeads;
        this->hasBias = hasBias;
        this->qSize = qSize;
        this->kSize = kSize;
        this->vSize = vSize;
        this->qProjSize = qProjSize;
        this->kProjSize = kProjSize;
        this->vProjSize = vProjSize;
        this->oProjSize = oProjSize;
        this->qoMaxSeqLength = qoMaxSeqLength;
        this->kvMaxSeqLength = kvMaxSeqLength;
        this->maxBatchSize = maxBatchSize;
        this->maxBeamSize = maxBeamSize;
        this->queryMapping = queryMapping;
        this->useAttentionDropOut = useAttentionDropOut;
        this->attentionDropOutProbability = attentionDropOutProbability;
        this->usePostAttentionDropOut = usePostAttentionDropOut;
        this->postAttentionDropOutProbability = postAttentionDropOutProbability;
    }

    Attention(SharedWeightsPackage sharedWeightsPackage, int64_t stampedId = -1)
        : TrainableWeightsBiasesLayer(sharedWeightsPackage, stampedId),
          numOutputFeatures(sharedWeightsPackage.weights.getDescriptor().getDimensions()[1]) {}

    virtual Optional<Tensor> createFeatureOutputTensor() {
        assert(!featureInputs.empty());
        assert(featureInputs.back().isPresent());

        return Tensor(featureInputs.back().get().getPlacement(),
                      TensorDescriptor(TensorDescriptor::DataType::FP16,
                                       {featureInputs[0].get().getDescriptor().getDimensions()[0], numOutputFeatures}));
    }

    virtual void createWeightsIfNecessary() {
        if (!usingSharedWeights && !weights.isInitialized()) {
            std::vector<unsigned long> weightsDimensions;
            Optional<Tensor> maybeAFeatureInput = getFirstPresentTensor(featureInputs);
            assert(maybeAFeatureInput.isPresent());
            Tensor aFeatureInput = maybeAFeatureInput.get();
            assert(aFeatureInput.getDimensions().size() == 2);
            weightsDimensions.push_back(aFeatureInput.getDescriptor().getDimensions()[1]);
            weightsDimensions.push_back(numOutputFeatures);
            TensorDescriptor weightsDescriptor = TensorDescriptor(TensorDescriptor::DataType::FP16, weightsDimensions);
            weights = Tensor(aFeatureInput.getPlacement(), weightsDescriptor);
            if (hasBias) {
                biases = Tensor(aFeatureInput.getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP16, {numOutputFeatures}));
            }
        }
    }

    virtual void compile() {
        int gpuNum;
        assert(!featureInputs.empty());
        assert(featureInputs[0].isPresent());
        assert(featureInputs[0].get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(!streams.empty());
        gpuNum = featureInputs[0].get().getPlacement().getDeviceNum();
        ScopedGpu scopedGpu(gpuNum);

        cudnnStatus_t cudnnStatus;
        batchSize = featureInputs[0].get().getDescriptor().getDimensions()[0];
        numInputFeatures = featureInputs[0].get().getDescriptor().getDimensions()[1];

        // FIXME: member objects:
        cudnnAttnDescriptor_t cudnnAttentionDescriptor;
        cudnnDropoutDescriptor_t attentionDropOutDescriptor;
        cudnnDropoutDescriptor_t postAttentionDropOutDescriptor,

            cudnnStatus = cudnnCreateAttnDescriptor(&cudnnAttentionDescriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        uint32_t attentionMode = 0;
        if (this->queryMapping = QUERY_MAPPING::ALL_TO_ONE)
            attentionMode |= CUDNN_ATTN_QUERYMAP_ALL_TO_ONE;
        else
            attentionMode |= CUDNN_ATTN_QUERYMAP_ONE_TO_ONE;
        if (hasBias)
            attentionMode |= CUDNN_ATTN_ENABLE_PROJ_BIASES;
        else
            attentionMode |= CUDNN_ATTN_DISABLE_PROJ_BIASES;
        random_device rd;
        if (useAttentionDropOut) {
            cudnnStatus = cudnnCreateDropoutDescriptor(&attentionDropOutDescriptor);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            size_t randomStateBytes = DropOut::getRandomStateSizeInBytes(streams[0].getCudnnHandle());
            attentionRandomState =
                Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {randomStateBytes}));

            uint64_t seed = rd() + chrono::system_clock::now().time_since_epoch().count() * 10000000;
            seed = Tensor::getThreadIdHash64(seed);
            cudnnStatus = cudnnSetDropoutDescriptor(attentionDropOutDescriptor,
                                                    streams[0].getCudnnHandle(),
                                                    attentionDropOutProbability,
                                                    attentionRandomState.getMemPtr(),
                                                    randomStateBytes,
                                                    seed);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            DropOut::getReservedSpaceSizeInBytes(featureInputs[0].get().getDimensions(), featureInputs[0].get().getDataType());
            attentionDropOutReserveSpace =
                Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {reserveSpaceBytes}));
        }
        if (usePostAttentionDropOut) {
            cudnnStatus = cudnnCreateDropoutDescriptor(&postAttentionDropOutDescriptor);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            size_t randomStateBytes = DropOut::getRandomStateSizeInBytes(streams[0].getCudnnHandle());
            postAttentionRandomState =
                Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {randomStateBytes}));

            uint64_t seed = rd() + chrono::system_clock::now().time_since_epoch().count() * 10000000;
            seed = Tensor::getThreadIdHash64(seed);
            cudnnStatus = cudnnSetDropoutDescriptor(postAttentionDropOutDescriptor,
                                                    streams[0].getCudnnHandle(),
                                                    postAttentionDropOutProbability,
                                                    postAttentionRandomState.getMemPtr(),
                                                    randomStateBytes,
                                                    seed);
            assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
            DropOut::getReservedSpaceSizeInBytes(featureInputs[0].get().getDimensions(), featureInputs[0].get().getDataType());
            postAttentionDropOutReserveSpace =
                Tensor(featureInputs[0].get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::UINT8, {reserveSpaceBytes}));
        }

        cudnnStatus = cudnnSetAttnDescriptor(cudnnAttentionDescriptor,
                                             attentionMode,
                                             numHeads,
                                             1.0,
                                             CUDNN_DATA_HALF,
                                             CUDNN_DATA_HALF,
                                             CUDNN_TENSOR_OP_MATH,
                                             useAttentionDropOut ? attentionDropOutDescriptor : nullptr,
                                             usePostAttentionDropOut ? postAttentionDropOutDescriptor : nullptr,
                                             this->qSize,
                                             this->kSize,
                                             this->vSize,
                                             this->qProjSize,
                                             this->kProjSize,
                                             this->vProjSize,
                                             this->oProjSize,
                                             this->qoMaxSeqLength,
                                             this->kvMaxSeqLength,
                                             this->maxBatchSize,
                                             this->maxBeamSize);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        if (!isInferenceOnly()) {
            CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
                gpuNum, batchSize, numInputFeatures, batchSize, numOutputFeatures, true, false, TensorDescriptor::DataType::FP16);

            uint64_t workspaceBackwardWeightsSizeInBytes =
                CublasMatrixMultiply::instance().getMatrixMultiplyWorkspaceSizeInBytes(gpuNum,
                                                                                       batchSize,
                                                                                       numInputFeatures,
                                                                                       batchSize,
                                                                                       numOutputFeatures,
                                                                                       true,
                                                                                       false,
                                                                                       TensorDescriptor::DataType::FP16,
                                                                                       kernelWillRunOnGpu);
            assert(kernelWillRunOnGpu);

            if (workspaceBackwardWeightsSizeInBytes > 0)
                workspaceBackwardWeights =
                    Tensor(featureInputs.front().get().getPlacement(),
                           TensorDescriptor(TensorDescriptor::DataType::UINT8, {workspaceBackwardWeightsSizeInBytes}));
        }

        if (hasBias) {
            createFeatureOutputCudnnTensorDescriptor();
            createBiasesCudnnTensorDescriptor();
        }
    }

    virtual void setOptimizer(Optional<std::shared_ptr<Optimizer>> optimizer) {
        TrainableWeightsBiasesLayer::setOptimizer(optimizer);

        if (!isInferenceOnly()) {
            assert(optimizer.isPresent());
            Optional<Tensor> anErrorInput = getFirstPresentTensor(errorInputs);
            assert(anErrorInput.isPresent());
            biasBatchReduce = std::unique_ptr<BatchReduce>(new BatchReduce(batchSize,
                                                                           batchSize,
                                                                           anErrorInput.get().getDimensions()[1],
                                                                           true,
                                                                           false,
                                                                           ThorImplementation::TensorDescriptor::DataType::FP16,
                                                                           ThorImplementation::TensorDescriptor::DataType::FP16,
                                                                           optimizer.get()->getGradientUpdateStream(),
                                                                           false));
        }
    }

    virtual void infer(Optional<Tensor> inputTensor,
                       Optional<Tensor> outputTensor,
                       Stream stream,
                       unsigned int connectionNumber,
                       Tensor weights,
                       Optional<Tensor> biases) {
        assert(inputTensor.isPresent());
        assert(outputTensor.isPresent());
        assert(inputTensor.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        CublasMatrixMultiply::instance().multiply(inputTensor,
                                                  weights,
                                                  outputTensor,
                                                  workspaceForward,
                                                  batchSize,
                                                  numInputFeatures,
                                                  numInputFeatures,
                                                  numOutputFeatures,
                                                  false,
                                                  false,
                                                  false,
                                                  false,
                                                  TensorDescriptor::DataType::FP16,
                                                  stream);

        if (hasBias) {
            assert(biases.isPresent());
            assert(cudnnBiasDescriptor.isPresent());
            assert(cudnnFeatureOutputDescriptor.isPresent());

            cudnnAddTensor(stream.getCudnnHandle(),
                           &ALPHA_NO_SCALE,
                           cudnnBiasDescriptor.get(),
                           biases.get().getMemPtr(),
                           &BETA_ACCUMULATE,
                           cudnnFeatureOutputDescriptor.get(),
                           outputTensor.get().getMemPtr());
        }
    }

    // Note: backward() syncs gradient stream with data stream prior to calling this to ensure error in is ready at end of gradient stream
    virtual void backProp(Optional<Tensor> dataIn,
                          Optional<Tensor> errorIn,
                          Optional<Tensor> errorOut,
                          Stream dataStream,
                          unsigned int connectionNumber,
                          bool accumulateGradient) {
        assert(errorIn.isPresent());
        assert(errorIn.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        if (errorOut.isPresent()) {
            assert(dataStream.isInitialized());

            CublasMatrixMultiply::instance().multiply(errorIn,
                                                      weights,
                                                      errorOut,
                                                      workspaceBackwardData,
                                                      batchSize,
                                                      numOutputFeatures,
                                                      numInputFeatures,
                                                      numOutputFeatures,
                                                      false,
                                                      true,
                                                      false,
                                                      false,
                                                      TensorDescriptor::DataType::FP16,
                                                      dataStream);
        }

        if (!isInferenceOnly()) {
            assert(optimizer.isPresent());

            // backward() syncs gradient stream with data stream prior to calling this to ensure error in is ready at end of gradient stream
            optimizer.get()->computeWeightsUpdate(dataIn, errorIn, accumulateGradient);

            // weights update cannot be applied to weights until errorOut has been computed since weights are part of that computation
            // so to enforce this gradientUpdateStream says that gradient is not ready to be applied until both errorOut and gradient are
            // computed
            optimizer.get()->getGradientUpdateStream().waitEvent(dataStream.putEvent());
            // Now at the end of gradientUpdateStream errorOut and gradients are ready from the updates for this connection.

            // Upon processing the last connection, schedule the upate to the weights memory.
            if (stillWaitingForErrorInputTensors.empty()) {
                optimizer.get()->updateWeights(weights, biases, batchSize);
            }

            // weights will be updated at the current end of the gradientUpdateStream
            // so Forward() must wait until gradientUpdateStream is finished.
            // This is accomplished in TrainableWeightsBiasesLayer::forward().
        }
    }

    // Compute the weights gradient for the specified connection number, accumulate as necessary.
    // This computation runs on optimizer.gradientUpdateStream.
    virtual void computeWeightsGradient(Optional<Tensor> weightsGradient,
                                        Optional<Tensor> biasesGradient,
                                        Optional<Tensor> featureIn,
                                        Optional<Tensor> errorIn,
                                        Stream gradientUpdateStream,
                                        bool accumulateGradient) {
        // Ensure all memory properly allocated
        assert(weightsGradient.isPresent());
        assert(weightsGradient.get().getDescriptor() == weights.getDescriptor());
        assert(weightsGradient.get().getPlacement() == weights.getPlacement());
        assert(weightsGradient.get().getMemPtr() != weights.getMemPtr());
        if (hasBias) {
            assert(biasesGradient.isPresent());
            assert(biases.isPresent());
            assert(biasesGradient.get().getDescriptor() == biasesGradient.get().getDescriptor());
            assert(biasesGradient.get().getMemPtr() != biases.get().getMemPtr());
            assert(biasesGradient.get().getPlacement() == biases.get().getPlacement());
        } else {
            assert(biasesGradient.isEmpty());
        }

        if (errorIn.isEmpty())
            return;
        assert(featureIn.isPresent());

        CublasMatrixMultiply::instance().multiply(featureIn,
                                                  errorIn,
                                                  weightsGradient,
                                                  workspaceBackwardWeights,
                                                  batchSize,
                                                  numInputFeatures,
                                                  batchSize,
                                                  numOutputFeatures,
                                                  true,
                                                  false,
                                                  accumulateGradient,
                                                  false,
                                                  TensorDescriptor::DataType::FP16,
                                                  gradientUpdateStream);

        if (hasBias) {
            biasBatchReduce->reduce(errorIn, biasesGradient, accumulateGradient);
        }
    }

    uint64_t flopsPerConnectionPerExample() {
        Optional<Tensor> anyFeatureInput = getFirstPresentTensor(featureInputs);
        Optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
        assert(anyFeatureInput.isPresent());
        assert(anyFeatureOutput.isPresent());
        uint64_t flops = 2 * numInputFeatures * numOutputFeatures - numOutputFeatures;
        if (hasBias)
            flops += numOutputFeatures;
        return flops;
    }

    uint64_t flopsPerGradientUpdatePerExample() {
        Optional<Tensor> anyFeatureInput = getFirstPresentTensor(featureInputs);
        Optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
        assert(anyFeatureInput.isPresent());
        assert(anyFeatureOutput.isPresent());
        uint64_t flops = numInputFeatures * numOutputFeatures;
        if (hasBias)
            flops += numOutputFeatures;
        return flops;
    }

    virtual uint64_t floatingPointOperationsPerExampleForward() {
        uint32_t connectionMultiplier = 0;
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (featureInputs[i].isPresent())
                connectionMultiplier += 1;
        }

        return connectionMultiplier * flopsPerConnectionPerExample();
    }

    virtual uint64_t floatingPointOperationsPerExampleBackward() {
        if (!isInferenceOnly())
            return 0;

        uint32_t connectionMultiplier = 0;
        uint32_t sums = 0;
        for (uint32_t i = 0; i < errorInputs.size(); ++i) {
            if (errorInputs[i].isPresent()) {
                if (connectionMultiplier == 0)
                    connectionMultiplier += 1;
                else
                    sums += 1;
            }
        }
        for (uint32_t i = 0; i < errorOutputs.size(); ++i) {
            if (errorOutputs[i].isPresent())
                connectionMultiplier += 1;
        }

        Optional<Tensor> anyErrorInput = getFirstPresentTensor(errorInputs);
        assert(anyErrorInput.isPresent());

        return connectionMultiplier * flopsPerConnectionPerExample() +
               (sums * anyErrorInput.get().getDescriptor().getTotalNumElements()) / batchSize + flopsPerGradientUpdatePerExample();
    }

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;
    static const float BETA_ACCUMULATE;

    void createBiasesCudnnTensorDescriptor() {
        cudnnStatus_t cudnnStatus;
        cudnnTensorDescriptor_t descriptor;

        cudnnStatus = cudnnCreateTensorDescriptor(&descriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        Optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
        assert(anyFeatureOutput.isPresent());
        uint32_t numOutputFeatures = anyFeatureOutput.get().getDescriptor().getDimensions()[1];
        cudnnStatus = cudnnSetTensor4dDescriptor(descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, numOutputFeatures, 1, 1);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnBiasDescriptor = descriptor;
    }

    void createFeatureOutputCudnnTensorDescriptor() {
        cudnnStatus_t cudnnStatus;
        cudnnTensorDescriptor_t descriptor;

        cudnnStatus = cudnnCreateTensorDescriptor(&descriptor);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);
        Optional<Tensor> anyFeatureOutput = getFirstPresentTensor(featureOutputs);
        assert(anyFeatureOutput.isPresent());
        std::vector<uint64_t> dimensions = anyFeatureOutput.get().getDescriptor().getDimensions();
        cudnnStatus = cudnnSetTensor4dDescriptor(descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, dimensions[0], dimensions[1], 1, 1);
        assert(cudnnStatus == CUDNN_STATUS_SUCCESS);

        cudnnFeatureOutputDescriptor = descriptor;
    }

    uint32_t numInputFeatures;
    const uint32_t numOutputFeatures;
    uint32_t batchSize;

    Optional<cudnnTensorDescriptor_t> cudnnBiasDescriptor;
    Optional<cudnnTensorDescriptor_t> cudnnFeatureOutputDescriptor;

    Optional<Tensor> workspaceForward;
    Optional<Tensor> workspaceBackwardData;
    Optional<Tensor> workspaceBackwardWeights;

    std::unique_ptr<BatchReduce> biasBatchReduce;
};

}  // namespace ThorImplementation
*/