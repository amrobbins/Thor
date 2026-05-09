#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Misc/ComputeBinaryAccuracy.h"

#include <chrono>
#include <thread>

namespace ThorImplementation {

/**
 * Returns the proportion of the predictions where the class with the highest prediction probability is the true class.
 */

class BinaryAccuracy : public Metric {
   public:
    ~BinaryAccuracy() override {}
    BinaryAccuracy() {}

    Optional<Tensor> createFeatureOutputTensor() override {
        TensorPlacement placement = featureInput.get().getPlacement();
        return Tensor(placement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1U}));
    }

    void compileImpl() override {
        Layer::compileImpl();
        THOR_THROW_IF_FALSE(labelsInput.isPresent());
        THOR_THROW_IF_FALSE(labelsInput.get().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());
        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        std::vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();
        std::vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        TensorDescriptor::DataType labelsDataType = labelsInput.get().getDescriptor().getDataType();
        THOR_THROW_IF_FALSE(labelDimensions.size() == 2);
        THOR_THROW_IF_FALSE(labelDimensions[1] == 1);
        THOR_THROW_IF_FALSE(featureInputDimensions == labelDimensions);
        THOR_THROW_IF_FALSE(labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::UINT16 ||
               labelsDataType == TensorDescriptor::DataType::UINT32 || labelsDataType == TensorDescriptor::DataType::INT8 ||
               labelsDataType == TensorDescriptor::DataType::INT16 || labelsDataType == TensorDescriptor::DataType::INT32 ||
               labelsDataType == TensorDescriptor::DataType::FP16 || labelsDataType == TensorDescriptor::DataType::FP32);

        batchSize = featureInput.get().getDescriptor().getDimensions()[0];

        workspace = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize}));

        batchReduce = createBinaryAccuracyBatchReduce(batchSize, stream);
    }

    void computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) override {
        if (predictions.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (half *)predictions.getMemPtr(),
                                            (uint8_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (half *)predictions.getMemPtr(),
                                            (uint16_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (half *)predictions.getMemPtr(),
                                            (uint32_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT8) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (half *)predictions.getMemPtr(),
                                            (int8_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT16) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (half *)predictions.getMemPtr(),
                                            (int16_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT32) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (half *)predictions.getMemPtr(),
                                            (int32_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchComputeBinaryAccuracy(
                    featureOutput, (half *)predictions.getMemPtr(), (half *)labels.getMemPtr(), workspace, batchSize, batchReduce, stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
                launchComputeBinaryAccuracy(
                    featureOutput, (half *)predictions.getMemPtr(), (float *)labels.getMemPtr(), workspace, batchSize, batchReduce, stream);
            } else {
                THOR_UNREACHABLE();
            }

        } else if (predictions.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (float *)predictions.getMemPtr(),
                                            (uint8_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (float *)predictions.getMemPtr(),
                                            (uint16_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (float *)predictions.getMemPtr(),
                                            (uint32_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT8) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (float *)predictions.getMemPtr(),
                                            (int8_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT16) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (float *)predictions.getMemPtr(),
                                            (int16_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT32) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (float *)predictions.getMemPtr(),
                                            (int32_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchComputeBinaryAccuracy(
                    featureOutput, (float *)predictions.getMemPtr(), (half *)labels.getMemPtr(), workspace, batchSize, batchReduce, stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
                launchComputeBinaryAccuracy(featureOutput,
                                            (float *)predictions.getMemPtr(),
                                            (float *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);
            } else {
                THOR_UNREACHABLE();
            }
        } else {
            THOR_UNREACHABLE();
        }
    }

    std::string toDisplayString(Tensor metric_h) override {
        THOR_THROW_IF_FALSE(metric_h.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        float accuracy = *((float *)metric_h.getMemPtr());
        return "Accuracy: " + std::to_string(accuracy);
    }

   private:
    unsigned int batchSize;
    Tensor workspace;
    std::shared_ptr<BatchReduce> batchReduce;
};

}  // namespace ThorImplementation
