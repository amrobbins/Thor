#pragma once

#include <optional>
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Metric.h"
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

    std::optional<Tensor> createFeatureOutputTensor() override {
        TensorPlacement placement = featureInput.value().getPlacement();
        return Tensor(placement, TensorDescriptor(DataType::FP32, {1U}));
    }

    void compileImpl() override {
        Layer::compileImpl();
        THOR_THROW_IF_FALSE(labelsInput.has_value());
        THOR_THROW_IF_FALSE(labelsInput.value().isInitialized());
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(labelsInput.value().getPlacement().getDeviceNum() == featureInput.value().getPlacement().getDeviceNum());
        THOR_THROW_IF_FALSE(featureInput.has_value());
        THOR_THROW_IF_FALSE(featureInput.value().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);

        std::vector<uint64_t> featureInputDimensions = featureInput.value().getDescriptor().getDimensions();
        std::vector<uint64_t> labelDimensions = labelsInput.value().getDescriptor().getDimensions();
        DataType labelsDataType = labelsInput.value().getDescriptor().getDataType();
        THOR_THROW_IF_FALSE(labelDimensions.size() == 2);
        THOR_THROW_IF_FALSE(labelDimensions[1] == 1);
        THOR_THROW_IF_FALSE(featureInputDimensions == labelDimensions);
        THOR_THROW_IF_FALSE(labelsDataType == DataType::UINT8 || labelsDataType == DataType::UINT16 ||
               labelsDataType == DataType::UINT32 || labelsDataType == DataType::INT8 ||
               labelsDataType == DataType::INT16 || labelsDataType == DataType::INT32 ||
               labelsDataType == DataType::FP16 || labelsDataType == DataType::FP32);

        batchSize = featureInput.value().getDescriptor().getDimensions()[0];

        workspace = Tensor(featureInput.value().getPlacement(), TensorDescriptor(DataType::FP16, {batchSize}));

        batchReduce = createBinaryAccuracyBatchReduce(batchSize, stream);
    }

    void computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) override {
        if (predictions.getDescriptor().getDataType() == DataType::FP16) {
            if (labels.getDescriptor().getDataType() == DataType::UINT8) {
                launchComputeBinaryAccuracy(metric,
                                            (half *)predictions.getMemPtr(),
                                            (uint8_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::UINT16) {
                launchComputeBinaryAccuracy(metric,
                                            (half *)predictions.getMemPtr(),
                                            (uint16_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::UINT32) {
                launchComputeBinaryAccuracy(metric,
                                            (half *)predictions.getMemPtr(),
                                            (uint32_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::INT8) {
                launchComputeBinaryAccuracy(metric,
                                            (half *)predictions.getMemPtr(),
                                            (int8_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::INT16) {
                launchComputeBinaryAccuracy(metric,
                                            (half *)predictions.getMemPtr(),
                                            (int16_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::INT32) {
                launchComputeBinaryAccuracy(metric,
                                            (half *)predictions.getMemPtr(),
                                            (int32_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::FP16) {
                launchComputeBinaryAccuracy(
                    metric, (half *)predictions.getMemPtr(), (half *)labels.getMemPtr(), workspace, batchSize, batchReduce, stream);

            } else if (labels.getDescriptor().getDataType() == DataType::FP32) {
                launchComputeBinaryAccuracy(
                    metric, (half *)predictions.getMemPtr(), (float *)labels.getMemPtr(), workspace, batchSize, batchReduce, stream);
            } else {
                THOR_UNREACHABLE();
            }

        } else if (predictions.getDescriptor().getDataType() == DataType::FP32) {
            if (labels.getDescriptor().getDataType() == DataType::UINT8) {
                launchComputeBinaryAccuracy(metric,
                                            (float *)predictions.getMemPtr(),
                                            (uint8_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::UINT16) {
                launchComputeBinaryAccuracy(metric,
                                            (float *)predictions.getMemPtr(),
                                            (uint16_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::UINT32) {
                launchComputeBinaryAccuracy(metric,
                                            (float *)predictions.getMemPtr(),
                                            (uint32_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::INT8) {
                launchComputeBinaryAccuracy(metric,
                                            (float *)predictions.getMemPtr(),
                                            (int8_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::INT16) {
                launchComputeBinaryAccuracy(metric,
                                            (float *)predictions.getMemPtr(),
                                            (int16_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::INT32) {
                launchComputeBinaryAccuracy(metric,
                                            (float *)predictions.getMemPtr(),
                                            (int32_t *)labels.getMemPtr(),
                                            workspace,
                                            batchSize,
                                            batchReduce,
                                            stream);

            } else if (labels.getDescriptor().getDataType() == DataType::FP16) {
                launchComputeBinaryAccuracy(
                    metric, (float *)predictions.getMemPtr(), (half *)labels.getMemPtr(), workspace, batchSize, batchReduce, stream);

            } else if (labels.getDescriptor().getDataType() == DataType::FP32) {
                launchComputeBinaryAccuracy(metric,
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
