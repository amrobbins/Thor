#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Misc/ComputeCategoricalAccuracy.h"

#include <chrono>
#include <thread>

namespace ThorImplementation {

/**
 * Returns the proportion of the predictions where the class with the highest prediction probability is the true class.
 */

class CategoricalAccuracy : public Metric {
   public:
    ~CategoricalAccuracy() override {}
    CategoricalAccuracy() {}

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

        std::vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();
        std::vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        TensorDescriptor::DataType labelsDataType = labelsInput.get().getDescriptor().getDataType();
        bool perClassLabels =
            featureInputDimensions == labelDimensions &&
            (labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::UINT16 ||
             labelsDataType == TensorDescriptor::DataType::UINT32 || labelsDataType == TensorDescriptor::DataType::INT8 ||
             labelsDataType == TensorDescriptor::DataType::INT16 || labelsDataType == TensorDescriptor::DataType::INT32 ||
             labelsDataType == TensorDescriptor::DataType::FP16 || labelsDataType == TensorDescriptor::DataType::FP32);
        bool classIndexLabels =
            labelDimensions.size() == 2 && featureInputDimensions[0] == labelDimensions[0] && labelDimensions[1] == 1 &&
            (labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::UINT16 ||
             labelsDataType == TensorDescriptor::DataType::UINT32 || labelsDataType == TensorDescriptor::DataType::INT8 ||
             labelsDataType == TensorDescriptor::DataType::INT16 || labelsDataType == TensorDescriptor::DataType::INT32);
        THOR_THROW_IF_FALSE(perClassLabels ^ classIndexLabels);
        if (perClassLabels)
            labelFormat = LABEL_FORMAT::INDICATOR_PER_CLASS_TYPE;
        else
            labelFormat = LABEL_FORMAT::INDEX_OF_CLASS_TYPE;

        THOR_THROW_IF_FALSE(featureInput.isPresent());
        THOR_THROW_IF_FALSE(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDimensions().size() == 2);

        THOR_THROW_IF_FALSE(featureInput.get().getDescriptor().getDimensions().size() >= 2);
        batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        numClasses = featureInput.get().getDescriptor().getDimensions()[1];

        // When there are two classes and the label is a single 1 or 0, binary accuracy can be used, instead of categorical accuracy.
        THOR_THROW_IF_FALSE(numClasses >= 2);

        workspace = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
    }

    void computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) override {
        if (labelFormat == LABEL_FORMAT::INDICATOR_PER_CLASS_TYPE) {
            computeMetricIndicatorPerClass(labels, predictions, metric, stream);
        } else if (labelFormat == LABEL_FORMAT::INDEX_OF_CLASS_TYPE) {
            computeMetricClassIndex(labels, predictions, metric, stream);
        } else {
            THOR_UNREACHABLE();
        }
    }

    std::string toDisplayString(Tensor metric_h) override {
        THOR_THROW_IF_FALSE(metric_h.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);
        float accuracy = *((float *)metric_h.getMemPtr());
        return "Accuracy: " + std::to_string(accuracy);
    }

    enum class LABEL_FORMAT { INDICATOR_PER_CLASS_TYPE = 7, INDEX_OF_CLASS_TYPE };

    LABEL_FORMAT confirmLabelFormat() { return labelFormat; }

   protected:
    LABEL_FORMAT labelFormat;

   private:
    void computeMetricIndicatorPerClass(Tensor labels, Tensor predictions, Tensor metric, Stream stream) {
        if (predictions.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (half *)predictions.getMemPtr(),
                                                                (uint8_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (half *)predictions.getMemPtr(),
                                                                (uint16_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (half *)predictions.getMemPtr(),
                                                                (uint32_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT8) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (half *)predictions.getMemPtr(),
                                                                (int8_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT16) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (half *)predictions.getMemPtr(),
                                                                (int16_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT32) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (half *)predictions.getMemPtr(),
                                                                (int32_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (half *)predictions.getMemPtr(),
                                                                (half *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (half *)predictions.getMemPtr(),
                                                                (float *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);
            } else {
                THOR_UNREACHABLE();
            }

        } else if (predictions.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (float *)predictions.getMemPtr(),
                                                                (uint8_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (float *)predictions.getMemPtr(),
                                                                (uint16_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (float *)predictions.getMemPtr(),
                                                                (uint32_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT8) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (float *)predictions.getMemPtr(),
                                                                (int8_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT16) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (float *)predictions.getMemPtr(),
                                                                (int16_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT32) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (float *)predictions.getMemPtr(),
                                                                (int32_t *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (float *)predictions.getMemPtr(),
                                                                (half *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
                launchComputeCategoricalAccuracy_perClassLabels((float *)featureOutput.get().getMemPtr(),
                                                                (float *)predictions.getMemPtr(),
                                                                (float *)labels.getMemPtr(),
                                                                (uint8_t *)workspace.getMemPtr(),
                                                                numClasses,
                                                                batchSize,
                                                                stream);
            } else {
                THOR_UNREACHABLE();
            }
        } else {
            THOR_UNREACHABLE();
        }
    }

    void computeMetricClassIndex(Tensor labels, Tensor predictions, Tensor metric, Stream stream) {
        if (predictions.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (half *)predictions.getMemPtr(),
                                                                  (uint8_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (half *)predictions.getMemPtr(),
                                                                  (uint16_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (half *)predictions.getMemPtr(),
                                                                  (uint32_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT8) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (half *)predictions.getMemPtr(),
                                                                  (int8_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);
            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT16) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (half *)predictions.getMemPtr(),
                                                                  (int16_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);
            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT32) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (half *)predictions.getMemPtr(),
                                                                  (int32_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);

            } else {
                THOR_UNREACHABLE();
            }

        } else if (predictions.getDescriptor().getDataType() == TensorDescriptor::DataType::FP32) {
            if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT8) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (float *)predictions.getMemPtr(),
                                                                  (uint8_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT16) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (float *)predictions.getMemPtr(),
                                                                  (uint16_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::UINT32) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (float *)predictions.getMemPtr(),
                                                                  (uint32_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);

            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT8) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (float *)predictions.getMemPtr(),
                                                                  (int8_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);
            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT16) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (float *)predictions.getMemPtr(),
                                                                  (int16_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);
            } else if (labels.getDescriptor().getDataType() == TensorDescriptor::DataType::INT32) {
                launchComputeCategoricalAccuracy_classIndexLabels((float *)featureOutput.get().getMemPtr(),
                                                                  (float *)predictions.getMemPtr(),
                                                                  (int32_t *)labels.getMemPtr(),
                                                                  (uint8_t *)workspace.getMemPtr(),
                                                                  numClasses,
                                                                  batchSize,
                                                                  stream);

            } else {
                THOR_UNREACHABLE();
            }
        } else {
            THOR_UNREACHABLE();
        }
    }

    unsigned int batchSize;
    unsigned int numClasses;

    Tensor workspace;
};

}  // namespace ThorImplementation
