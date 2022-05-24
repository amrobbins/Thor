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
    virtual ~CategoricalAccuracy() {}
    CategoricalAccuracy() {}

    virtual void compile() {
        assert(labelsInput.isPresent());
        assert(labelsInput.get().isInitialized());
        assert(labelsInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(labelsInput.get().getPlacement().getDeviceNum() == featureInput.get().getPlacement().getDeviceNum());

        vector<uint64_t> featureInputDimensions = featureInput.get().getDescriptor().getDimensions();
        vector<uint64_t> labelDimensions = labelsInput.get().getDescriptor().getDimensions();
        TensorDescriptor::DataType labelsDataType = labelsInput.get().getDescriptor().getDataType();
        bool perClassLabels = featureInputDimensions == labelDimensions &&
                              (labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::FP16 ||
                               labelsDataType == TensorDescriptor::DataType::FP32);
        bool classIndexLabels =
            labelDimensions.size() == 2 && featureInputDimensions[0] == labelDimensions[0] && labelDimensions[1] == 1 &&
            (labelsDataType == TensorDescriptor::DataType::UINT8 || labelsDataType == TensorDescriptor::DataType::UINT16 ||
             labelsDataType == TensorDescriptor::DataType::UINT32);
        assert(perClassLabels ^ classIndexLabels);
        if (perClassLabels)
            labelFormat = LABEL_FORMAT::INDICATOR_PER_CLASS_TYPE;
        else
            labelFormat = LABEL_FORMAT::INDEX_OF_CLASS_TYPE;

        assert(featureInput.isPresent());
        assert(featureInput.get().getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
        assert(featureInput.get().getDescriptor().getDimensions().size() == 2);

        assert(featureInput.get().getDescriptor().getDimensions().size() >= 2);
        batchSize = featureInput.get().getDescriptor().getDimensions()[0];
        numClasses = featureInput.get().getDescriptor().getDimensions()[1];

        // When there are two classes and the label is a single 1 or 0, binary accuracy can be used, instead of categorical accuracy.
        assert(numClasses >= 2);

        workspace = Tensor(featureInput.get().getPlacement(), TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
    }

    virtual void computeMetric(Tensor labels, Tensor predictions, Tensor metric, Stream stream) {
        if (labelFormat == LABEL_FORMAT::INDICATOR_PER_CLASS_TYPE) {
            computeMetricIndicatorPerClass(labels, predictions, metric, stream);

            // FIXME: TEMP
            /*
            Tensor labels_h = labels.clone(TensorPlacement::MemDevices::CPU);
            labels_h.copyFromAsync(labels, stream);
            uint8_t *labelArray = (uint8_t *)labels_h.getMemPtr();
            Tensor predictions_h = predictions.clone(TensorPlacement::MemDevices::CPU);
            predictions_h.copyFromAsync(predictions, stream);
            float *predictionArray = (float *)predictions_h.getMemPtr();
            labelsStream.synchronize();

            vector<uint32_t> labelVector;
            vector<uint32_t> predictionVector;
            vector<float> bestPredictionVector;
            vector<uint8_t> bestLabelVector;
            for (uint32_t b = 0; b < 6; ++b) {
                uint8_t bestLabel = labelArray[b * numClasses];
                labelVector.push_back(0);
                float bestPrediction = predictionArray[b * numClasses];
                predictionVector.push_back(0);
                for (uint32_t c = 1; c < numClasses; ++c) {
                    uint8_t classLabel = labelArray[b * numClasses + c];
                    if (classLabel > bestLabel) {
                        labelVector.pop_back();
                        labelVector.push_back(c);
                        bestLabel = classLabel;
                    }
                    float classPrediction = predictionArray[b * numClasses + c];
                    if (classPrediction > bestPrediction) {
                        predictionVector.back() = c;
                        bestPrediction = classPrediction;
                    }
                }
                bestLabelVector.push_back(bestLabel);
                bestPredictionVector.push_back(bestPrediction);
            }
            printf("\rLabels:      ");
            for (uint32_t i = 0; i < labelVector.size(); ++i)
                printf("%d(%d) ", labelVector[i], (uint32_t)bestLabelVector[i]);
            printf("\n");
            printf("\rPredictions: ");
            for (uint32_t i = 0; i < predictionVector.size(); ++i)
                printf("%d(%.2f) ", predictionVector[i], bestPredictionVector[i]);
            printf("\n\n");
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            */
        } else if (labelFormat == LABEL_FORMAT::INDEX_OF_CLASS_TYPE) {
            computeMetricClassIndex(labels, predictions, metric, stream);
        } else {
            assert(false);
        }
    }

    virtual std::string toDisplayString(Tensor metric_h) {
        assert(metric_h.getPlacement() == TensorPlacement::MemDevices::CPU);
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
                assert(false);
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
                assert(false);
            }
        } else {
            assert(false);
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

            } else {
                assert(false);
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

            } else {
                assert(false);
            }
        } else {
            assert(false);
        }
    }

    unsigned int batchSize;
    unsigned int numClasses;

    Tensor workspace;
};

}  // namespace ThorImplementation
