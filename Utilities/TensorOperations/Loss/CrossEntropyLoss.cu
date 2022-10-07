#include "CrossEntropyLoss.h"

using namespace std;

/**
 *
 * @param numClasses is ignored for binary cross entropy
 * @param indexLabels is ignored for binary cross entropy
 */
template <typename LABEL_OR_INDEX_TYPE, typename PROBABILITY_TYPE, typename LOSS_TYPE>
void launchElementWiseCrossEntropyLoss(void *labelsOrClassOfHotLabels_d,
                                       void *probabilities_d,
                                       void *loss_d,
                                       void *gradient_d,
                                       uint32_t numClasses,
                                       uint32_t batchSize,
                                       bool computeGradient,
                                       uint32_t lossScalingFactor,
                                       CrossEntropyLossType crossEntropyLossType,
                                       bool indexLabels,
                                       Stream stream) {
    assert(crossEntropyLossType == CrossEntropyLossType::BINARY || crossEntropyLossType == CrossEntropyLossType::CATEGORICAL);

    if (crossEntropyLossType == CrossEntropyLossType::BINARY) {
        launchElementWiseBinaryCrossEntropyLoss<LABEL_OR_INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>(
            labelsOrClassOfHotLabels_d, probabilities_d, loss_d, gradient_d, batchSize, computeGradient, lossScalingFactor, stream);
    } else {
        if (indexLabels) {
            // This version takes in an integer per item in the batch that specifies the true class of the example.
            launchElementWiseCategoricalCrossEntropyLoss_classIndexLabels<LABEL_OR_INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>(
                labelsOrClassOfHotLabels_d,
                probabilities_d,
                loss_d,
                gradient_d,
                numClasses,
                batchSize,
                computeGradient,
                lossScalingFactor,
                stream);
        } else {
            launchElementWiseCategoricalCrossEntropyLoss_oneHotLabels<LABEL_OR_INDEX_TYPE, PROBABILITY_TYPE, LOSS_TYPE>(
                labelsOrClassOfHotLabels_d,
                probabilities_d,
                loss_d,
                gradient_d,
                numClasses,
                batchSize,
                computeGradient,
                lossScalingFactor,
                stream);
        }
    }
}

template void launchElementWiseCrossEntropyLoss<bool, half, half>(void *labelsOrClassOfHotLabels_d,
                                                                  void *probabilities_d,
                                                                  void *loss_d,
                                                                  void *gradient_d,
                                                                  uint32_t numClasses,
                                                                  uint32_t batchSize,
                                                                  bool computeGradient,
                                                                  uint32_t lossScalingFactor,
                                                                  CrossEntropyLossType crossEntropyLossType,
                                                                  bool indexLabels,
                                                                  Stream stream);

template void launchElementWiseCrossEntropyLoss<bool, half, float>(void *labelsOrClassOfHotLabels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   CrossEntropyLossType crossEntropyLossType,
                                                                   bool indexLabels,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<bool, float, half>(void *labelsOrClassOfHotLabels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   CrossEntropyLossType crossEntropyLossType,
                                                                   bool indexLabels,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<bool, float, float>(void *labelsOrClassOfHotLabels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    CrossEntropyLossType crossEntropyLossType,
                                                                    bool indexLabels,
                                                                    Stream stream);

template void launchElementWiseCrossEntropyLoss<uint8_t, half, half>(void *labelsOrClassOfHotLabels_d,
                                                                     void *probabilities_d,
                                                                     void *loss_d,
                                                                     void *gradient_d,
                                                                     uint32_t numClasses,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     uint32_t lossScalingFactor,
                                                                     CrossEntropyLossType crossEntropyLossType,
                                                                     bool indexLabels,
                                                                     Stream stream);

template void launchElementWiseCrossEntropyLoss<uint8_t, half, float>(void *labelsOrClassOfHotLabels_d,
                                                                      void *probabilities_d,
                                                                      void *loss_d,
                                                                      void *gradient_d,
                                                                      uint32_t numClasses,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      uint32_t lossScalingFactor,
                                                                      CrossEntropyLossType crossEntropyLossType,
                                                                      bool indexLabels,
                                                                      Stream stream);

template void launchElementWiseCrossEntropyLoss<uint8_t, float, half>(void *labelsOrClassOfHotLabels_d,
                                                                      void *probabilities_d,
                                                                      void *loss_d,
                                                                      void *gradient_d,
                                                                      uint32_t numClasses,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      uint32_t lossScalingFactor,
                                                                      CrossEntropyLossType crossEntropyLossType,
                                                                      bool indexLabels,
                                                                      Stream stream);

template void launchElementWiseCrossEntropyLoss<uint8_t, float, float>(void *labelsOrClassOfHotLabels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       CrossEntropyLossType crossEntropyLossType,
                                                                       bool indexLabels,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint16_t, half, half>(void *labelsOrClassOfHotLabels_d,
                                                                      void *probabilities_d,
                                                                      void *loss_d,
                                                                      void *gradient_d,
                                                                      uint32_t numClasses,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      uint32_t lossScalingFactor,
                                                                      CrossEntropyLossType crossEntropyLossType,
                                                                      bool indexLabels,
                                                                      Stream stream);

template void launchElementWiseCrossEntropyLoss<uint16_t, half, float>(void *labelsOrClassOfHotLabels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       CrossEntropyLossType crossEntropyLossType,
                                                                       bool indexLabels,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint16_t, float, half>(void *labelsOrClassOfHotLabels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       CrossEntropyLossType crossEntropyLossType,
                                                                       bool indexLabels,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint16_t, float, float>(void *labelsOrClassOfHotLabels_d,
                                                                        void *probabilities_d,
                                                                        void *loss_d,
                                                                        void *gradient_d,
                                                                        uint32_t numClasses,
                                                                        uint32_t batchSize,
                                                                        bool computeGradient,
                                                                        uint32_t lossScalingFactor,
                                                                        CrossEntropyLossType crossEntropyLossType,
                                                                        bool indexLabels,
                                                                        Stream stream);

template void launchElementWiseCrossEntropyLoss<uint32_t, half, half>(void *labelsOrClassOfHotLabels_d,
                                                                      void *probabilities_d,
                                                                      void *loss_d,
                                                                      void *gradient_d,
                                                                      uint32_t numClasses,
                                                                      uint32_t batchSize,
                                                                      bool computeGradient,
                                                                      uint32_t lossScalingFactor,
                                                                      CrossEntropyLossType crossEntropyLossType,
                                                                      bool indexLabels,
                                                                      Stream stream);

template void launchElementWiseCrossEntropyLoss<uint32_t, half, float>(void *labelsOrClassOfHotLabels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       CrossEntropyLossType crossEntropyLossType,
                                                                       bool indexLabels,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint32_t, float, half>(void *labelsOrClassOfHotLabels_d,
                                                                       void *probabilities_d,
                                                                       void *loss_d,
                                                                       void *gradient_d,
                                                                       uint32_t numClasses,
                                                                       uint32_t batchSize,
                                                                       bool computeGradient,
                                                                       uint32_t lossScalingFactor,
                                                                       CrossEntropyLossType crossEntropyLossType,
                                                                       bool indexLabels,
                                                                       Stream stream);

template void launchElementWiseCrossEntropyLoss<uint32_t, float, float>(void *labelsOrClassOfHotLabels_d,
                                                                        void *probabilities_d,
                                                                        void *loss_d,
                                                                        void *gradient_d,
                                                                        uint32_t numClasses,
                                                                        uint32_t batchSize,
                                                                        bool computeGradient,
                                                                        uint32_t lossScalingFactor,
                                                                        CrossEntropyLossType crossEntropyLossType,
                                                                        bool indexLabels,
                                                                        Stream stream);

template void launchElementWiseCrossEntropyLoss<half, half, half>(void *labelsOrClassOfHotLabels_d,
                                                                  void *probabilities_d,
                                                                  void *loss_d,
                                                                  void *gradient_d,
                                                                  uint32_t numClasses,
                                                                  uint32_t batchSize,
                                                                  bool computeGradient,
                                                                  uint32_t lossScalingFactor,
                                                                  CrossEntropyLossType crossEntropyLossType,
                                                                  bool indexLabels,
                                                                  Stream stream);

template void launchElementWiseCrossEntropyLoss<half, half, float>(void *labelsOrClassOfHotLabels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   CrossEntropyLossType crossEntropyLossType,
                                                                   bool indexLabels,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<half, float, half>(void *labelsOrClassOfHotLabels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   CrossEntropyLossType crossEntropyLossType,
                                                                   bool indexLabels,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<half, float, float>(void *labelsOrClassOfHotLabels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    CrossEntropyLossType crossEntropyLossType,
                                                                    bool indexLabels,
                                                                    Stream stream);

template void launchElementWiseCrossEntropyLoss<float, half, half>(void *labelsOrClassOfHotLabels_d,
                                                                   void *probabilities_d,
                                                                   void *loss_d,
                                                                   void *gradient_d,
                                                                   uint32_t numClasses,
                                                                   uint32_t batchSize,
                                                                   bool computeGradient,
                                                                   uint32_t lossScalingFactor,
                                                                   CrossEntropyLossType crossEntropyLossType,
                                                                   bool indexLabels,
                                                                   Stream stream);

template void launchElementWiseCrossEntropyLoss<float, half, float>(void *labelsOrClassOfHotLabels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    CrossEntropyLossType crossEntropyLossType,
                                                                    bool indexLabels,
                                                                    Stream stream);

template void launchElementWiseCrossEntropyLoss<float, float, half>(void *labelsOrClassOfHotLabels_d,
                                                                    void *probabilities_d,
                                                                    void *loss_d,
                                                                    void *gradient_d,
                                                                    uint32_t numClasses,
                                                                    uint32_t batchSize,
                                                                    bool computeGradient,
                                                                    uint32_t lossScalingFactor,
                                                                    CrossEntropyLossType crossEntropyLossType,
                                                                    bool indexLabels,
                                                                    Stream stream);

template void launchElementWiseCrossEntropyLoss<float, float, float>(void *labelsOrClassOfHotLabels_d,
                                                                     void *probabilities_d,
                                                                     void *loss_d,
                                                                     void *gradient_d,
                                                                     uint32_t numClasses,
                                                                     uint32_t batchSize,
                                                                     bool computeGradient,
                                                                     uint32_t lossScalingFactor,
                                                                     CrossEntropyLossType crossEntropyLossType,
                                                                     bool indexLabels,
                                                                     Stream stream);
