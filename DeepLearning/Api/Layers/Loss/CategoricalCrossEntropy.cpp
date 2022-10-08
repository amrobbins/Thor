#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"

using namespace Thor;

void CategoricalCrossEntropy::convertToSingleLayersAndAddToNetwork() {
    Tensor currentPredictionsTensor = predictionsTensor;

    CategoricalCrossEntropy::Builder categoricalCrossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                          .network(*network)
                                                                          .predictions(currentPredictionsTensor)
                                                                          .labels(labelsTensor)
                                                                          .reportsRawLoss()
                                                                          .lossDataType(lossTensor.getDataType());
    if (labelType == LabelType::INDEX) {
        categoricalCrossEntropyBuilder.receivesClassIndexLabels();
    } else {
        assert(labelType == LabelType::ONE_HOT);
        categoricalCrossEntropyBuilder.receivesOneHotLabels();
    }
    CategoricalCrossEntropy crossEntropy = categoricalCrossEntropyBuilder.build();
    currentPredictionsTensor = crossEntropy.getLoss();

    if (lossType == ThorImplementation::Loss::LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentPredictionsTensor).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentPredictionsTensor).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::ELEMENTWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(currentPredictionsTensor).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossType == ThorImplementation::Loss::LossType::RAW);
        lossTensor = crossEntropy.getLoss();
    }
}