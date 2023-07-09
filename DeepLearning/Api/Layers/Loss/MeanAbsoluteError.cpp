#include "MeanAbsoluteError.h"

using namespace Thor;
using namespace std;

void MeanAbsoluteError::buildSupportLayersAndAddToNetwork() {
    Tensor currentFeatureInput = predictionsTensor;

    MeanAbsoluteError meanAbsoluteError =
        MeanAbsoluteError::Builder().network(*network).predictions(predictionsTensor).labels(labelsTensor).reportsRawLoss().build();

    if (lossType == ThorImplementation::Loss::LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(meanAbsoluteError.getLoss()).reportsBatchLoss().build();
        // Replace the output on the compound layer to be the output of the last stage
        // i.e. tunnel the actual input to actual output of the compound layer,
        // Network uses single layers, user uses compound layer.
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::ELEMENTWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(meanAbsoluteError.getLoss()).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::CLASSWISE) {
        LossShaper lossShaper =
            LossShaper::Builder().network(*network).lossInput(meanAbsoluteError.getLoss()).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed
        assert(ThorImplementation::Loss::LossType::RAW);
    }
}