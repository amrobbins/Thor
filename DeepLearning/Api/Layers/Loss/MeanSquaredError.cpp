#include "MeanSquaredError.h"

using namespace Thor;

void MeanSquaredError::convertToSingleLayersAndAddToNetwork() {
    assert(isMultiLayer());

    Tensor currentFeatureInput = predictionsTensor;

    MeanSquaredError meanSquaredError = MeanSquaredError::Builder()
                                            .network(*network)
                                            .predictions(predictionsTensor)
                                            .labels(labelsTensor)
                                            .reportsRawLoss()
                                            .lossDataType(lossTensor.getDataType())
                                            .build();

    if (lossType == ThorImplementation::Loss::LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossTensor).reportsBatchLoss().build();
        // Replace the output on the compound layer to be the output of the last stage
        // i.e. tunnel the actual input to actual output of the compound layer,
        // Network uses single layers, user uses compound layer.
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossTensor).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(lossTensor).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed
        assert(ThorImplementation::Loss::LossType::RAW);
    }
}