#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"

using namespace Thor;

void BinaryCrossEntropy::convertToSingleLayersAndAddToNetwork() {
    Tensor currentFeatureInput = featureInput;

    if (!softmaxRemoved) {
        Softmax::Builder softmaxBuilder = Softmax::Builder();
        softmaxBuilder.network(*network);
        softmaxBuilder.featureInput(currentFeatureInput);
        shared_ptr<Layer> softmax = softmaxBuilder.build();
        currentFeatureInput = softmax->getFeatureOutput();
    }

    BinaryCrossEntropy::Builder binaryCrossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                                .network(*network)
                                                                .predictions(currentFeatureInput)
                                                                .labels(labelsTensor)
                                                                .removeSoftmax()
                                                                .reportsElementwiseLoss();
    BinaryCrossEntropy crossEntropy = binaryCrossEntropyBuilder.build();
    currentFeatureInput = crossEntropy.getFeatureOutput();

    if (lossType == ThorImplementation::Loss::LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsBatchLoss().build();
        featureOutput = lossShaper.getFeatureOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsClasswiseLoss().build();
        featureOutput = lossShaper.getFeatureOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsElementwiseLoss().build();
        featureOutput = lossShaper.getFeatureOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossType == ThorImplementation::Loss::LossType::RAW);
        featureOutput = crossEntropy.getFeatureOutput();
    }
}