#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"

using namespace Thor;
using namespace std;

void BinaryCrossEntropy::convertToSingleLayersAndAddToNetwork() {
    Tensor currentFeatureInput = predictionsTensor;

    assert(!sigmoidStamped);
    Sigmoid::Builder sigmoidBuilder = Sigmoid::Builder();
    sigmoidBuilder.network(*network);
    sigmoidBuilder.featureInput(currentFeatureInput);
    sigmoidBuilder.backwardComputedExternally();
    shared_ptr<Layer> sigmoid = sigmoidBuilder.build();
    currentFeatureInput = sigmoid->getFeatureOutput();

    BinaryCrossEntropy::Builder binaryCrossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                                .network(*network)
                                                                .predictions(currentFeatureInput)
                                                                .labels(labelsTensor)
                                                                .sigmoidStamped()
                                                                .reportsElementwiseLoss();
    BinaryCrossEntropy crossEntropy = binaryCrossEntropyBuilder.build();
    currentFeatureInput = crossEntropy.getFeatureOutput();

    if (lossType == ThorImplementation::Loss::LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsBatchLoss().build();
        featureOutput = lossShaper.getFeatureOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossType == ThorImplementation::Loss::LossType::ELEMENTWISE);
        featureOutput = crossEntropy.getFeatureOutput();
    }
}