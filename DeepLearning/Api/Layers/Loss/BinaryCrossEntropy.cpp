#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"

using namespace Thor;
using namespace std;

void BinaryCrossEntropy::buildSupportLayersAndAddToNetwork() {
    Tensor currentFeatureInput = predictionsTensor;

    assert(!sigmoidStamped);
    Sigmoid::Builder sigmoidBuilder = Sigmoid::Builder();
    sigmoidBuilder.network(*network);
    sigmoidBuilder.featureInput(currentFeatureInput);
    sigmoidBuilder.backwardComputedExternally();
    shared_ptr<Layer> sigmoid = sigmoidBuilder.build();
    sigmoidOutput = sigmoid->getFeatureOutput();
    currentFeatureInput = sigmoidOutput;

    BinaryCrossEntropy::Builder binaryCrossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                                .network(*network)
                                                                .predictions(currentFeatureInput)
                                                                .labels(labelsTensor)
                                                                .sigmoidStamped()
                                                                .reportsElementwiseLoss()
                                                                .lossDataType(lossDataType);
    BinaryCrossEntropy crossEntropy = binaryCrossEntropyBuilder.build();
    currentFeatureInput = crossEntropy.getLoss();

    if (lossType == LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getFeatureOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossType == LossType::ELEMENTWISE);
        lossTensor = currentFeatureInput;
    }
}
