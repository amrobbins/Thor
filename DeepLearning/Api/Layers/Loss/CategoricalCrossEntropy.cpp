#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"

using namespace Thor;
using namespace std;

void CategoricalCrossEntropy::buildSupportLayersAndAddToNetwork() {
    Tensor currentFeatureInput = predictionsTensor;

    assert(!softmaxStamped);
    Softmax::Builder softmaxBuilder = Softmax::Builder();
    softmaxBuilder.network(*network);
    softmaxBuilder.featureInput(currentFeatureInput);
    softmaxBuilder.backwardComputedExternally();
    shared_ptr<Layer> softmax = softmaxBuilder.build();
    softmaxOutput = softmax->getFeatureOutput();
    currentFeatureInput = softmaxOutput;

    CategoricalCrossEntropy::Builder categoricalCrossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                          .network(*network)
                                                                          .predictions(currentFeatureInput)
                                                                          .labels(labelsTensor)
                                                                          .softmaxStamped()
                                                                          .reportsRawLoss()
                                                                          .lossDataType(lossDataType);
    if (labelType == LabelType::INDEX) {
        categoricalCrossEntropyBuilder.receivesClassIndexLabels(numClasses);
    } else {
        assert(labelType == LabelType::ONE_HOT);
        categoricalCrossEntropyBuilder.receivesOneHotLabels();
    }
    CategoricalCrossEntropy crossEntropy = categoricalCrossEntropyBuilder.build();
    currentFeatureInput = crossEntropy.getLoss();

    if (lossType == ThorImplementation::Loss::LossType::BATCH) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsBatchLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::CLASSWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsClasswiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else if (lossType == ThorImplementation::Loss::LossType::ELEMENTWISE) {
        LossShaper lossShaper = LossShaper::Builder().network(*network).lossInput(currentFeatureInput).reportsElementwiseLoss().build();
        lossTensor = lossShaper.getLossOutput();
    } else {
        // No loss shaper needed in this case
        assert(lossType == ThorImplementation::Loss::LossType::RAW);
        lossTensor = currentFeatureInput;
    }
}