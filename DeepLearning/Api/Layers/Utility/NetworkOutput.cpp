#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"

using namespace Thor;

using namespace std;

void NetworkOutput::convertToSingleLayersAndAddToNetwork() {
    assert(isMultiLayer());

    Tensor currentFeatureInput = featureInput.get();

    // Force the input tensor to this type of layer to be FP16
    if (featureInput.get().getDataType() != dataType) {
        currentFeatureInput =
            TypeConverter::Builder().network(*network).featureInput(currentFeatureInput).newDataType(dataType).build().getFeatureOutput();
    }

    currentFeatureInput = NetworkOutput::Builder()
                              .network(*network)
                              .name(name)
                              .inputTensor(currentFeatureInput)
                              .dataType(dataType)
                              .build()
                              .getFeatureOutput();

    // Replace the output on the compound layer to be the outputs of the last stage
    // i.e. tunnel the actual inputs to actual output of the compound layer,
    // Network uses single layers, user uses compound layer.
    featureOutput = currentFeatureInput;
}