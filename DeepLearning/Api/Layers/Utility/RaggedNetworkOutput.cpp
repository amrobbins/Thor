#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"

#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

namespace Thor {

RaggedNetworkOutput RaggedNetworkOutput::Builder::build() {
    THOR_THROW_IF_FALSE(network_.has_value());
    THOR_THROW_IF_FALSE(name_.has_value());
    THOR_THROW_IF_FALSE(input_.has_value());
    THOR_THROW_IF_FALSE(!network_.value()->hasRaggedNetworkOutput(name_.value()));
    for (const std::string& externalOutputName : network_.value()->getExternalNetworkOutputNames()) {
        THOR_THROW_IF_FALSE(externalOutputName != name_.value());
    }

    const std::string valuesOutputName = "__thor_ragged_output." + name_.value() + ".values";
    const std::string offsetsOutputName = "__thor_ragged_output." + name_.value() + ".offsets";

    NetworkOutput valuesOutput = NetworkOutput::Builder()
                                     .network(*network_.value())
                                     .name(valuesOutputName)
                                     .inputTensor(input_->getValues())
                                     .dataType(input_->getValuesDataType())
                                     .external(false)
                                     .build();
    NetworkOutput offsetsOutput = NetworkOutput::Builder()
                                      .network(*network_.value())
                                      .name(offsetsOutputName)
                                      .inputTensor(input_->getOffsets())
                                      .dataType(input_->getOffsetsDataType())
                                      .external(false)
                                      .build();

    RaggedNetworkOutput result;
    result.name_ = name_.value();
    result.input_ = input_.value();
    result.output_ = input_->hasMaxValuesPerRow()
        ? RaggedTensor(valuesOutput.getFeatureOutput().value(),
                       offsetsOutput.getFeatureOutput().value(),
                       input_->getMaxValuesPerRow())
        : RaggedTensor(valuesOutput.getFeatureOutput().value(), offsetsOutput.getFeatureOutput().value());
    network_.value()->registerRaggedNetworkOutput(
        result.name_, result.output_, valuesOutputName, offsetsOutputName);
    return result;
}

}  // namespace Thor
