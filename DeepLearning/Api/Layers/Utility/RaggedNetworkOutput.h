#pragma once

#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <optional>
#include <string>

namespace Thor {

class Network;

// Exposes a ragged tensor without extending its logical extent. RP7 emits only a
// physical values NetworkOutput; the authoritative host row partition is carried
// as metadata on that values tensor and inferLogical() reconstructs the runtime
// compatibility offsets view from it. Values beyond hostOffsets[B] remain
// undefined capacity.
class RaggedNetworkOutput {
   public:
    class Builder;

    RaggedNetworkOutput() = default;

    const std::string& getName() const { return name_; }
    RaggedTensor getInput() const { return input_; }
    RaggedTensor getFeatureOutput() const { return output_; }

   private:
    std::string name_;
    RaggedTensor input_;
    RaggedTensor output_;

    friend class Builder;
};

class RaggedNetworkOutput::Builder {
   public:
    RaggedNetworkOutput build();

    Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!network_.has_value());
        network_ = &network;
        return *this;
    }

    Builder& name(const std::string& name) {
        THOR_THROW_IF_FALSE(!name.empty());
        THOR_THROW_IF_FALSE(!name_.has_value());
        name_ = name;
        return *this;
    }

    Builder& inputTensor(const RaggedTensor& input) {
        THOR_THROW_IF_FALSE(input.isInitialized());
        THOR_THROW_IF_FALSE(!input_.has_value());
        input_ = input;
        return *this;
    }

   private:
    std::optional<Network*> network_;
    std::optional<std::string> name_;
    std::optional<RaggedTensor> input_;
};

}  // namespace Thor
