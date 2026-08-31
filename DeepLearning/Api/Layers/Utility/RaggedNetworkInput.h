#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class Network;

// Declares a logical ragged network boundary. The default form owns a new
// external values + offsets pair; Builder::partition(...) declares a values-only
// boundary that reuses an existing logical input's exact row partition. The row
// partition remains authoritative: values beyond offsets[B] are inactive,
// undefined capacity. RaggedNetworkInput copies packed values without inspecting
// or canonicalizing that inactive storage; consumers that over-read own their
// required sanitation.
class RaggedNetworkInput {
   public:
    class Builder;
};

class RaggedNetworkInput::Builder {
   public:
    virtual RaggedTensor build();

    virtual RaggedNetworkInput::Builder& network(Network& network);
    virtual RaggedNetworkInput::Builder& name(const std::string& name);
    virtual RaggedNetworkInput::Builder& valuesDataType(DataType dataType);
    virtual RaggedNetworkInput::Builder& offsetsDataType(DataType dataType);
    virtual RaggedNetworkInput::Builder& trailingDimensions(const std::vector<uint64_t>& dimensions);
    virtual RaggedNetworkInput::Builder& maxTotalValues(uint64_t maxTotalValues);
    virtual RaggedNetworkInput::Builder& maxValuesPerRow(uint64_t maxValuesPerRow);
    virtual RaggedNetworkInput::Builder& batchSize(uint64_t batchSize);
    // Reuse the exact row partition of an existing logical RaggedNetworkInput.
    // In this mode only <name>.values is declared as a new external boundary;
    // batch/offset/capacity metadata is inherited from partition and must not be
    // redundantly specified on this builder.
    virtual RaggedNetworkInput::Builder& partition(const RaggedTensor& partition);

   private:
    std::optional<Network*> network_;
    std::optional<std::string> name_;
    std::optional<DataType> valuesDataType_;
    std::optional<DataType> offsetsDataType_;
    std::vector<uint64_t> trailingDimensions_;
    std::optional<uint64_t> maxTotalValues_;
    std::optional<uint64_t> maxValuesPerRow_;
    std::optional<uint64_t> batchSize_;
    std::optional<RaggedTensor> partition_;
};

}  // namespace Thor
