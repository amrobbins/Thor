#pragma once

#include "DeepLearning/Api/Data/DatasetFieldMaterializationRequirement.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class NamedDataset;
class Network;
class NetworkInput;

struct CompiledDatasetInputBindings {
    std::vector<TrainingInputBinding> trainingInputBindings;
    DatasetFieldMaterializationRequirements fieldRequirements;

    bool operator==(const CompiledDatasetInputBindings&) const = default;
};

/**
 * Explicit, conversion-free bindings between immutable dataset fields and
 * logical external network inputs.
 *
 * Ordinary NetworkInput ports bind directly. A RaggedNetworkInput binds as one
 * logical endpoint even though the API graph contains physical .values and
 * .offsets NetworkInputs underneath it. Physical ragged components are not
 * independently bindable dataset inputs.
 */
class DatasetInputBindings {
   public:
    DatasetInputBindings() = default;

    DatasetInputBindings &bind(const NetworkInput &networkInput, const DatasetField &field);
    DatasetInputBindings &bind(const Network &network,
                               const RaggedTensor &raggedNetworkInput,
                               const DatasetField &field);

    [[nodiscard]] static DatasetInputBindings byExactName(const Network &network,
                                                          const NamedDataset &dataset);

    [[nodiscard]] CompiledDatasetInputBindings compile(const Network &network,
                                                       const NamedDataset &dataset,
                                                       uint64_t batchSize) const;

    /**
     * Resolve a Network's logical external inputs directly against a dataset
     * schema. Dataset fields that are not consumed by the Network are ignored;
     * every consumed logical input must resolve to one dataset field.
     *
     * Explicit TrainingInputBinding remaps logical input names. For ragged
     * inputs this is the RaggedNetworkInput name, never the physical .values or
     * .offsets implementation names.
     */
    [[nodiscard]] static CompiledDatasetInputBindings compileByName(
        const Network &network,
        const NamedDataset &dataset,
        uint64_t batchSize,
        const std::vector<TrainingInputBinding> &explicitBindings = {});

    [[nodiscard]] uint64_t size() const { return static_cast<uint64_t>(entries.size()); }
    [[nodiscard]] bool empty() const { return entries.empty(); }

   private:
    enum class EntryKind { DENSE, RAGGED };

    struct Entry {
        EntryKind kind = EntryKind::DENSE;
        uint64_t networkInputLayerId = 0;
        uint64_t raggedTensorId = 0;
        std::string networkInputName;
        ThorImplementation::DataType networkInputDataType = ThorImplementation::DataType::FP32;
        std::vector<uint64_t> networkInputDimensions;
        bool dimensionsIncludeBatch = false;
        std::optional<ThorImplementation::RaggedTensorDescriptor> raggedDescriptor;
        DatasetField field;
    };

    std::vector<Entry> entries;
};

}  // namespace Thor
