#pragma once

#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace Thor {

class NamedDataset;
class Network;
class NetworkInput;

struct CompiledDatasetInputBindings {
    std::vector<TrainingInputBinding> trainingInputBindings;
    std::set<DatasetFieldId> requiredFieldIds;
};

/**
 * Explicit, conversion-free bindings between immutable dataset fields and
 * external NetworkInput ports.
 *
 * A binding records typed endpoint identity only. It never renames, casts,
 * reshapes, or otherwise changes the dataset field. A different model-side
 * name is supported only through an explicit bind() call.
 */
class DatasetInputBindings {
   public:
    DatasetInputBindings() = default;

    DatasetInputBindings &bind(const NetworkInput &networkInput, const DatasetField &field);

    [[nodiscard]] static DatasetInputBindings byExactName(const Network &network,
                                                          const NamedDataset &dataset);

    [[nodiscard]] CompiledDatasetInputBindings compile(const Network &network,
                                                       const NamedDataset &dataset,
                                                       uint64_t batchSize) const;

    /**
     * Resolve a Network's external inputs directly against a dataset schema.
     * Dataset fields that are not consumed by the Network are intentionally
     * ignored; every consumed external input must resolve to one dataset field.
     *
     * Inputs use exact-name binding unless an explicit TrainingInputBinding
     * remaps the NetworkInput name to a dataset field name.  This form is used
     * for composed TrainingPhase graphs, whose NetworkInput layer identities do
     * not exist until the currently enabled phases are joined.
     */
    [[nodiscard]] static CompiledDatasetInputBindings compileByName(
        const Network &network,
        const NamedDataset &dataset,
        uint64_t batchSize,
        const std::vector<TrainingInputBinding> &explicitBindings = {});

    [[nodiscard]] uint64_t size() const { return static_cast<uint64_t>(entries.size()); }
    [[nodiscard]] bool empty() const { return entries.empty(); }

   private:
    struct Entry {
        uint64_t networkInputLayerId = 0;
        std::string networkInputName;
        ThorImplementation::DataType networkInputDataType = ThorImplementation::DataType::FP32;
        std::vector<uint64_t> networkInputDimensions;
        bool dimensionsIncludeBatch = false;
        DatasetField field;
    };

    std::vector<Entry> entries;
};

}  // namespace Thor
