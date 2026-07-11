#include "DeepLearning/Api/Training/DatasetInputBindings.h"

#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"

#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace Thor {
namespace {

std::string dimensionsToString(const std::vector<uint64_t> &dimensions) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << dimensions[i];
    }
    out << ']';
    return out.str();
}

std::vector<uint64_t> expectedNetworkInputDimensions(const DatasetField &field,
                                                     bool dimensionsIncludeBatch,
                                                     uint64_t batchSize) {
    if (!dimensionsIncludeBatch) {
        return field.dimensions;
    }
    std::vector<uint64_t> dimensions;
    dimensions.reserve(field.dimensions.size() + 1);
    dimensions.push_back(batchSize);
    dimensions.insert(dimensions.end(), field.dimensions.begin(), field.dimensions.end());
    return dimensions;
}

}  // namespace

DatasetInputBindings &DatasetInputBindings::bind(const NetworkInput &networkInput,
                                                 const DatasetField &field) {
    if (!networkInput.isExternal() || networkInput.hasPassThroughSource()) {
        throw std::runtime_error("DatasetInputBindings can bind only external NetworkInput ports: '" +
                                 networkInput.getName() + "'.");
    }
    for (const Entry &entry : entries) {
        if (entry.networkInputLayerId == networkInput.getId() ||
            entry.networkInputName == networkInput.getName()) {
            throw std::runtime_error("DatasetInputBindings contains a duplicate binding for NetworkInput '" +
                                     networkInput.getName() + "'.");
        }
        if (entry.field.id == field.id || entry.field.name == field.name) {
            throw std::runtime_error("DatasetInputBindings contains a duplicate binding for dataset field '" +
                                     field.name + "'. Use graph fanout when one field must feed multiple consumers.");
        }
    }

    entries.push_back(Entry{.networkInputLayerId = networkInput.getId(),
                            .networkInputName = networkInput.getName(),
                            .networkInputDataType = networkInput.getDataType(),
                            .networkInputDimensions = networkInput.getDimensions(),
                            .dimensionsIncludeBatch = networkInput.dimensionsIncludeBatch(),
                            .field = field});
    return *this;
}

DatasetInputBindings DatasetInputBindings::byExactName(const Network &network,
                                                       const NamedDataset &dataset) {
    DatasetInputBindings bindings;
    for (const std::shared_ptr<NetworkInput> &input : network.getExternalNetworkInputs()) {
        if (!dataset.getSchema().contains(input->getName())) {
            throw std::runtime_error("DatasetInputBindings.byExactName could not find dataset field '" +
                                     input->getName() + "' required by NetworkInput of the same name.");
        }
        bindings.bind(*input, dataset.getField(input->getName()));
    }
    return bindings;
}

CompiledDatasetInputBindings DatasetInputBindings::compile(const Network &network,
                                                            const NamedDataset &dataset,
                                                            uint64_t batchSize) const {
    if (batchSize == 0) {
        throw std::runtime_error("DatasetInputBindings batch size must be >= 1.");
    }

    std::map<std::string, std::shared_ptr<NetworkInput>> externalInputsByName;
    std::map<uint64_t, std::shared_ptr<NetworkInput>> externalInputsById;
    for (const std::shared_ptr<NetworkInput> &input : network.getExternalNetworkInputs()) {
        if (!externalInputsByName.emplace(input->getName(), input).second) {
            throw std::runtime_error("Network contains duplicate external NetworkInput name '" +
                                     input->getName() + "'.");
        }
        externalInputsById.emplace(input->getId(), input);
    }

    CompiledDatasetInputBindings compiled;
    std::set<std::string> boundNetworkInputNames;
    for (const Entry &entry : entries) {
        auto inputById = externalInputsById.find(entry.networkInputLayerId);
        if (inputById == externalInputsById.end() ||
            inputById->second->getName() != entry.networkInputName) {
            throw std::runtime_error("DatasetInputBindings binding for NetworkInput '" +
                                     entry.networkInputName +
                                     "' does not belong to the Network being compiled.");
        }
        const NetworkInput &input = *inputById->second;
        if (input.getDataType() != entry.networkInputDataType ||
            input.getDimensions() != entry.networkInputDimensions ||
            input.dimensionsIncludeBatch() != entry.dimensionsIncludeBatch) {
            throw std::runtime_error("DatasetInputBindings NetworkInput contract changed after binding: '" +
                                     entry.networkInputName + "'.");
        }

        const DatasetField *canonicalField = nullptr;
        try {
            canonicalField = &dataset.getSchema().getField(entry.field.id);
        } catch (const std::exception &) {
            throw std::runtime_error("DatasetInputBindings field '" + entry.field.name +
                                     "' does not belong to the dataset being compiled.");
        }
        if (*canonicalField != entry.field) {
            throw std::runtime_error("DatasetInputBindings field contract for '" + entry.field.name +
                                     "' does not match the dataset being compiled.");
        }

        if (input.getDataType() != canonicalField->dataType) {
            throw std::runtime_error("DatasetInputBindings dtype mismatch: dataset field '" +
                                     canonicalField->name + "' cannot bind NetworkInput '" +
                                     input.getName() + "'. Add an explicit graph TypeConversion after a matching NetworkInput.");
        }
        const std::vector<uint64_t> expectedDimensions = expectedNetworkInputDimensions(
            *canonicalField, input.dimensionsIncludeBatch(), batchSize);
        if (input.getDimensions() != expectedDimensions) {
            throw std::runtime_error("DatasetInputBindings shape mismatch: dataset field '" +
                                     canonicalField->name + "' has per-example shape " +
                                     dimensionsToString(canonicalField->dimensions) +
                                     " but NetworkInput '" + input.getName() + "' declares " +
                                     dimensionsToString(input.getDimensions()) + ".");
        }

        if (!boundNetworkInputNames.insert(input.getName()).second) {
            throw std::runtime_error("DatasetInputBindings resolved duplicate NetworkInput '" +
                                     input.getName() + "'.");
        }
        if (!compiled.requiredFieldIds.insert(canonicalField->id).second) {
            throw std::runtime_error("DatasetInputBindings resolved duplicate dataset field '" +
                                     canonicalField->name + "'.");
        }
        compiled.trainingInputBindings.emplace_back(input.getName(), canonicalField->name);
    }

    for (const auto &[name, input] : externalInputsByName) {
        (void)input;
        if (!boundNetworkInputNames.contains(name)) {
            throw std::runtime_error("DatasetInputBindings is missing required external NetworkInput '" +
                                     name + "'.");
        }
    }

    return compiled;
}

CompiledDatasetInputBindings DatasetInputBindings::compileByName(
    const Network &network,
    const NamedDataset &dataset,
    uint64_t batchSize,
    const std::vector<TrainingInputBinding> &explicitBindings) {
    if (batchSize == 0) {
        throw std::runtime_error("DatasetInputBindings batch size must be >= 1.");
    }

    std::map<std::string, std::shared_ptr<NetworkInput>> externalInputsByName;
    for (const std::shared_ptr<NetworkInput> &input : network.getExternalNetworkInputs()) {
        if (!externalInputsByName.emplace(input->getName(), input).second) {
            throw std::runtime_error("Network contains duplicate external NetworkInput name '" +
                                     input->getName() + "'.");
        }
    }

    std::map<std::string, std::string> datasetFieldNameByNetworkInput;
    for (const TrainingInputBinding &binding : explicitBindings) {
        if (!binding.isInitialized()) {
            throw std::runtime_error("Dataset input bindings must all be initialized.");
        }
        if (!externalInputsByName.contains(binding.getNetworkInputName())) {
            throw std::runtime_error("Dataset input binding references unknown external NetworkInput '" +
                                     binding.getNetworkInputName() + "'.");
        }
        auto [it, inserted] = datasetFieldNameByNetworkInput.emplace(
            binding.getNetworkInputName(), binding.getBatchInputName());
        if (!inserted && it->second != binding.getBatchInputName()) {
            throw std::runtime_error("Dataset input bindings contain conflicting mappings for NetworkInput '" +
                                     binding.getNetworkInputName() + "'.");
        }
    }

    CompiledDatasetInputBindings compiled;
    std::set<std::string> usedDatasetFieldNames;
    for (const auto &[inputName, input] : externalInputsByName) {
        const auto explicitIt = datasetFieldNameByNetworkInput.find(inputName);
        const std::string fieldName = explicitIt == datasetFieldNameByNetworkInput.end()
                                          ? inputName
                                          : explicitIt->second;
        if (!dataset.getSchema().contains(fieldName)) {
            throw std::runtime_error("Dataset does not contain field '" + fieldName +
                                     "' required by external NetworkInput '" + inputName + "'.");
        }
        const DatasetField &field = dataset.getField(fieldName);
        if (input->getDataType() != field.dataType) {
            throw std::runtime_error("DatasetInputBindings dtype mismatch: dataset field '" +
                                     field.name + "' cannot bind NetworkInput '" + inputName +
                                     "'. Add an explicit graph TypeConversion after a matching NetworkInput.");
        }
        const std::vector<uint64_t> expectedDimensions = expectedNetworkInputDimensions(
            field, input->dimensionsIncludeBatch(), batchSize);
        if (input->getDimensions() != expectedDimensions) {
            throw std::runtime_error("DatasetInputBindings shape mismatch: dataset field '" +
                                     field.name + "' has per-example shape " +
                                     dimensionsToString(field.dimensions) +
                                     " but NetworkInput '" + inputName + "' declares " +
                                     dimensionsToString(input->getDimensions()) + ".");
        }
        if (!usedDatasetFieldNames.insert(field.name).second) {
            throw std::runtime_error("Dataset field '" + field.name +
                                     "' is bound to more than one external NetworkInput. Use graph fanout from one input instead.");
        }
        compiled.trainingInputBindings.emplace_back(inputName, field.name);
        compiled.requiredFieldIds.insert(field.id);
    }

    return compiled;
}

}  // namespace Thor
