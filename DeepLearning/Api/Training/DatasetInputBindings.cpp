#include "DeepLearning/Api/Training/DatasetInputBindings.h"

#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"

#include <map>
#include <memory>
#include <set>
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

struct LogicalNetworkInputs {
    std::map<std::string, std::shared_ptr<NetworkInput>> denseByName;
    std::map<uint64_t, std::shared_ptr<NetworkInput>> denseById;
    std::map<std::string, RaggedNetworkInputReference> raggedByName;
    std::map<uint64_t, RaggedNetworkInputReference> raggedByTensorId;
};

LogicalNetworkInputs collectLogicalNetworkInputs(const Network &network) {
    LogicalNetworkInputs logical;
    std::set<std::string> raggedPhysicalNames;
    for (const RaggedNetworkInputReference &ragged : network.getExternalRaggedNetworkInputs()) {
        if (!logical.raggedByName.emplace(ragged.name, ragged).second) {
            throw std::runtime_error("Network contains duplicate logical RaggedNetworkInput name '" +
                                     ragged.name + "'.");
        }
        if (!logical.raggedByTensorId.emplace(ragged.raggedTensor.getId(), ragged).second) {
            throw std::runtime_error("Network contains duplicate logical RaggedNetworkInput tensor identity.");
        }
        if (!raggedPhysicalNames.insert(ragged.valuesInputName).second ||
            !raggedPhysicalNames.insert(ragged.offsetsInputName).second) {
            throw std::runtime_error("Network contains overlapping RaggedNetworkInput physical components.");
        }
    }

    for (const std::shared_ptr<NetworkInput> &input : network.getExternalNetworkInputs()) {
        if (raggedPhysicalNames.contains(input->getName())) {
            continue;
        }
        if (logical.raggedByName.contains(input->getName())) {
            throw std::runtime_error("Network logical input name '" + input->getName() +
                                     "' is shared by dense and ragged inputs.");
        }
        if (!logical.denseByName.emplace(input->getName(), input).second) {
            throw std::runtime_error("Network contains duplicate external NetworkInput name '" +
                                     input->getName() + "'.");
        }
        logical.denseById.emplace(input->getId(), input);
    }
    return logical;
}

const DatasetField &canonicalFieldFor(const NamedDataset &dataset, const DatasetField &boundField) {
    const DatasetField *canonicalField = nullptr;
    try {
        canonicalField = &dataset.getSchema().getField(boundField.id);
    } catch (const std::exception &) {
        throw std::runtime_error("DatasetInputBindings field '" + boundField.name +
                                 "' does not belong to the dataset being compiled.");
    }
    if (*canonicalField != boundField) {
        throw std::runtime_error("DatasetInputBindings field contract for '" + boundField.name +
                                 "' does not match the dataset being compiled.");
    }
    return *canonicalField;
}

void validateDenseBinding(const NetworkInput &input,
                          const DatasetField &field,
                          uint64_t batchSize) {
    if (field.kind == DatasetFieldKind::RAGGED) {
        throw std::runtime_error("DatasetInputBindings cannot bind ragged dataset field '" + field.name +
                                 "' to ordinary NetworkInput '" + input.getName() +
                                 "'. Bind it to the logical RaggedNetworkInput instead.");
    }
    if (input.getDataType() != field.dataType) {
        throw std::runtime_error("DatasetInputBindings dtype mismatch: dataset field '" +
                                 field.name + "' cannot bind NetworkInput '" +
                                 input.getName() + "'. Add an explicit graph TypeConversion after a matching NetworkInput.");
    }
    const std::vector<uint64_t> expectedDimensions = expectedNetworkInputDimensions(
        field, input.dimensionsIncludeBatch(), batchSize);
    if (input.getDimensions() != expectedDimensions) {
        throw std::runtime_error("DatasetInputBindings shape mismatch: dataset field '" +
                                 field.name + "' has per-example shape " +
                                 dimensionsToString(field.dimensions) +
                                 " but NetworkInput '" + input.getName() + "' declares " +
                                 dimensionsToString(input.getDimensions()) + ".");
    }
}

void validateRaggedBinding(const RaggedNetworkInputReference &input,
                           const DatasetField &field,
                           uint64_t batchSize) {
    if (field.kind != DatasetFieldKind::RAGGED) {
        throw std::runtime_error("DatasetInputBindings logical RaggedNetworkInput '" + input.name +
                                 "' requires a RAGGED dataset field, but field '" + field.name +
                                 "' is not ragged.");
    }
    const ThorImplementation::RaggedTensorDescriptor descriptor = input.raggedTensor.getDescriptor();
    if (descriptor.getBatchSize() != batchSize) {
        throw std::runtime_error("DatasetInputBindings ragged batch-size mismatch: logical input '" +
                                 input.name + "' declares batch capacity " +
                                 std::to_string(descriptor.getBatchSize()) +
                                 " but the dataset batch policy uses " + std::to_string(batchSize) + ".");
    }
    if (descriptor.getValuesDataType() != field.dataType) {
        throw std::runtime_error("DatasetInputBindings ragged dtype mismatch: dataset field '" +
                                 field.name + "' cannot bind logical RaggedNetworkInput '" + input.name + "'.");
    }
    if (descriptor.getTrailingDimensions() != field.dimensions) {
        throw std::runtime_error("DatasetInputBindings ragged trailing-dimension mismatch: dataset field '" +
                                 field.name + "' has per-value shape " + dimensionsToString(field.dimensions) +
                                 " but logical RaggedNetworkInput '" + input.name + "' declares " +
                                 dimensionsToString(descriptor.getTrailingDimensions()) + ".");
    }
}

void insertRequirement(CompiledDatasetInputBindings &compiled,
                       const DatasetField &field,
                       std::optional<ThorImplementation::RaggedTensorDescriptor> raggedDescriptor) {
    DatasetFieldMaterializationRequirement requirement = raggedDescriptor.has_value()
        ? DatasetFieldMaterializationRequirement::ragged(field.id, raggedDescriptor.value())
        : DatasetFieldMaterializationRequirement::dense(field.id);
    if (!compiled.fieldRequirements.emplace(field.id, std::move(requirement)).second) {
        throw std::runtime_error("DatasetInputBindings resolved duplicate dataset field '" + field.name + "'.");
    }
}

}  // namespace

DatasetInputBindings &DatasetInputBindings::bind(const NetworkInput &networkInput,
                                                 const DatasetField &field) {
    if (!networkInput.isExternal() || networkInput.hasPassThroughSource()) {
        throw std::runtime_error("DatasetInputBindings can bind only external NetworkInput ports: '" +
                                 networkInput.getName() + "'.");
    }
    for (const Entry &entry : entries) {
        if ((entry.kind == EntryKind::DENSE && entry.networkInputLayerId == networkInput.getId()) ||
            entry.networkInputName == networkInput.getName()) {
            throw std::runtime_error("DatasetInputBindings contains a duplicate binding for network input '" +
                                     networkInput.getName() + "'.");
        }
        if (entry.field.id == field.id || entry.field.name == field.name) {
            throw std::runtime_error("DatasetInputBindings contains a duplicate binding for dataset field '" +
                                     field.name + "'. Use graph fanout when one field must feed multiple consumers.");
        }
    }

    entries.push_back(Entry{.kind = EntryKind::DENSE,
                            .networkInputLayerId = networkInput.getId(),
                            .networkInputName = networkInput.getName(),
                            .networkInputDataType = networkInput.getDataType(),
                            .networkInputDimensions = networkInput.getDimensions(),
                            .dimensionsIncludeBatch = networkInput.dimensionsIncludeBatch(),
                            .field = field});
    return *this;
}

DatasetInputBindings &DatasetInputBindings::bind(const Network &network,
                                                 const RaggedTensor &raggedNetworkInput,
                                                 const DatasetField &field) {
    const std::vector<RaggedNetworkInputReference> raggedInputs = network.getExternalRaggedNetworkInputs();
    const RaggedNetworkInputReference *match = nullptr;
    for (const RaggedNetworkInputReference &candidate : raggedInputs) {
        if (candidate.raggedTensor == raggedNetworkInput) {
            match = &candidate;
            break;
        }
    }
    if (match == nullptr) {
        throw std::runtime_error("DatasetInputBindings ragged input does not belong to the Network being bound.");
    }
    for (const Entry &entry : entries) {
        if ((entry.kind == EntryKind::RAGGED && entry.raggedTensorId == raggedNetworkInput.getId()) ||
            entry.networkInputName == match->name) {
            throw std::runtime_error("DatasetInputBindings contains a duplicate binding for logical RaggedNetworkInput '" +
                                     match->name + "'.");
        }
        if (entry.field.id == field.id || entry.field.name == field.name) {
            throw std::runtime_error("DatasetInputBindings contains a duplicate binding for dataset field '" +
                                     field.name + "'. Use graph fanout when one field must feed multiple consumers.");
        }
    }

    entries.push_back(Entry{.kind = EntryKind::RAGGED,
                            .raggedTensorId = raggedNetworkInput.getId(),
                            .networkInputName = match->name,
                            .networkInputDataType = match->raggedTensor.getValuesDataType(),
                            .networkInputDimensions = match->raggedTensor.getTrailingDimensions(),
                            .raggedDescriptor = match->raggedTensor.getDescriptor(),
                            .field = field});
    return *this;
}

DatasetInputBindings DatasetInputBindings::byExactName(const Network &network,
                                                       const NamedDataset &dataset) {
    DatasetInputBindings bindings;
    const LogicalNetworkInputs logical = collectLogicalNetworkInputs(network);
    for (const auto &[name, input] : logical.denseByName) {
        if (!dataset.getSchema().contains(name)) {
            throw std::runtime_error("DatasetInputBindings.byExactName could not find dataset field '" +
                                     name + "' required by NetworkInput of the same name.");
        }
        bindings.bind(*input, dataset.getField(name));
    }
    for (const auto &[name, input] : logical.raggedByName) {
        if (!dataset.getSchema().contains(name)) {
            throw std::runtime_error("DatasetInputBindings.byExactName could not find ragged dataset field '" +
                                     name + "' required by logical RaggedNetworkInput of the same name.");
        }
        bindings.bind(network, input.raggedTensor, dataset.getField(name));
    }
    return bindings;
}

CompiledDatasetInputBindings DatasetInputBindings::compile(const Network &network,
                                                            const NamedDataset &dataset,
                                                            uint64_t batchSize) const {
    if (batchSize == 0) {
        throw std::runtime_error("DatasetInputBindings batch size must be >= 1.");
    }

    const LogicalNetworkInputs logical = collectLogicalNetworkInputs(network);
    CompiledDatasetInputBindings compiled;
    std::set<std::string> boundNetworkInputNames;

    for (const Entry &entry : entries) {
        const DatasetField &field = canonicalFieldFor(dataset, entry.field);
        if (entry.kind == EntryKind::DENSE) {
            auto inputById = logical.denseById.find(entry.networkInputLayerId);
            if (inputById == logical.denseById.end() ||
                inputById->second->getName() != entry.networkInputName) {
                throw std::runtime_error("DatasetInputBindings binding for NetworkInput '" +
                                         entry.networkInputName +
                                         "' does not belong to the Network being compiled. Physical .values/.offsets "
                                         "ports of a RaggedNetworkInput are not dataset endpoints.");
            }
            const NetworkInput &input = *inputById->second;
            if (input.getDataType() != entry.networkInputDataType ||
                input.getDimensions() != entry.networkInputDimensions ||
                input.dimensionsIncludeBatch() != entry.dimensionsIncludeBatch) {
                throw std::runtime_error("DatasetInputBindings NetworkInput contract changed after binding: '" +
                                         entry.networkInputName + "'.");
            }
            validateDenseBinding(input, field, batchSize);
            insertRequirement(compiled, field, std::nullopt);
        } else {
            auto raggedIt = logical.raggedByName.find(entry.networkInputName);
            if (raggedIt == logical.raggedByName.end() ||
                raggedIt->second.raggedTensor.getId() != entry.raggedTensorId) {
                throw std::runtime_error("DatasetInputBindings binding for logical RaggedNetworkInput '" +
                                         entry.networkInputName +
                                         "' does not belong to the Network being compiled.");
            }
            const RaggedNetworkInputReference &input = raggedIt->second;
            const ThorImplementation::RaggedTensorDescriptor descriptor = input.raggedTensor.getDescriptor();
            if (!entry.raggedDescriptor.has_value() || descriptor != entry.raggedDescriptor.value() ||
                input.raggedTensor.getValuesDataType() != entry.networkInputDataType ||
                input.raggedTensor.getTrailingDimensions() != entry.networkInputDimensions) {
                throw std::runtime_error("DatasetInputBindings RaggedNetworkInput contract changed after binding: '" +
                                         entry.networkInputName + "'.");
            }
            validateRaggedBinding(input, field, batchSize);
            insertRequirement(compiled, field, descriptor);
        }

        if (!boundNetworkInputNames.insert(entry.networkInputName).second) {
            throw std::runtime_error("DatasetInputBindings resolved duplicate logical network input '" +
                                     entry.networkInputName + "'.");
        }
        compiled.trainingInputBindings.emplace_back(entry.networkInputName, field.name);
    }

    for (const auto &[name, input] : logical.denseByName) {
        (void)input;
        if (!boundNetworkInputNames.contains(name)) {
            throw std::runtime_error("DatasetInputBindings is missing required external NetworkInput '" + name + "'.");
        }
    }
    for (const auto &[name, input] : logical.raggedByName) {
        (void)input;
        if (!boundNetworkInputNames.contains(name)) {
            throw std::runtime_error("DatasetInputBindings is missing required logical RaggedNetworkInput '" + name + "'.");
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

    const LogicalNetworkInputs logical = collectLogicalNetworkInputs(network);
    auto containsLogicalInput = [&logical](const std::string &name) {
        return logical.denseByName.contains(name) || logical.raggedByName.contains(name);
    };

    std::map<std::string, std::string> datasetFieldNameByNetworkInput;
    for (const TrainingInputBinding &binding : explicitBindings) {
        if (!binding.isInitialized()) {
            throw std::runtime_error("Dataset input bindings must all be initialized.");
        }
        if (!containsLogicalInput(binding.getNetworkInputName())) {
            throw std::runtime_error("Dataset input binding references unknown logical external network input '" +
                                     binding.getNetworkInputName() +
                                     "'. Physical .values/.offsets names of a RaggedNetworkInput are not bindable.");
        }
        auto [it, inserted] = datasetFieldNameByNetworkInput.emplace(
            binding.getNetworkInputName(), binding.getBatchInputName());
        if (!inserted && it->second != binding.getBatchInputName()) {
            throw std::runtime_error("Dataset input bindings contain conflicting mappings for network input '" +
                                     binding.getNetworkInputName() + "'.");
        }
    }

    CompiledDatasetInputBindings compiled;
    std::set<std::string> usedDatasetFieldNames;
    auto resolveField = [&](const std::string &inputName) -> const DatasetField& {
        const auto explicitIt = datasetFieldNameByNetworkInput.find(inputName);
        const std::string fieldName = explicitIt == datasetFieldNameByNetworkInput.end()
                                          ? inputName
                                          : explicitIt->second;
        if (!dataset.getSchema().contains(fieldName)) {
            throw std::runtime_error("Dataset does not contain field '" + fieldName +
                                     "' required by logical external network input '" + inputName + "'.");
        }
        const DatasetField &field = dataset.getField(fieldName);
        if (!usedDatasetFieldNames.insert(field.name).second) {
            throw std::runtime_error("Dataset field '" + field.name +
                                     "' is bound to more than one logical external network input. Use graph fanout from one input instead.");
        }
        return field;
    };

    for (const auto &[inputName, input] : logical.denseByName) {
        const DatasetField &field = resolveField(inputName);
        validateDenseBinding(*input, field, batchSize);
        compiled.trainingInputBindings.emplace_back(inputName, field.name);
        insertRequirement(compiled, field, std::nullopt);
    }
    for (const auto &[inputName, input] : logical.raggedByName) {
        const DatasetField &field = resolveField(inputName);
        validateRaggedBinding(input, field, batchSize);
        compiled.trainingInputBindings.emplace_back(inputName, field.name);
        insertRequirement(compiled, field, input.raggedTensor.getDescriptor());
    }

    return compiled;
}

}  // namespace Thor
