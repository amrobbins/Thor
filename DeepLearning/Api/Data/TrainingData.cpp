#include "DeepLearning/Api/Data/TrainingData.h"

#include <stdexcept>
#include <utility>

namespace Thor {
namespace {

DatasetFieldMaterializationRequirements allFieldRequirementsOrThrow(const DatasetSchema& schema) {
    DatasetFieldMaterializationRequirements requirements;
    for (const DatasetField& field : schema.getFields()) {
        if (field.kind == DatasetFieldKind::RAGGED) {
            throw std::runtime_error(
                "TrainingData cannot open all fields when the dataset contains ragged field '" + field.name +
                "'. Ragged batch materialization requires a consuming RaggedNetworkInput descriptor.");
        }
        requirements.emplace(field.id, DatasetFieldMaterializationRequirement::dense(field.id));
    }
    return requirements;
}

void validateFieldRequirements(const DatasetSchema& schema,
                               const DatasetFieldMaterializationRequirements& requirements,
                               uint64_t batchSize) {
    for (const auto& [fieldId, requirement] : requirements) {
        if (fieldId != requirement.fieldId) {
            throw std::runtime_error("Dataset field materialization requirement key/id mismatch.");
        }
        const DatasetField& field = schema.getField(fieldId);
        if (field.kind == DatasetFieldKind::RAGGED) {
            if (!requirement.raggedTensorDescriptor.has_value()) {
                throw std::runtime_error("Ragged dataset field '" + field.name +
                                         "' requires a RaggedTensor materialization descriptor.");
            }
            const auto& descriptor = requirement.raggedTensorDescriptor.value();
            if (descriptor.getValuesDataType() != field.dataType ||
                descriptor.getTrailingDimensions() != field.dimensions ||
                descriptor.getBatchSize() != batchSize) {
                throw std::runtime_error("Ragged dataset field materialization contract does not match field '" +
                                         field.name + "'.");
            }
        } else if (requirement.raggedTensorDescriptor.has_value()) {
            throw std::runtime_error("Non-ragged dataset field '" + field.name +
                                     "' cannot carry a RaggedTensor materialization descriptor.");
        }
    }
}

}  // namespace

TrainingData::TrainingData(std::shared_ptr<const NamedDataset> dataset,
                           DatasetSplitManifest splits,
                           BatchPolicy batching,
                           DatasetAccessPolicy accessPolicy,
                           std::string datasetName)
    : TrainingData(std::move(dataset),
                   std::optional<DatasetSplitManifest>(std::move(splits)),
                   std::move(batching),
                   accessPolicy,
                   std::move(datasetName)) {}

TrainingData::TrainingData(std::shared_ptr<const NamedDataset> dataset,
                           std::optional<DatasetSplitManifest> splits,
                           BatchPolicy batching,
                           DatasetAccessPolicy accessPolicy,
                           std::string datasetName)
    : dataset(std::move(dataset)),
      splits(std::move(splits)),
      batching(std::move(batching)),
      accessPolicy(accessPolicy),
      datasetName(std::move(datasetName)) {
    if (this->dataset == nullptr) {
        throw std::runtime_error("TrainingData dataset must not be null.");
    }
    if (this->datasetName.empty()) {
        throw std::runtime_error("TrainingData dataset_name must not be empty.");
    }
    if (this->splits.has_value()) {
        this->splits->validateAgainst(*this->dataset);
    }
}

const DatasetSplitManifest &TrainingData::getSplits() const {
    if (!splits.has_value()) {
        throw std::runtime_error(
            "TrainingData has no dataset split manifest. Provide splits for a new training run, or use a "
            "repository-backed TrainingRuns resume with a previously persisted split.");
    }
    return splits.value();
}

void TrainingData::requireNonEmptyPartition(ExampleType exampleType, const std::string& context) const {
    const ExampleIndexSet* partition = nullptr;
    const char* partitionName = nullptr;
    switch (exampleType) {
        case ExampleType::TRAIN:
            partition = &getSplits().getTrain();
            partitionName = "train";
            break;
        case ExampleType::VALIDATE:
            partition = &getSplits().getValidate();
            partitionName = "validate";
            break;
        case ExampleType::TEST:
            partition = &getSplits().getTest();
            partitionName = "test";
            break;
        default:
            throw std::runtime_error(context + " requires a concrete dataset partition.");
    }
    if (partition->empty()) {
        throw std::runtime_error(context + " requires a non-empty " + partitionName + " partition.");
    }
}

std::shared_ptr<BatchSession> TrainingData::openSession(uint64_t maxInFlightBatches) const {
    return openSession(maxInFlightBatches, allFieldRequirementsOrThrow(dataset->getSchema()));
}

std::shared_ptr<BatchSession> TrainingData::openSession(
    uint64_t maxInFlightBatches,
    const DatasetFieldMaterializationRequirements& fieldRequirements) const {
    if (maxInFlightBatches == 0) {
        throw std::runtime_error("TrainingData max_in_flight_batches must be >= 1.");
    }
    DatasetFieldMaterializationRequirements effectiveRequirements = fieldRequirements.empty()
        ? allFieldRequirementsOrThrow(dataset->getSchema())
        : fieldRequirements;
    validateFieldRequirements(dataset->getSchema(), effectiveRequirements, batching.getBatchSize());
    std::shared_ptr<BatchSession> session = dataset->openBatchSession(
        getSplits(), batching, accessPolicy, maxInFlightBatches, effectiveRequirements);
    if (session == nullptr) {
        throw std::runtime_error("NamedDataset backend returned a null BatchSession.");
    }
    session->setDatasetName(datasetName);
    return session;
}

std::shared_ptr<BatchSession> TrainingData::openValidationSession(
    const std::string& validationPopulation,
    uint64_t maxInFlightBatches) const {
    return openValidationSession(
        validationPopulation, maxInFlightBatches, allFieldRequirementsOrThrow(dataset->getSchema()));
}

std::shared_ptr<BatchSession> TrainingData::openValidationSession(
    const std::string& validationPopulation,
    uint64_t maxInFlightBatches,
    const DatasetFieldMaterializationRequirements& fieldRequirements) const {
    if (maxInFlightBatches == 0) {
        throw std::runtime_error("TrainingData max_in_flight_batches must be >= 1.");
    }
    (void)getSplits().getValidation(validationPopulation);
    DatasetFieldMaterializationRequirements effectiveRequirements = fieldRequirements.empty()
        ? allFieldRequirementsOrThrow(dataset->getSchema())
        : fieldRequirements;
    validateFieldRequirements(dataset->getSchema(), effectiveRequirements, batching.getBatchSize());
    DatasetSplitManifest selectedSplits = getSplits().withDefaultValidation(validationPopulation);
    std::shared_ptr<BatchSession> session = dataset->openBatchSession(
        selectedSplits, batching, accessPolicy, maxInFlightBatches, effectiveRequirements);
    if (session == nullptr) {
        throw std::runtime_error("NamedDataset backend returned a null BatchSession.");
    }
    session->setDatasetName(datasetName + ":validation:" + validationPopulation);
    return session;
}

}  // namespace Thor
