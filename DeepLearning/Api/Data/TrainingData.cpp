#include "DeepLearning/Api/Data/TrainingData.h"

#include <stdexcept>
#include <utility>

namespace Thor {

TrainingData::TrainingData(std::shared_ptr<const NamedDataset> dataset,
                           DatasetSplitManifest splits,
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
    this->splits.validateAgainst(*this->dataset);
}


void TrainingData::requireNonEmptyPartition(ExampleType exampleType, const std::string& context) const {
    const ExampleIndexSet* partition = nullptr;
    const char* partitionName = nullptr;
    switch (exampleType) {
        case ExampleType::TRAIN:
            partition = &splits.getTrain();
            partitionName = "train";
            break;
        case ExampleType::VALIDATE:
            partition = &splits.getValidate();
            partitionName = "validate";
            break;
        case ExampleType::TEST:
            partition = &splits.getTest();
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
    std::set<DatasetFieldId> allFields;
    for (const DatasetField& field : dataset->getSchema().getFields()) {
        allFields.insert(field.id);
    }
    return openSession(maxInFlightBatches, allFields);
}

std::shared_ptr<BatchSession> TrainingData::openSession(
    uint64_t maxInFlightBatches,
    const std::set<DatasetFieldId>& requiredFieldIds) const {
    if (maxInFlightBatches == 0) {
        throw std::runtime_error("TrainingData max_in_flight_batches must be >= 1.");
    }
    for (DatasetFieldId fieldId : requiredFieldIds) {
        (void)dataset->getSchema().getField(fieldId);
    }
    std::shared_ptr<BatchSession> session = dataset->openBatchSession(
        splits, batching, accessPolicy, maxInFlightBatches, requiredFieldIds);
    if (session == nullptr) {
        throw std::runtime_error("NamedDataset backend returned a null BatchSession.");
    }
    session->setDatasetName(datasetName);
    return session;
}

}  // namespace Thor
