#include "DeepLearning/Api/Data/TrainingData.h"

#include "DeepLearning/Api/Loaders/IndexedNamedBatchSession.h"
#include "DeepLearning/Api/Data/LocalNamedDataset.h"

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
    if (this->splits.getTrain().empty()) {
        throw std::runtime_error("TrainingData train partition must contain at least one row index.");
    }
}


TrainingData::TrainingData(std::shared_ptr<const NamedDataset> dataset,
                           DatasetSplitManifest splits,
                           BatchPolicy batching,
                           std::string datasetName)
    : TrainingData(std::move(dataset),
                   std::move(splits),
                   std::move(batching),
                   DatasetAccessPolicy{},
                   std::move(datasetName)) {}

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
    std::shared_ptr<const LocalNamedDataset> localDataset =
        std::dynamic_pointer_cast<const LocalNamedDataset>(dataset);
    if (localDataset == nullptr) {
        throw std::runtime_error("TrainingData has no BatchSession implementation for this dataset backend.");
    }
    auto session = std::make_shared<IndexedNamedBatchSession>(
        localDataset, splits, batching, maxInFlightBatches, requiredFieldIds);
    session->setDatasetName(datasetName);
    return session;
}

}  // namespace Thor
