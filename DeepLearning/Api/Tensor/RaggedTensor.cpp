#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace Thor;
using json = nlohmann::json;

std::atomic<uint64_t> RaggedTensor::nextId(20000);

RaggedTensor::RaggedTensor(Tensor values, Tensor partitionToken, uint64_t maxValuesPerRow)
    : id(nextId.fetch_add(1)), originalId(id), values(std::move(values)),
      rowPartitionToken(std::move(partitionToken)), maxValuesPerRow(maxValuesPerRow) {
    constructLogical();
}

RaggedTensor RaggedTensor::logical(Tensor values, Tensor rowPartitionToken, uint64_t maxValuesPerRow) {
    return RaggedTensor(std::move(values), std::move(rowPartitionToken), maxValuesPerRow);
}

void RaggedTensor::constructLogical() {
    THOR_THROW_IF_FALSE(values.isInitialized());
    THOR_THROW_IF_FALSE(rowPartitionToken.isInitialized());
    THOR_THROW_IF_FALSE(offsetsDataTypeValid(rowPartitionToken.getDataType()));
    THOR_THROW_IF_FALSE(maxValuesPerRow > 0);

    const std::vector<uint64_t> valuesDimensions = values.getDimensions();
    THOR_THROW_IF_FALSE(!valuesDimensions.empty());
    THOR_THROW_IF_FALSE(valuesDimensions.front() == maxValuesPerRow);
    THOR_THROW_IF_FALSE(rowPartitionToken.getDimensions() == std::vector<uint64_t>{1});
    for (uint64_t dimension : valuesDimensions) THOR_THROW_IF_FALSE(dimension > 0);

    rowPartitionId = rowPartitionToken.getId();
    THOR_THROW_IF_FALSE(rowPartitionId != 0);
    initialized = true;
}

RaggedTensor RaggedTensor::withValues(Tensor newValues) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(newValues.isInitialized());
    const std::vector<uint64_t> dimensions = newValues.getDimensions();
    THOR_THROW_IF_FALSE(!dimensions.empty() && dimensions.front() == maxValuesPerRow);
    RaggedTensor result = RaggedTensor::logical(std::move(newValues), rowPartitionToken, maxValuesPerRow);
    result.rowPartitionId = rowPartitionId;
    return result;
}

std::vector<uint64_t> RaggedTensor::getTrailingDimensions() const {
    THOR_THROW_IF_FALSE(initialized);
    std::vector<uint64_t> valuesDimensions = values.getDimensions();
    THOR_THROW_IF_FALSE(!valuesDimensions.empty());
    return std::vector<uint64_t>(valuesDimensions.begin() + 1, valuesDimensions.end());
}

ThorImplementation::RaggedTensorDescriptor RaggedTensor::getDescriptorForPlacement(uint64_t placementBatchSize) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(placementBatchSize > 0);
    THOR_THROW_IF_FALSE(hasMaxValuesPerRow());
    THOR_THROW_IF_FALSE(maxValuesPerRow <= std::numeric_limits<uint64_t>::max() / placementBatchSize);
    const uint64_t placementMaxTotalValues = placementBatchSize * maxValuesPerRow;
    THOR_THROW_IF_FALSE(ThorImplementation::canonicalRowPartitionOffsetCanRepresent(
        rowPartitionToken.getDataType(), placementMaxTotalValues));
    return ThorImplementation::RaggedTensorDescriptor(values.getDataType(),
                                                      getTrailingDimensions(),
                                                      placementBatchSize,
                                                      placementMaxTotalValues,
                                                      maxValuesPerRow,
                                                      rowPartitionToken.getDataType());
}

json RaggedTensor::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"version", getVersion()},
                {"id", getId()},
                {"ragged_rank", getRaggedRank()},
                {"max_values_per_row", getMaxValuesPerRow()},
                {"values", values.architectureJson()},
                {"offsets", rowPartitionToken.architectureJson()}};
}

json RaggedTensor::serialize(thor_file::TarWriter &archiveWriter) const {
    THOR_THROW_IF_FALSE(initialized);
    return json{{"version", getVersion()},
                {"id", getId()},
                {"ragged_rank", getRaggedRank()},
                {"max_values_per_row", getMaxValuesPerRow()},
                {"values", values.serialize(archiveWriter)},
                {"offsets", rowPartitionToken.serialize(archiveWriter)}};
}

RaggedTensor RaggedTensor::deserialize(const json &j, thor_file::TarReader *archiveReader) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported version in RaggedTensor::deserialize: " + version);
    }
    if (j.at("ragged_rank").get<uint32_t>() != 1) {
        throw std::runtime_error("Unsupported ragged_rank in RaggedTensor::deserialize: " + std::to_string(j.at("ragged_rank").get<uint32_t>()));
    }
    // Old version strings describe input grammar, not a fixed-batch model.
    // A historical packed capacity is not sufficient to infer a semantic M.
    if (!j.contains("max_values_per_row")) {
        throw std::runtime_error("RaggedTensor::deserialize: historical archive has no independently recoverable max_values_per_row");
    }
    const uint64_t m = j.at("max_values_per_row").get<uint64_t>();
    THOR_THROW_IF_FALSE(m > 0);

    const json &valuesJson = j.at("values");
    const json &offsetsJson = j.at("offsets");
    const auto archivedValuesDimensions = valuesJson.at("dimensions").get<std::vector<uint64_t>>();
    const auto archivedOffsetsDimensions = offsetsJson.at("dimensions").get<std::vector<uint64_t>>();
    THOR_THROW_IF_FALSE(!archivedValuesDimensions.empty());
    THOR_THROW_IF_FALSE(archivedOffsetsDimensions.size() == 1 && archivedOffsetsDimensions.front() > 0);
    for (uint64_t dim : archivedValuesDimensions) THOR_THROW_IF_FALSE(dim > 0);

    // The canonical [1] partition-token representation is already logical,
    // so its values must agree with M. Older [B+1] offsets are only physical
    // storage hints; do not interpret either them or top-level B/total fields
    // as logical geometry or as placement restrictions.
    if (archivedOffsetsDimensions.front() == 1) {
        THOR_THROW_IF_FALSE(archivedValuesDimensions.front() == m);
    }
    std::vector<uint64_t> logicalValuesDimensions = archivedValuesDimensions;
    logicalValuesDimensions.front() = m;
    Tensor logicalValues = Tensor::deserializeWithDimensions(valuesJson, logicalValuesDimensions, archiveReader);
    Tensor logicalPartition = Tensor::deserializeWithDimensions(offsetsJson, {1}, archiveReader);
    RaggedTensor ragged = RaggedTensor::logical(std::move(logicalValues), std::move(logicalPartition), m);
    ragged.originalId = j.at("id").get<uint64_t>();
    return ragged;
}
