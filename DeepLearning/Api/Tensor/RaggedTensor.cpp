#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace Thor;
using json = nlohmann::json;

std::atomic<uint64_t> RaggedTensor::nextId(20000);

RaggedTensor::RaggedTensor(Tensor values, Tensor offsets)
    : id(nextId.fetch_add(1)), originalId(id), values(values), rowPartitionToken(offsets) {
    constructFromValuesAndOffsets();
}

RaggedTensor::RaggedTensor(Tensor values, Tensor offsets, uint64_t maxValuesPerRow)
    : id(nextId.fetch_add(1)), originalId(id), values(values), rowPartitionToken(offsets), maxValuesPerRow(maxValuesPerRow) {
    constructFromValuesAndOffsets();
    THOR_THROW_IF_FALSE(maxValuesPerRow > 0 && maxValuesPerRow <= maxTotalValues);
}

RaggedTensor::RaggedTensor(DataType valuesDataType,
                           const std::vector<uint64_t> &trailingDimensions,
                           uint64_t batchSize,
                           uint64_t maxTotalValues,
                           DataType offsetsDataType)
    : RaggedTensor(Tensor(valuesDataType, makeValuesDimensions(maxTotalValues, trailingDimensions)),
                   Tensor(offsetsDataType, {batchSize + 1})) {}

RaggedTensor::RaggedTensor(DataType valuesDataType,
                           const std::vector<uint64_t> &trailingDimensions,
                           uint64_t batchSize,
                           uint64_t maxTotalValues,
                           uint64_t maxValuesPerRow,
                           DataType offsetsDataType)
    : RaggedTensor(Tensor(valuesDataType, makeValuesDimensions(maxTotalValues, trailingDimensions)),
                   Tensor(offsetsDataType, {batchSize + 1}),
                   maxValuesPerRow) {}

RaggedTensor RaggedTensor::placementNeutral(Tensor values, Tensor rowPartitionToken, uint64_t maxValuesPerRow) {
    RaggedTensor ragged;
    ragged.id = nextId.fetch_add(1);
    ragged.originalId = ragged.id;
    ragged.values = std::move(values);
    ragged.rowPartitionToken = std::move(rowPartitionToken);
    ragged.maxValuesPerRow = maxValuesPerRow;
    ragged.geometryMode = GeometryMode::PLACEMENT_NEUTRAL;
    ragged.constructPlacementNeutral();
    return ragged;
}

std::vector<uint64_t> RaggedTensor::makeValuesDimensions(uint64_t maxTotalValues, const std::vector<uint64_t> &trailingDimensions) {
    THOR_THROW_IF_FALSE(maxTotalValues > 0);
    std::vector<uint64_t> dimensions;
    dimensions.reserve(trailingDimensions.size() + 1);
    dimensions.push_back(maxTotalValues);
    for (uint64_t dim : trailingDimensions) {
        THOR_THROW_IF_FALSE(dim > 0);
        dimensions.push_back(dim);
    }
    return dimensions;
}

void RaggedTensor::constructFromValuesAndOffsets() {
    THOR_THROW_IF_FALSE(values.isInitialized());
    THOR_THROW_IF_FALSE(rowPartitionToken.isInitialized());
    THOR_THROW_IF_FALSE(offsetsDataTypeValid(rowPartitionToken.getDataType()));

    const std::vector<uint64_t> valuesDimensions = values.getDimensions();
    const std::vector<uint64_t> offsetsDimensions = rowPartitionToken.getDimensions();
    THOR_THROW_IF_FALSE(!valuesDimensions.empty());
    THOR_THROW_IF_FALSE(offsetsDimensions.size() == 1);
    THOR_THROW_IF_FALSE(offsetsDimensions[0] >= 1);

    maxTotalValues = valuesDimensions[0];
    batchSize = offsetsDimensions[0] - 1;
    geometryMode = GeometryMode::FIXED;
    rowPartitionId = rowPartitionToken.getId();
    THOR_THROW_IF_FALSE(rowPartitionId != 0);
    THOR_THROW_IF_FALSE(maxTotalValues > 0);
    initialized = true;
}

void RaggedTensor::constructPlacementNeutral() {
    THOR_THROW_IF_FALSE(values.isInitialized());
    THOR_THROW_IF_FALSE(rowPartitionToken.isInitialized());
    THOR_THROW_IF_FALSE(offsetsDataTypeValid(rowPartitionToken.getDataType()));
    THOR_THROW_IF_FALSE(maxValuesPerRow > 0);

    const std::vector<uint64_t> valuesDimensions = values.getDimensions();
    const std::vector<uint64_t> tokenDimensions = rowPartitionToken.getDimensions();
    THOR_THROW_IF_FALSE(!valuesDimensions.empty());
    THOR_THROW_IF_FALSE(valuesDimensions.front() == maxValuesPerRow);
    THOR_THROW_IF_FALSE(tokenDimensions == std::vector<uint64_t>{1});
    for (uint64_t dimension : valuesDimensions) THOR_THROW_IF_FALSE(dimension > 0);

    batchSize = 0;
    maxTotalValues = 0;
    rowPartitionId = rowPartitionToken.getId();
    THOR_THROW_IF_FALSE(rowPartitionId != 0);
    initialized = true;
}

RaggedTensor RaggedTensor::withValues(Tensor newValues) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(newValues.isInitialized());
    const std::vector<uint64_t> dimensions = newValues.getDimensions();
    THOR_THROW_IF_FALSE(!dimensions.empty());
    RaggedTensor result;
    if (isPlacementNeutral()) {
        THOR_THROW_IF_FALSE(dimensions.front() == maxValuesPerRow);
        result = RaggedTensor::placementNeutral(std::move(newValues), rowPartitionToken, maxValuesPerRow);
    } else {
        THOR_THROW_IF_FALSE(dimensions.front() == maxTotalValues);
        result = hasMaxValuesPerRow() ? RaggedTensor(std::move(newValues), rowPartitionToken, maxValuesPerRow)
                                      : RaggedTensor(std::move(newValues), rowPartitionToken);
    }
    result.rowPartitionId = rowPartitionId;
    return result;
}

std::vector<uint64_t> RaggedTensor::getTrailingDimensions() const {
    THOR_THROW_IF_FALSE(initialized);
    std::vector<uint64_t> valuesDimensions = values.getDimensions();
    THOR_THROW_IF_FALSE(!valuesDimensions.empty());
    return std::vector<uint64_t>(valuesDimensions.begin() + 1, valuesDimensions.end());
}

ThorImplementation::RaggedTensorDescriptor RaggedTensor::getDescriptor() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(geometryMode == GeometryMode::FIXED);
    if (hasMaxValuesPerRow()) {
        return ThorImplementation::RaggedTensorDescriptor(values.getDataType(),
                                                          getTrailingDimensions(),
                                                          batchSize,
                                                          maxTotalValues,
                                                          maxValuesPerRow,
                                                          rowPartitionToken.getDataType());
    }
    return ThorImplementation::RaggedTensorDescriptor(
        values.getDataType(), getTrailingDimensions(), batchSize, maxTotalValues, rowPartitionToken.getDataType());
}

ThorImplementation::RaggedTensorDescriptor RaggedTensor::getDescriptorForPlacement(uint64_t placementBatchSize) const {
    THOR_THROW_IF_FALSE(initialized);
    if (geometryMode == GeometryMode::FIXED) {
        THOR_THROW_IF_FALSE(batchSize == placementBatchSize);
        return getDescriptor();
    }

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
    THOR_THROW_IF_FALSE(geometryMode == GeometryMode::FIXED);
    json out{{"version", getVersion()},
             {"id", getId()},
             {"ragged_rank", getRaggedRank()},
             {"batch_size", getBatchSize()},
             {"max_total_values", getMaxTotalValues()},
             {"values", values.architectureJson()},
             {"offsets", rowPartitionToken.architectureJson()}};
    if (hasMaxValuesPerRow()) out["max_values_per_row"] = getMaxValuesPerRow();
    return out;
}

json RaggedTensor::serialize(thor_file::TarWriter &archiveWriter) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(geometryMode == GeometryMode::FIXED);
    json out{{"version", getVersion()},
             {"id", getId()},
             {"ragged_rank", getRaggedRank()},
             {"batch_size", getBatchSize()},
             {"max_total_values", getMaxTotalValues()},
             {"values", values.serialize(archiveWriter)},
             {"offsets", rowPartitionToken.serialize(archiveWriter)}};
    if (hasMaxValuesPerRow()) out["max_values_per_row"] = getMaxValuesPerRow();
    return out;
}

RaggedTensor RaggedTensor::deserialize(const json &j, thor_file::TarReader *archiveReader) {
    const std::string version = j.at("version").get<std::string>();
    if (version != "1.0.0" && version != "1.1.0") {
        throw std::runtime_error("Unsupported version in RaggedTensor::deserialize: " + version);
    }
    if (j.at("ragged_rank").get<uint32_t>() != 1) {
        throw std::runtime_error("Unsupported ragged_rank in RaggedTensor::deserialize: " + std::to_string(j.at("ragged_rank").get<uint32_t>()));
    }

    Tensor values = Tensor::deserialize(j.at("values"), archiveReader);
    Tensor offsets = Tensor::deserialize(j.at("offsets"), archiveReader);
    RaggedTensor ragged = j.contains("max_values_per_row")
        ? RaggedTensor(values, offsets, j.at("max_values_per_row").get<uint64_t>())
        : RaggedTensor(values, offsets);

    const uint64_t expectedBatchSize = j.at("batch_size").get<uint64_t>();
    const uint64_t expectedMaxTotalValues = j.at("max_total_values").get<uint64_t>();
    THOR_THROW_IF_FALSE(ragged.getBatchSize() == expectedBatchSize);
    THOR_THROW_IF_FALSE(ragged.getMaxTotalValues() == expectedMaxTotalValues);

    ragged.originalId = j.at("id").get<uint64_t>();
    return ragged;
}
