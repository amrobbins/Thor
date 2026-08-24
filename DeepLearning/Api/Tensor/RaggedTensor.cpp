#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <stdexcept>
#include <string>
#include <utility>

using namespace Thor;
using json = nlohmann::json;

std::atomic<uint64_t> RaggedTensor::nextId(20000);

RaggedTensor::RaggedTensor(Tensor values, Tensor offsets)
    : id(nextId.fetch_add(1)), originalId(id), values(values), offsets(offsets) {
    constructFromValuesAndOffsets();
}

RaggedTensor::RaggedTensor(Tensor values, Tensor offsets, uint64_t maxValuesPerRow)
    : id(nextId.fetch_add(1)), originalId(id), values(values), offsets(offsets), maxValuesPerRow(maxValuesPerRow) {
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
    THOR_THROW_IF_FALSE(offsets.isInitialized());
    THOR_THROW_IF_FALSE(offsetsDataTypeValid(offsets.getDataType()));

    const std::vector<uint64_t> valuesDimensions = values.getDimensions();
    const std::vector<uint64_t> offsetsDimensions = offsets.getDimensions();
    THOR_THROW_IF_FALSE(!valuesDimensions.empty());
    THOR_THROW_IF_FALSE(offsetsDimensions.size() == 1);
    THOR_THROW_IF_FALSE(offsetsDimensions[0] >= 1);

    maxTotalValues = valuesDimensions[0];
    batchSize = offsetsDimensions[0] - 1;
    THOR_THROW_IF_FALSE(maxTotalValues > 0);
    initialized = true;
}

RaggedTensor RaggedTensor::withValues(Tensor newValues) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(newValues.isInitialized());
    const std::vector<uint64_t> dimensions = newValues.getDimensions();
    THOR_THROW_IF_FALSE(!dimensions.empty());
    THOR_THROW_IF_FALSE(dimensions.front() == maxTotalValues);
    return hasMaxValuesPerRow() ? RaggedTensor(std::move(newValues), offsets, maxValuesPerRow)
                                : RaggedTensor(std::move(newValues), offsets);
}

std::vector<uint64_t> RaggedTensor::getTrailingDimensions() const {
    THOR_THROW_IF_FALSE(initialized);
    std::vector<uint64_t> valuesDimensions = values.getDimensions();
    THOR_THROW_IF_FALSE(!valuesDimensions.empty());
    return std::vector<uint64_t>(valuesDimensions.begin() + 1, valuesDimensions.end());
}

ThorImplementation::RaggedTensorDescriptor RaggedTensor::getDescriptor() const {
    THOR_THROW_IF_FALSE(initialized);
    if (hasMaxValuesPerRow()) {
        return ThorImplementation::RaggedTensorDescriptor(values.getDataType(),
                                                          getTrailingDimensions(),
                                                          batchSize,
                                                          maxTotalValues,
                                                          maxValuesPerRow,
                                                          offsets.getDataType());
    }
    return ThorImplementation::RaggedTensorDescriptor(
        values.getDataType(), getTrailingDimensions(), batchSize, maxTotalValues, offsets.getDataType());
}

json RaggedTensor::architectureJson() const {
    THOR_THROW_IF_FALSE(initialized);
    json out{{"version", getVersion()},
             {"id", getId()},
             {"ragged_rank", getRaggedRank()},
             {"batch_size", getBatchSize()},
             {"max_total_values", getMaxTotalValues()},
             {"values", values.architectureJson()},
             {"offsets", offsets.architectureJson()}};
    if (hasMaxValuesPerRow()) out["max_values_per_row"] = getMaxValuesPerRow();
    return out;
}

json RaggedTensor::serialize(thor_file::TarWriter &archiveWriter) const {
    THOR_THROW_IF_FALSE(initialized);
    json out{{"version", getVersion()},
             {"id", getId()},
             {"ragged_rank", getRaggedRank()},
             {"batch_size", getBatchSize()},
             {"max_total_values", getMaxTotalValues()},
             {"values", values.serialize(archiveWriter)},
             {"offsets", offsets.serialize(archiveWriter)}};
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
