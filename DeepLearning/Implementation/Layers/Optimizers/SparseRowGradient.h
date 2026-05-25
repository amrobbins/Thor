#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {

struct SparseRowGradient {

    // rows has one extra staging slot so CUB RLE can write the optional invalid-row sentinel directly into this tensor.
    // Only the first numRows entries are valid sparse-gradient rows; values remains exactly [capacity, embeddingDim].
    Tensor rows;     // [capacity + 1], row ids for the materialized unique rows plus optional sentinel scratch.
    Tensor values;   // [capacity, embeddingDim], accumulated gradient values for each unique row.
    Tensor numRows;  // [1], runtime count of valid entries in rows/values.

    uint64_t capacity = 0;
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;
    DataType rowDataType = DataType::UINT64;
    DataType accumulationDataType = DataType::FP32;

    [[nodiscard]] bool isInitialized() const { return rows.isInitialized() && values.isInitialized() && numRows.isInitialized(); }

    [[nodiscard]] static DataType chooseRowDataType(uint64_t vocabularySize) {
        // The embedding sparse-gradient producer uses vocabularySize as the invalid-row sentinel during sort/RLE.
        // Therefore the chosen row dtype must be able to represent both every valid row id and the sentinel itself.
        if (vocabularySize <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) {
            return DataType::UINT16;
        }
        if (vocabularySize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            return DataType::UINT32;
        }
        return DataType::UINT64;
    }

    static SparseRowGradient allocate(const TensorPlacement& placement,
                                      uint64_t capacity,
                                      uint64_t vocabularySize,
                                      uint64_t embeddingDim,
                                      DataType accumulationDataType = DataType::FP32,
                                      DataType rowDataType = DataType::UINT64) {
        if (capacity == 0) {
            throw std::invalid_argument("SparseRowGradient capacity must be non-zero.");
        }
        if (vocabularySize == 0) {
            throw std::invalid_argument("SparseRowGradient vocabulary size must be non-zero.");
        }
        if (embeddingDim == 0) {
            throw std::invalid_argument("SparseRowGradient embedding dimension must be non-zero.");
        }
        if (!isSupportedRowDataType(rowDataType)) {
            throw std::invalid_argument("SparseRowGradient row dtype must be uint16, uint32, or uint64. Got " + dataTypeName(rowDataType) + ".");
        }
        if (!rowDataTypeCanRepresentVocabularySentinel(rowDataType, vocabularySize)) {
            throw std::invalid_argument("SparseRowGradient row dtype " + dataTypeName(rowDataType) +
                                        " cannot represent vocabulary_size as the invalid-row sentinel.");
        }
        if (!isSupportedAccumulationDataType(accumulationDataType)) {
            throw std::invalid_argument("SparseRowGradient accumulation dtype must be fp32. Got " + dataTypeName(accumulationDataType) + ".");
        }
        if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::invalid_argument("SparseRowGradient storage must be allocated on GPU.");
        }

        SparseRowGradient gradient;
        gradient.rows = Tensor(placement, TensorDescriptor(rowDataType, {rowStorageCapacityFor(capacity)}));
        gradient.values = Tensor(placement, TensorDescriptor(accumulationDataType, {capacity, embeddingDim}));
        gradient.numRows = Tensor(placement, TensorDescriptor(rowDataType, {1}));
        gradient.capacity = capacity;
        gradient.vocabularySize = vocabularySize;
        gradient.embeddingDim = embeddingDim;
        gradient.rowDataType = rowDataType;
        gradient.accumulationDataType = accumulationDataType;
        gradient.validate();
        return gradient;
    }

    void validate() const {
        if (!isInitialized()) {
            throw std::invalid_argument("SparseRowGradient tensors must all be initialized.");
        }
        if (capacity == 0 || vocabularySize == 0 || embeddingDim == 0) {
            throw std::invalid_argument("SparseRowGradient capacity, vocabulary size, and embedding dimension must be non-zero.");
        }
        if (capacity > vocabularySize) {
            throw std::invalid_argument("SparseRowGradient capacity cannot exceed the embedding vocabulary size.");
        }
        if (!isSupportedRowDataType(rowDataType)) {
            throw std::invalid_argument("SparseRowGradient row dtype must be uint16, uint32, or uint64. Got " + dataTypeName(rowDataType) + ".");
        }
        if (!rowDataTypeCanRepresentVocabularySentinel(rowDataType, vocabularySize)) {
            throw std::invalid_argument("SparseRowGradient row dtype " + dataTypeName(rowDataType) +
                                        " cannot represent vocabulary_size as the invalid-row sentinel.");
        }
        if (!isSupportedAccumulationDataType(accumulationDataType)) {
            throw std::invalid_argument("SparseRowGradient accumulation dtype must be fp32. Got " + dataTypeName(accumulationDataType) + ".");
        }
        if (rows.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
            values.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
            numRows.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
            throw std::invalid_argument("SparseRowGradient tensors must all live on GPU.");
        }
        if (rows.getPlacement() != values.getPlacement() || rows.getPlacement() != numRows.getPlacement()) {
            throw std::invalid_argument("SparseRowGradient tensors must all live on the same GPU placement.");
        }
        if (rows.getDataType() != rowDataType || numRows.getDataType() != rowDataType) {
            throw std::invalid_argument("SparseRowGradient rows and numRows tensors must match rowDataType.");
        }
        if (values.getDataType() != accumulationDataType) {
            throw std::invalid_argument("SparseRowGradient values tensor must match accumulationDataType.");
        }
        if (rows.getDimensions() != std::vector<uint64_t>{rowStorageCapacityFor(capacity)}) {
            throw std::invalid_argument("SparseRowGradient rows tensor must have shape [capacity + 1] for direct CUB RLE sentinel staging.");
        }
        if (values.getDimensions() != std::vector<uint64_t>{capacity, embeddingDim}) {
            throw std::invalid_argument("SparseRowGradient values tensor must have shape [capacity, embedding_dim].");
        }
        if (numRows.getDimensions() != std::vector<uint64_t>{1}) {
            throw std::invalid_argument("SparseRowGradient numRows tensor must have shape [1].");
        }
        if (rows.hasCustomStrides() || !rows.isDenseContiguous() || values.hasCustomStrides() || !values.isDenseContiguous() ||
            numRows.hasCustomStrides() || !numRows.isDenseContiguous()) {
            throw std::invalid_argument("SparseRowGradient tensors must be dense contiguous.");
        }
    }

   private:
    static uint64_t rowStorageCapacityFor(uint64_t capacity) {
        if (capacity == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("SparseRowGradient row storage capacity exceeds uint64_t range.");
        }
        return capacity + 1;
    }

    static bool isSupportedRowDataType(DataType dtype) {
        return dtype == DataType::UINT16 || dtype == DataType::UINT32 || dtype == DataType::UINT64;
    }

    static bool rowDataTypeCanRepresentVocabularySentinel(DataType dtype, uint64_t vocabularySize) {
        switch (dtype) {
            case DataType::UINT16:
                return vocabularySize <= static_cast<uint64_t>(std::numeric_limits<uint16_t>::max());
            case DataType::UINT32:
                return vocabularySize <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
            case DataType::UINT64:
                return true;
            default:
                return false;
        }
    }

    static bool isSupportedAccumulationDataType(DataType dtype) {
        // Reduced embedding gradients should accumulate in fp32. Mixed-precision weight/state handling belongs to the optimizer update stage.
        return dtype == DataType::FP32;
    }

    static std::string dataTypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }
};

}  // namespace ThorImplementation
