#include "bindings/python/src/core/training/NumpyDataset.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/DatasetAccessPolicy.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Api/Data/NamedDataset.h"
#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Implementation/Data/Materialization/MaterializedNamedDatasetSnapshot.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Random/FullPeriodRandom.h"
#include "Utilities/TensorOperations/Ragged/RowPartitionDTypePolicy.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"
#include "bindings/python/src/core/cast.h"
#include "bindings/python/src/core/physical/NumpyDTypeMapping.h"
#include "bindings/python/src/core/training/PythonRaggedBatch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace Thor::PythonBindings {
namespace {

struct NumpyFieldStorage {
    nb::object owner;
    const uint8_t *data = nullptr;
    uint64_t bytesPerExample = 0;
};

struct NumpyRaggedFieldStorage {
    nb::object valuesOwner;
    nb::object offsetsOwner;
    const uint8_t *values = nullptr;
    const void *offsets = nullptr;
    ThorImplementation::DataType offsetsDataType =
        ThorImplementation::kDefaultRowPartitionOffsetDataType;
    uint64_t numValues = 0;
    uint64_t bytesPerValue = 0;

    [[nodiscard]] uint64_t offsetAt(uint64_t index) const {
        if (offsetsDataType == ThorImplementation::DataType::UINT32) {
            uint32_t value = 0;
            std::memcpy(&value,
                        static_cast<const uint8_t *>(offsets) + index * sizeof(uint32_t),
                        sizeof(value));
            return value;
        }
        if (offsetsDataType == ThorImplementation::DataType::UINT64) {
            uint64_t value = 0;
            std::memcpy(&value,
                        static_cast<const uint8_t *>(offsets) + index * sizeof(uint64_t),
                        sizeof(value));
            return value;
        }
        throw std::runtime_error("NumpyDataset ragged offsets dtype is not canonical.");
    }
};

struct NumpyRaggedQueue {
    std::unique_ptr<AsyncTensorQueue> values;
    std::unique_ptr<AsyncTensorQueue> offsets;
};

class NumpyDataset;

class NumpyBatchSession final : public BatchSession {
   public:
    NumpyBatchSession(std::shared_ptr<const NumpyDataset> dataset,
                      DatasetSplitManifest splits,
                      BatchPolicy batching,
                      uint64_t queueDepth,
                      DatasetFieldMaterializationRequirements fieldRequirements);
    ~NumpyBatchSession() override;

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;
    const DatasetFieldMaterializationRequirements& getDatasetFieldMaterializationRequirements() const override {
        return fieldRequirements;
    }
    void cancel() override;

   private:
    Batch acquireBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void recycleBatch(ExampleType exampleType, Batch &&batch) override;
    void setBatchTailModeForRuntimeImpl(ThorImplementation::BatchTailMode mode) override;
    struct SplitState {
        std::shared_ptr<const ExampleIndexSet> indices;
        uint64_t nextBatchNum = 0;
        uint64_t nextLogicalPosition = 0;
        std::unique_ptr<FullPeriodRandom> randomizer;
        std::map<DatasetFieldId, std::unique_ptr<AsyncTensorQueue>> queues;
        std::map<DatasetFieldId, NumpyRaggedQueue> raggedQueues;
    };

    std::shared_ptr<const NumpyDataset> dataset;
    DatasetSplitManifest splits;
    DatasetFieldMaterializationRequirements fieldRequirements;
    uint64_t queueDepth;
    bool randomizeTrain;
    std::atomic<bool> cancelled{false};
    SplitState train;
    SplitState validate;
    SplitState test;

    SplitState &mutableSplit(ExampleType exampleType);
    const SplitState &immutableSplit(ExampleType exampleType) const;
    void initializeSplit(SplitState &split, const ExampleIndexSet &indices, bool randomized, std::optional<uint64_t> seed);
    void closeSplit(SplitState &split) noexcept;
};

class NumpyDataset final : public NamedDataset {
   public:
    NumpyDataset(nb::dict tensors, nb::dict raggedTensors);
    ~NumpyDataset() override;

    const DatasetId &getId() const override { return id; }
    uint64_t getNumExamples() const override { return numExamples; }
    const DatasetSchema &getSchema() const override { return *schema; }
    const DatasetField &getField(std::string_view name) const override { return schema->getField(name); }
    const NumpyFieldStorage &storage(DatasetFieldId id) const { return storageById.at(id); }
    const NumpyRaggedFieldStorage &raggedStorage(DatasetFieldId id) const {
        return raggedStorageById.at(id);
    }

   private:
    [[nodiscard]] std::unique_ptr<DatasetMaterializationDescription>
    describeMaterializationForRuntime() const override;
    [[nodiscard]] MaterializedNamedDatasetSnapshot
    materializeSnapshotForRuntime(uint64_t readerQueueDepth) const override;

    std::shared_ptr<BatchSession> openBatchSession(const DatasetSplitManifest &splits,
                                                   const BatchPolicy &batching,
                                                   const DatasetAccessPolicy &accessPolicy,
                                                   uint64_t maxInFlightBatches,
                                                   const DatasetFieldMaterializationRequirements &fieldRequirements) const override;

    DatasetId id;
    uint64_t numExamples = 0;
    std::optional<DatasetSchema> schema;
    DatasetLayout layout;
    std::map<DatasetFieldId, NumpyFieldStorage> storageById;
    std::map<DatasetFieldId, NumpyRaggedFieldStorage> raggedStorageById;
};

std::string tensorName(nb::handle key) {
    std::string name = castOrTypeError<std::string>(key, "NumpyDataset tensor name", "str", false);
    if (name.empty()) {
        throw nb::value_error("NumpyDataset tensor names must be non-empty");
    }
    return name;
}

NumpyDataset::NumpyDataset(nb::dict tensors, nb::dict raggedTensors)
    : id(DatasetId::generate()) {
    if (nb::len(tensors) == 0 && nb::len(raggedTensors) == 0) {
        throw nb::value_error("NumpyDataset must contain at least one dense or ragged field");
    }

    nb::object numpy = nb::module_::import_("numpy");
    nb::object ndarrayType = numpy.attr("ndarray");
    std::vector<DatasetField> fields;
    fields.reserve(nb::len(tensors) + nb::len(raggedTensors));
    std::vector<DatasetLayout::TensorShape> tensorShapes;
    tensorShapes.reserve(nb::len(tensors));
    std::vector<DatasetLayout::RaggedTensorShape> raggedTensorShapes;
    raggedTensorShapes.reserve(nb::len(raggedTensors));
    std::set<std::string> fieldNames;
    bool haveNumExamples = false;
    DatasetFieldId nextFieldId = 1;

    auto acceptNumExamples = [&](uint64_t fieldExamples, const std::string &context) {
        if (fieldExamples == 0) {
            throw nb::value_error((context + " must contain at least one example").c_str());
        }
        if (!haveNumExamples) {
            numExamples = fieldExamples;
            haveNumExamples = true;
        } else if (fieldExamples != numExamples) {
            throw nb::value_error(
                "NumpyDataset fields must all have the same leading dimension and the same example count");
        }
    };

    for (auto item : tensors) {
        const std::string name = tensorName(item.first);
        if (!fieldNames.insert(name).second) {
            throw nb::value_error(("NumpyDataset duplicate field name: " + name).c_str());
        }
        const std::string context = "NumpyDataset tensors['" + name + "']";
        if (!nb::isinstance(item.second, ndarrayType)) {
            throw nb::type_error((context + " must be a numpy.ndarray").c_str());
        }

        nb::object owner = nb::borrow<nb::object>(item.second);
        if (!nb::cast<bool>(owner.attr("flags").attr("c_contiguous"))) {
            throw nb::type_error((context + " must be C-contiguous").c_str());
        }
        const CanonicalNumpyArrayView array = canonicalNumpyArrayViewNoCopy(owner, context);
        if (array.dimensions.empty() || array.dimensions.front() == 0) {
            throw nb::value_error((context + " must have shape [N, ...] with N >= 1").c_str());
        }

        const ThorImplementation::DataType dataType = array.dataType;
        const uint64_t elementBytes = thorStorageDataTypeSizeBytes(dataType);
        const uint64_t fieldExamples = array.dimensions.front();
        acceptNumExamples(fieldExamples, context);

        std::vector<uint64_t> dimensions;
        uint64_t elementsPerExample = 1;
        if (array.dimensions.size() == 1) {
            dimensions.push_back(1);
        } else {
            for (size_t i = 1; i < array.dimensions.size(); ++i) {
                if (array.dimensions[i] == 0) {
                    throw nb::value_error((context + " dimensions must all be positive").c_str());
                }
                const uint64_t dimension = array.dimensions[i];
                if (elementsPerExample > std::numeric_limits<uint64_t>::max() / dimension) {
                    throw nb::value_error((context + " dimensions overflow uint64_t").c_str());
                }
                dimensions.push_back(dimension);
                elementsPerExample *= dimension;
            }
        }

        owner.attr("setflags")("write"_a = false);
        DatasetField field{.id = nextFieldId,
                           .name = name,
                           .dataType = dataType,
                           .dimensions = std::move(dimensions),
                           .kind = DatasetFieldKind::DENSE};
        NumpyFieldStorage storage{.owner = std::move(owner),
                                  .data = reinterpret_cast<const uint8_t *>(array.data),
                                  .bytesPerExample = elementsPerExample * elementBytes};
        storageById.emplace(nextFieldId, std::move(storage));
        tensorShapes.emplace_back(name, field.dimensions, dataType);
        fields.push_back(std::move(field));
        ++nextFieldId;
    }

    for (auto item : raggedTensors) {
        const std::string name = tensorName(item.first);
        if (!fieldNames.insert(name).second) {
            throw nb::value_error(("NumpyDataset duplicate field name: " + name).c_str());
        }
        const std::string context = "NumpyDataset ragged_tensors['" + name + "']";
        PythonRaggedBatch batch = castOrTypeError<PythonRaggedBatch>(
            item.second, context, "thor.data.RaggedBatch", false);

        if (!nb::isinstance(batch.values, ndarrayType)) {
            throw nb::type_error((context + ".values must be a numpy.ndarray").c_str());
        }
        if (!nb::isinstance(batch.offsets, ndarrayType)) {
            throw nb::type_error((context + ".offsets must be a numpy.ndarray").c_str());
        }
        if (!nb::cast<bool>(batch.values.attr("flags").attr("c_contiguous"))) {
            throw nb::type_error((context + ".values must be C-contiguous").c_str());
        }
        if (!nb::cast<bool>(batch.offsets.attr("flags").attr("c_contiguous"))) {
            throw nb::type_error((context + ".offsets must be C-contiguous").c_str());
        }

        const CanonicalNumpyArrayView values =
            canonicalNumpyArrayViewNoCopy(batch.values, context + ".values");
        const CanonicalNumpyArrayView offsets =
            canonicalNumpyArrayViewNoCopy(batch.offsets, context + ".offsets");
        if (values.dimensions.empty()) {
            throw nb::value_error((context + ".values must have shape [total_values, ...]").c_str());
        }
        if (offsets.dimensions.size() != 1 || offsets.dimensions.front() < 2) {
            throw nb::value_error((context + ".offsets must have shape [N + 1] with N >= 1").c_str());
        }
        if (!ThorImplementation::RowPartitionDescriptor::isValidOffsetsDataType(offsets.dataType)) {
            throw nb::type_error((context + ".offsets dtype must be numpy.uint32 or numpy.uint64").c_str());
        }

        const uint64_t fieldExamples = offsets.dimensions.front() - 1;
        acceptNumExamples(fieldExamples, context);
        const uint64_t numValues = values.dimensions.front();
        std::vector<uint64_t> valueDimensions(values.dimensions.begin() + 1, values.dimensions.end());
        uint64_t elementsPerValue = 1;
        for (uint64_t dimension : valueDimensions) {
            if (dimension == 0) {
                throw nb::value_error((context + ".values trailing dimensions must all be positive").c_str());
            }
            if (elementsPerValue > std::numeric_limits<uint64_t>::max() / dimension) {
                throw nb::value_error((context + ".values trailing dimensions overflow uint64_t").c_str());
            }
            elementsPerValue *= dimension;
        }
        const uint64_t elementBytes = thorStorageDataTypeSizeBytes(values.dataType);
        if (elementsPerValue > std::numeric_limits<uint64_t>::max() / elementBytes) {
            throw nb::value_error((context + ".values bytes per logical value overflow uint64_t").c_str());
        }
        const uint64_t bytesPerValue = elementsPerValue * elementBytes;

        NumpyRaggedFieldStorage storage{
            .valuesOwner = batch.values,
            .offsetsOwner = batch.offsets,
            .values = reinterpret_cast<const uint8_t *>(values.data),
            .offsets = offsets.data,
            .offsetsDataType = offsets.dataType,
            .numValues = numValues,
            .bytesPerValue = bytesPerValue,
        };
        uint64_t previous = storage.offsetAt(0);
        if (previous != 0) {
            throw nb::value_error((context + ".offsets[0] must be zero").c_str());
        }
        for (uint64_t row = 0; row < fieldExamples; ++row) {
            const uint64_t next = storage.offsetAt(row + 1);
            if (next < previous) {
                throw nb::value_error((context + ".offsets must be nondecreasing").c_str());
            }
            if (next > numValues) {
                throw nb::value_error((context + ".offsets cannot reference past values.shape[0]").c_str());
            }
            previous = next;
        }
        if (previous != numValues) {
            throw nb::value_error((context + ".offsets[-1] must equal values.shape[0]").c_str());
        }

        batch.values.attr("setflags")("write"_a = false);
        batch.offsets.attr("setflags")("write"_a = false);
        DatasetField field{.id = nextFieldId,
                           .name = name,
                           .dataType = values.dataType,
                           .dimensions = valueDimensions,
                           .kind = DatasetFieldKind::RAGGED};
        raggedStorageById.emplace(nextFieldId, std::move(storage));
        raggedTensorShapes.emplace_back(name, valueDimensions, values.dataType);
        fields.push_back(std::move(field));
        ++nextFieldId;
    }

    schema.emplace(std::move(fields));
    layout = DatasetLayout::fromTensorShapes(tensorShapes, raggedTensorShapes);
}

NumpyDataset::~NumpyDataset() {
    if (Py_IsInitialized()) {
        nb::gil_scoped_acquire gil;
        storageById.clear();
        raggedStorageById.clear();
        return;
    }

    // Python has already finalized, so releasing references is no longer safe.
    // Detach the handles and let interpreter shutdown reclaim the objects.
    for (auto &[fieldId, storage] : storageById) {
        (void)fieldId;
        (void)storage.owner.release();
    }
    storageById.clear();
    for (auto &[fieldId, storage] : raggedStorageById) {
        (void)fieldId;
        (void)storage.valuesOwner.release();
        (void)storage.offsetsOwner.release();
    }
    raggedStorageById.clear();
}

std::unique_ptr<DatasetMaterializationDescription>
NumpyDataset::describeMaterializationForRuntime() const {
    return std::make_unique<DatasetMaterializationDescription>(
        std::filesystem::path{},
        id,
        *schema,
        layout,
        numExamples,
        DatasetMaterializationSource::MEMORY);
}

MaterializedNamedDatasetSnapshot NumpyDataset::materializeSnapshotForRuntime(
    uint64_t readerQueueDepth) const {
    if (readerQueueDepth == 0) {
        throw std::runtime_error(
            "NumpyDataset materialization reader_queue_depth must be >= 1.");
    }
    const auto started = std::chrono::steady_clock::now();
    MaterializedNamedDatasetSnapshot snapshot(id, *schema, layout, numExamples);
    ThorImplementation::TensorPlacement cpuPlacement(
        ThorImplementation::TensorPlacement::MemDevices::CPU);

    for (const DatasetField &field : schema->getFields()) {
        if (field.kind == DatasetFieldKind::RAGGED) {
            const NumpyRaggedFieldStorage &source = raggedStorage(field.id);
            MaterializedRaggedFieldSnapshot ragged;
            ragged.valuesDataType = field.dataType;
            ragged.trailingDimensions = field.dimensions;
            ragged.storedValueCount = source.numValues;
            ragged.valueBytes = source.bytesPerValue;

            ragged.offsets = ThorImplementation::Tensor(
                cpuPlacement,
                ThorImplementation::TensorDescriptor(
                    source.offsetsDataType, {numExamples + 1}));
            const uint64_t offsetBytes =
                source.offsetsDataType == ThorImplementation::DataType::UINT32
                    ? sizeof(uint32_t)
                    : sizeof(uint64_t);
            std::memcpy(
                ragged.offsets.getMemPtr<void>(),
                source.offsets,
                static_cast<size_t>((numExamples + 1) * offsetBytes));

            if (source.numValues != 0) {
                std::vector<uint64_t> valueDimensions;
                valueDimensions.reserve(field.dimensions.size() + 1);
                valueDimensions.push_back(source.numValues);
                valueDimensions.insert(
                    valueDimensions.end(), field.dimensions.begin(), field.dimensions.end());
                ragged.values = ThorImplementation::Tensor(
                    cpuPlacement,
                    ThorImplementation::TensorDescriptor(field.dataType, valueDimensions));
                const uint64_t expectedBytes = source.numValues * source.bytesPerValue;
                if (ragged.values.getArraySizeInBytes() != expectedBytes) {
                    throw std::runtime_error(
                        "NumpyDataset ragged field storage changed after dataset construction: " +
                        field.name);
                }
                std::memcpy(ragged.values.getMemPtr<void>(), source.values, expectedBytes);
            }
            snapshot.raggedFields.emplace(field.id, std::move(ragged));
            continue;
        }

        const NumpyFieldStorage &source = storage(field.id);
        std::vector<uint64_t> dimensions;
        dimensions.reserve(field.dimensions.size() + 1);
        dimensions.push_back(numExamples);
        dimensions.insert(
            dimensions.end(), field.dimensions.begin(), field.dimensions.end());
        ThorImplementation::Tensor tensor(
            cpuPlacement,
            ThorImplementation::TensorDescriptor(field.dataType, dimensions));
        const uint64_t expectedBytes = tensor.getArraySizeInBytes();
        if (source.bytesPerExample != expectedBytes / numExamples) {
            throw std::runtime_error(
                "NumpyDataset field storage changed after dataset construction: " +
                field.name);
        }
        std::memcpy(tensor.getMemPtr<void>(), source.data, expectedBytes);
        snapshot.fields.emplace(field.id, std::move(tensor));
    }

    snapshot.materializationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return snapshot;
}

std::shared_ptr<BatchSession> NumpyDataset::openBatchSession(
    const DatasetSplitManifest &splits,
    const BatchPolicy &batching,
    const DatasetAccessPolicy &,
    uint64_t maxInFlightBatches,
    const DatasetFieldMaterializationRequirements &fieldRequirements) const {
    std::shared_ptr<const NumpyDataset> self =
        std::dynamic_pointer_cast<const NumpyDataset>(shared_from_this());
    if (self == nullptr) {
        throw std::runtime_error("NumpyDataset must be owned by std::shared_ptr before opening a session.");
    }
    return std::make_shared<NumpyBatchSession>(
        std::move(self), splits, batching, maxInFlightBatches, fieldRequirements);
}

NumpyBatchSession::NumpyBatchSession(std::shared_ptr<const NumpyDataset> dataset,
                                     DatasetSplitManifest splits,
                                     BatchPolicy batching,
                                     uint64_t queueDepth,
                                     DatasetFieldMaterializationRequirements fieldRequirements)
    : dataset(std::move(dataset)),
      splits(std::move(splits)),
      fieldRequirements(std::move(fieldRequirements)),
      queueDepth(queueDepth),
      randomizeTrain(batching.getRandomizeTrain()) {
    if (this->dataset == nullptr) {
        throw std::runtime_error("NumpyBatchSession dataset must not be null.");
    }
    if (queueDepth == 0) {
        throw std::runtime_error("NumpyBatchSession queue depth must be >= 1.");
    }
    this->splits.validateAgainst(*this->dataset);
    this->batchSize = batching.getBatchSize();
    if (this->fieldRequirements.empty()) {
        for (const DatasetField &field : this->dataset->getSchema().getFields()) {
            if (field.kind == DatasetFieldKind::RAGGED) {
                throw std::runtime_error(
                    "NumpyBatchSession requires an explicit materialization descriptor for ragged field '" +
                    field.name + "'.");
            }
            this->fieldRequirements.emplace(
                field.id, DatasetFieldMaterializationRequirement::dense(field.id));
        }
    }
    for (const auto& [fieldId, requirement] : this->fieldRequirements) {
        if (fieldId != requirement.fieldId) {
            throw std::runtime_error("NumpyBatchSession field requirement key/id mismatch.");
        }
        const DatasetField &field = this->dataset->getSchema().getField(fieldId);
        if (field.kind == DatasetFieldKind::RAGGED) {
            if (!requirement.raggedTensorDescriptor.has_value()) {
                throw std::runtime_error(
                    "NumpyBatchSession ragged field '" + field.name +
                    "' requires a materialization descriptor.");
            }
            const ThorImplementation::RaggedTensorDescriptor &descriptor =
                requirement.raggedTensorDescriptor.value();
            if (descriptor.getValuesDataType() != field.dataType ||
                descriptor.getTrailingDimensions() != field.dimensions ||
                descriptor.getBatchSize() != this->batchSize) {
                throw std::runtime_error(
                    "NumpyBatchSession ragged materialization contract does not match field '" +
                    field.name + "'.");
            }
        } else if (requirement.raggedTensorDescriptor.has_value()) {
            throw std::runtime_error(
                "NumpyBatchSession non-ragged field '" + field.name +
                "' cannot carry a RaggedTensor materialization descriptor.");
        }
    }
    initializeSplit(train, this->splits.getTrain(), randomizeTrain, batching.getRandomSeed());
    initializeSplit(validate, this->splits.getValidate(), false, std::nullopt);
    initializeSplit(test, this->splits.getTest(), false, std::nullopt);
}

NumpyBatchSession::~NumpyBatchSession() {
    closeSplit(train);
    closeSplit(validate);
    closeSplit(test);
}

void NumpyBatchSession::initializeSplit(SplitState &split,
                                            const ExampleIndexSet &indices,
                                            bool randomized,
                                            std::optional<uint64_t> seed) {
    split.indices = std::make_shared<const ExampleIndexSet>(indices);
    if (split.indices->empty()) {
        return;
    }
    if (randomized) {
        split.randomizer = std::make_unique<FullPeriodRandom>(split.indices->size(), false);
        if (seed.has_value()) {
            split.randomizer->reseed(*seed);
        }
    }
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    for (const auto& [fieldId, requirement] : fieldRequirements) {
        const DatasetField &field = dataset->getSchema().getField(fieldId);
        if (field.kind == DatasetFieldKind::RAGGED) {
            THOR_THROW_IF_FALSE(requirement.raggedTensorDescriptor.has_value());
            const ThorImplementation::RaggedTensorDescriptor &descriptor =
                requirement.raggedTensorDescriptor.value();
            NumpyRaggedQueue queues;
            queues.values = std::make_unique<AsyncTensorQueue>(
                queueDepth, descriptor.getValuesDescriptor(), cpuPlacement);
            queues.offsets = std::make_unique<AsyncTensorQueue>(
                queueDepth, descriptor.getOffsetsDescriptor(), cpuPlacement);
            queues.values->open();
            queues.offsets->open();
            split.raggedQueues.emplace(fieldId, std::move(queues));
        } else {
            std::vector<uint64_t> batchShape{batchSize};
            batchShape.insert(batchShape.end(), field.dimensions.begin(), field.dimensions.end());
            ThorImplementation::TensorDescriptor descriptor(field.dataType, std::move(batchShape));
            auto queue = std::make_unique<AsyncTensorQueue>(queueDepth, descriptor, cpuPlacement);
            queue->open();
            split.queues.emplace(fieldId, std::move(queue));
        }
    }
}

void NumpyBatchSession::closeSplit(SplitState &split) noexcept {
    for (auto &[fieldId, queue] : split.queues) {
        (void)fieldId;
        if (queue != nullptr) {
            queue->close();
        }
    }
    split.queues.clear();
    for (auto &[fieldId, queues] : split.raggedQueues) {
        (void)fieldId;
        if (queues.values != nullptr) {
            queues.values->close();
        }
        if (queues.offsets != nullptr) {
            queues.offsets->close();
        }
    }
    split.raggedQueues.clear();
}

NumpyBatchSession::SplitState &NumpyBatchSession::mutableSplit(ExampleType exampleType) {
    if (exampleType == ExampleType::TRAIN) {
        return train;
    }
    if (exampleType == ExampleType::VALIDATE) {
        return validate;
    }
    if (exampleType == ExampleType::TEST) {
        return test;
    }
    throw std::runtime_error("Unsupported ExampleType");
}

const NumpyBatchSession::SplitState &NumpyBatchSession::immutableSplit(ExampleType exampleType) const {
    if (exampleType == ExampleType::TRAIN) {
        return train;
    }
    if (exampleType == ExampleType::VALIDATE) {
        return validate;
    }
    if (exampleType == ExampleType::TEST) {
        return test;
    }
    throw std::runtime_error("Unsupported ExampleType");
}

void writeNumpyRaggedOffset(ThorImplementation::Tensor &offsets,
                            uint64_t index,
                            uint64_t value) {
    const ThorImplementation::DataType dataType = offsets.getDescriptor().getDataType();
    if (dataType == ThorImplementation::DataType::UINT32) {
        THOR_THROW_IF_FALSE(value <= std::numeric_limits<uint32_t>::max());
        static_cast<uint32_t *>(offsets.getMemPtr<void>())[index] = static_cast<uint32_t>(value);
        return;
    }
    if (dataType == ThorImplementation::DataType::UINT64) {
        static_cast<uint64_t *>(offsets.getMemPtr<void>())[index] = value;
        return;
    }
    throw std::runtime_error("NumpyBatchSession ragged offsets dtype is not canonical.");
}

Batch NumpyBatchSession::acquireBatch(ExampleType exampleType, uint64_t &batchNum) {
    if (cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("NumpyBatchSession has been cancelled.");
    }
    SplitState &split = mutableSplit(exampleType);
    if (split.indices == nullptr || split.indices->empty()) {
        throw std::runtime_error("NumpyBatchSession cannot get a batch from an empty split.");
    }
    const uint64_t batchesPerEpoch = getNumBatchesPerEpoch(exampleType);
    if (batchNum >= batchesPerEpoch) {
        batchNum = split.nextBatchNum;
    }

    const uint64_t firstLogicalIndex = batchNum * batchSize;
    const bool wrapTail = usesWrappedBatchTailForRuntime();
    const uint64_t validExampleCount = wrapTail
        ? batchSize
        : std::min<uint64_t>(batchSize, split.indices->size() - firstLogicalIndex);
    THOR_THROW_IF_FALSE(validExampleCount > 0);
    const bool randomized = exampleType == ExampleType::TRAIN && randomizeTrain;

    std::vector<uint64_t> selectedExampleIndices;
    selectedExampleIndices.reserve(validExampleCount);
    for (uint64_t row = 0; row < validExampleCount; ++row) {
        uint64_t logicalIndex = 0;
        if (randomized) {
            logicalIndex = split.randomizer->getRandomNumber();
        } else if (wrapTail) {
            logicalIndex = split.nextLogicalPosition;
            split.nextLogicalPosition =
                (split.nextLogicalPosition + 1) % split.indices->size();
        } else {
            logicalIndex = firstLogicalIndex + row;
        }
        selectedExampleIndices.push_back(split.indices->at(logicalIndex));
    }

    // Preflight ragged capacities before acquiring reusable queue buffers. A
    // network chooses maxTotalValues through its RaggedNetworkInput contract;
    // the in-memory dataset stores only the actual variable-length rows.
    std::map<DatasetFieldId, uint64_t> raggedActiveValueCounts;
    std::map<DatasetFieldId, uint64_t> raggedMaxActiveRowLengths;
    for (const auto &[fieldId, queues] : split.raggedQueues) {
        (void)queues;
        const DatasetField &field = dataset->getSchema().getField(fieldId);
        const NumpyRaggedFieldStorage &source = dataset->raggedStorage(fieldId);
        const ThorImplementation::RaggedTensorDescriptor &descriptor =
            fieldRequirements.at(fieldId).raggedTensorDescriptor.value();
        uint64_t activeValueCount = 0;
        uint64_t maxActiveRowLength = 0;
        for (uint64_t row = 0; row < selectedExampleIndices.size(); ++row) {
            const uint64_t exampleIndex = selectedExampleIndices[row];
            const uint64_t begin = source.offsetAt(exampleIndex);
            const uint64_t end = source.offsetAt(exampleIndex + 1);
            THOR_THROW_IF_FALSE(end >= begin);
            const uint64_t rowValueCount = end - begin;
            if (descriptor.hasMaxValuesPerRow() &&
                rowValueCount > descriptor.getMaxValuesPerRow()) {
                throw std::runtime_error(
                    "NumpyBatchSession ragged field '" + field.name +
                    "' selected row " + std::to_string(row) +
                    " (dataset example " + std::to_string(exampleIndex) +
                    ") has " + std::to_string(rowValueCount) +
                    " values, exceeding maxValuesPerRow=" +
                    std::to_string(descriptor.getMaxValuesPerRow()) + ".");
            }
            if (rowValueCount > descriptor.getMaxTotalValues() - activeValueCount) {
                throw std::runtime_error(
                    "NumpyBatchSession ragged field '" + field.name +
                    "' exceeds maxTotalValues=" + std::to_string(descriptor.getMaxTotalValues()) +
                    " while materializing selected row " + std::to_string(row) +
                    " (dataset example " + std::to_string(exampleIndex) +
                    ", active_before=" + std::to_string(activeValueCount) +
                    ", row_values=" + std::to_string(rowValueCount) + ").");
            }
            activeValueCount += rowValueCount;
            maxActiveRowLength = std::max(maxActiveRowLength, rowValueCount);
        }
        raggedActiveValueCounts.emplace(fieldId, activeValueCount);
        raggedMaxActiveRowLengths.emplace(fieldId, maxActiveRowLength);
    }

    std::map<DatasetFieldId, ThorImplementation::Tensor> tensors;
    for (auto &[fieldId, queue] : split.queues) {
        ThorImplementation::Tensor tensor;
        THOR_THROW_IF_FALSE(queue->getBufferToLoad(tensor));
        tensors.emplace(fieldId, tensor);
    }
    std::map<DatasetFieldId, ThorImplementation::RaggedTensor> raggedTensors;
    for (auto &[fieldId, queues] : split.raggedQueues) {
        ThorImplementation::Tensor values;
        ThorImplementation::Tensor offsets;
        THOR_THROW_IF_FALSE(queues.values->getBufferToLoad(values));
        THOR_THROW_IF_FALSE(queues.offsets->getBufferToLoad(offsets));
        const ThorImplementation::RaggedTensorDescriptor &descriptor =
            fieldRequirements.at(fieldId).raggedTensorDescriptor.value();
        ThorImplementation::RowPartitionRuntime rowPartition(
            offsets, descriptor.getRowPartition());
        raggedTensors.emplace(
            fieldId, ThorImplementation::RaggedTensor(values, std::move(rowPartition)));
    }

    const uint64_t finalValidExampleIndex = selectedExampleIndices.back();
    for (auto &[fieldId, tensor] : tensors) {
        const NumpyFieldStorage &source = dataset->storage(fieldId);
        uint8_t *destination = static_cast<uint8_t *>(tensor.getMemPtr<void>());
        for (uint64_t row = 0; row < selectedExampleIndices.size(); ++row) {
            const uint64_t exampleIndex = selectedExampleIndices[row];
            std::memcpy(destination + row * source.bytesPerExample,
                        source.data + exampleIndex * source.bytesPerExample,
                        source.bytesPerExample);
        }
        // Dense exact-tail storage retains Thor's historical fixed-capacity
        // behavior. validExampleCount prevents these rows from becoming logical
        // examples. Ragged exact tails below are represented as empty rows.
        for (uint64_t row = validExampleCount; row < batchSize; ++row) {
            std::memcpy(destination + row * source.bytesPerExample,
                        source.data + finalValidExampleIndex * source.bytesPerExample,
                        source.bytesPerExample);
        }
    }

    for (auto &[fieldId, ragged] : raggedTensors) {
        const NumpyRaggedFieldStorage &source = dataset->raggedStorage(fieldId);
        ThorImplementation::Tensor values = ragged.getValues();
        ThorImplementation::Tensor offsets = ragged.getOffsets();
        uint8_t *destination = static_cast<uint8_t *>(values.getMemPtr<void>());
        uint64_t activeValueCount = 0;
        writeNumpyRaggedOffset(offsets, 0, 0);
        for (uint64_t row = 0; row < selectedExampleIndices.size(); ++row) {
            const uint64_t exampleIndex = selectedExampleIndices[row];
            const uint64_t begin = source.offsetAt(exampleIndex);
            const uint64_t end = source.offsetAt(exampleIndex + 1);
            const uint64_t rowValueCount = end - begin;
            if (rowValueCount != 0) {
                std::memcpy(destination + activeValueCount * source.bytesPerValue,
                            source.values + begin * source.bytesPerValue,
                            rowValueCount * source.bytesPerValue);
            }
            activeValueCount += rowValueCount;
            writeNumpyRaggedOffset(offsets, row + 1, activeValueCount);
        }
        THOR_THROW_IF_FALSE(activeValueCount == raggedActiveValueCounts.at(fieldId));
        for (uint64_t row = validExampleCount; row < batchSize; ++row) {
            writeNumpyRaggedOffset(offsets, row + 1, activeValueCount);
        }
        ThorImplementation::RowPartitionRuntime &rowPartition =
            ragged.getRowPartitionRuntime();
        rowPartition.setHostActiveValueCount(activeValueCount);
        rowPartition.setHostMaxActiveRowLength(
            raggedMaxActiveRowLengths.at(fieldId));
    }

    split.nextBatchNum = (batchNum + 1) % batchesPerEpoch;

    for (auto &[fieldId, queue] : split.queues) {
        THOR_THROW_IF_FALSE(queue->bufferLoaded(tensors.at(fieldId)));
    }
    for (auto &[fieldId, queues] : split.raggedQueues) {
        ThorImplementation::RaggedTensor &ragged = raggedTensors.at(fieldId);
        THOR_THROW_IF_FALSE(queues.values->bufferLoaded(ragged.getValues()));
        THOR_THROW_IF_FALSE(queues.offsets->bufferLoaded(ragged.getOffsets()));
    }
    for (auto &[fieldId, queue] : split.queues) {
        THOR_THROW_IF_FALSE(queue->getBufferToUnload(tensors.at(fieldId)));
    }
    for (auto &[fieldId, queues] : split.raggedQueues) {
        ThorImplementation::Tensor values;
        ThorImplementation::Tensor offsets;
        THOR_THROW_IF_FALSE(queues.values->getBufferToUnload(values));
        THOR_THROW_IF_FALSE(queues.offsets->getBufferToUnload(offsets));
        THOR_THROW_IF_FALSE(values == raggedTensors.at(fieldId).getValues());
        THOR_THROW_IF_FALSE(offsets == raggedTensors.at(fieldId).getOffsets());
    }

    Batch batch;
    if (validExampleCount < batchSize) {
        batch.setValidExampleCount(static_cast<uint32_t>(validExampleCount));
    }
    for (auto &[fieldId, tensor] : tensors) {
        batch.insert(dataset->getSchema().getField(fieldId).name, tensor);
    }
    for (auto &[fieldId, ragged] : raggedTensors) {
        batch.insert(dataset->getSchema().getField(fieldId).name, ragged);
    }
    return batch;
}

void NumpyBatchSession::setBatchTailModeForRuntimeImpl(
    ThorImplementation::BatchTailMode mode) {
    (void)mode;
    for (SplitState *split : {&train, &validate, &test}) {
        THOR_THROW_IF_FALSE(split->nextBatchNum == 0);
        split->nextLogicalPosition = 0;
    }
}

void NumpyBatchSession::recycleBatch(ExampleType exampleType, Batch &&batch) {
    if (cancelled.load(std::memory_order_acquire)) {
        return;
    }
    SplitState &split = mutableSplit(exampleType);
    if (batch.size() != fieldRequirements.size()) {
        throw std::runtime_error("NumpyBatchSession returned batch has unexpected tensor count.");
    }
    for (auto &[fieldId, queue] : split.queues) {
        const std::string &name = dataset->getSchema().getField(fieldId).name;
        if (!batch.contains(name)) {
            throw std::runtime_error("NumpyBatchSession returned batch is missing tensor '" + name + "'.");
        }
        THOR_THROW_IF_FALSE(queue->bufferUnloaded(batch.getTensor(name)));
    }
    for (auto &[fieldId, queues] : split.raggedQueues) {
        const std::string &name = dataset->getSchema().getField(fieldId).name;
        if (!batch.contains(name) || !batch.isRaggedTensor(name)) {
            throw std::runtime_error(
                "NumpyBatchSession returned batch is missing ragged tensor '" + name + "'.");
        }
        const ThorImplementation::RaggedTensor &ragged = batch.getRaggedTensor(name);
        THOR_THROW_IF_FALSE(queues.values->bufferUnloaded(ragged.getValues()));
        THOR_THROW_IF_FALSE(queues.offsets->bufferUnloaded(ragged.getOffsets()));
    }
}

uint64_t NumpyBatchSession::getNumExamples(ExampleType exampleType) {
    const SplitState &split = immutableSplit(exampleType);
    return split.indices == nullptr ? 0 : split.indices->size();
}

uint64_t NumpyBatchSession::getNumBatchesPerEpoch(ExampleType exampleType) {
    const uint64_t examples = getNumExamples(exampleType);
    return examples == 0 ? 0 : (examples + batchSize - 1) / batchSize;
}

uint64_t NumpyBatchSession::getNextBatchNum(ExampleType exampleType) {
    return immutableSplit(exampleType).nextBatchNum;
}

void NumpyBatchSession::cancel() {
    if (cancelled.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    closeSplit(train);
    closeSplit(validate);
    closeSplit(test);
}

}  // namespace

void bindNumpyDataset(nb::module_ &training) {
    auto numpyDataset = nb::class_<NumpyDataset, NamedDataset>(
        training, "NumpyDataset", nb::is_weak_referenceable());
    numpyDataset.attr("__module__") = "thor.data";
    numpyDataset.def_static(
        "__new__",
        [](nb::handle cls, nb::dict tensors, nb::dict raggedTensors) -> std::shared_ptr<NumpyDataset> {
            (void)cls;
            return std::make_shared<NumpyDataset>(std::move(tensors), std::move(raggedTensors));
        },
        "cls"_a,
        "tensors"_a = nb::dict(),
        "ragged_tensors"_a = nb::dict(),
        R"nbdoc(
Create an immutable in-memory dataset over canonical dense and ragged NumPy arrays.

Dense fields are C-contiguous ``[N, ...]`` ndarrays. Ragged fields are
``thor.data.RaggedBatch`` objects whose values are packed ``[total_values, ...]``
arrays and whose canonical uint32/uint64 offsets have shape ``[N + 1]``. Dense
and ragged fields share the same example count. Thor marks all supplied arrays
read-only and retains them for the dataset lifetime; callers must not mutate the
underlying allocations through other aliases.

Host-backed batching supports dense/ragged mixtures, arbitrary dataset splits,
randomized training order, exact partial batches, and wrapped tails. Device-
resident ragged NumpyDataset storage is intentionally deferred; BEST_EFFORT
device storage falls back to the host-backed session and STRICT reports the
unsupported resident-ragged backend.
        )nbdoc");
    numpyDataset.def(
        "__init__",
        [](NumpyDataset *, nb::dict, nb::dict) {},
        "tensors"_a = nb::dict(),
        "ragged_tensors"_a = nb::dict());
}

}  // namespace Thor::PythonBindings
