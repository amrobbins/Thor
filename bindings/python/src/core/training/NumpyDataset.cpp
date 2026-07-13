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
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Random/FullPeriodRandom.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"
#include "bindings/python/src/core/cast.h"
#include "bindings/python/src/core/physical/NumpyDTypeMapping.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

class NumpyDataset;

class NumpyBatchSession final : public BatchSession {
   public:
    NumpyBatchSession(std::shared_ptr<const NumpyDataset> dataset,
                      DatasetSplitManifest splits,
                      BatchPolicy batching,
                      uint64_t queueDepth,
                      std::set<DatasetFieldId> requiredFieldIds);
    ~NumpyBatchSession() override;

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;
    const std::set<DatasetFieldId> &getRequiredDatasetFieldIds() const override { return requiredFieldIds; }
    void cancel() override;

   private:
    Batch acquireBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void recycleBatch(ExampleType exampleType, Batch &&batch) override;
    struct SplitState {
        std::shared_ptr<const ExampleIndexSet> indices;
        uint64_t nextBatchNum = 0;
        std::unique_ptr<FullPeriodRandom> randomizer;
        std::map<DatasetFieldId, std::unique_ptr<AsyncTensorQueue>> queues;
    };

    std::shared_ptr<const NumpyDataset> dataset;
    DatasetSplitManifest splits;
    std::set<DatasetFieldId> requiredFieldIds;
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
    explicit NumpyDataset(nb::dict tensors);
    ~NumpyDataset() override;

    const DatasetId &getId() const override { return id; }
    uint64_t getNumExamples() const override { return numExamples; }
    const DatasetSchema &getSchema() const override { return *schema; }
    const DatasetField &getField(std::string_view name) const override { return schema->getField(name); }
    const NumpyFieldStorage &storage(DatasetFieldId id) const { return storageById.at(id); }

   private:
    [[nodiscard]] std::unique_ptr<DatasetMaterializationDescription>
    describeMaterializationForRuntime() const override;
    [[nodiscard]] MaterializedNamedDatasetSnapshot
    materializeSnapshotForRuntime(uint64_t readerQueueDepth) const override;

    std::shared_ptr<BatchSession> openBatchSession(const DatasetSplitManifest &splits,
                                                   const BatchPolicy &batching,
                                                   const DatasetAccessPolicy &accessPolicy,
                                                   uint64_t maxInFlightBatches,
                                                   const std::set<DatasetFieldId> &requiredFieldIds) const override;

    DatasetId id;
    uint64_t numExamples = 0;
    std::optional<DatasetSchema> schema;
    DatasetLayout layout;
    std::map<DatasetFieldId, NumpyFieldStorage> storageById;
};

std::string tensorName(nb::handle key) {
    std::string name = castOrTypeError<std::string>(key, "NumpyDataset tensor name", "str", false);
    if (name.empty()) {
        throw nb::value_error("NumpyDataset tensor names must be non-empty");
    }
    return name;
}

NumpyDataset::NumpyDataset(nb::dict tensors)
    : id(DatasetId::generate()) {
    if (nb::len(tensors) == 0) {
        throw nb::value_error("NumpyDataset tensors must contain at least one field");
    }

    nb::object numpy = nb::module_::import_("numpy");
    nb::object ndarrayType = numpy.attr("ndarray");
    std::vector<DatasetField> fields;
    fields.reserve(nb::len(tensors));
    std::vector<DatasetLayout::TensorShape> tensorShapes;
    tensorShapes.reserve(nb::len(tensors));
    bool haveNumExamples = false;
    DatasetFieldId nextFieldId = 1;

    for (auto item : tensors) {
        const std::string name = tensorName(item.first);
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
        if (!haveNumExamples) {
            numExamples = fieldExamples;
            haveNumExamples = true;
        } else if (fieldExamples != numExamples) {
            throw nb::value_error("NumpyDataset tensors must all have the same leading dimension");
        }

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
    schema.emplace(std::move(fields));
    layout = DatasetLayout::fromTensorShapes(tensorShapes);
}

NumpyDataset::~NumpyDataset() {
    if (Py_IsInitialized()) {
        nb::gil_scoped_acquire gil;
        storageById.clear();
        return;
    }

    // Python has already finalized, so releasing references is no longer safe.
    // Detach the handles and let interpreter shutdown reclaim the objects.
    for (auto &[fieldId, storage] : storageById) {
        (void)fieldId;
        (void)storage.owner.release();
    }
    storageById.clear();
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
    const std::set<DatasetFieldId> &requiredFieldIds) const {
    std::shared_ptr<const NumpyDataset> self =
        std::dynamic_pointer_cast<const NumpyDataset>(shared_from_this());
    if (self == nullptr) {
        throw std::runtime_error("NumpyDataset must be owned by std::shared_ptr before opening a session.");
    }
    return std::make_shared<NumpyBatchSession>(
        std::move(self), splits, batching, maxInFlightBatches, requiredFieldIds);
}

NumpyBatchSession::NumpyBatchSession(std::shared_ptr<const NumpyDataset> dataset,
                                     DatasetSplitManifest splits,
                                     BatchPolicy batching,
                                     uint64_t queueDepth,
                                     std::set<DatasetFieldId> requiredFieldIds)
    : dataset(std::move(dataset)),
      splits(std::move(splits)),
      requiredFieldIds(std::move(requiredFieldIds)),
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
    if (this->requiredFieldIds.empty()) {
        for (const DatasetField &field : this->dataset->getSchema().getFields()) {
            this->requiredFieldIds.insert(field.id);
        }
    }
    for (DatasetFieldId fieldId : this->requiredFieldIds) {
        (void)this->dataset->getSchema().getField(fieldId);
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
    for (DatasetFieldId fieldId : requiredFieldIds) {
        const DatasetField &field = dataset->getSchema().getField(fieldId);
        std::vector<uint64_t> batchShape{batchSize};
        batchShape.insert(batchShape.end(), field.dimensions.begin(), field.dimensions.end());
        ThorImplementation::TensorDescriptor descriptor(field.dataType, std::move(batchShape));
        auto queue = std::make_unique<AsyncTensorQueue>(queueDepth, descriptor, cpuPlacement);
        queue->open();
        split.queues.emplace(fieldId, std::move(queue));
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

    std::map<DatasetFieldId, ThorImplementation::Tensor> tensors;
    for (auto &[fieldId, queue] : split.queues) {
        ThorImplementation::Tensor tensor;
        THOR_THROW_IF_FALSE(queue->getBufferToLoad(tensor));
        tensors.emplace(fieldId, tensor);
    }

    const uint64_t firstLogicalIndex = batchNum * batchSize;
    const bool randomized = exampleType == ExampleType::TRAIN && randomizeTrain;
    for (uint64_t row = 0; row < batchSize; ++row) {
        const uint64_t logicalIndex = randomized
            ? split.randomizer->getRandomNumber()
            : (firstLogicalIndex + row) % split.indices->size();
        const uint64_t exampleIndex = split.indices->at(logicalIndex);
        for (DatasetFieldId fieldId : requiredFieldIds) {
            const NumpyFieldStorage &source = dataset->storage(fieldId);
            // This is a raw byte copy into a tensor whose logical dtype is described by the dataset field.
            // Use the untyped pointer accessor so Tensor does not interpret uint8_t as the field dtype.
            uint8_t *destination = static_cast<uint8_t *>(tensors.at(fieldId).getMemPtr<void>());
            std::memcpy(destination + row * source.bytesPerExample,
                        source.data + exampleIndex * source.bytesPerExample,
                        source.bytesPerExample);
        }
    }
    split.nextBatchNum = (batchNum + 1) % batchesPerEpoch;

    for (auto &[fieldId, queue] : split.queues) {
        THOR_THROW_IF_FALSE(queue->bufferLoaded(tensors.at(fieldId)));
    }
    for (auto &[fieldId, queue] : split.queues) {
        THOR_THROW_IF_FALSE(queue->getBufferToUnload(tensors.at(fieldId)));
    }

    Batch batch;
    for (auto &[fieldId, tensor] : tensors) {
        batch.insert(dataset->getSchema().getField(fieldId).name, tensor);
    }
    return batch;
}

void NumpyBatchSession::recycleBatch(ExampleType exampleType, Batch &&batch) {
    if (cancelled.load(std::memory_order_acquire)) {
        return;
    }
    SplitState &split = mutableSplit(exampleType);
    if (batch.size() != requiredFieldIds.size()) {
        throw std::runtime_error("NumpyBatchSession returned batch has unexpected tensor count.");
    }
    for (auto &[fieldId, queue] : split.queues) {
        const std::string &name = dataset->getSchema().getField(fieldId).name;
        if (!batch.contains(name)) {
            throw std::runtime_error("NumpyBatchSession returned batch is missing tensor '" + name + "'.");
        }
        THOR_THROW_IF_FALSE(queue->bufferUnloaded(batch.getTensor(name)));
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
        [](nb::handle cls, nb::dict tensors) -> std::shared_ptr<NumpyDataset> {
            (void)cls;
            return std::make_shared<NumpyDataset>(std::move(tensors));
        },
        "cls"_a,
        "tensors"_a,
        R"nbdoc(
Create an immutable in-memory dataset over one canonical table of NumPy arrays.

Each field must be a C-contiguous ndarray using the canonical NumPy or
ml_dtypes representation of a storable Thor dtype. All fields share one leading
example dimension. Thor marks the arrays read-only and retains them for the
dataset lifetime; callers must not mutate the underlying allocations through
other aliases. Split membership and batching belong to DatasetSplitManifest and
TrainingData. Device residency is selected transparently by
TrainingData.device_storage; off is the Python binding default.
        )nbdoc");
    numpyDataset.def("__init__", [](NumpyDataset *, nb::dict) {}, "tensors"_a);
}

}  // namespace Thor::PythonBindings
