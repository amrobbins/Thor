#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Loaders/LocalBatchLoader.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Training/StepExecutable.h"
#include "DeepLearning/Api/Training/Trainer.h"
#include "DeepLearning/Api/Training/TrainingRuns.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingPhase.h"
#include "DeepLearning/Api/Training/TrainingStep.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Loaders/NoOpDataProcessor.h"
#include "Utilities/Loaders/ShardedRawDatasetCreator.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"
#include "bindings/python/src/core/physical/NanobindDTypes.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

namespace {

using Float32Array = nb::ndarray<const float, nb::numpy, nb::c_contig>;
using Float16Array = nb::ndarray<const half, nb::numpy, nb::c_contig>;


template <typename ScalarT>
struct InMemoryNumpySplit {
    std::vector<uint8_t> examples;
    std::vector<uint8_t> labels;
    std::vector<uint64_t> exampleShape;
    std::vector<uint64_t> labelShape;
    ThorImplementation::TensorDescriptor exampleBatchDescriptor;
    ThorImplementation::TensorDescriptor labelBatchDescriptor;
    std::unique_ptr<AsyncTensorQueue> exampleQueue;
    std::unique_ptr<AsyncTensorQueue> labelQueue;
    uint64_t numExamples = 0;
    uint64_t nextBatchNum = 0;
};

template <typename ArrayT>
std::vector<uint64_t> shapeWithoutBatchDim(const ArrayT& array, const std::string& name) {
    if (array.ndim() < 2) {
        throw nb::value_error((name + " must have shape [N, ...] with at least one feature/label dimension").c_str());
    }
    std::vector<uint64_t> shape;
    for (size_t i = 1; i < array.ndim(); ++i) {
        if (array.shape(i) == 0) {
            throw nb::value_error((name + " dimensions must all be positive").c_str());
        }
        shape.push_back(static_cast<uint64_t>(array.shape(i)));
    }
    return shape;
}

uint64_t product(const std::vector<uint64_t>& values) {
    uint64_t result = 1;
    for (uint64_t value : values) {
        result *= value;
    }
    return result;
}

template <typename ScalarT, typename ArrayT>
void copyNumpyArrayBytes(std::vector<uint8_t>& dest, const ArrayT& src) {
    const size_t bytes = src.size() * sizeof(ScalarT);
    dest.resize(bytes);
    std::memcpy(dest.data(), src.data(), bytes);
}

template <typename ScalarT, typename ArrayT>
InMemoryNumpySplit<ScalarT> makeNumpySplit(const ArrayT& examples, const ArrayT& labels, const std::string& splitName) {
    if (examples.ndim() < 2) {
        throw nb::value_error((splitName + " examples must have shape [N, ...]").c_str());
    }
    if (labels.ndim() < 2) {
        throw nb::value_error((splitName + " labels must have shape [N, ...]").c_str());
    }
    if (examples.shape(0) == 0) {
        throw nb::value_error((splitName + " examples must contain at least one example").c_str());
    }
    if (examples.shape(0) != labels.shape(0)) {
        throw nb::value_error((splitName + " examples and labels must have the same leading dimension").c_str());
    }

    InMemoryNumpySplit<ScalarT> split;
    split.numExamples = static_cast<uint64_t>(examples.shape(0));
    split.exampleShape = shapeWithoutBatchDim(examples, splitName + " examples");
    split.labelShape = shapeWithoutBatchDim(labels, splitName + " labels");
    copyNumpyArrayBytes<ScalarT>(split.examples, examples);
    copyNumpyArrayBytes<ScalarT>(split.labels, labels);
    return split;
}

template <typename ScalarT, ThorImplementation::DataType TensorDataType, typename ArrayT>
class NumpyBatchLoader : public Loader {
   public:
    NumpyBatchLoader(ArrayT trainExamples,
                     ArrayT trainLabels,
                     ArrayT validateExamples,
                     ArrayT validateLabels,
                     uint64_t batchSize,
                     std::string exampleInputName,
                     std::string labelInputName,
                     std::string loaderClassName)
        : exampleInputName(std::move(exampleInputName)),
          labelInputName(std::move(labelInputName)),
          loaderClassName(std::move(loaderClassName)) {
        if (batchSize == 0) {
            throw nb::value_error((this->loaderClassName + " batch_size must be >= 1").c_str());
        }
        if (this->exampleInputName.empty() || this->labelInputName.empty()) {
            throw nb::value_error((this->loaderClassName + " input names must be non-empty").c_str());
        }
        this->batchSize = batchSize;
        train = makeNumpySplit<ScalarT>(trainExamples, trainLabels, "train");
        validate = makeNumpySplit<ScalarT>(validateExamples, validateLabels, "validate");
        // InMemoryNumpySplit owns AsyncTensorQueue instances, so it is intentionally
        // not copy-assignable once queues are part of the split state.  Keep TEST as
        // a distinct split backed by the validation arrays rather than copying the
        // VALIDATE split object.  This preserves independent queue ownership and
        // independent nextBatchNum state for validate vs. test.
        test = makeNumpySplit<ScalarT>(validateExamples, validateLabels, "test");

        if (train.exampleShape != validate.exampleShape || train.labelShape != validate.labelShape) {
            throw nb::value_error("train and validate arrays must have matching non-batch shapes");
        }

        initializeSplitQueues(train);
        initializeSplitQueues(validate);
        initializeSplitQueues(test);
    }

    ~NumpyBatchLoader() override {
        closeSplitQueues(train);
        closeSplitQueues(validate);
        closeSplitQueues(test);
    }

    Batch getBatch(ExampleType exampleType, uint64_t& batchNum) override {
        InMemoryNumpySplit<ScalarT>& split = mutableSplit(exampleType);
        const uint64_t batchesPerEpoch = getNumBatchesPerEpoch(exampleType);
        if (batchNum >= batchesPerEpoch) {
            batchNum = split.nextBatchNum;
        }

        ThorImplementation::Tensor examples;
        ThorImplementation::Tensor labels;
        bool queueOpen = split.exampleQueue->getBufferToLoad(examples);
        THOR_THROW_IF_FALSE(queueOpen);
        queueOpen = split.labelQueue->getBufferToLoad(labels);
        THOR_THROW_IF_FALSE(queueOpen);

        const uint64_t exampleElements = product(split.exampleShape);
        const uint64_t labelElements = product(split.labelShape);
        ScalarT* exampleDest = examples.getMemPtr<ScalarT>();
        ScalarT* labelDest = labels.getMemPtr<ScalarT>();
        const ScalarT* exampleSrc = reinterpret_cast<const ScalarT*>(split.examples.data());
        const ScalarT* labelSrc = reinterpret_cast<const ScalarT*>(split.labels.data());

        const uint64_t firstExample = batchNum * batchSize;
        for (uint64_t i = 0; i < batchSize; ++i) {
            const uint64_t exampleIndex = (firstExample + i) % split.numExamples;
            std::memcpy(
                exampleDest + (i * exampleElements), exampleSrc + (exampleIndex * exampleElements), exampleElements * sizeof(ScalarT));
            std::memcpy(labelDest + (i * labelElements), labelSrc + (exampleIndex * labelElements), labelElements * sizeof(ScalarT));
        }

        split.nextBatchNum = (batchNum + 1) % batchesPerEpoch;

        queueOpen = split.exampleQueue->bufferLoaded(examples);
        THOR_THROW_IF_FALSE(queueOpen);
        queueOpen = split.labelQueue->bufferLoaded(labels);
        THOR_THROW_IF_FALSE(queueOpen);
        queueOpen = split.exampleQueue->getBufferToUnload(examples);
        THOR_THROW_IF_FALSE(queueOpen);
        queueOpen = split.labelQueue->getBufferToUnload(labels);
        THOR_THROW_IF_FALSE(queueOpen);

        Batch batch;
        batch.insert(exampleInputName, examples);
        batch.insert(labelInputName, labels);
        return batch;
    }

    void returnBatchBuffers(ExampleType exampleType, Batch&& batch) override {
        InMemoryNumpySplit<ScalarT>& split = mutableSplit(exampleType);
        THOR_THROW_IF_FALSE(batch.contains(exampleInputName));
        THOR_THROW_IF_FALSE(batch.contains(labelInputName));

        bool queueOpen = split.exampleQueue->bufferUnloaded(batch.getTensor(exampleInputName));
        THOR_THROW_IF_FALSE(queueOpen);
        queueOpen = split.labelQueue->bufferUnloaded(batch.getTensor(labelInputName));
        THOR_THROW_IF_FALSE(queueOpen);
    }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        const InMemoryNumpySplit<ScalarT>& split = immutableSplit(exampleType);
        return (split.numExamples + batchSize - 1) / batchSize;
    }

    uint64_t getNumExamples(ExampleType exampleType) override { return immutableSplit(exampleType).numExamples; }

    uint64_t getNextBatchNum(ExampleType exampleType) override { return immutableSplit(exampleType).nextBatchNum; }

   private:
    static constexpr uint64_t tensorQueueSize = 32;

    void initializeSplitQueues(InMemoryNumpySplit<ScalarT>& split) {
        std::vector<uint64_t> exampleBatchShape{batchSize};
        exampleBatchShape.insert(exampleBatchShape.end(), split.exampleShape.begin(), split.exampleShape.end());
        std::vector<uint64_t> labelBatchShape{batchSize};
        labelBatchShape.insert(labelBatchShape.end(), split.labelShape.begin(), split.labelShape.end());

        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        split.exampleBatchDescriptor = ThorImplementation::TensorDescriptor(TensorDataType, exampleBatchShape);
        split.labelBatchDescriptor = ThorImplementation::TensorDescriptor(TensorDataType, labelBatchShape);
        split.exampleQueue = std::make_unique<AsyncTensorQueue>(tensorQueueSize, split.exampleBatchDescriptor, cpuPlacement);
        split.labelQueue = std::make_unique<AsyncTensorQueue>(tensorQueueSize, split.labelBatchDescriptor, cpuPlacement);
        split.exampleQueue->open();
        split.labelQueue->open();
    }

    void closeSplitQueues(InMemoryNumpySplit<ScalarT>& split) {
        if (split.exampleQueue != nullptr) {
            split.exampleQueue->close();
        }
        if (split.labelQueue != nullptr) {
            split.labelQueue->close();
        }
    }

    InMemoryNumpySplit<ScalarT>& mutableSplit(ExampleType exampleType) {
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

    const InMemoryNumpySplit<ScalarT>& immutableSplit(ExampleType exampleType) const {
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

    std::string exampleInputName;
    std::string labelInputName;
    std::string loaderClassName;
    InMemoryNumpySplit<ScalarT> train;
    InMemoryNumpySplit<ScalarT> validate;
    InMemoryNumpySplit<ScalarT> test;
};

using NumpyFloat32BatchLoader = NumpyBatchLoader<float, ThorImplementation::DataType::FP32, Float32Array>;
using NumpyFloat16BatchLoader = NumpyBatchLoader<half, ThorImplementation::DataType::FP16, Float16Array>;

std::set<std::string> stringSetFromVector(std::vector<std::string> values) { return std::set<std::string>(values.begin(), values.end()); }

std::unordered_set<std::string> unorderedStringSetFromVector(const std::vector<std::string>& values) {
    return std::unordered_set<std::string>(values.begin(), values.end());
}

LineStatsColorMode lineStatsColorModeFromString(const std::string& value) {
    if (value == "always") {
        return LineStatsColorMode::ALWAYS;
    }
    if (value == "auto") {
        return LineStatsColorMode::AUTO;
    }
    if (value == "never") {
        return LineStatsColorMode::NEVER;
    }
    throw nb::value_error("stats_color must be one of: 'always', 'auto', 'never'");
}

TrainingEventPhase trainingEventPhaseFromString(const std::string& value) {
    if (value == "train") {
        return TrainingEventPhase::TRAIN;
    }
    if (value == "validate" || value == "validation") {
        return TrainingEventPhase::VALIDATE;
    }
    if (value == "test") {
        return TrainingEventPhase::TEST;
    }
    throw nb::value_error("phase must be one of: 'train', 'validate', 'validation', 'test'");
}

TrainingRunsFailurePolicy trainingRunsFailurePolicyFromString(const std::string& value) {
    if (value == "cancel_siblings") {
        return TrainingRunsFailurePolicy::CANCEL_SIBLINGS;
    }
    if (value == "continue") {
        return TrainingRunsFailurePolicy::CONTINUE;
    }
    throw nb::value_error("failure_policy must be one of: 'cancel_siblings', 'continue'");
}

std::vector<TrainingRunsSpec> trainingRunsSpecsFromPython(nb::iterable runs) {
    std::vector<TrainingRunsSpec> specs;
    for (nb::handle item : runs) {
        nb::sequence entry = nb::cast<nb::sequence>(item);
        const size_t entrySize = nb::len(entry);
        if (entrySize < 2 || entrySize > 4) {
            throw nb::value_error(
                "TrainingRuns entries must be (run_name, trainer), (run_name, trainer, ensemble_group), or "
                "(run_name, trainer, ensemble_group, ensemble_weight)");
        }

        std::string runName = nb::cast<std::string>(entry[0]);
        std::shared_ptr<Trainer> trainer = nb::cast<std::shared_ptr<Trainer>>(entry[1]);
        TrainingRunsSpec spec(std::move(runName), std::move(trainer));
        if (entrySize >= 3 && !entry[2].is_none()) {
            spec.ensembleGroup = nb::cast<std::string>(entry[2]);
        }
        if (entrySize >= 4 && !entry[3].is_none()) {
            spec.ensembleWeight = nb::cast<double>(entry[3]);
        }
        specs.push_back(std::move(spec));
    }
    return specs;
}

nb::object optionalDouble(std::optional<double> value) {
    if (!value.has_value()) {
        return nb::none();
    }
    return nb::cast(*value);
}

nb::object optionalUint64FromStats(const std::optional<TrainingStatsSnapshot>& stats, uint64_t TrainingStatsSnapshot::*field) {
    if (!stats.has_value()) {
        return nb::none();
    }
    return nb::cast((*stats).*field);
}

nb::object optionalLossFromStats(const std::optional<TrainingStatsSnapshot>& stats) {
    if (!stats.has_value()) {
        return nb::none();
    }
    return optionalDouble(stats->loss);
}

}  // namespace

void bind_training(nb::module_& training) {
    training.doc() = "Thor training program scaffolding";

    auto loader = nb::class_<Loader>(training, "Loader");
    loader.attr("__module__") = "thor.training";
    loader.def("get_batch_size", &Loader::getBatchSize);
    loader.def("get_dataset_name", &Loader::getDatasetName);
    loader.def("set_dataset_name", &Loader::setDatasetName, "dataset_name"_a);

    auto numpy_float32_batch_loader = nb::class_<NumpyFloat32BatchLoader, Loader>(training, "NumpyFloat32BatchLoader");
    numpy_float32_batch_loader.attr("__module__") = "thor.training";
    numpy_float32_batch_loader.def_static(
        "__new__",
        [](nb::handle cls,
           Float32Array train_examples,
           Float32Array train_labels,
           Float32Array validate_examples,
           Float32Array validate_labels,
           uint64_t batch_size,
           const std::string& example_input_name,
           const std::string& label_input_name,
           const std::string& dataset_name) -> std::shared_ptr<NumpyFloat32BatchLoader> {
            (void)cls;
            auto loader = std::make_shared<NumpyFloat32BatchLoader>(train_examples,
                                                                    train_labels,
                                                                    validate_examples,
                                                                    validate_labels,
                                                                    batch_size,
                                                                    example_input_name,
                                                                    label_input_name,
                                                                    "NumpyFloat32BatchLoader");
            loader->setDatasetName(dataset_name);
            return loader;
        },
        "cls"_a,
        "train_examples"_a,
        "train_labels"_a,
        "validate_examples"_a,
        "validate_labels"_a,
        "batch_size"_a,
        "example_input_name"_a = "examples",
        "label_input_name"_a = "labels",
        "dataset_name"_a = "numpy");
    numpy_float32_batch_loader.def(
        "__init__",
        [](NumpyFloat32BatchLoader*,
           Float32Array,
           Float32Array,
           Float32Array,
           Float32Array,
           uint64_t,
           const std::string&,
           const std::string&,
           const std::string&) {},
        "train_examples"_a,
        "train_labels"_a,
        "validate_examples"_a,
        "validate_labels"_a,
        "batch_size"_a,
        "example_input_name"_a = "examples",
        "label_input_name"_a = "labels",
        "dataset_name"_a = "numpy");
    numpy_float32_batch_loader.def("get_num_train_examples",
                                   [](NumpyFloat32BatchLoader& self) { return self.getNumExamples(ExampleType::TRAIN); });
    numpy_float32_batch_loader.def("get_num_validate_examples",
                                   [](NumpyFloat32BatchLoader& self) { return self.getNumExamples(ExampleType::VALIDATE); });
    numpy_float32_batch_loader.def("get_num_train_batches",
                                   [](NumpyFloat32BatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::TRAIN); });
    numpy_float32_batch_loader.def("get_num_validate_batches",
                                   [](NumpyFloat32BatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::VALIDATE); });

    auto numpy_float16_batch_loader = nb::class_<NumpyFloat16BatchLoader, Loader>(training, "NumpyFloat16BatchLoader");
    numpy_float16_batch_loader.attr("__module__") = "thor.training";
    numpy_float16_batch_loader.def_static(
        "__new__",
        [](nb::handle cls,
           Float16Array train_examples,
           Float16Array train_labels,
           Float16Array validate_examples,
           Float16Array validate_labels,
           uint64_t batch_size,
           const std::string& example_input_name,
           const std::string& label_input_name,
           const std::string& dataset_name) -> std::shared_ptr<NumpyFloat16BatchLoader> {
            (void)cls;
            auto loader = std::make_shared<NumpyFloat16BatchLoader>(train_examples,
                                                                    train_labels,
                                                                    validate_examples,
                                                                    validate_labels,
                                                                    batch_size,
                                                                    example_input_name,
                                                                    label_input_name,
                                                                    "NumpyFloat16BatchLoader");
            loader->setDatasetName(dataset_name);
            return loader;
        },
        "cls"_a,
        "train_examples"_a,
        "train_labels"_a,
        "validate_examples"_a,
        "validate_labels"_a,
        "batch_size"_a,
        "example_input_name"_a = "examples",
        "label_input_name"_a = "labels",
        "dataset_name"_a = "numpy");
    numpy_float16_batch_loader.def(
        "__init__",
        [](NumpyFloat16BatchLoader*,
           Float16Array,
           Float16Array,
           Float16Array,
           Float16Array,
           uint64_t,
           const std::string&,
           const std::string&,
           const std::string&) {},
        "train_examples"_a,
        "train_labels"_a,
        "validate_examples"_a,
        "validate_labels"_a,
        "batch_size"_a,
        "example_input_name"_a = "examples",
        "label_input_name"_a = "labels",
        "dataset_name"_a = "numpy");
    numpy_float16_batch_loader.def("get_num_train_examples",
                                   [](NumpyFloat16BatchLoader& self) { return self.getNumExamples(ExampleType::TRAIN); });
    numpy_float16_batch_loader.def("get_num_validate_examples",
                                   [](NumpyFloat16BatchLoader& self) { return self.getNumExamples(ExampleType::VALIDATE); });
    numpy_float16_batch_loader.def("get_num_train_batches",
                                   [](NumpyFloat16BatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::TRAIN); });
    numpy_float16_batch_loader.def("get_num_validate_batches",
                                   [](NumpyFloat16BatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::VALIDATE); });

    training.def(
        "create_sharded_raw_dataset",
        [](const std::vector<std::string>& source_directories,
           const std::vector<std::string>& dest_directories,
           const std::string& base_dataset_file_name,
           uint64_t example_size_in_bytes,
           ThorImplementation::DataType data_type,
           uint32_t max_classes) {
            if (example_size_in_bytes == 0) {
                throw nb::value_error("example_size_in_bytes must be > 0");
            }
            std::vector<std::shared_ptr<Shard>> shards;
            ShardedRawDatasetCreator creator(unorderedStringSetFromVector(source_directories),
                                             unorderedStringSetFromVector(dest_directories),
                                             base_dataset_file_name,
                                             max_classes);
            creator.createDataset(std::make_unique<NoOpDataProcessor>(example_size_in_bytes, data_type), shards);
            std::vector<std::string> shard_paths;
            shard_paths.reserve(shards.size());
            for (const auto& shard : shards) {
                shard_paths.push_back(shard->getFilename());
            }
            return shard_paths;
        },
        "source_directories"_a,
        "dest_directories"_a,
        "base_dataset_file_name"_a,
        "example_size_in_bytes"_a,
        "data_type"_a,
        "max_classes"_a = 0,
        R"nbdoc(
Create Thor shard files from an already preprocessed raw dataset directory.

The source directories must contain train/, validate/, and test/ subdirectories,
each with one subdirectory per class. Every example file is expected to already
contain exactly example_size_in_bytes bytes in the target tensor layout and dtype.
NoOpDataProcessor is used, so image decoding/normalization should happen before
calling this helper.
        )nbdoc");

    auto local_batch_loader = nb::class_<LocalBatchLoader, Loader>(training, "LocalBatchLoader");
    local_batch_loader.attr("__module__") = "thor.training";
    local_batch_loader.def_static(
        "__new__",
        [](nb::handle cls,
           std::vector<std::string> shard_paths,
           std::vector<uint64_t> example_shape,
           ThorImplementation::DataType example_data_type,
           std::vector<uint64_t> label_shape,
           ThorImplementation::DataType label_data_type,
           uint64_t batch_size,
           const std::string& dataset_name,
           uint64_t batch_queue_depth) -> std::shared_ptr<LocalBatchLoader> {
            (void)cls;
            if (shard_paths.empty()) {
                throw nb::value_error("LocalBatchLoader requires at least one shard path");
            }
            if (example_shape.empty() || label_shape.empty()) {
                throw nb::value_error("LocalBatchLoader example_shape and label_shape must be non-empty");
            }
            if (batch_size == 0) {
                throw nb::value_error("LocalBatchLoader batch_size must be >= 1");
            }
            if (batch_queue_depth == 0) {
                throw nb::value_error("LocalBatchLoader batch_queue_depth must be >= 1");
            }
            auto loader =
                std::make_shared<LocalBatchLoader>(stringSetFromVector(std::move(shard_paths)),
                                                   ThorImplementation::TensorDescriptor(example_data_type, std::move(example_shape)),
                                                   ThorImplementation::TensorDescriptor(label_data_type, std::move(label_shape)),
                                                   batch_size,
                                                   batch_queue_depth);
            loader->setDatasetName(dataset_name);
            return loader;
        },
        "cls"_a,
        "shard_paths"_a,
        "example_shape"_a,
        "example_data_type"_a,
        "label_shape"_a,
        "label_data_type"_a,
        "batch_size"_a,
        "dataset_name"_a = "local_shards",
        "batch_queue_depth"_a = 32);
    local_batch_loader.def(
        "__init__",
        [](LocalBatchLoader*,
           std::vector<std::string>,
           std::vector<uint64_t>,
           ThorImplementation::DataType,
           std::vector<uint64_t>,
           ThorImplementation::DataType,
           uint64_t,
           const std::string&,
           uint64_t) {},
        "shard_paths"_a,
        "example_shape"_a,
        "example_data_type"_a,
        "label_shape"_a,
        "label_data_type"_a,
        "batch_size"_a,
        "dataset_name"_a = "local_shards",
        "batch_queue_depth"_a = 32);
    local_batch_loader.def("get_num_train_examples", [](LocalBatchLoader& self) { return self.getNumExamples(ExampleType::TRAIN); });
    local_batch_loader.def("get_num_validate_examples", [](LocalBatchLoader& self) { return self.getNumExamples(ExampleType::VALIDATE); });
    local_batch_loader.def("get_num_train_batches", [](LocalBatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::TRAIN); });
    local_batch_loader.def("get_num_validate_batches",
                           [](LocalBatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::VALIDATE); });

    auto trainer_fit_options = nb::class_<TrainerFitOptions>(training, "TrainerFitOptions");
    trainer_fit_options.attr("__module__") = "thor.training";
    trainer_fit_options.def(nb::init<>()).def_rw("epochs", &TrainerFitOptions::epochs);

    auto trainer = nb::class_<Trainer>(training, "Trainer");
    trainer.attr("__module__") = "thor.training";
    trainer.def_static(
        "__new__",
        [](nb::handle cls,
           std::shared_ptr<Network> network,
           std::shared_ptr<Loader> loader,
           std::shared_ptr<Optimizer> optimizer,
           nb::object training_program,
           bool debug_synchronous,
           bool stats,
           double stats_interval_s,
           uint64_t max_in_flight_batches,
           std::vector<std::string> scalar_tensors_to_report,
           bool stats_stderr_also,
           std::string stats_color,
           std::optional<std::string> save_model_dir,
           bool save_model_overwrite,
           bool save_optimizer_state) -> std::shared_ptr<Trainer> {
            (void)cls;
            Trainer::Builder builder;
            builder.network(std::move(network))
                .loader(std::move(loader))
                .statsEnabled(stats)
                .statsIntervalSeconds(stats_interval_s)
                .statsStderrAlso(stats_stderr_also)
                .statsColorMode(lineStatsColorModeFromString(stats_color))
                .maxInFlightBatches(max_in_flight_batches)
                .scalarTensorsToReport(stringSetFromVector(std::move(scalar_tensors_to_report)))
                .saveModelDirectory(std::move(save_model_dir))
                .saveModelOverwrite(save_model_overwrite)
                .saveOptimizerState(save_optimizer_state);
            if (optimizer != nullptr) {
                builder.optimizer(std::move(optimizer));
            }
            if (!training_program.is_none()) {
                builder.trainingProgram(nb::cast<std::shared_ptr<TrainingProgram>>(training_program));
            }
            if (debug_synchronous) {
                builder.debugSynchronousExecutor();
            }
            return std::make_shared<Trainer>(builder.build());
        },
        "cls"_a,
        "network"_a,
        "loader"_a,
        "optimizer"_a.none() = nb::none(),
        "training_program"_a.none() = nb::none(),
        "debug_synchronous"_a = false,
        "stats"_a = true,
        "stats_interval_s"_a = 10.0,
        "max_in_flight_batches"_a = 32,
        "scalar_tensors_to_report"_a = std::vector<std::string>{"loss"},
        "stats_stderr_also"_a = false,
        "stats_color"_a = "auto",
        "save_model_dir"_a.none() = nb::none(),
        "save_model_overwrite"_a = false,
        "save_optimizer_state"_a = true);
    trainer.def(
        "__init__",
        [](Trainer*,
           std::shared_ptr<Network>,
           std::shared_ptr<Loader>,
           std::shared_ptr<Optimizer>,
           nb::object,
           bool,
           bool,
           double,
           uint64_t,
           std::vector<std::string>,
           bool,
           std::string,
           std::optional<std::string>,
           bool,
           bool) {},
        "network"_a,
        "loader"_a,
        "optimizer"_a.none() = nb::none(),
        "training_program"_a.none() = nb::none(),
        "debug_synchronous"_a = false,
        "stats"_a = true,
        "stats_interval_s"_a = 10.0,
        "max_in_flight_batches"_a = 32,
        "scalar_tensors_to_report"_a = std::vector<std::string>{"loss"},
        "stats_stderr_also"_a = false,
        "stats_color"_a = "auto",
        "save_model_dir"_a.none() = nb::none(),
        "save_model_overwrite"_a = false,
        "save_optimizer_state"_a = true);
    trainer.def("fit", [](Trainer& self, uint32_t epochs) {
        nb::gil_scoped_release release;
        self.fit(epochs);
    }, "epochs"_a);

    auto training_event_phase = nb::enum_<TrainingEventPhase>(training, "TrainingEventPhase")
                                    .value("unknown", TrainingEventPhase::UNKNOWN)
                                    .value("train", TrainingEventPhase::TRAIN)
                                    .value("validate", TrainingEventPhase::VALIDATE)
                                    .value("test", TrainingEventPhase::TEST);
    training_event_phase.attr("__module__") = "thor.training";

    auto training_stats_snapshot = nb::class_<TrainingStatsSnapshot>(training, "TrainingStatsSnapshot");
    training_stats_snapshot.attr("__module__") = "thor.training";
    training_stats_snapshot.def_ro("network_name", &TrainingStatsSnapshot::networkName);
    training_stats_snapshot.def_ro("dataset_name", &TrainingStatsSnapshot::datasetName);
    training_stats_snapshot.def_ro("phase", &TrainingStatsSnapshot::phase);
    training_stats_snapshot.def_ro("epoch", &TrainingStatsSnapshot::epoch);
    training_stats_snapshot.def_ro("epochs", &TrainingStatsSnapshot::epochs);
    training_stats_snapshot.def_ro("step", &TrainingStatsSnapshot::step);
    training_stats_snapshot.def_ro("step_in_epoch", &TrainingStatsSnapshot::stepInEpoch);
    training_stats_snapshot.def_ro("steps_per_epoch", &TrainingStatsSnapshot::stepsPerEpoch);
    training_stats_snapshot.def_ro("batch_size", &TrainingStatsSnapshot::batchSize);
    training_stats_snapshot.def_ro("samples_processed", &TrainingStatsSnapshot::samplesProcessed);
    training_stats_snapshot.def_ro("in_flight_batches", &TrainingStatsSnapshot::inFlightBatches);
    training_stats_snapshot.def_ro("elapsed_seconds", &TrainingStatsSnapshot::elapsedSeconds);
    training_stats_snapshot.def_ro("samples_per_second", &TrainingStatsSnapshot::samplesPerSecond);
    training_stats_snapshot.def_ro("batches_per_second", &TrainingStatsSnapshot::batchesPerSecond);
    training_stats_snapshot.def_ro("floating_point_operations_per_batch",
                                   &TrainingStatsSnapshot::floatingPointOperationsPerBatch);
    training_stats_snapshot.def_ro("floating_point_operations_per_second",
                                   &TrainingStatsSnapshot::floatingPointOperationsPerSecond);
    training_stats_snapshot.def_prop_ro("loss", [](const TrainingStatsSnapshot& self) { return optionalDouble(self.loss); });
    training_stats_snapshot.def_prop_ro("accuracy", [](const TrainingStatsSnapshot& self) { return optionalDouble(self.accuracy); });
    training_stats_snapshot.def_prop_ro("learning_rate",
                                        [](const TrainingStatsSnapshot& self) { return optionalDouble(self.learningRate); });
    training_stats_snapshot.def_prop_ro("momentum", [](const TrainingStatsSnapshot& self) { return optionalDouble(self.momentum); });
    training_stats_snapshot.def_ro("metrics", &TrainingStatsSnapshot::metrics);

    auto training_run_status = nb::enum_<TrainingRunStatus>(training, "TrainingRunStatus")
                                   .value("not_started", TrainingRunStatus::NOT_STARTED)
                                   .value("running", TrainingRunStatus::RUNNING)
                                   .value("completed", TrainingRunStatus::COMPLETED)
                                   .value("failed", TrainingRunStatus::FAILED)
                                   .value("cancelled", TrainingRunStatus::CANCELLED)
                                   .value("interrupted", TrainingRunStatus::INTERRUPTED)
                                   .value("oom", TrainingRunStatus::OUT_OF_MEMORY);
    training_run_status.attr("__module__") = "thor.training";

    auto training_runs_failure_policy = nb::enum_<TrainingRunsFailurePolicy>(training, "TrainingRunsFailurePolicy")
                                            .value("continue_", TrainingRunsFailurePolicy::CONTINUE)
                                            .value("cancel_siblings", TrainingRunsFailurePolicy::CANCEL_SIBLINGS);
    training_runs_failure_policy.attr("__module__") = "thor.training";

    auto training_run_result = nb::class_<TrainingRunResult>(training, "TrainingRunResult");
    training_run_result.attr("__module__") = "thor.training";
    training_run_result.def_prop_ro("run_name", [](const TrainingRunResult& self) { return self.runName; });
    training_run_result.def_prop_ro("ensemble_group", [](const TrainingRunResult& self) -> nb::object {
        if (!self.ensembleGroup.has_value()) {
            return nb::none();
        }
        return nb::cast(*self.ensembleGroup);
    });
    training_run_result.def_ro("ensemble_weight", &TrainingRunResult::ensembleWeight);
    training_run_result.def_prop_ro("status", [](const TrainingRunResult& self) { return trainingRunStatusName(self.status); });
    training_run_result.def_prop_ro("status_enum", [](const TrainingRunResult& self) { return self.status; });
    training_run_result.def_prop_ro("exception_type", [](const TrainingRunResult& self) { return self.exception.type; });
    training_run_result.def_prop_ro("exception_message", [](const TrainingRunResult& self) { return self.exception.message; });
    training_run_result.def_prop_ro("final_training_stats", [](const TrainingRunResult& self) { return self.finalTrainingStats; });
    training_run_result.def_prop_ro("final_validation_stats", [](const TrainingRunResult& self) { return self.finalValidationStats; });
    training_run_result.def_prop_ro("final_test_stats", [](const TrainingRunResult& self) { return self.finalTestStats; });
    training_run_result.def_prop_ro("final_training_loss",
                                    [](const TrainingRunResult& self) { return optionalLossFromStats(self.finalTrainingStats); });
    training_run_result.def_prop_ro("final_validation_loss",
                                    [](const TrainingRunResult& self) { return optionalLossFromStats(self.finalValidationStats); });
    training_run_result.def_prop_ro("final_test_loss",
                                    [](const TrainingRunResult& self) { return optionalLossFromStats(self.finalTestStats); });
    training_run_result.def("final_loss", [](const TrainingRunResult& self, const std::string& phase) {
        return optionalLossFromStats(self.finalStatsForPhase(trainingEventPhaseFromString(phase)));
    }, "phase"_a);
    training_run_result.def_prop_ro("final_training_step", [](const TrainingRunResult& self) {
        return optionalUint64FromStats(self.finalTrainingStats, &TrainingStatsSnapshot::step);
    });
    training_run_result.def_prop_ro("final_validation_step", [](const TrainingRunResult& self) {
        return optionalUint64FromStats(self.finalValidationStats, &TrainingStatsSnapshot::step);
    });
    training_run_result.def_prop_ro("final_test_step", [](const TrainingRunResult& self) {
        return optionalUint64FromStats(self.finalTestStats, &TrainingStatsSnapshot::step);
    });
    training_run_result.def("completed", &TrainingRunResult::completed);
    training_run_result.def("failed", &TrainingRunResult::failed);
    training_run_result.def("cancelled", &TrainingRunResult::cancelled);

    auto training_run_output_signature = nb::class_<TrainingRunOutputSignature>(training, "TrainingRunOutputSignature");
    training_run_output_signature.attr("__module__") = "thor.training";
    training_run_output_signature.def_prop_ro("output_name", [](const TrainingRunOutputSignature& self) { return self.outputName; });
    training_run_output_signature.def_prop_ro("dimensions", [](const TrainingRunOutputSignature& self) { return self.dimensions; });
    training_run_output_signature.def_prop_ro("data_type", [](const TrainingRunOutputSignature& self) { return self.dataType; });

    auto training_ensemble_member_result = nb::class_<TrainingEnsembleMemberResult>(training, "TrainingEnsembleMemberResult");
    training_ensemble_member_result.attr("__module__") = "thor.training";
    training_ensemble_member_result.def_prop_ro("run_name", [](const TrainingEnsembleMemberResult& self) { return self.runName; });
    training_ensemble_member_result.def_ro("weight", &TrainingEnsembleMemberResult::weight);
    training_ensemble_member_result.def_prop_ro("status", [](const TrainingEnsembleMemberResult& self) { return trainingRunStatusName(self.status); });
    training_ensemble_member_result.def_prop_ro("status_enum", [](const TrainingEnsembleMemberResult& self) { return self.status; });
    training_ensemble_member_result.def_prop_ro("final_training_loss", [](const TrainingEnsembleMemberResult& self) { return optionalDouble(self.finalTrainingLoss); });
    training_ensemble_member_result.def_prop_ro("final_validation_loss", [](const TrainingEnsembleMemberResult& self) { return optionalDouble(self.finalValidationLoss); });
    training_ensemble_member_result.def_prop_ro("final_test_loss", [](const TrainingEnsembleMemberResult& self) { return optionalDouble(self.finalTestLoss); });

    auto training_ensemble_result = nb::class_<TrainingEnsembleResult>(training, "TrainingEnsembleResult");
    training_ensemble_result.attr("__module__") = "thor.training";
    training_ensemble_result.def_prop_ro("ensemble_group", [](const TrainingEnsembleResult& self) { return self.ensembleGroup; });
    training_ensemble_result.def_prop_ro("members", [](const TrainingEnsembleResult& self) { return self.members; });
    training_ensemble_result.def_prop_ro("output_signature", [](const TrainingEnsembleResult& self) { return self.outputSignature; });
    training_ensemble_result.def("__len__", &TrainingEnsembleResult::size);
    training_ensemble_result.def("__bool__", [](const TrainingEnsembleResult& self) { return !self.empty(); });
    training_ensemble_result.def("all_completed", &TrainingEnsembleResult::allCompleted);
    training_ensemble_result.def("any_failed", &TrainingEnsembleResult::anyFailed);
    training_ensemble_result.def_prop_ro("total_weight", &TrainingEnsembleResult::totalWeight);
    training_ensemble_result.def_prop_ro("status_counts", &TrainingEnsembleResult::statusCounts);
    training_ensemble_result.def("weighted_final_loss", [](const TrainingEnsembleResult& self, const std::string& phase) {
        return optionalDouble(self.weightedFinalLossForPhase(trainingEventPhaseFromString(phase)));
    }, "phase"_a);
    training_ensemble_result.def_prop_ro("weighted_final_training_loss", [](const TrainingEnsembleResult& self) {
        return optionalDouble(self.weightedFinalTrainingLoss());
    });
    training_ensemble_result.def_prop_ro("weighted_member_training_loss", [](const TrainingEnsembleResult& self) {
        return optionalDouble(self.weightedFinalTrainingLoss());
    });
    training_ensemble_result.def_prop_ro("weighted_final_validation_loss", [](const TrainingEnsembleResult& self) {
        return optionalDouble(self.weightedFinalValidationLoss());
    });
    training_ensemble_result.def_prop_ro("weighted_member_validation_loss", [](const TrainingEnsembleResult& self) {
        return optionalDouble(self.weightedFinalValidationLoss());
    });
    training_ensemble_result.def_prop_ro("weighted_final_test_loss", [](const TrainingEnsembleResult& self) {
        return optionalDouble(self.weightedFinalTestLoss());
    });
    training_ensemble_result.def_prop_ro("weighted_member_test_loss", [](const TrainingEnsembleResult& self) {
        return optionalDouble(self.weightedFinalTestLoss());
    });

    auto training_runs_result = nb::class_<TrainingRunsResult>(training, "TrainingRunsResult");
    training_runs_result.attr("__module__") = "thor.training";
    training_runs_result.def("__len__", &TrainingRunsResult::size);
    training_runs_result.def("__bool__", [](const TrainingRunsResult& self) { return !self.empty(); });
    training_runs_result.def("__getitem__", [](const TrainingRunsResult& self, int64_t index) -> const TrainingRunResult& {
        int64_t resolvedIndex = index;
        if (resolvedIndex < 0) {
            resolvedIndex += static_cast<int64_t>(self.size());
        }
        if (resolvedIndex < 0) {
            throw nb::index_error("TrainingRunsResult index is out of range");
        }
        return self.at(static_cast<size_t>(resolvedIndex));
    }, nb::rv_policy::reference_internal);
    training_runs_result.def("__getitem__", [](const TrainingRunsResult& self, const std::string& runName) -> const TrainingRunResult& {
        return self.at(runName);
    }, nb::rv_policy::reference_internal);
    training_runs_result.def_prop_ro("runs", [](const TrainingRunsResult& self) { return self.runs(); });
    training_runs_result.def_prop_ro("ensembles", [](const TrainingRunsResult& self) { return self.ensembles(); });
    training_runs_result.def_prop_ro("has_ensembles", &TrainingRunsResult::hasEnsembles);
    training_runs_result.def("ensemble", [](const TrainingRunsResult& self, const std::string& ensembleGroup) -> const TrainingEnsembleResult& {
        return self.ensemble(ensembleGroup);
    }, nb::rv_policy::reference_internal, "ensemble_group"_a);
    training_runs_result.def("all_completed", &TrainingRunsResult::allCompleted);
    training_runs_result.def("any_failed", &TrainingRunsResult::anyFailed);
    training_runs_result.def("any_cancelled", &TrainingRunsResult::anyCancelled);
    training_runs_result.def_prop_ro("status_counts", &TrainingRunsResult::statusCounts);
    training_runs_result.def("get_status_counts", &TrainingRunsResult::statusCounts);

    auto training_runs = nb::class_<TrainingRuns>(training, "TrainingRuns");
    training_runs.attr("__module__") = "thor.training";
    training_runs.def_static(
        "__new__",
        [](nb::handle cls,
           nb::iterable runs,
           const std::string& failure_policy,
           double max_summary_logs_per_second,
           std::optional<size_t> max_parallel_runs) -> std::shared_ptr<TrainingRuns> {
            (void)cls;
            return std::make_shared<TrainingRuns>(trainingRunsSpecsFromPython(runs),
                                                  trainingRunsFailurePolicyFromString(failure_policy),
                                                  max_summary_logs_per_second,
                                                  max_parallel_runs);
        },
        "cls"_a,
        "runs"_a,
        "failure_policy"_a = "cancel_siblings",
        "max_summary_logs_per_second"_a = 2.0,
        "max_parallel_runs"_a.none() = nb::none());
    training_runs.def(
        "__init__",
        [](TrainingRuns*, nb::iterable, const std::string&, double, std::optional<size_t>) {},
        "runs"_a,
        "failure_policy"_a = "cancel_siblings",
        "max_summary_logs_per_second"_a = 2.0,
        "max_parallel_runs"_a.none() = nb::none());
    training_runs.def("fit", [](TrainingRuns& self, uint32_t epochs) {
        nb::gil_scoped_release release;
        return self.fit(epochs);
    }, "epochs"_a);

    auto gradient_clear_policy = nb::enum_<TrainingStep::GradientClearPolicy>(training, "GradientClearPolicy")
                                     .value("clear_before_step", TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP)
                                     .value("accumulate", TrainingStep::GradientClearPolicy::ACCUMULATE);
    gradient_clear_policy.attr("__module__") = "thor.training";

    auto training_input_binding = nb::class_<TrainingInputBinding>(training, "TrainingInputBinding");
    training_input_binding.attr("__module__") = "thor.training";
    training_input_binding.def(
        "__init__",
        [](TrainingInputBinding* self, const std::string& network_input_name, const std::string& batch_input_name) {
            new (self) TrainingInputBinding(network_input_name, batch_input_name);
        },
        "network_input_name"_a,
        "batch_input_name"_a);
    training_input_binding.def_prop_ro("network_input_name", &TrainingInputBinding::getNetworkInputName);
    training_input_binding.def_prop_ro("batch_input_name", &TrainingInputBinding::getBatchInputName);
    training_input_binding.def("is_initialized", &TrainingInputBinding::isInitialized);
    training_input_binding.def("get_architecture_json", &TrainingInputBinding::architectureJsonString);
    training_input_binding.def_static(
        "deserialize",
        [](const std::string& payload) { return TrainingInputBinding::deserialize(nlohmann::json::parse(payload)); },
        "architecture_json"_a);
    training_input_binding.def("__eq__", &TrainingInputBinding::operator==);

    auto training_phase = nb::class_<TrainingPhase>(training, "TrainingPhase");
    training_phase.attr("__module__") = "thor.training";
    training_phase.def_static(
        "__new__",
        [](nb::handle cls,
           const std::string& name,
           std::vector<Tensor> loss_roots,
           std::map<std::string, Tensor> outputs,
           std::vector<std::string> depends_on,
           bool enabled) -> std::shared_ptr<TrainingPhase> {
            (void)cls;
            return std::make_shared<TrainingPhase>(
                name, std::move(loss_roots), std::move(outputs), std::move(depends_on), enabled);
        },
        "cls"_a,
        "name"_a,
        "loss_roots"_a = std::vector<Tensor>{},
        "outputs"_a = std::map<std::string, Tensor>{},
        "depends_on"_a = std::vector<std::string>{},
        "enabled"_a = true);
    training_phase.def(
        "__init__",
        [](TrainingPhase*, const std::string&, std::vector<Tensor>, std::map<std::string, Tensor>, std::vector<std::string>, bool) {},
        "name"_a,
        "loss_roots"_a = std::vector<Tensor>{},
        "outputs"_a = std::map<std::string, Tensor>{},
        "depends_on"_a = std::vector<std::string>{},
        "enabled"_a = true);
    training_phase.def_prop_ro("name", &TrainingPhase::getName);
    training_phase.def_prop_rw("enabled", &TrainingPhase::isEnabled, &TrainingPhase::setEnabled);
    training_phase.def("is_initialized", &TrainingPhase::isInitialized);
    training_phase.def("is_enabled", &TrainingPhase::isEnabled);
    training_phase.def("enable", &TrainingPhase::enable);
    training_phase.def("disable", &TrainingPhase::disable);
    training_phase.def("set_enabled", &TrainingPhase::setEnabled, "enabled"_a);
    training_phase.def("get_loss_roots", &TrainingPhase::getLossRoots, nb::rv_policy::reference_internal);
    training_phase.def("get_outputs", &TrainingPhase::getOutputs, nb::rv_policy::reference_internal);
    training_phase.def("get_depends_on", &TrainingPhase::getDependsOn, nb::rv_policy::reference_internal);
    training_phase.def("get_architecture_json", &TrainingPhase::architectureJsonString);
    training_phase.def_static(
        "deserialize",
        [](const std::string& payload) {
            std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
            return std::make_shared<TrainingPhase>(TrainingPhase::deserialize(nlohmann::json::parse(payload), archiveReader));
        },
        "architecture_json"_a);

    auto training_step = nb::class_<TrainingStep>(training, "TrainingStep");
    training_step.attr("__module__") = "thor.training";
    training_step.def_static(
        "__new__",
        [](nb::handle cls,
           const std::string& name,
           nb::object loss_roots,
           std::shared_ptr<Optimizer> optimizer,
           std::vector<ParameterReference> update_parameters,
           uint32_t repeat_count,
           TrainingStep::GradientClearPolicy gradient_clear_policy,
           std::vector<TrainingInputBinding> input_bindings,
           bool enabled,
           nb::object phases) -> std::shared_ptr<TrainingStep> {
            (void)cls;
            const bool hasLossRoots = !loss_roots.is_none();
            const bool hasPhases = !phases.is_none();
            if (hasLossRoots == hasPhases) {
                throw nb::value_error("TrainingStep requires exactly one of loss_roots or phases.");
            }
            if (hasPhases) {
                return std::make_shared<TrainingStep>(name,
                                                      nb::cast<std::vector<std::shared_ptr<TrainingPhase>>>(phases),
                                                      std::move(optimizer),
                                                      std::move(update_parameters),
                                                      repeat_count,
                                                      gradient_clear_policy,
                                                      std::move(input_bindings),
                                                      enabled);
            }
            return std::make_shared<TrainingStep>(name,
                                                  nb::cast<std::vector<Tensor>>(loss_roots),
                                                  std::move(optimizer),
                                                  std::move(update_parameters),
                                                  repeat_count,
                                                  gradient_clear_policy,
                                                  std::move(input_bindings),
                                                  enabled);
        },
        "cls"_a,
        "name"_a,
        "loss_roots"_a.none() = nb::none(),
        "optimizer"_a.none() = nb::none(),
        "update_parameters"_a = std::vector<ParameterReference>{},
        "repeat_count"_a = 1,
        "gradient_clear_policy"_a = TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
        "input_bindings"_a = std::vector<TrainingInputBinding>{},
        "enabled"_a = true,
        "phases"_a.none() = nb::none());
    training_step.def(
        "__init__",
        [](TrainingStep*,
           const std::string&,
           nb::object,
           std::shared_ptr<Optimizer>,
           std::vector<ParameterReference>,
           uint32_t,
           TrainingStep::GradientClearPolicy,
           std::vector<TrainingInputBinding>,
           bool,
           nb::object) {},
        "name"_a,
        "loss_roots"_a.none() = nb::none(),
        "optimizer"_a.none() = nb::none(),
        "update_parameters"_a = std::vector<ParameterReference>{},
        "repeat_count"_a = 1,
        "gradient_clear_policy"_a = TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
        "input_bindings"_a = std::vector<TrainingInputBinding>{},
        "enabled"_a = true,
        "phases"_a.none() = nb::none());
    training_step.def_prop_ro("name", &TrainingStep::getName);
    training_step.def_prop_ro("repeat_count", &TrainingStep::getRepeatCount);
    training_step.def_prop_ro("gradient_clear_policy", &TrainingStep::getGradientClearPolicy);
    training_step.def_prop_rw("enabled", &TrainingStep::isEnabled, &TrainingStep::setEnabled);
    training_step.def("is_initialized", &TrainingStep::isInitialized);
    training_step.def("is_enabled", &TrainingStep::isEnabled);
    training_step.def("enable", &TrainingStep::enable);
    training_step.def("disable", &TrainingStep::disable);
    training_step.def("set_enabled", &TrainingStep::setEnabled, "enabled"_a);
    training_step.def("validate_enabled_phase_dependencies", &TrainingStep::validateEnabledPhaseDependencies);
    training_step.def("get_loss_roots", &TrainingStep::getLossRoots, nb::rv_policy::reference_internal);
    training_step.def("get_active_loss_roots", &TrainingStep::getActiveLossRoots);
    training_step.def("get_active_phase_names", &TrainingStep::getActivePhaseNames);
    training_step.def("get_phases", &TrainingStep::getPhases, nb::rv_policy::reference_internal);
    training_step.def("get_optimizer", &TrainingStep::getOptimizer);
    training_step.def("get_update_parameters", &TrainingStep::getUpdateParameters, nb::rv_policy::reference_internal);
    training_step.def("get_input_bindings", &TrainingStep::getInputBindings, nb::rv_policy::reference_internal);
    training_step.def("updates_parameter", &TrainingStep::updatesParameter, "parameter"_a);
    training_step.def("get_architecture_json", &TrainingStep::architectureJsonString);
    training_step.def_static(
        "deserialize",
        [](const std::string& payload) {
            std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
            return std::make_shared<TrainingStep>(TrainingStep::deserialize(nlohmann::json::parse(payload), archiveReader, nullptr));
        },
        "architecture_json"_a);

    auto step_executable = nb::class_<StepExecutable>(training, "StepExecutable");
    step_executable.attr("__module__") = "thor.training";
    step_executable.def("is_initialized", &StepExecutable::isInitialized);
    step_executable.def_prop_ro("name", &StepExecutable::getName);
    step_executable.def_prop_ro("repeat_count", &StepExecutable::getRepeatCount);
    step_executable.def_prop_ro("gradient_clear_policy", &StepExecutable::getGradientClearPolicy);
    step_executable.def("get_loss_roots", &StepExecutable::getLossRoots, nb::rv_policy::reference_internal);
    step_executable.def("get_resolved_loss_roots", &StepExecutable::getResolvedLossRoots, nb::rv_policy::reference_internal);
    step_executable.def("get_active_phase_names", &StepExecutable::getActivePhaseNames, nb::rv_policy::reference_internal);
    step_executable.def("get_optimizer", &StepExecutable::getOptimizer);
    step_executable.def(
        "get_update_parameter_references", &StepExecutable::getUpdateParameterReferences, nb::rv_policy::reference_internal);
    step_executable.def("get_resolved_update_parameters", &StepExecutable::getResolvedUpdateParameters, nb::rv_policy::reference_internal);
    step_executable.def("get_input_bindings", &StepExecutable::getInputBindings, nb::rv_policy::reference_internal);
    step_executable.def("get_resolved_input_bindings", &StepExecutable::getResolvedInputBindings, nb::rv_policy::reference_internal);
    step_executable.def("get_required_batch_input_names", &StepExecutable::getRequiredBatchInputNames, nb::rv_policy::reference_internal);
    step_executable.def("get_architecture_json", &StepExecutable::architectureJsonString);

    auto training_program = nb::class_<TrainingProgram>(training, "TrainingProgram");
    training_program.attr("__module__") = "thor.training";
    training_program.def_static(
        "__new__",
        [](nb::handle cls, nb::object steps) -> std::shared_ptr<TrainingProgram> {
            (void)cls;
            if (steps.is_none()) {
                return std::make_shared<TrainingProgram>();
            }
            return std::make_shared<TrainingProgram>(nb::cast<std::vector<std::shared_ptr<TrainingStep>>>(steps));
        },
        "cls"_a,
        "steps"_a.none() = nb::none());
    training_program.def("__init__", [](TrainingProgram*, nb::object) {}, "steps"_a.none() = nb::none());
    training_program.def("add_step", &TrainingProgram::addStep, "step"_a);
    training_program.def("get_num_steps", &TrainingProgram::getNumSteps);
    training_program.def("get_step", [](TrainingProgram& self, uint64_t index) { return self.getStepReference(index); }, "index"_a);
    training_program.def("get_steps", &TrainingProgram::getSteps, nb::rv_policy::reference_internal);
    training_program.def("is_initialized", &TrainingProgram::isInitialized);
    training_program.def("get_architecture_json", &TrainingProgram::architectureJsonString);
    training_program.def_static(
        "deserialize",
        [](const std::string& payload) {
            std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
            return std::make_shared<TrainingProgram>(TrainingProgram::deserialize(nlohmann::json::parse(payload), archiveReader, nullptr));
        },
        "architecture_json"_a);
    training_program.def("compile", &TrainingProgram::compile, "placed_network"_a);
}
