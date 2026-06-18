#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Loaders/LocalBatchLoader.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Training/StepExecutable.h"
#include "DeepLearning/Api/Training/Trainer.h"
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
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
    trainer.def(
        "__init__",
        [](Trainer* self,
           Network& network,
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
           bool save_optimizer_state) {
            Trainer::Builder builder;
            builder.network(network)
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
                builder.trainingProgram(nb::cast<TrainingProgram>(training_program));
            }
            if (debug_synchronous) {
                builder.debugSynchronousExecutor();
            }
            new (self) Trainer(builder.build());
        },
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
    trainer.def("fit", nb::overload_cast<uint32_t>(&Trainer::fit), "epochs"_a);

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

    auto training_step = nb::class_<TrainingStep>(training, "TrainingStep");
    training_step.attr("__module__") = "thor.training";
    training_step.def(
        "__init__",
        [](TrainingStep* self,
           const std::string& name,
           std::vector<Tensor> loss_roots,
           std::shared_ptr<Optimizer> optimizer,
           std::vector<ParameterReference> update_parameters,
           uint32_t repeat_count,
           TrainingStep::GradientClearPolicy gradient_clear_policy,
           std::vector<TrainingInputBinding> input_bindings) {
            new (self) TrainingStep(name,
                                    std::move(loss_roots),
                                    std::move(optimizer),
                                    std::move(update_parameters),
                                    repeat_count,
                                    gradient_clear_policy,
                                    std::move(input_bindings));
        },
        "name"_a,
        "loss_roots"_a,
        "optimizer"_a.none() = nb::none(),
        "update_parameters"_a = std::vector<ParameterReference>{},
        "repeat_count"_a = 1,
        "gradient_clear_policy"_a = TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
        "input_bindings"_a = std::vector<TrainingInputBinding>{});
    training_step.def_prop_ro("name", &TrainingStep::getName);
    training_step.def_prop_ro("repeat_count", &TrainingStep::getRepeatCount);
    training_step.def_prop_ro("gradient_clear_policy", &TrainingStep::getGradientClearPolicy);
    training_step.def("is_initialized", &TrainingStep::isInitialized);
    training_step.def("get_loss_roots", &TrainingStep::getLossRoots, nb::rv_policy::reference_internal);
    training_step.def("get_optimizer", &TrainingStep::getOptimizer);
    training_step.def("get_update_parameters", &TrainingStep::getUpdateParameters, nb::rv_policy::reference_internal);
    training_step.def("get_input_bindings", &TrainingStep::getInputBindings, nb::rv_policy::reference_internal);
    training_step.def("updates_parameter", &TrainingStep::updatesParameter, "parameter"_a);
    training_step.def("get_architecture_json", &TrainingStep::architectureJsonString);
    training_step.def_static(
        "deserialize",
        [](const std::string& payload) {
            std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
            return TrainingStep::deserialize(nlohmann::json::parse(payload), archiveReader, nullptr);
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
    training_program.def(nb::init<>());
    training_program.def(nb::init<std::vector<TrainingStep>>(), "steps"_a);
    training_program.def("add_step", &TrainingProgram::addStep, "step"_a);
    training_program.def("get_num_steps", &TrainingProgram::getNumSteps);
    training_program.def("get_step", &TrainingProgram::getStep, "index"_a, nb::rv_policy::reference_internal);
    training_program.def("get_steps", &TrainingProgram::getSteps, nb::rv_policy::reference_internal);
    training_program.def("is_initialized", &TrainingProgram::isInitialized);
    training_program.def("get_architecture_json", &TrainingProgram::architectureJsonString);
    training_program.def_static(
        "deserialize",
        [](const std::string& payload) {
            std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
            return TrainingProgram::deserialize(nlohmann::json::parse(payload), archiveReader, nullptr);
        },
        "architecture_json"_a);
    training_program.def("compile", &TrainingProgram::compile, "placed_network"_a);
}
