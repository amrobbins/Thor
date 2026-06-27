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
#include <cstdint>
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
#include "DeepLearning/Api/Training/TrainingInputBinding.h"
#include "DeepLearning/Api/Training/TrainingPhase.h"
#include "DeepLearning/Api/Training/TrainingProgram.h"
#include "DeepLearning/Api/Training/TrainingRuns.h"
#include "DeepLearning/Api/Training/TrainingStep.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Loaders/NoOpDataProcessor.h"
#include "Utilities/Loaders/ShardedRawDatasetCreator.h"
#include "Utilities/Random/FullPeriodRandom.h"
#include "Utilities/WorkQueue/AsyncTensorQueue.h"
#include "bindings/python/src/core/cast.h"
#include "bindings/python/src/core/physical/NanobindDTypes.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;
namespace pybind = Thor::PythonBindings;

namespace {

using Float32Array = nb::ndarray<const float, nb::numpy, nb::c_contig>;
using Float16Array = nb::ndarray<const half, nb::numpy, nb::c_contig>;
using Int64Array = nb::ndarray<const int64_t, nb::numpy, nb::c_contig>;

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

std::optional<std::string> optionalPathStringFromPython(const nb::object& obj, const std::string& argumentName) {
    if (obj.is_none()) {
        return std::nullopt;
    }

    nb::object pathObject;
    try {
        pathObject = nb::module_::import_("os").attr("fspath")(obj);
    } catch (const nb::python_error&) {
        throw nb::type_error((argumentName + " must be str, bytes, os.PathLike, or None").c_str());
    }

    if (nb::isinstance<nb::bytes>(pathObject)) {
        pathObject = nb::module_::import_("os").attr("fsdecode")(pathObject);
    }
    std::string path = pybind::castOrTypeError<std::string>(pathObject, argumentName, "str", false);
    if (path.empty()) {
        throw nb::value_error((argumentName + " must not be empty").c_str());
    }
    return path;
}

std::string pathStringFromPython(const nb::object& obj, const std::string& argumentName) {
    std::optional<std::string> path = optionalPathStringFromPython(obj, argumentName);
    if (!path.has_value()) {
        throw nb::type_error((argumentName + " must be str, bytes, or os.PathLike").c_str());
    }
    return *path;
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

struct NamedFloat32NumpyTensor {
    std::vector<uint8_t> data;
    std::vector<uint64_t> shapeWithoutBatch;
    ThorImplementation::TensorDescriptor batchDescriptor;
    std::unique_ptr<AsyncTensorQueue> queue;
    uint64_t elementsPerExample = 0;
};

struct InMemoryNumpyDictSplit {
    std::map<std::string, NamedFloat32NumpyTensor> tensors;
    uint64_t numExamples = 0;
    uint64_t nextBatchNum = 0;
    std::unique_ptr<FullPeriodRandom> randomizer;
};

Float32Array asContiguousFloat32Array(nb::handle value, const std::string& context) {
    nb::object numpy = nb::module_::import_("numpy");
    nb::object arrayObject;
    try {
        arrayObject = numpy.attr("ascontiguousarray")(value, numpy.attr("float32"));
    } catch (const nb::python_error&) {
        throw nb::type_error((context + " must be convertible to a contiguous numpy.float32 array").c_str());
    }

    return pybind::castOrTypeError<Float32Array>(
        arrayObject, context, "a contiguous numpy.float32 array", false);
}

std::vector<uint64_t> float32ArrayShapeWithoutBatchDim(const Float32Array& array, const std::string& name) {
    if (array.ndim() < 1) {
        throw nb::value_error((name + " must have shape [N, ...]").c_str());
    }
    if (array.shape(0) == 0) {
        throw nb::value_error((name + " must contain at least one example").c_str());
    }

    std::vector<uint64_t> shape;
    if (array.ndim() == 1) {
        // NetworkInput requires a non-empty non-batch shape.  Treat a flat [N]
        // vector as [N, 1], which is the natural representation for scalar
        // tensors such as example weights.
        shape.push_back(1);
        return shape;
    }

    for (size_t i = 1; i < array.ndim(); ++i) {
        if (array.shape(i) == 0) {
            throw nb::value_error((name + " dimensions must all be positive").c_str());
        }
        shape.push_back(static_cast<uint64_t>(array.shape(i)));
    }
    return shape;
}

std::string tensorNameFromPythonKey(nb::handle key, const std::string& context) {
    std::string name = pybind::castOrTypeError<std::string>(key, context + " key", "str", false);
    if (name.empty()) {
        throw nb::value_error((context + " tensor names must be non-empty").c_str());
    }
    return name;
}

InMemoryNumpyDictSplit makeFloat32NumpyDictSplit(const nb::dict& tensors,
                                                 const std::string& splitName,
                                                 const std::string& loaderClassName) {
    if (nb::len(tensors) == 0) {
        throw nb::value_error((loaderClassName + " " + splitName + " dict must contain at least one tensor").c_str());
    }

    InMemoryNumpyDictSplit split;
    bool haveNumExamples = false;
    for (auto item : tensors) {
        const std::string name = tensorNameFromPythonKey(item.first, loaderClassName + " " + splitName);
        const std::string context = loaderClassName + " " + splitName + "['" + name + "']";
        Float32Array array = asContiguousFloat32Array(item.second, context);
        const uint64_t numExamples = static_cast<uint64_t>(array.shape(0));
        if (!haveNumExamples) {
            split.numExamples = numExamples;
            haveNumExamples = true;
        } else if (split.numExamples != numExamples) {
            throw nb::value_error((loaderClassName + " " + splitName + " tensors must all have the same leading dimension; tensor '" +
                                   name + "' has a different N")
                                      .c_str());
        }

        NamedFloat32NumpyTensor tensor;
        tensor.shapeWithoutBatch = float32ArrayShapeWithoutBatchDim(array, context);
        tensor.elementsPerExample = product(tensor.shapeWithoutBatch);
        const size_t bytes = array.size() * sizeof(float);
        tensor.data.resize(bytes);
        std::memcpy(tensor.data.data(), array.data(), bytes);

        auto [it, inserted] = split.tensors.emplace(name, std::move(tensor));
        (void)it;
        if (!inserted) {
            throw nb::value_error((loaderClassName + " " + splitName + " duplicate tensor name '" + name + "'").c_str());
        }
    }

    return split;
}

void validateFloat32NumpyDictSchemas(const InMemoryNumpyDictSplit& train,
                                     const InMemoryNumpyDictSplit& candidate,
                                     const std::string& candidateSplitName,
                                     const std::string& loaderClassName) {
    if (train.tensors.size() != candidate.tensors.size()) {
        throw nb::value_error((loaderClassName + " train and " + candidateSplitName + " dicts must have the same tensor names").c_str());
    }
    for (const auto& [name, trainTensor] : train.tensors) {
        auto candidateIt = candidate.tensors.find(name);
        if (candidateIt == candidate.tensors.end()) {
            throw nb::value_error((loaderClassName + " " + candidateSplitName + " dict is missing tensor '" + name + "'").c_str());
        }
        if (trainTensor.shapeWithoutBatch != candidateIt->second.shapeWithoutBatch) {
            throw nb::value_error(
                (loaderClassName + " train and " + candidateSplitName + " tensor '" + name + "' must have matching non-batch shapes")
                    .c_str());
        }
    }
}

nb::dict requireOptionalFloat32NumpyDictSplit(nb::object tensors, const std::string& splitName, const std::string& loaderClassName) {
    if (!nb::isinstance<nb::dict>(tensors)) {
        throw nb::type_error((loaderClassName + " " + splitName + " must be a dict or None").c_str());
    }
    return pybind::castOrTypeError<nb::dict>(tensors, loaderClassName + " " + splitName, "dict or None", false);
}

std::optional<uint64_t> optionalUint64FromPython(nb::object value, const std::string& name) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return pybind::castOrTypeError<uint64_t>(value, name, "int or None", false);
}

struct SharedFloat32NumpyTensor {
    std::optional<Float32Array> array;
    const float* data = nullptr;
    std::vector<uint64_t> shapeWithoutBatch;
    ThorImplementation::TensorDescriptor batchDescriptor;
    uint64_t elementsPerExample = 0;
};

struct IndexedNumpyDictSplit {
    std::vector<uint64_t> indices;
    uint64_t numExamples = 0;
    uint64_t nextBatchNum = 0;
    std::unique_ptr<FullPeriodRandom> randomizer;
    std::map<std::string, std::unique_ptr<AsyncTensorQueue>> queues;
};

Float32Array requireFloat32NumpyArrayNoCopy(nb::handle value, const std::string& context) {
    return pybind::castOrTypeError<Float32Array>(
        value, context, "a C-contiguous numpy.float32 array", false);
}

void markNumpyArrayReadOnly(nb::handle value, const std::string& context) {
    try {
        nb::object arrayObject = nb::borrow<nb::object>(value);
        arrayObject.attr("setflags")("write"_a = false);
    } catch (const nb::python_error& e) {
        throw nb::value_error((context + " could not be marked read-only").c_str());
    }
}

std::map<std::string, SharedFloat32NumpyTensor> makeSharedFloat32NumpyDictTensors(const nb::dict& tensors,
                                                                                 const std::string& loaderClassName) {
    if (nb::len(tensors) == 0) {
        throw nb::value_error((loaderClassName + " tensors dict must contain at least one tensor").c_str());
    }

    std::map<std::string, SharedFloat32NumpyTensor> sharedTensors;
    bool haveNumExamples = false;
    uint64_t expectedNumExamples = 0;
    for (auto item : tensors) {
        const std::string name = tensorNameFromPythonKey(item.first, loaderClassName + " tensors");
        const std::string context = loaderClassName + " tensors['" + name + "']";
        Float32Array array = requireFloat32NumpyArrayNoCopy(item.second, context);
        markNumpyArrayReadOnly(item.second, context);
        const uint64_t numExamples = static_cast<uint64_t>(array.shape(0));
        if (!haveNumExamples) {
            expectedNumExamples = numExamples;
            haveNumExamples = true;
        } else if (expectedNumExamples != numExamples) {
            throw nb::value_error((loaderClassName + " tensors must all have the same leading dimension; tensor '" + name +
                                   "' has a different N")
                                      .c_str());
        }

        SharedFloat32NumpyTensor tensor;
        tensor.shapeWithoutBatch = float32ArrayShapeWithoutBatchDim(array, context);
        tensor.elementsPerExample = product(tensor.shapeWithoutBatch);
        tensor.data = array.data();
        tensor.array.emplace(std::move(array));

        auto [it, inserted] = sharedTensors.emplace(name, std::move(tensor));
        (void)it;
        if (!inserted) {
            throw nb::value_error((loaderClassName + " tensors duplicate tensor name '" + name + "'").c_str());
        }
    }

    return sharedTensors;
}

std::vector<uint64_t> uint64IndicesFromPython(nb::object indices,
                                              const std::string& context,
                                              uint64_t maxExclusive) {
    nb::object numpy = nb::module_::import_("numpy");
    nb::object sourceObject;
    nb::object arrayObject;
    try {
        sourceObject = numpy.attr("asarray")(indices);
        const bool isInteger = pybind::castOrTypeError<bool>(
            numpy.attr("issubdtype")(sourceObject.attr("dtype"), numpy.attr("integer")),
            context + " dtype check",
            "bool",
            false);
        if (!isInteger) {
            throw nb::type_error((context + " must be an integer index array").c_str());
        }
        arrayObject = numpy.attr("ascontiguousarray")(sourceObject, numpy.attr("int64"));
    } catch (const nb::python_error&) {
        throw nb::type_error((context + " must be a one-dimensional integer index array").c_str());
    }

    Int64Array array = pybind::castOrTypeError<Int64Array>(
        arrayObject, context, "a one-dimensional integer index array", false);
    if (array.ndim() != 1) {
        throw nb::value_error((context + " must be one-dimensional").c_str());
    }
    if (array.shape(0) == 0) {
        throw nb::value_error((context + " must contain at least one row index").c_str());
    }

    std::vector<uint64_t> out;
    out.reserve(static_cast<size_t>(array.shape(0)));
    for (size_t i = 0; i < array.shape(0); ++i) {
        const int64_t raw = array.data()[i];
        if (raw < 0) {
            throw nb::value_error((context + " contains a negative row index").c_str());
        }
        const uint64_t index = static_cast<uint64_t>(raw);
        if (index >= maxExclusive) {
            throw nb::value_error((context + " contains a row index outside the tensor leading dimension").c_str());
        }
        out.push_back(index);
    }
    return out;
}

IndexedNumpyDictSplit makeIndexedNumpyDictSplit(nb::object indices,
                                                const std::string& splitName,
                                                const std::string& loaderClassName,
                                                uint64_t maxExclusive) {
    IndexedNumpyDictSplit split;
    split.indices = uint64IndicesFromPython(std::move(indices), loaderClassName + " " + splitName + "_indices", maxExclusive);
    split.numExamples = static_cast<uint64_t>(split.indices.size());
    return split;
}

class IndexedNumpyFloat32DictBatchLoader : public Loader {
   public:
    IndexedNumpyFloat32DictBatchLoader(nb::dict tensors,
                                       nb::object trainIndices,
                                       nb::object validateIndices,
                                       nb::object testIndices,
                                       uint64_t batchSize,
                                       std::string loaderClassName,
                                       bool randomizeTrain,
                                       uint64_t batchQueueDepth,
                                       std::optional<uint64_t> randomSeed)
        : loaderClassName(std::move(loaderClassName)),
          randomizeTrain(randomizeTrain),
          batchQueueDepth(batchQueueDepth),
          randomSeed(randomSeed),
          explicitTestSplit(!testIndices.is_none()) {
        if (batchSize == 0) {
            throw nb::value_error((this->loaderClassName + " batch_size must be >= 1").c_str());
        }
        if (batchQueueDepth == 0) {
            throw nb::value_error((this->loaderClassName + " batch_queue_depth must be >= 1").c_str());
        }
        this->batchSize = batchSize;

        sharedTensors = makeSharedFloat32NumpyDictTensors(tensors, this->loaderClassName);
        const uint64_t maxExclusive = static_cast<uint64_t>(sharedTensors.begin()->second.array.value().shape(0));

        train = makeIndexedNumpyDictSplit(std::move(trainIndices), "train", this->loaderClassName, maxExclusive);
        validate = makeIndexedNumpyDictSplit(std::move(validateIndices), "validate", this->loaderClassName, maxExclusive);
        if (explicitTestSplit) {
            test = makeIndexedNumpyDictSplit(std::move(testIndices), "test", this->loaderClassName, maxExclusive);
        } else {
            test.indices = validate.indices;
            test.numExamples = validate.numExamples;
        }

        if (this->randomizeTrain) {
            train.randomizer = std::make_unique<FullPeriodRandom>(train.numExamples, false);
            if (this->randomSeed.has_value()) {
                train.randomizer->reseed(this->randomSeed.value());
            }
        } else if (this->randomSeed.has_value()) {
            throw nb::value_error((this->loaderClassName + " random_seed requires randomize_train=True").c_str());
        }

        initializeSplitQueues(train);
        initializeSplitQueues(validate);
        initializeSplitQueues(test);
    }

    ~IndexedNumpyFloat32DictBatchLoader() override {
        closeSplitQueues(train);
        closeSplitQueues(validate);
        closeSplitQueues(test);
    }

    Batch getBatch(ExampleType exampleType, uint64_t& batchNum) override {
        IndexedNumpyDictSplit& split = mutableSplit(exampleType);
        const uint64_t batchesPerEpoch = getNumBatchesPerEpoch(exampleType);
        if (batchNum >= batchesPerEpoch) {
            batchNum = split.nextBatchNum;
        }

        std::map<std::string, ThorImplementation::Tensor> loadedTensors;
        for (auto& [name, queue] : split.queues) {
            ThorImplementation::Tensor tensor;
            bool queueOpen = queue->getBufferToLoad(tensor);
            THOR_THROW_IF_FALSE(queueOpen);
            loadedTensors.emplace(name, tensor);
        }

        const uint64_t firstExample = batchNum * batchSize;
        const bool useRandomizer = exampleType == ExampleType::TRAIN && randomizeTrain;
        for (uint64_t i = 0; i < batchSize; ++i) {
            const uint64_t logicalIndex = useRandomizer ? split.randomizer->getRandomNumber() : (firstExample + i) % split.numExamples;
            const uint64_t exampleIndex = split.indices[logicalIndex];
            for (const auto& [name, spec] : sharedTensors) {
                ThorImplementation::Tensor& tensor = loadedTensors.at(name);
                float* dest = tensor.getMemPtr<float>();
                std::memcpy(dest + (i * spec.elementsPerExample),
                            spec.data + (exampleIndex * spec.elementsPerExample),
                            spec.elementsPerExample * sizeof(float));
            }
        }

        split.nextBatchNum = (batchNum + 1) % batchesPerEpoch;

        for (auto& [name, queue] : split.queues) {
            ThorImplementation::Tensor& tensor = loadedTensors.at(name);
            bool queueOpen = queue->bufferLoaded(tensor);
            THOR_THROW_IF_FALSE(queueOpen);
        }
        for (auto& [name, queue] : split.queues) {
            ThorImplementation::Tensor& tensor = loadedTensors.at(name);
            bool queueOpen = queue->getBufferToUnload(tensor);
            THOR_THROW_IF_FALSE(queueOpen);
        }

        Batch batch;
        for (auto& [name, tensor] : loadedTensors) {
            batch.insert(name, tensor);
        }
        return batch;
    }

    void returnBatchBuffers(ExampleType exampleType, Batch&& batch) override {
        IndexedNumpyDictSplit& split = mutableSplit(exampleType);
        if (batch.size() != sharedTensors.size()) {
            throw std::runtime_error(loaderClassName + " returned batch has unexpected tensor count");
        }
        for (auto& [name, queue] : split.queues) {
            if (!batch.contains(name)) {
                throw std::runtime_error(loaderClassName + " returned batch is missing tensor '" + name + "'");
            }
            bool queueOpen = queue->bufferUnloaded(batch.getTensor(name));
            THOR_THROW_IF_FALSE(queueOpen);
        }
    }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        const IndexedNumpyDictSplit& split = immutableSplit(exampleType);
        return (split.numExamples + batchSize - 1) / batchSize;
    }

    uint64_t getNumExamples(ExampleType exampleType) override { return immutableSplit(exampleType).numExamples; }

    uint64_t getNextBatchNum(ExampleType exampleType) override { return immutableSplit(exampleType).nextBatchNum; }

    std::vector<std::string> getTensorNames() const {
        std::vector<std::string> names;
        names.reserve(sharedTensors.size());
        for (const auto& [name, spec] : sharedTensors) {
            (void)spec;
            names.push_back(name);
        }
        return names;
    }

    std::map<std::string, std::vector<uint64_t>> getTensorShapes() const {
        std::map<std::string, std::vector<uint64_t>> shapes;
        for (const auto& [name, spec] : sharedTensors) {
            shapes.emplace(name, spec.shapeWithoutBatch);
        }
        return shapes;
    }

    uint64_t getBatchQueueDepth() const { return batchQueueDepth; }

    bool getRandomizeTrain() const { return randomizeTrain; }

    nb::object getRandomSeed() const {
        if (!randomSeed.has_value()) {
            return nb::none();
        }
        return nb::int_(randomSeed.value());
    }

    bool hasExplicitTestSplit() const { return explicitTestSplit; }

   private:
    void initializeSplitQueues(IndexedNumpyDictSplit& split) {
        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        for (auto& [name, spec] : sharedTensors) {
            std::vector<uint64_t> batchShape{batchSize};
            batchShape.insert(batchShape.end(), spec.shapeWithoutBatch.begin(), spec.shapeWithoutBatch.end());
            spec.batchDescriptor = ThorImplementation::TensorDescriptor(ThorImplementation::DataType::FP32, batchShape);
            auto queue = std::make_unique<AsyncTensorQueue>(batchQueueDepth, spec.batchDescriptor, cpuPlacement);
            queue->open();
            split.queues.emplace(name, std::move(queue));
        }
    }

    void closeSplitQueues(IndexedNumpyDictSplit& split) {
        for (auto& [name, queue] : split.queues) {
            (void)name;
            if (queue != nullptr) {
                queue->close();
            }
        }
    }

    IndexedNumpyDictSplit& mutableSplit(ExampleType exampleType) {
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

    const IndexedNumpyDictSplit& immutableSplit(ExampleType exampleType) const {
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

    std::string loaderClassName;
    bool randomizeTrain = true;
    uint64_t batchQueueDepth = 32;
    std::optional<uint64_t> randomSeed;
    bool explicitTestSplit = false;
    std::map<std::string, SharedFloat32NumpyTensor> sharedTensors;
    IndexedNumpyDictSplit train;
    IndexedNumpyDictSplit validate;
    IndexedNumpyDictSplit test;
};

class NumpyFloat32DictBatchLoader : public Loader {
   public:
    NumpyFloat32DictBatchLoader(nb::dict trainTensors,
                                nb::dict validateTensors,
                                nb::object testTensors,
                                uint64_t batchSize,
                                std::string loaderClassName,
                                bool randomizeTrain,
                                uint64_t batchQueueDepth,
                                std::optional<uint64_t> randomSeed)
        : loaderClassName(std::move(loaderClassName)),
          randomizeTrain(randomizeTrain),
          batchQueueDepth(batchQueueDepth),
          randomSeed(randomSeed),
          explicitTestSplit(!testTensors.is_none()) {
        if (batchSize == 0) {
            throw nb::value_error((this->loaderClassName + " batch_size must be >= 1").c_str());
        }
        if (batchQueueDepth == 0) {
            throw nb::value_error((this->loaderClassName + " batch_queue_depth must be >= 1").c_str());
        }
        this->batchSize = batchSize;

        train = makeFloat32NumpyDictSplit(trainTensors, "train", this->loaderClassName);
        validate = makeFloat32NumpyDictSplit(validateTensors, "validate", this->loaderClassName);
        if (explicitTestSplit) {
            test = makeFloat32NumpyDictSplit(
                requireOptionalFloat32NumpyDictSplit(testTensors, "test", this->loaderClassName), "test", this->loaderClassName);
        } else {
            // Backward compatible default: if no explicit holdout test split is supplied,
            // TEST remains backed by a distinct queue over the validation arrays.  This
            // preserves independent VALIDATE and TEST nextBatchNum/queue state while making
            // real train/validate/test demand-forecasting manifests possible.
            test = makeFloat32NumpyDictSplit(validateTensors, "test", this->loaderClassName);
        }
        validateFloat32NumpyDictSchemas(train, validate, "validate", this->loaderClassName);
        validateFloat32NumpyDictSchemas(train, test, "test", this->loaderClassName);

        if (this->randomizeTrain) {
            train.randomizer = std::make_unique<FullPeriodRandom>(train.numExamples, false);
            if (this->randomSeed.has_value()) {
                train.randomizer->reseed(this->randomSeed.value());
            }
        } else if (this->randomSeed.has_value()) {
            throw nb::value_error((this->loaderClassName + " random_seed requires randomize_train=True").c_str());
        }

        initializeSplitQueues(train);
        initializeSplitQueues(validate);
        initializeSplitQueues(test);
    }

    ~NumpyFloat32DictBatchLoader() override {
        closeSplitQueues(train);
        closeSplitQueues(validate);
        closeSplitQueues(test);
    }

    Batch getBatch(ExampleType exampleType, uint64_t& batchNum) override {
        InMemoryNumpyDictSplit& split = mutableSplit(exampleType);
        const uint64_t batchesPerEpoch = getNumBatchesPerEpoch(exampleType);
        if (batchNum >= batchesPerEpoch) {
            batchNum = split.nextBatchNum;
        }

        std::map<std::string, ThorImplementation::Tensor> loadedTensors;
        for (auto& [name, spec] : split.tensors) {
            ThorImplementation::Tensor tensor;
            bool queueOpen = spec.queue->getBufferToLoad(tensor);
            THOR_THROW_IF_FALSE(queueOpen);
            loadedTensors.emplace(name, tensor);
        }

        const uint64_t firstExample = batchNum * batchSize;
        const bool useRandomizer = exampleType == ExampleType::TRAIN && randomizeTrain;
        for (uint64_t i = 0; i < batchSize; ++i) {
            const uint64_t exampleIndex = useRandomizer ? split.randomizer->getRandomNumber() : (firstExample + i) % split.numExamples;
            for (const auto& [name, spec] : split.tensors) {
                ThorImplementation::Tensor& tensor = loadedTensors.at(name);
                float* dest = tensor.getMemPtr<float>();
                const float* src = reinterpret_cast<const float*>(spec.data.data());
                std::memcpy(dest + (i * spec.elementsPerExample),
                            src + (exampleIndex * spec.elementsPerExample),
                            spec.elementsPerExample * sizeof(float));
            }
        }

        split.nextBatchNum = (batchNum + 1) % batchesPerEpoch;

        for (auto& [name, spec] : split.tensors) {
            ThorImplementation::Tensor& tensor = loadedTensors.at(name);
            bool queueOpen = spec.queue->bufferLoaded(tensor);
            THOR_THROW_IF_FALSE(queueOpen);
        }
        for (auto& [name, spec] : split.tensors) {
            ThorImplementation::Tensor& tensor = loadedTensors.at(name);
            bool queueOpen = spec.queue->getBufferToUnload(tensor);
            THOR_THROW_IF_FALSE(queueOpen);
        }

        Batch batch;
        for (auto& [name, tensor] : loadedTensors) {
            batch.insert(name, tensor);
        }
        return batch;
    }

    void returnBatchBuffers(ExampleType exampleType, Batch&& batch) override {
        InMemoryNumpyDictSplit& split = mutableSplit(exampleType);
        if (batch.size() != split.tensors.size()) {
            throw std::runtime_error(loaderClassName + " returned batch has unexpected tensor count");
        }
        for (auto& [name, spec] : split.tensors) {
            if (!batch.contains(name)) {
                throw std::runtime_error(loaderClassName + " returned batch is missing tensor '" + name + "'");
            }
            bool queueOpen = spec.queue->bufferUnloaded(batch.getTensor(name));
            THOR_THROW_IF_FALSE(queueOpen);
        }
    }

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override {
        const InMemoryNumpyDictSplit& split = immutableSplit(exampleType);
        return (split.numExamples + batchSize - 1) / batchSize;
    }

    uint64_t getNumExamples(ExampleType exampleType) override { return immutableSplit(exampleType).numExamples; }

    uint64_t getNextBatchNum(ExampleType exampleType) override { return immutableSplit(exampleType).nextBatchNum; }

    std::vector<std::string> getTensorNames() const {
        std::vector<std::string> names;
        names.reserve(train.tensors.size());
        for (const auto& [name, spec] : train.tensors) {
            (void)spec;
            names.push_back(name);
        }
        return names;
    }

    std::map<std::string, std::vector<uint64_t>> getTensorShapes() const {
        std::map<std::string, std::vector<uint64_t>> shapes;
        for (const auto& [name, spec] : train.tensors) {
            shapes.emplace(name, spec.shapeWithoutBatch);
        }
        return shapes;
    }

    uint64_t getBatchQueueDepth() const { return batchQueueDepth; }

    bool getRandomizeTrain() const { return randomizeTrain; }

    nb::object getRandomSeed() const {
        if (!randomSeed.has_value()) {
            return nb::none();
        }
        return nb::int_(randomSeed.value());
    }

    bool hasExplicitTestSplit() const { return explicitTestSplit; }

   private:
    void initializeSplitQueues(InMemoryNumpyDictSplit& split) {
        ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
        for (auto& [name, spec] : split.tensors) {
            std::vector<uint64_t> batchShape{batchSize};
            batchShape.insert(batchShape.end(), spec.shapeWithoutBatch.begin(), spec.shapeWithoutBatch.end());
            spec.batchDescriptor = ThorImplementation::TensorDescriptor(ThorImplementation::DataType::FP32, batchShape);
            spec.queue = std::make_unique<AsyncTensorQueue>(batchQueueDepth, spec.batchDescriptor, cpuPlacement);
            spec.queue->open();
        }
    }

    void closeSplitQueues(InMemoryNumpyDictSplit& split) {
        for (auto& [name, spec] : split.tensors) {
            (void)name;
            if (spec.queue != nullptr) {
                spec.queue->close();
            }
        }
    }

    InMemoryNumpyDictSplit& mutableSplit(ExampleType exampleType) {
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

    const InMemoryNumpyDictSplit& immutableSplit(ExampleType exampleType) const {
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

    std::string loaderClassName;
    bool randomizeTrain = true;
    uint64_t batchQueueDepth = 32;
    std::optional<uint64_t> randomSeed;
    bool explicitTestSplit = false;
    InMemoryNumpyDictSplit train;
    InMemoryNumpyDictSplit validate;
    InMemoryNumpyDictSplit test;
};

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

std::map<std::string, size_t> trainingRunsMinSuccessfulModelsFromPython(nb::object minSuccessfulModels) {
    if (minSuccessfulModels.is_none()) {
        return {};
    }
    if (!nb::isinstance<nb::dict>(minSuccessfulModels)) {
        throw nb::type_error(
            "TrainingRuns min_successful_models must be a dict mapping ensemble_group names to positive integers, or None.");
    }

    nb::dict mapping = pybind::castOrTypeError<nb::dict>(
        minSuccessfulModels, "TrainingRuns min_successful_models", "dict mapping ensemble_group names to positive integers or None", false);
    std::map<std::string, size_t> result;
    for (auto item : mapping) {
        if (!nb::isinstance<nb::str>(item.first)) {
            throw nb::type_error("TrainingRuns min_successful_models keys must be ensemble_group strings.");
        }
        const std::string groupName = pybind::castOrTypeError<std::string>(
            item.first, "TrainingRuns min_successful_models key", "str", false);
        int64_t minimum = pybind::castOrTypeError<int64_t>(
            item.second, "TrainingRuns min_successful_models[\"" + groupName + "\"]", "positive int", false);
        if (minimum <= 0) {
            throw nb::value_error("TrainingRuns min_successful_models values must be >= 1.");
        }
        result.emplace(groupName, static_cast<size_t>(minimum));
    }
    return result;
}

std::vector<std::string> trainingRunsReportNameListFromPython(nb::handle value,
                                                                   const std::string& context,
                                                                   const std::string& itemName = "report") {
    if (value.is_none()) {
        return {};
    }
    PyObject* iterator = nb::isinstance<nb::str>(value) ? nullptr : PyObject_GetIter(value.ptr());
    if (iterator == nullptr) {
        PyErr_Clear();
        throw nb::type_error((context + " must be an iterable of " + itemName + "-name strings or None.").c_str());
    }
    Py_DECREF(iterator);

    std::vector<std::string> names;
    size_t index = 0;
    for (nb::handle nameObj : pybind::castOrTypeError<nb::iterable>(
             value, context, "iterable of " + itemName + "-name strings or None", false)) {
        if (!nb::isinstance<nb::str>(nameObj)) {
            throw nb::type_error((context + "[" + std::to_string(index) + "] must be a string.").c_str());
        }
        names.push_back(pybind::castOrTypeError<std::string>(
            nameObj, context + "[" + std::to_string(index) + "]", "str", false));
        ++index;
    }
    return names;
}

std::map<std::string, std::vector<std::string>> trainingRunsReportsFromPython(nb::object reports,
                                                                              const std::vector<TrainingRunsSpec>& specs) {
    if (reports.is_none()) {
        return {};
    }

    auto defaultTargets = [&specs]() {
        std::set<std::string> ensembleGroups;
        std::vector<std::string> runNames;
        runNames.reserve(specs.size());
        for (const TrainingRunsSpec& spec : specs) {
            runNames.push_back(spec.runName);
            if (spec.ensembleGroup.has_value()) {
                ensembleGroups.insert(*spec.ensembleGroup);
            }
        }
        std::vector<std::string> targets;
        if (!ensembleGroups.empty()) {
            targets.assign(ensembleGroups.begin(), ensembleGroups.end());
        } else {
            targets = std::move(runNames);
        }
        return targets;
    };

    std::map<std::string, std::vector<std::string>> result;
    if (nb::isinstance<nb::dict>(reports)) {
        nb::dict mapping = pybind::castOrTypeError<nb::dict>(reports, "TrainingRuns reports", "dict or iterable of report-name strings or None", false);
        for (auto item : mapping) {
            if (!nb::isinstance<nb::str>(item.first)) {
                throw nb::type_error("TrainingRuns reports keys must be run_name or ensemble_group strings.");
            }
            const std::string targetName = pybind::castOrTypeError<std::string>(item.first, "TrainingRuns reports key", "str", false);
            result.emplace(targetName,
                           trainingRunsReportNameListFromPython(
                               item.second,
                               "TrainingRuns reports['" + targetName + "']"));
        }
        return result;
    }

    std::vector<std::string> reportNames = trainingRunsReportNameListFromPython(reports, "TrainingRuns reports");
    for (const std::string& targetName : defaultTargets()) {
        result.emplace(targetName, reportNames);
    }
    return result;
}

std::vector<TrainingRunsSpec> trainingRunsSpecsFromPython(nb::iterable runs) {
    std::vector<TrainingRunsSpec> specs;
    for (nb::handle item : runs) {
        nb::sequence entry = pybind::castOrTypeError<nb::sequence>(
            item, "TrainingRuns runs entry", "sequence (run_name, trainer[, ensemble_group[, ensemble_weight]])", false);
        const size_t entrySize = nb::len(entry);
        if (entrySize < 2 || entrySize > 4) {
            throw nb::value_error(
                "TrainingRuns entries must be (run_name, trainer), (run_name, trainer, ensemble_group), or "
                "(run_name, trainer, ensemble_group, ensemble_weight)");
        }

        std::string runName = pybind::castOrTypeError<std::string>(entry[0], "TrainingRuns runs entry[0]", "str run_name", false);
        std::shared_ptr<Trainer> trainer = pybind::castOrTypeError<std::shared_ptr<Trainer>>(
            entry[1], "TrainingRuns runs entry[1]", "thor.training.Trainer", false);
        TrainingRunsSpec spec(std::move(runName), std::move(trainer));
        if (entrySize >= 3 && !entry[2].is_none()) {
            spec.ensembleGroup = pybind::castOrTypeError<std::string>(
                entry[2], "TrainingRuns runs entry[2]", "str ensemble_group or None", false);
        }
        if (entrySize >= 4 && !entry[3].is_none()) {
            spec.ensembleWeight = pybind::castOrTypeError<double>(
                entry[3], "TrainingRuns runs entry[3]", "float ensemble_weight or None", false);
        }
        specs.push_back(std::move(spec));
    }
    return specs;
}

void warnPythonRuntimeWarning(const std::string& message) {
    if (PyErr_WarnEx(PyExc_RuntimeWarning, message.c_str(), 1) < 0) {
        throw nb::python_error();
    }
}

std::vector<TrainingRestartPolicy> trainingRestartPoliciesFromPython(nb::object restartConditions, bool trainerScope) {
    if (restartConditions.is_none()) {
        return {};
    }

    PyObject* iterator = PyObject_GetIter(restartConditions.ptr());
    if (iterator == nullptr) {
        PyErr_Clear();
        throw nb::type_error("restart_conditions must be an iterable of RestartPolicy objects.");
    }
    Py_DECREF(iterator);

    std::vector<TrainingRestartPolicy> policies;
    size_t conditionIndex = 0;
    for (nb::handle conditionObj : pybind::castOrTypeError<nb::iterable>(
             restartConditions, "restart_conditions", "iterable of RestartPolicy objects or None", false)) {
        if (!nb::isinstance<TrainingRestartPolicy>(conditionObj)) {
            throw nb::type_error(("restart_conditions[" + std::to_string(conditionIndex) + "] must be a RestartPolicy object.").c_str());
        }
        TrainingRestartPolicy policy = pybind::castOrTypeError<TrainingRestartPolicy>(
            conditionObj, "restart_conditions[" + std::to_string(conditionIndex) + "]", "thor.training.RestartPolicy", false);
        if (trainerScope && (policy.runName.has_value() || policy.ensembleGroup.has_value())) {
            std::string ignoredFields;
            if (policy.runName.has_value()) {
                ignoredFields += "run_name";
            }
            if (policy.ensembleGroup.has_value()) {
                if (!ignoredFields.empty()) {
                    ignoredFields += " and ";
                }
                ignoredFields += "ensemble_group";
            }
            warnPythonRuntimeWarning("Trainer restart_conditions ignore RestartPolicy " + ignoredFields +
                                     "; targeting is only meaningful when the policy is passed to TrainingRuns.");
            policy = policy.withoutTarget();
        }
        policies.push_back(std::move(policy));
        ++conditionIndex;
    }
    return policies;
}

nb::object optionalDouble(std::optional<double> value);
nb::object optionalString(std::optional<std::string> value);

class GilSafePythonObject {
   public:
    explicit GilSafePythonObject(nb::handle object) : object(object.ptr()) {
        nb::gil_scoped_acquire gil;
        Py_XINCREF(this->object);
    }

    GilSafePythonObject(const GilSafePythonObject&) = delete;
    GilSafePythonObject& operator=(const GilSafePythonObject&) = delete;

    ~GilSafePythonObject() {
        if (object == nullptr) {
            return;
        }
        nb::gil_scoped_acquire gil;
        Py_XDECREF(object);
    }

    nb::handle get() const { return nb::handle(object); }

   private:
    PyObject* object = nullptr;
};

struct BoundPythonEarlyCompletionPolicy {
    TrainingEarlyCompletionPolicy policy;
    nb::object callbackHolder;
};

nb::object makeWeakrefableCallbackHolder(nb::object callback) {
    nb::object builtins = nb::module_::import_("builtins");
    nb::dict classDict;
    classDict["__slots__"] = nb::make_tuple("callback", "__weakref__");
    nb::object holderClass = builtins.attr("type")("_ThorCallbackHolder", nb::make_tuple(builtins.attr("object")), classDict);
    nb::object holder = holderClass();
    holder.attr("callback") = std::move(callback);
    return holder;
}

BoundPythonEarlyCompletionPolicy trainingEarlyCompletionPolicyFromWeakCallable(nb::object completionCondition) {
    if (completionCondition.is_none() || !PyCallable_Check(completionCondition.ptr())) {
        throw nb::type_error("completion_condition must be callable");
    }

    nb::object holder = makeWeakrefableCallbackHolder(std::move(completionCondition));
    nb::object weakref = nb::module_::import_("weakref").attr("ref")(holder);
    auto weakrefObject = std::make_shared<GilSafePythonObject>(weakref);

    TrainingEarlyCompletionPolicy policy{
        [weakrefObject = std::move(weakrefObject)](double currentScore, double bestScore, uint64_t currentEpoch, uint64_t bestEpoch) {
            nb::gil_scoped_acquire acquire;
            nb::object holder = nb::borrow<nb::object>(weakrefObject->get())();
            if (holder.is_none()) {
                throw std::runtime_error(
                    "Python early-completion callback is no longer alive. This is an internal binding lifetime error: "
                    "the owning Trainer/TrainingRuns object should retain the callback holder while the C++ callback is active.");
            }
            nb::object callableObject = holder.attr("callback");
            nb::object result = callableObject(currentScore, bestScore, currentEpoch, bestEpoch);
            return pybind::castOrTypeError<bool>(result, "completion_condition return value", "bool", false);
        }};

    return BoundPythonEarlyCompletionPolicy{std::move(policy), std::move(holder)};
}

std::vector<nb::object> callbackRefsFromObject(nb::handle object) {
    std::vector<nb::object> refs;
    if (PyObject_HasAttrString(object.ptr(), "_thor_callback_refs") == 0) {
        return refs;
    }
    nb::object refsObject = nb::borrow<nb::object>(object).attr("_thor_callback_refs");
    if (refsObject.is_none()) {
        return refs;
    }
    for (nb::handle ref : pybind::castOrTypeError<nb::iterable>(
             refsObject, "_thor_callback_refs", "iterable callback-reference list", false)) {
        refs.emplace_back(nb::borrow<nb::object>(ref));
    }
    return refs;
}

void attachCallbackRefs(nb::object owner, const std::vector<nb::object>& refs) {
    if (refs.empty()) {
        return;
    }
    nb::list refsList;
    for (const nb::object& ref : refs) {
        refsList.append(ref);
    }
    owner.attr("_thor_callback_refs") = std::move(refsList);
}

TrainingEarlyCompletionPolicy trainingEarlyCompletionPolicyFromCallable(nb::object completionCondition) {
    if (completionCondition.is_none() || !PyCallable_Check(completionCondition.ptr())) {
        throw nb::type_error("completion_condition must be callable");
    }

    PyObject* callable = completionCondition.ptr();
    Py_INCREF(callable);
    auto callback = std::shared_ptr<PyObject>(callable, [](PyObject* object) {
        // The last policy copy can be destroyed from a native TrainingRuns worker thread
        // after fit() has released the GIL, so Python refcount updates must acquire it.
        nb::gil_scoped_acquire acquire;
        Py_DECREF(object);
    });

    return TrainingEarlyCompletionPolicy{
        [callback = std::move(callback)](double currentScore, double bestScore, uint64_t currentEpoch, uint64_t bestEpoch) {
            // Trainer::fit() / TrainingRuns::fit() release the GIL while native training runs.
            // TrainingRuns may then evaluate this callback on a native worker thread, so reacquire
            // the GIL for every Python completion-condition invocation.
            nb::gil_scoped_acquire acquire;
            nb::object callableObject = nb::borrow<nb::object>(nb::handle(callback.get()));
            nb::object result = callableObject(currentScore, bestScore, currentEpoch, bestEpoch);
            return pybind::castOrTypeError<bool>(result, "completion_condition return value", "bool", false);
        }};
}

TrainingModelSelectionScore trainingModelSelectionScoreFromPython(nb::object modelSelectionScore) {
    if (modelSelectionScore.is_none()) {
        return TrainingModelSelectionScore{};
    }
    if (!PyCallable_Check(modelSelectionScore.ptr())) {
        throw nb::type_error("model_selection_score must be callable");
    }

    PyObject* callable = modelSelectionScore.ptr();
    Py_INCREF(callable);
    auto callback = std::shared_ptr<PyObject>(callable, [](PyObject* object) {
        nb::gil_scoped_acquire acquire;
        Py_DECREF(object);
    });

    return TrainingModelSelectionScore{[callback = std::move(callback)](std::optional<double> validationLoss,
                                                                        std::optional<double> trainingLoss,
                                                                        uint64_t epoch) -> std::optional<double> {
        // Trainer::fit() / TrainingRuns::fit() release the GIL while native training runs.
        // The model-selection callback can also run from a native TrainingRuns worker thread.
        nb::gil_scoped_acquire acquire;
        nb::object callableObject = nb::borrow<nb::object>(nb::handle(callback.get()));
        nb::object result = callableObject(optionalDouble(validationLoss), optionalDouble(trainingLoss), epoch);
        if (result.is_none()) {
            return std::nullopt;
        }
        return pybind::castOrTypeError<double>(result, "model_selection_score return value", "float or None", false);
    }};
}

struct TrainingEarlyCompletionPoliciesBinding {
    std::vector<TrainingEarlyCompletionPolicy> policies;
    std::vector<nb::object> callbackRefs;
};

TrainingEarlyCompletionPoliciesBinding trainingEarlyCompletionPoliciesFromPython(nb::object earlyCompletionPolicies) {
    TrainingEarlyCompletionPoliciesBinding out;
    if (earlyCompletionPolicies.is_none()) {
        return out;
    }

    size_t policyIndex = 0;
    for (nb::handle policyObject : pybind::castOrTypeError<nb::iterable>(
             earlyCompletionPolicies, "early_completion_policies", "iterable of EarlyCompletionPolicy objects or None", false)) {
        if (!nb::isinstance<TrainingEarlyCompletionPolicy>(policyObject)) {
            throw nb::type_error(
                ("Trainer early_completion_policies[" + std::to_string(policyIndex) + "] must be an EarlyCompletionPolicy object.")
                    .c_str());
        }
        out.policies.push_back(pybind::castOrTypeError<TrainingEarlyCompletionPolicy>(
            policyObject, "early_completion_policies[" + std::to_string(policyIndex) + "]", "thor.training.EarlyCompletionPolicy", false));
        std::vector<nb::object> refs = callbackRefsFromObject(policyObject);
        out.callbackRefs.insert(out.callbackRefs.end(), refs.begin(), refs.end());
        ++policyIndex;
    }
    return out;
}

struct TrainingRunsEarlyCompletionRulesBinding {
    std::vector<TrainingRunsEarlyCompletionRule> rules;
    std::vector<nb::object> callbackRefs;
};

TrainingRunsEarlyCompletionRulesBinding trainingRunsEarlyCompletionRulesFromPython(nb::object earlyCompletionRules) {
    TrainingRunsEarlyCompletionRulesBinding out;
    if (earlyCompletionRules.is_none()) {
        return out;
    }

    size_t ruleIndex = 0;
    for (nb::handle ruleObject : pybind::castOrTypeError<nb::iterable>(
             earlyCompletionRules, "early_completion_rules", "iterable of TrainingRunsEarlyCompletionRule objects or None", false)) {
        out.rules.push_back(pybind::castOrTypeError<TrainingRunsEarlyCompletionRule>(
            ruleObject, "early_completion_rules[" + std::to_string(ruleIndex) + "]", "thor.training.TrainingRunsEarlyCompletionRule", false));
        std::vector<nb::object> refs = callbackRefsFromObject(ruleObject);
        out.callbackRefs.insert(out.callbackRefs.end(), refs.begin(), refs.end());
        ++ruleIndex;
    }
    return out;
}

nb::object optionalDouble(std::optional<double> value) {
    if (!value.has_value()) {
        return nb::none();
    }
    return nb::cast(*value);
}

nb::object optionalString(std::optional<std::string> value) {
    if (!value.has_value()) {
        return nb::none();
    }
    return nb::cast(*value);
}

nb::object optionalUint64(std::optional<uint64_t> value) {
    if (!value.has_value()) {
        return nb::none();
    }
    return nb::cast(*value);
}

nb::object optionalUint64FromStats(const std::optional<TrainingStatsSnapshot>& stats, uint64_t TrainingStatsSnapshot::* field) {
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

    auto numpy_float32_dict_batch_loader = nb::class_<NumpyFloat32DictBatchLoader, Loader>(training, "NumpyFloat32DictBatchLoader");
    numpy_float32_dict_batch_loader.attr("__module__") = "thor.training";
    numpy_float32_dict_batch_loader.def_static(
        "__new__",
        [](nb::handle cls,
           nb::dict train,
           nb::dict validate,
           uint64_t batch_size,
           const std::string& dataset_name,
           bool randomize_train,
           uint64_t batch_queue_depth,
           nb::object test,
           nb::object random_seed) -> std::shared_ptr<NumpyFloat32DictBatchLoader> {
            (void)cls;
            auto loader = std::make_shared<NumpyFloat32DictBatchLoader>(std::move(train),
                                                                        std::move(validate),
                                                                        std::move(test),
                                                                        batch_size,
                                                                        "NumpyFloat32DictBatchLoader",
                                                                        randomize_train,
                                                                        batch_queue_depth,
                                                                        optionalUint64FromPython(std::move(random_seed), "random_seed"));
            loader->setDatasetName(dataset_name);
            return loader;
        },
        "cls"_a,
        "train"_a,
        "validate"_a,
        "batch_size"_a,
        "dataset_name"_a = "numpy_dict",
        "randomize_train"_a = true,
        "batch_queue_depth"_a = 32,
        "test"_a = nb::none(),
        "random_seed"_a = nb::none(),
        R"nbdoc(
Create an eager in-memory batch loader from dictionaries of named float32 tensors.

The train, validate, and optional test dictionaries must have the same string
keys and matching non-batch shapes.  If test is omitted, TEST uses a distinct
queue backed by the validate arrays for backward compatibility.  Values may be
numpy arrays or objects convertible with
``numpy.ascontiguousarray(value, dtype=numpy.float32)``.  The loader copies all
arrays into C++-owned memory during construction and allocates fixed-size CPU
batch tensor queues up front, so no Python callbacks or per-batch numpy work are
needed in the training hot path.

The leading dimension is the example dimension.  A one-dimensional array with
shape ``[N]`` is exposed to the network as non-batch shape ``[1]``, which is
convenient for scalar tensors such as ``example_weights``.  ``batch_queue_depth``
controls how many fixed CPU batch buffers are allocated up front per named tensor.
        )nbdoc");
    numpy_float32_dict_batch_loader.def(
        "__init__",
        [](NumpyFloat32DictBatchLoader*, nb::dict, nb::dict, uint64_t, const std::string&, bool, uint64_t, nb::object, nb::object) {},
        "train"_a,
        "validate"_a,
        "batch_size"_a,
        "dataset_name"_a = "numpy_dict",
        "randomize_train"_a = true,
        "batch_queue_depth"_a = 32,
        "test"_a = nb::none(),
        "random_seed"_a = nb::none());
    numpy_float32_dict_batch_loader.def("get_num_train_examples",
                                        [](NumpyFloat32DictBatchLoader& self) { return self.getNumExamples(ExampleType::TRAIN); });
    numpy_float32_dict_batch_loader.def("get_num_validate_examples",
                                        [](NumpyFloat32DictBatchLoader& self) { return self.getNumExamples(ExampleType::VALIDATE); });
    numpy_float32_dict_batch_loader.def("get_num_test_examples",
                                        [](NumpyFloat32DictBatchLoader& self) { return self.getNumExamples(ExampleType::TEST); });
    numpy_float32_dict_batch_loader.def("get_num_train_batches",
                                        [](NumpyFloat32DictBatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::TRAIN); });
    numpy_float32_dict_batch_loader.def(
        "get_num_validate_batches", [](NumpyFloat32DictBatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::VALIDATE); });
    numpy_float32_dict_batch_loader.def("get_num_test_batches",
                                        [](NumpyFloat32DictBatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::TEST); });
    numpy_float32_dict_batch_loader.def("get_tensor_names", &NumpyFloat32DictBatchLoader::getTensorNames);
    numpy_float32_dict_batch_loader.def("get_tensor_shapes", &NumpyFloat32DictBatchLoader::getTensorShapes);
    numpy_float32_dict_batch_loader.def("get_batch_queue_depth", &NumpyFloat32DictBatchLoader::getBatchQueueDepth);
    numpy_float32_dict_batch_loader.def("get_randomize_train", &NumpyFloat32DictBatchLoader::getRandomizeTrain);
    numpy_float32_dict_batch_loader.def("get_random_seed", &NumpyFloat32DictBatchLoader::getRandomSeed);
    numpy_float32_dict_batch_loader.def("has_explicit_test_split", &NumpyFloat32DictBatchLoader::hasExplicitTestSplit);

    auto indexed_numpy_float32_dict_batch_loader =
        nb::class_<IndexedNumpyFloat32DictBatchLoader, Loader>(training, "IndexedNumpyFloat32DictBatchLoader");
    indexed_numpy_float32_dict_batch_loader.attr("__module__") = "thor.training";
    indexed_numpy_float32_dict_batch_loader.def_static(
        "__new__",
        [](nb::handle cls,
           nb::dict tensors,
           nb::object train_indices,
           nb::object validate_indices,
           uint64_t batch_size,
           const std::string& dataset_name,
           bool randomize_train,
           uint64_t batch_queue_depth,
           nb::object test_indices,
           nb::object random_seed) -> std::shared_ptr<IndexedNumpyFloat32DictBatchLoader> {
            (void)cls;
            auto loader = std::make_shared<IndexedNumpyFloat32DictBatchLoader>(
                std::move(tensors),
                std::move(train_indices),
                std::move(validate_indices),
                std::move(test_indices),
                batch_size,
                "IndexedNumpyFloat32DictBatchLoader",
                randomize_train,
                batch_queue_depth,
                optionalUint64FromPython(std::move(random_seed), "random_seed"));
            loader->setDatasetName(dataset_name);
            return loader;
        },
        "cls"_a,
        "tensors"_a,
        "train_indices"_a,
        "validate_indices"_a,
        "batch_size"_a,
        "dataset_name"_a = "indexed_numpy_dict",
        "randomize_train"_a = true,
        "batch_queue_depth"_a = 32,
        "test_indices"_a = nb::none(),
        "random_seed"_a = nb::none(),
        R"nbdoc(
Create a no-copy batch loader from one shared dictionary of named float32 tensors
and train/validate[/test] row-index arrays.

The tensor dictionary must contain C-contiguous numpy.float32 arrays with matching
leading dimensions.  Unlike ``NumpyFloat32DictBatchLoader``, this loader does
not call ``numpy.ascontiguousarray`` and does not copy tensor data into
loader-owned split buffers.  It marks the provided ndarray objects read-only,
keeps them alive, and copies rows from the canonical tensors into batch buffers
through the supplied integer indices.

If test_indices is omitted, TEST uses a distinct queue over validate_indices for
backward-compatible train/validate/test semantics without duplicating tensor
data.
        )nbdoc");
    indexed_numpy_float32_dict_batch_loader.def(
        "__init__",
        [](IndexedNumpyFloat32DictBatchLoader*, nb::dict, nb::object, nb::object, uint64_t, const std::string&, bool, uint64_t, nb::object, nb::object) {},
        "tensors"_a,
        "train_indices"_a,
        "validate_indices"_a,
        "batch_size"_a,
        "dataset_name"_a = "indexed_numpy_dict",
        "randomize_train"_a = true,
        "batch_queue_depth"_a = 32,
        "test_indices"_a = nb::none(),
        "random_seed"_a = nb::none());
    indexed_numpy_float32_dict_batch_loader.def(
        "get_num_train_examples", [](IndexedNumpyFloat32DictBatchLoader& self) { return self.getNumExamples(ExampleType::TRAIN); });
    indexed_numpy_float32_dict_batch_loader.def(
        "get_num_validate_examples", [](IndexedNumpyFloat32DictBatchLoader& self) { return self.getNumExamples(ExampleType::VALIDATE); });
    indexed_numpy_float32_dict_batch_loader.def(
        "get_num_test_examples", [](IndexedNumpyFloat32DictBatchLoader& self) { return self.getNumExamples(ExampleType::TEST); });
    indexed_numpy_float32_dict_batch_loader.def(
        "get_num_train_batches", [](IndexedNumpyFloat32DictBatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::TRAIN); });
    indexed_numpy_float32_dict_batch_loader.def(
        "get_num_validate_batches", [](IndexedNumpyFloat32DictBatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::VALIDATE); });
    indexed_numpy_float32_dict_batch_loader.def(
        "get_num_test_batches", [](IndexedNumpyFloat32DictBatchLoader& self) { return self.getNumBatchesPerEpoch(ExampleType::TEST); });
    indexed_numpy_float32_dict_batch_loader.def("get_tensor_names", &IndexedNumpyFloat32DictBatchLoader::getTensorNames);
    indexed_numpy_float32_dict_batch_loader.def("get_tensor_shapes", &IndexedNumpyFloat32DictBatchLoader::getTensorShapes);
    indexed_numpy_float32_dict_batch_loader.def("get_batch_queue_depth", &IndexedNumpyFloat32DictBatchLoader::getBatchQueueDepth);
    indexed_numpy_float32_dict_batch_loader.def("get_randomize_train", &IndexedNumpyFloat32DictBatchLoader::getRandomizeTrain);
    indexed_numpy_float32_dict_batch_loader.def("get_random_seed", &IndexedNumpyFloat32DictBatchLoader::getRandomSeed);
    indexed_numpy_float32_dict_batch_loader.def("has_explicit_test_split", &IndexedNumpyFloat32DictBatchLoader::hasExplicitTestSplit);

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
    trainer_fit_options.def(nb::init<>())
        .def_rw("epochs", &TrainerFitOptions::epochs)
        .def_rw("check_best_model_every_epochs", &TrainerFitOptions::checkBestModelEveryEpochs)
        .def_rw("min_early_completion_epochs", &TrainerFitOptions::minEarlyCompletionEpochs);

    auto trainer = nb::class_<Trainer>(training, "Trainer", nb::dynamic_attr());
    trainer.attr("__module__") = "thor.training";
    trainer.def_static(
        "__new__",
        [](nb::handle cls,
           std::shared_ptr<Network> network,
           std::shared_ptr<Loader> loader,
           std::shared_ptr<Optimizer> optimizer,
           nb::object training_program,
           bool debug_synchronous,
           double stats_interval_s,
           uint64_t max_in_flight_batches,
           std::vector<std::string> scalar_tensors_to_report,
           bool stats_stderr_also,
           std::string stats_color,
           nb::object save_model_dir,
           bool save_model_overwrite,
           nb::object model_selection_score) -> nb::object {
            (void)cls;
            Trainer::Builder builder;
            builder.network(std::move(network))
                .loader(std::move(loader))
                .statsIntervalSeconds(stats_interval_s)
                .statsStderrAlso(stats_stderr_also)
                .statsColorMode(lineStatsColorModeFromString(stats_color))
                .maxInFlightBatches(max_in_flight_batches)
                .scalarTensorsToReport(stringSetFromVector(std::move(scalar_tensors_to_report)))
                .saveModelDirectory(optionalPathStringFromPython(save_model_dir, "save_model_dir"))
                .saveModelOverwrite(save_model_overwrite)
                .modelSelectionScore(trainingModelSelectionScoreFromPython(model_selection_score));
            if (optimizer != nullptr) {
                builder.optimizer(std::move(optimizer));
            }
            if (!training_program.is_none()) {
                builder.trainingProgram(pybind::castArgument<std::shared_ptr<TrainingProgram>>(
                    training_program, "Trainer.__new__", "training_program", "thor.training.TrainingProgram or None", false));
            }
            if (debug_synchronous) {
                builder.debugSynchronousExecutor();
            }
            nb::object object = nb::cast(std::make_shared<Trainer>(builder.build()));
            return object;
        },
        "cls"_a,
        "network"_a,
        "loader"_a,
        "optimizer"_a.none() = nb::none(),
        "training_program"_a.none() = nb::none(),
        "debug_synchronous"_a = false,
        "stats_interval_s"_a = 10.0,
        "max_in_flight_batches"_a = 32,
        "scalar_tensors_to_report"_a = std::vector<std::string>{"loss"},
        "stats_stderr_also"_a = false,
        "stats_color"_a = "auto",
        "save_model_dir"_a.none() = nb::none(),
        "save_model_overwrite"_a = false,
        "model_selection_score"_a.none() = nb::none());
    trainer.def(
        "__init__",
        [](Trainer*,
           std::shared_ptr<Network>,
           std::shared_ptr<Loader>,
           std::shared_ptr<Optimizer>,
           nb::object,
           bool,
           double,
           uint64_t,
           std::vector<std::string>,
           bool,
           std::string,
           nb::object,
           bool,
           nb::object) {},
        "network"_a,
        "loader"_a,
        "optimizer"_a.none() = nb::none(),
        "training_program"_a.none() = nb::none(),
        "debug_synchronous"_a = false,
        "stats_interval_s"_a = 10.0,
        "max_in_flight_batches"_a = 32,
        "scalar_tensors_to_report"_a = std::vector<std::string>{"loss"},
        "stats_stderr_also"_a = false,
        "stats_color"_a = "auto",
        "save_model_dir"_a.none() = nb::none(),
        "save_model_overwrite"_a = false,
        "model_selection_score"_a.none() = nb::none());
    trainer.def(
        "fit",
        [](Trainer& self,
           uint32_t epochs,
           uint32_t check_best_model_every_epochs,
           uint64_t min_early_completion_epochs,
           nb::object restart_conditions,
           nb::object early_completion_policies) -> nb::object {
            TrainerFitOptions options;
            options.epochs = epochs;
            options.checkBestModelEveryEpochs = check_best_model_every_epochs;
            options.minEarlyCompletionEpochs = min_early_completion_epochs;
            options.restartConditions = trainingRestartPoliciesFromPython(restart_conditions, /*trainerScope=*/true);
            TrainingEarlyCompletionPoliciesBinding earlyPolicies = trainingEarlyCompletionPoliciesFromPython(early_completion_policies);
            options.earlyCompletionPolicies = std::move(earlyPolicies.policies);
            TrainingRunResult result;
            {
                nb::gil_scoped_release release;
                result = self.fit(options);
            }
            return nb::cast(std::move(result));
        },
        "epochs"_a,
        "check_best_model_every_epochs"_a = 0,
        "min_early_completion_epochs"_a = 0,
        "restart_conditions"_a.none() = nb::none(),
        "early_completion_policies"_a.none() = nb::none());
    trainer.def(
        "save_model",
        [](Trainer& self, nb::object directory, bool overwrite, bool save_optimizer_state) {
            std::string path = pathStringFromPython(directory, "directory");
            nb::gil_scoped_release release;
            self.saveModel(path, overwrite, save_optimizer_state);
        },
        "directory"_a,
        "overwrite"_a = false,
        "save_optimizer_state"_a = true);
    trainer.def_prop_ro("completed_training_epochs", &Trainer::getCompletedTrainingEpochs);

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
    training_stats_snapshot.def_ro("floating_point_operations_per_batch", &TrainingStatsSnapshot::floatingPointOperationsPerBatch);
    training_stats_snapshot.def_ro("floating_point_operations_per_second", &TrainingStatsSnapshot::floatingPointOperationsPerSecond);
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

    auto training_run_completion_reason = nb::enum_<TrainingRunCompletionReason>(training, "TrainingRunCompletionReason")
                                              .value("completed", TrainingRunCompletionReason::COMPLETED)
                                              .value("early_completed", TrainingRunCompletionReason::EARLY_COMPLETED);
    training_run_completion_reason.attr("__module__") = "thor.training";

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
    training_run_result.def_prop_ro("result", [](const TrainingRunResult& self) { return self.resultName(); });
    training_run_result.def_prop_ro("completion_reason",
                                    [](const TrainingRunResult& self) { return trainingRunCompletionReasonName(self.completionReason); });
    training_run_result.def_prop_ro("completion_reason_enum", [](const TrainingRunResult& self) { return self.completionReason; });
    training_run_result.def_prop_ro("early_completed", &TrainingRunResult::earlyCompleted);
    training_run_result.def_prop_ro("completed_epoch", [](const TrainingRunResult& self) { return optionalUint64(self.completedEpoch); });
    training_run_result.def_prop_ro("best_epoch", [](const TrainingRunResult& self) { return optionalUint64(self.bestEpoch); });
    training_run_result.def_prop_ro("best_score", [](const TrainingRunResult& self) { return optionalDouble(self.bestScore); });
    training_run_result.def_prop_ro("saved_model_dir", [](const TrainingRunResult& self) -> nb::object {
        if (!self.savedModelDirectory.has_value()) {
            return nb::none();
        }
        return nb::cast(*self.savedModelDirectory);
    });
    training_run_result.def_prop_ro("saved_model_network_name", [](const TrainingRunResult& self) -> nb::object {
        if (!self.savedModelNetworkName.has_value()) {
            return nb::none();
        }
        return nb::cast(*self.savedModelNetworkName);
    });
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
    training_run_result.def(
        "final_loss",
        [](const TrainingRunResult& self, const std::string& phase) {
            return optionalLossFromStats(self.finalStatsForPhase(trainingEventPhaseFromString(phase)));
        },
        "phase"_a);
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

    auto training_run_input_signature = nb::class_<TrainingRunInputSignature>(training, "TrainingRunInputSignature");
    training_run_input_signature.attr("__module__") = "thor.training";
    training_run_input_signature.def_prop_ro("input_name", [](const TrainingRunInputSignature& self) { return self.inputName; });
    training_run_input_signature.def_prop_ro("dimensions", [](const TrainingRunInputSignature& self) { return self.dimensions; });
    training_run_input_signature.def_prop_ro("data_type", [](const TrainingRunInputSignature& self) { return self.dataType; });
    training_run_input_signature.def_prop_ro("dimensions_include_batch",
                                             [](const TrainingRunInputSignature& self) { return self.dimensionsIncludeBatch; });

    auto training_run_output_signature = nb::class_<TrainingRunOutputSignature>(training, "TrainingRunOutputSignature");
    training_run_output_signature.attr("__module__") = "thor.training";
    training_run_output_signature.def_prop_ro("output_name", [](const TrainingRunOutputSignature& self) { return self.outputName; });
    training_run_output_signature.def_prop_ro("dimensions", [](const TrainingRunOutputSignature& self) { return self.dimensions; });
    training_run_output_signature.def_prop_ro("data_type", [](const TrainingRunOutputSignature& self) { return self.dataType; });

    auto training_ensemble_member_result = nb::class_<TrainingEnsembleMemberResult>(training, "TrainingEnsembleMemberResult");
    training_ensemble_member_result.attr("__module__") = "thor.training";
    training_ensemble_member_result.def_prop_ro("run_name", [](const TrainingEnsembleMemberResult& self) { return self.runName; });
    training_ensemble_member_result.def_ro("weight", &TrainingEnsembleMemberResult::weight);
    training_ensemble_member_result.def_prop_ro(
        "status", [](const TrainingEnsembleMemberResult& self) { return trainingRunStatusName(self.status); });
    training_ensemble_member_result.def_prop_ro("status_enum", [](const TrainingEnsembleMemberResult& self) { return self.status; });
    training_ensemble_member_result.def_prop_ro(
        "final_training_loss", [](const TrainingEnsembleMemberResult& self) { return optionalDouble(self.finalTrainingLoss); });
    training_ensemble_member_result.def_prop_ro(
        "final_validation_loss", [](const TrainingEnsembleMemberResult& self) { return optionalDouble(self.finalValidationLoss); });
    training_ensemble_member_result.def_prop_ro(
        "final_test_loss", [](const TrainingEnsembleMemberResult& self) { return optionalDouble(self.finalTestLoss); });
    training_ensemble_member_result.def_prop_ro("final_test_metrics",
                                                [](const TrainingEnsembleMemberResult& self) { return self.finalTestMetrics; });

    auto training_named_metric_result = nb::class_<TrainingNamedMetricResult>(training, "TrainingNamedMetricResult");
    training_named_metric_result.attr("__module__") = "thor.training";
    training_named_metric_result.def_prop_ro("name", [](const TrainingNamedMetricResult& self) { return self.name; });
    training_named_metric_result.def_prop_ro("train_value",
                                             [](const TrainingNamedMetricResult& self) { return optionalDouble(self.trainValue); });
    training_named_metric_result.def_prop_ro("test_value",
                                             [](const TrainingNamedMetricResult& self) { return optionalDouble(self.testValue); });
    training_named_metric_result.def("has_value", &TrainingNamedMetricResult::hasValue);

    auto training_ensemble_result = nb::class_<TrainingEnsembleResult>(training, "TrainingEnsembleResult");
    training_ensemble_result.attr("__module__") = "thor.training";
    training_ensemble_result.def_prop_ro("ensemble_group", [](const TrainingEnsembleResult& self) { return self.ensembleGroup; });
    training_ensemble_result.def_prop_ro("members", [](const TrainingEnsembleResult& self) { return self.members; });
    training_ensemble_result.def_prop_ro("input_signature", [](const TrainingEnsembleResult& self) { return self.inputSignature; });
    training_ensemble_result.def_prop_ro("output_signature", [](const TrainingEnsembleResult& self) { return self.outputSignature; });
    training_ensemble_result.def("__len__", &TrainingEnsembleResult::size);
    training_ensemble_result.def("__bool__", [](const TrainingEnsembleResult& self) { return !self.empty(); });
    training_ensemble_result.def("all_completed", &TrainingEnsembleResult::allCompleted);
    training_ensemble_result.def("any_failed", &TrainingEnsembleResult::anyFailed);
    training_ensemble_result.def("has_enough_successful_models", &TrainingEnsembleResult::hasEnoughSuccessfulModels);
    training_ensemble_result.def_prop_ro("successful_models", &TrainingEnsembleResult::successfulModels);
    training_ensemble_result.def_prop_ro("required_successful_models", &TrainingEnsembleResult::requiredSuccessfulModels);
    training_ensemble_result.def_prop_ro("min_successful_models", &TrainingEnsembleResult::requiredSuccessfulModels);
    training_ensemble_result.def_prop_ro("target_num_members", &TrainingEnsembleResult::size);
    training_ensemble_result.def_prop_ro("actual_num_members", &TrainingEnsembleResult::successfulModels);
    training_ensemble_result.def_prop_ro("total_weight", &TrainingEnsembleResult::totalWeight);
    training_ensemble_result.def_prop_ro("status_counts", &TrainingEnsembleResult::statusCounts);
    training_ensemble_result.def_prop_ro("named_metrics", [](const TrainingEnsembleResult& self) { return self.namedMetrics; });
    training_ensemble_result.def_prop_ro("graph_metrics", [](const TrainingEnsembleResult& self) { return self.namedGraphMetrics; });
    training_ensemble_result.def_prop_ro("reported_metrics", [](const TrainingEnsembleResult& self) { return self.namedGraphMetrics; });
    training_ensemble_result.def("has_named_metric_values", &TrainingEnsembleResult::hasNamedMetricValues);
    training_ensemble_result.def("has_graph_metric_values", &TrainingEnsembleResult::hasNamedGraphMetricValues);
    training_ensemble_result.def("has_ensemble_evaluation_metrics", &TrainingEnsembleResult::hasEnsembleEvaluationMetrics);
    training_ensemble_result.def_prop_ro(
        "ensemble_training_loss", [](const TrainingEnsembleResult& self) { return optionalDouble(self.ensembleFinalTrainingLoss()); });
    training_ensemble_result.def_prop_ro(
        "ensemble_train_loss", [](const TrainingEnsembleResult& self) { return optionalDouble(self.ensembleFinalTrainingLoss()); });
    training_ensemble_result.def_prop_ro("ensemble_test_loss",
                                         [](const TrainingEnsembleResult& self) { return optionalDouble(self.ensembleFinalTestLoss()); });

    auto training_restart_policy = nb::class_<TrainingRestartPolicy>(training, "TrainingRestartPolicy");
    training_restart_policy.attr("__module__") = "thor.training";
    training_restart_policy.def(
        "__init__",
        [](TrainingRestartPolicy* self,
           std::optional<std::string> run_name,
           std::optional<std::string> ensemble_group,
           uint32_t progress_check_epochs,
           double progress_improvement_min_percentage,
           uint32_t max_restarts) {
            new (self) TrainingRestartPolicy();
            self->runName = std::move(run_name);
            self->ensembleGroup = std::move(ensemble_group);
            self->progressCheckEpochs = progress_check_epochs;
            self->progressImprovementMinPercentage = progress_improvement_min_percentage;
            self->maxRestarts = max_restarts;
        },
        "run_name"_a.none() = nb::none(),
        "ensemble_group"_a.none() = nb::none(),
        "progress_check_epochs"_a = 3,
        "progress_improvement_min_percentage"_a = 5.0,
        "max_restarts"_a = 5);
    training_restart_policy.def_prop_ro("run_name", [](const TrainingRestartPolicy& self) -> nb::object {
        if (!self.runName.has_value()) {
            return nb::none();
        }
        return nb::cast(*self.runName);
    });
    training_restart_policy.def_prop_ro("ensemble_group", [](const TrainingRestartPolicy& self) -> nb::object {
        if (!self.ensembleGroup.has_value()) {
            return nb::none();
        }
        return nb::cast(*self.ensembleGroup);
    });
    training_restart_policy.def_prop_ro("progress_check_epochs",
                                        [](const TrainingRestartPolicy& self) { return self.progressCheckEpochs; });
    training_restart_policy.def_prop_ro("progress_improvement_min_percentage",
                                        [](const TrainingRestartPolicy& self) { return self.progressImprovementMinPercentage; });
    training_restart_policy.def_prop_ro("max_restarts", [](const TrainingRestartPolicy& self) { return self.maxRestarts; });

    // Backward-compatible names are aliases to the single public restart-policy type.
    training.attr("RestartPolicy") = training.attr("TrainingRestartPolicy");
    training.attr("TrainingRestartCondition") = training.attr("TrainingRestartPolicy");
    training.attr("RestartCondition") = training.attr("TrainingRestartPolicy");
    training.attr("TrainingRunsRestartPolicy") = training.attr("TrainingRestartPolicy");
    training.attr("TrainingRunsRestartCondition") = training.attr("TrainingRestartPolicy");

    auto training_early_completion_policy =
        nb::class_<TrainingEarlyCompletionPolicy>(training, "TrainingEarlyCompletionPolicy", nb::dynamic_attr());
    training_early_completion_policy.attr("__module__") = "thor.training";
    training_early_completion_policy.def_static(
        "__new__",
        [](nb::handle cls, nb::object completion_condition) -> nb::object {
            (void)cls;
            BoundPythonEarlyCompletionPolicy bound = trainingEarlyCompletionPolicyFromWeakCallable(std::move(completion_condition));
            nb::object object = nb::cast(std::move(bound.policy));
            attachCallbackRefs(object, {bound.callbackHolder});
            return object;
        },
        "cls"_a,
        "completion_condition"_a);
    training.attr("EarlyCompletionPolicy") = training.attr("TrainingEarlyCompletionPolicy");

    auto training_runs_early_completion_rule = nb::class_<TrainingRunsEarlyCompletionRule, TrainingEarlyCompletionPolicy>(
        training, "TrainingRunsEarlyCompletionRule", nb::dynamic_attr());
    training_runs_early_completion_rule.attr("__module__") = "thor.training";
    training_runs_early_completion_rule.def_static(
        "__new__",
        [](nb::handle cls, nb::object completion_condition, std::optional<std::string> run_name, std::optional<std::string> ensemble_group)
            -> nb::object {
            (void)cls;
            BoundPythonEarlyCompletionPolicy bound = trainingEarlyCompletionPolicyFromWeakCallable(std::move(completion_condition));
            TrainingRunsEarlyCompletionRule rule(std::move(bound.policy.completionCondition));
            rule.runName = std::move(run_name);
            rule.ensembleGroup = std::move(ensemble_group);
            nb::object object = nb::cast(std::move(rule));
            attachCallbackRefs(object, {bound.callbackHolder});
            return object;
        },
        "cls"_a,
        "completion_condition"_a,
        "run_name"_a.none() = nb::none(),
        "ensemble_group"_a.none() = nb::none());
    training_runs_early_completion_rule.def_prop_ro("run_name", [](const TrainingRunsEarlyCompletionRule& self) -> nb::object {
        if (!self.runName.has_value()) {
            return nb::none();
        }
        return nb::cast(*self.runName);
    });
    training_runs_early_completion_rule.def_prop_ro("ensemble_group", [](const TrainingRunsEarlyCompletionRule& self) -> nb::object {
        if (!self.ensembleGroup.has_value()) {
            return nb::none();
        }
        return nb::cast(*self.ensembleGroup);
    });
    training.attr("EarlyCompletionRule") = training.attr("TrainingRunsEarlyCompletionRule");
    training.attr("TrainingRunsEarlyCompletionPolicy") = training.attr("TrainingRunsEarlyCompletionRule");

    auto training_runs_result = nb::class_<TrainingRunsResult>(training, "TrainingRunsResult");
    training_runs_result.attr("__module__") = "thor.training";
    training_runs_result.def("__len__", &TrainingRunsResult::size);
    training_runs_result.def("__bool__", [](const TrainingRunsResult& self) { return !self.empty(); });
    training_runs_result.def(
        "__getitem__",
        [](const TrainingRunsResult& self, int64_t index) -> const TrainingRunResult& {
            int64_t resolvedIndex = index;
            if (resolvedIndex < 0) {
                resolvedIndex += static_cast<int64_t>(self.size());
            }
            if (resolvedIndex < 0) {
                throw nb::index_error("TrainingRunsResult index is out of range");
            }
            return self.at(static_cast<size_t>(resolvedIndex));
        },
        nb::rv_policy::reference_internal);
    training_runs_result.def(
        "__getitem__",
        [](const TrainingRunsResult& self, const std::string& runName) -> const TrainingRunResult& { return self.at(runName); },
        nb::rv_policy::reference_internal);
    training_runs_result.def_prop_ro("runs", [](const TrainingRunsResult& self) { return self.runs(); });
    training_runs_result.def_prop_ro("ensembles", [](const TrainingRunsResult& self) { return self.ensembles(); });
    training_runs_result.def_prop_ro("has_ensembles", &TrainingRunsResult::hasEnsembles);
    training_runs_result.def(
        "ensemble",
        [](const TrainingRunsResult& self, const std::string& ensembleGroup) -> const TrainingEnsembleResult& {
            return self.ensemble(ensembleGroup);
        },
        nb::rv_policy::reference_internal,
        "ensemble_group"_a);
    training_runs_result.def("all_completed", &TrainingRunsResult::allCompleted);
    training_runs_result.def("any_failed", &TrainingRunsResult::anyFailed);
    training_runs_result.def("any_cancelled", &TrainingRunsResult::anyCancelled);
    training_runs_result.def_prop_ro("status_counts", &TrainingRunsResult::statusCounts);
    training_runs_result.def("get_status_counts", &TrainingRunsResult::statusCounts);
    training_runs_result.def(
        "save_ensemble",
        [](const TrainingRunsResult& self,
           const std::string& ensemble_group,
           nb::object path,
           const std::string& aggregation,
           bool overwrite) { return self.saveEnsemble(ensemble_group, pathStringFromPython(path, "path"), aggregation, overwrite); },
        "ensemble_group"_a,
        "path"_a,
        "aggregation"_a = "auto",
        "overwrite"_a = false);

    auto training_runs = nb::class_<TrainingRuns>(training, "TrainingRuns", nb::dynamic_attr());
    training_runs.attr("__module__") = "thor.training";
    training_runs.def_static(
        "__new__",
        [](nb::handle cls,
           nb::iterable runs,
           const std::string& failure_policy,
           double max_summary_logs_per_second,
           std::optional<size_t> max_parallel_runs,
           nb::object min_successful_models) -> nb::object {
            (void)cls;
            std::vector<TrainingRunsSpec> specs = trainingRunsSpecsFromPython(runs);
            auto self = std::make_shared<TrainingRuns>(std::move(specs),
                                                       trainingRunsFailurePolicyFromString(failure_policy),
                                                       max_summary_logs_per_second,
                                                       max_parallel_runs,
                                                       trainingRunsMinSuccessfulModelsFromPython(min_successful_models));
            nb::object object = nb::cast(std::move(self));
            return object;
        },
        "cls"_a,
        "runs"_a,
        "failure_policy"_a = "cancel_siblings",
        "max_summary_logs_per_second"_a = 2.0,
        "max_parallel_runs"_a.none() = nb::none(),
        "min_successful_models"_a.none() = nb::none());
    training_runs.def(
        "__init__",
        [](TrainingRuns*,
           nb::iterable,
           const std::string&,
           double,
           std::optional<size_t>,
           nb::object) {},
        "runs"_a,
        "failure_policy"_a = "cancel_siblings",
        "max_summary_logs_per_second"_a = 2.0,
        "max_parallel_runs"_a.none() = nb::none(),
        "min_successful_models"_a.none() = nb::none());
    training_runs.def_prop_ro("reports", [](const TrainingRuns& self) { return self.getReports(); });
    training_runs.def(
        "fit",
        [](TrainingRuns& self,
           uint32_t epochs,
           std::shared_ptr<Loader> test_loader,
           uint32_t check_best_model_every_epochs,
           uint64_t min_early_completion_epochs,
           nb::object restart_conditions,
           nb::object early_completion_rules,
           nb::object reports,
           bool evaluate_training_population) {
            TrainerFitOptions options;
            options.epochs = epochs;
            options.checkBestModelEveryEpochs = check_best_model_every_epochs;
            options.minEarlyCompletionEpochs = min_early_completion_epochs;
            TrainingRunsSessionOptions sessionOptions;
            sessionOptions.restartConditions = trainingRestartPoliciesFromPython(restart_conditions, /*trainerScope=*/false);
            TrainingRunsEarlyCompletionRulesBinding earlyRules = trainingRunsEarlyCompletionRulesFromPython(early_completion_rules);
            sessionOptions.earlyCompletionRules = std::move(earlyRules.rules);
            sessionOptions.reports = trainingRunsReportsFromPython(reports, self.getRuns());
            sessionOptions.evaluation.testLoader = std::move(test_loader);
            sessionOptions.evaluation.evaluateTrainingPopulation = evaluate_training_population;
            nb::gil_scoped_release release;
            return self.fit(options, sessionOptions);
        },
        "epochs"_a,
        "test_loader"_a.none() = nb::none(),
        "check_best_model_every_epochs"_a = 0,
        "min_early_completion_epochs"_a = 0,
        "restart_conditions"_a.none() = nb::none(),
        "early_completion_rules"_a.none() = nb::none(),
        "reports"_a.none() = nb::none(),
        "evaluate_training_population"_a = true);

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
        [](nb::handle cls, const std::string& name, std::shared_ptr<Network> network, bool enabled) -> std::shared_ptr<TrainingPhase> {
            (void)cls;
            return std::make_shared<TrainingPhase>(name, std::move(network), enabled);
        },
        "cls"_a,
        "name"_a,
        "network"_a,
        "enabled"_a = true);
    training_phase.def(
        "__init__", [](TrainingPhase*, const std::string&, std::shared_ptr<Network>, bool) {}, "name"_a, "network"_a, "enabled"_a = true);
    training_phase.def_prop_ro("name", &TrainingPhase::getName);
    training_phase.def_prop_rw("enabled", &TrainingPhase::isEnabled, &TrainingPhase::setEnabled);
    training_phase.def("is_initialized", &TrainingPhase::isInitialized);
    training_phase.def("is_enabled", &TrainingPhase::isEnabled);
    training_phase.def("enable", &TrainingPhase::enable);
    training_phase.def("disable", &TrainingPhase::disable);
    training_phase.def("set_enabled", &TrainingPhase::setEnabled, "enabled"_a);
    training_phase.def("get_network", &TrainingPhase::getNetwork);
    training_phase.def("get_outputs", &TrainingPhase::getOutputs, nb::rv_policy::reference_internal);
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
           std::vector<std::shared_ptr<TrainingPhase>> phases,
           std::shared_ptr<Optimizer> optimizer,
           std::vector<ParameterReference> update_parameters,
           uint32_t repeat_count,
           TrainingStep::GradientClearPolicy gradient_clear_policy,
           std::vector<TrainingInputBinding> input_bindings,
           bool enabled) -> std::shared_ptr<TrainingStep> {
            (void)cls;
            return std::make_shared<TrainingStep>(name,
                                                  std::move(phases),
                                                  std::move(optimizer),
                                                  std::move(update_parameters),
                                                  repeat_count,
                                                  gradient_clear_policy,
                                                  std::move(input_bindings),
                                                  enabled);
        },
        "cls"_a,
        "name"_a,
        "phases"_a,
        "optimizer"_a.none() = nb::none(),
        "update_parameters"_a = std::vector<ParameterReference>{},
        "repeat_count"_a = 1,
        "gradient_clear_policy"_a = TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
        "input_bindings"_a = std::vector<TrainingInputBinding>{},
        "enabled"_a = true);
    training_step.def(
        "__init__",
        [](TrainingStep*,
           const std::string&,
           std::vector<std::shared_ptr<TrainingPhase>>,
           std::shared_ptr<Optimizer>,
           std::vector<ParameterReference>,
           uint32_t,
           TrainingStep::GradientClearPolicy,
           std::vector<TrainingInputBinding>,
           bool) {},
        "name"_a,
        "phases"_a,
        "optimizer"_a.none() = nb::none(),
        "update_parameters"_a = std::vector<ParameterReference>{},
        "repeat_count"_a = 1,
        "gradient_clear_policy"_a = TrainingStep::GradientClearPolicy::CLEAR_BEFORE_STEP,
        "input_bindings"_a = std::vector<TrainingInputBinding>{},
        "enabled"_a = true);
    training_step.def_prop_ro("name", &TrainingStep::getName);
    training_step.def_prop_ro("repeat_count", &TrainingStep::getRepeatCount);
    training_step.def_prop_ro("gradient_clear_policy", &TrainingStep::getGradientClearPolicy);
    training_step.def_prop_rw("enabled", &TrainingStep::isEnabled, &TrainingStep::setEnabled);
    training_step.def("is_initialized", &TrainingStep::isInitialized);
    training_step.def("is_enabled", &TrainingStep::isEnabled);
    training_step.def("enable", &TrainingStep::enable);
    training_step.def("disable", &TrainingStep::disable);
    training_step.def("set_enabled", &TrainingStep::setEnabled, "enabled"_a);
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
            return std::make_shared<TrainingProgram>(pybind::castArgument<std::vector<std::shared_ptr<TrainingStep>>>(
                steps, "TrainingProgram.__new__", "steps", "sequence of thor.training.TrainingStep objects or None", false));
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
    training_program.def(
        "compile", &TrainingProgram::compile, "placed_network"_a, "resolve_empty_update_parameters_as_all_trainable"_a = true);
}
