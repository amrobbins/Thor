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
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/FileDataset.h"
#include "DeepLearning/Api/Data/TrainingData.h"
#include "DeepLearning/Api/Data/DatasetAccessPolicy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/ParameterReference.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Training/DatasetInputBindings.h"
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
#include "DeepLearning/Api/Data/DatasetWriter.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "bindings/python/src/core/cast.h"
#include "bindings/python/src/core/physical/NumpyDTypeMapping.h"
#include "bindings/python/src/core/training/NumpyDataset.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;
namespace pybind = Thor::PythonBindings;

namespace {

using Int32Array = nb::ndarray<const int32_t, nb::numpy, nb::c_contig>;
using UInt32Array = nb::ndarray<const uint32_t, nb::numpy, nb::c_contig>;
using Int64Array = nb::ndarray<const int64_t, nb::numpy, nb::c_contig>;
using UInt64Array = nb::ndarray<const uint64_t, nb::numpy, nb::c_contig>;

struct PythonConstantPad {
    PythonConstantPad() = default;
    explicit PythonConstantPad(double value) : value(value) {}

    double value = 0.0;
};

struct PythonTensorLayout {
    std::vector<uint64_t> shape;
    ThorImplementation::DataType dataType = ThorImplementation::DataType::FP32;
};

struct PythonWindowedTensorSourceLayout {
    std::vector<uint64_t> stepShape;
    ThorImplementation::DataType dataType = ThorImplementation::DataType::FP32;
    ThorImplementation::DataType keyType = ThorImplementation::DataType::UINT64;
};

struct PythonWindowedTensorLayout {
    std::vector<uint64_t> shape;
    std::string source;
    ThorImplementation::DataType indexType = ThorImplementation::DataType::INT32;
    double padValue = 0.0;
    std::optional<std::string> maskName;
    DatasetLayout::WindowedTensorReferenceMode referenceMode = DatasetLayout::WindowedTensorReferenceMode::INDEXED;
};

struct PythonWindowedTensorChunk {
    nb::object key;
    nb::object start;
};

struct PythonAffineWindowedTensorChunk {
    nb::object key;
    int64_t base = 0;
    int64_t stride = 1;
    int64_t fieldOffset = 0;
};

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

std::vector<uint64_t> uint64IndicesFromPython(nb::object indices,
                                              const std::string& context,
                                              uint64_t maxExclusive,
                                              bool allowEmpty = false) {
    nb::object numpy = nb::module_::import_("numpy");
    nb::object sourceObject;
    nb::object arrayObject;
    try {
        sourceObject = numpy.attr("asarray")(indices);

        // NumPy gives an empty Python list dtype=float64 by default. Empty
        // validate/test partitions are valid, so recognize an empty 1-D input
        // before requiring an integer dtype.
        if (allowEmpty) {
            const int ndim = pybind::castOrTypeError<int>(sourceObject.attr("ndim"), context + " ndim", "int", false);
            const uint64_t size =
                pybind::castOrTypeError<uint64_t>(sourceObject.attr("size"), context + " size", "int", false);
            if (ndim == 1 && size == 0) {
                return {};
            }
        }

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
    if (array.shape(0) == 0 && !allowEmpty) {
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
            throw nb::value_error((context + " contains a row index outside the dataset row count").c_str());
        }
        out.push_back(index);
    }
    return out;
}

std::string tensorNameFromPythonKey(nb::handle key, const std::string& context) {
    std::string name = pybind::castOrTypeError<std::string>(key, context + " key", "str", false);
    if (name.empty()) {
        throw nb::value_error((context + " tensor names must be non-empty").c_str());
    }
    return name;
}


std::optional<uint64_t> optionalUint64FromPython(nb::object value, const std::string& name) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return pybind::castOrTypeError<uint64_t>(value, name, "int or None", false);
}

std::vector<uint64_t> uint64ShapeFromPython(nb::handle value, const std::string& context) {
    if (nb::isinstance<nb::str>(value)) {
        throw nb::type_error((context + " must be an iterable of positive integers, not str").c_str());
    }
    std::vector<uint64_t> shape;
    uint64_t index = 0;
    for (nb::handle dimObject : pybind::castOrTypeError<nb::iterable>(
             value, context, "iterable of positive integers", false)) {
        const uint64_t dim = pybind::castOrTypeError<uint64_t>(
            dimObject, context + "[" + std::to_string(index) + "]", "positive int", false);
        if (dim == 0) {
            throw nb::value_error((context + " dimensions must be positive").c_str());
        }
        shape.push_back(dim);
        ++index;
    }
    if (shape.empty()) {
        throw nb::value_error((context + " must contain at least one dimension").c_str());
    }
    return shape;
}

std::vector<uint64_t> uint64OptionalShapeFromPython(nb::handle value, const std::string& context) {
    if (nb::isinstance<nb::str>(value)) {
        throw nb::type_error((context + " must be an iterable of positive integers, not str").c_str());
    }
    std::vector<uint64_t> shape;
    uint64_t index = 0;
    for (nb::handle dimObject : pybind::castOrTypeError<nb::iterable>(
             value, context, "iterable of positive integers", false)) {
        const uint64_t dim = pybind::castOrTypeError<uint64_t>(
            dimObject, context + "[" + std::to_string(index) + "]", "positive int", false);
        if (dim == 0) {
            throw nb::value_error((context + " dimensions must be positive").c_str());
        }
        shape.push_back(dim);
        ++index;
    }
    return shape;
}

std::vector<DatasetLayout::TensorShape> datasetTensorShapesFromPython(
    const nb::dict& tensors,
    const std::string& context) {
    std::vector<DatasetLayout::TensorShape> out;
    out.reserve(nb::len(tensors));
    for (auto item : tensors) {
        const std::string name = tensorNameFromPythonKey(item.first, context);
        PythonTensorLayout entry = pybind::castOrTypeError<PythonTensorLayout>(
            item.second, context + "['" + name + "']", "thor.data.TensorLayout", false);
        out.emplace_back(name, std::move(entry.shape), entry.dataType);
    }
    return out;
}

std::optional<std::string> optionalStringFromPython(const nb::object& obj, const std::string& argumentName) {
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::string value = pybind::castOrTypeError<std::string>(obj, argumentName, "str or None", false);
    if (value.empty()) {
        throw nb::value_error((argumentName + " must not be empty").c_str());
    }
    return value;
}

bool dictContainsString(const nb::dict& dict, const std::string& key) {
    return PyDict_GetItemString(dict.ptr(), key.c_str()) != nullptr;
}

nb::object dictGetString(const nb::dict& dict, const std::string& key, const std::string& context) {
    PyObject* item = PyDict_GetItemString(dict.ptr(), key.c_str());
    if (item == nullptr) {
        throw nb::value_error((context + " missing key '" + key + "'").c_str());
    }
    return nb::borrow<nb::object>(item);
}

bool pythonHasAttribute(nb::handle obj, const char* name) {
    return PyObject_HasAttrString(obj.ptr(), name) == 1;
}

DatasetLayout::WindowedTensorReferenceMode windowedReferenceModeFromPython(
    const std::string& value,
    const std::string& argumentName) {
    if (value == "indexed") {
        return DatasetLayout::WindowedTensorReferenceMode::INDEXED;
    }
    if (value == "affine") {
        return DatasetLayout::WindowedTensorReferenceMode::AFFINE;
    }
    throw nb::value_error((argumentName + " must be 'indexed' or 'affine'").c_str());
}

double padValueFromPython(const nb::object& obj, const std::string& argumentName) {
    if (obj.is_none()) {
        return 0.0;
    }
    PythonConstantPad pad = pybind::castOrTypeError<PythonConstantPad>(obj, argumentName, "thor.data.ConstantPad", false);
    return pad.value;
}

std::vector<DatasetLayout::WindowedTensorSourceShape> datasetWindowedTensorSourceShapesFromPython(
    const nb::object& maybeWindowSources,
    const std::string& context) {
    if (maybeWindowSources.is_none()) {
        return {};
    }
    nb::dict windowSources = pybind::castOrTypeError<nb::dict>(
        maybeWindowSources, context + " window_sources", "dict or None", false);
    std::vector<DatasetLayout::WindowedTensorSourceShape> out;
    out.reserve(nb::len(windowSources));
    for (auto item : windowSources) {
        const std::string name = tensorNameFromPythonKey(item.first, context + " window_sources");
        PythonWindowedTensorSourceLayout entry = pybind::castOrTypeError<PythonWindowedTensorSourceLayout>(
            item.second,
            context + " window_sources['" + name + "']",
            "thor.data.WindowedTensorSourceLayout",
            false);
        out.emplace_back(name, std::move(entry.stepShape), entry.dataType, entry.keyType);
    }
    return out;
}

std::vector<DatasetLayout::WindowedTensorShape> datasetWindowedTensorShapesFromPython(
    const nb::object& maybeWindowedTensors,
    const std::string& context) {
    if (maybeWindowedTensors.is_none()) {
        return {};
    }
    nb::dict windowedTensors = pybind::castOrTypeError<nb::dict>(
        maybeWindowedTensors, context + " windowed_tensors", "dict or None", false);
    std::vector<DatasetLayout::WindowedTensorShape> out;
    out.reserve(nb::len(windowedTensors));
    for (auto item : windowedTensors) {
        const std::string name = tensorNameFromPythonKey(item.first, context + " windowed_tensors");
        PythonWindowedTensorLayout entry = pybind::castOrTypeError<PythonWindowedTensorLayout>(
            item.second,
            context + " windowed_tensors['" + name + "']",
            "thor.data.WindowedTensorLayout",
            false);
        out.emplace_back(name,
                         std::move(entry.shape),
                         std::move(entry.source),
                         entry.indexType,
                         entry.padValue,
                         std::move(entry.maskName),
                         entry.referenceMode);
    }
    return out;
}

nb::list uint64VectorToPythonList(const std::vector<uint64_t>& values) {
    nb::list out;
    for (uint64_t value : values) {
        out.append(nb::int_(value));
    }
    return out;
}

nb::dict datasetLayoutTensorSpecsToPythonDict(const DatasetLayout& layout) {
    nb::dict out;
    for (const DatasetLayout::TensorSpec& spec : layout.tensors()) {
        nb::dict specDict;
        specDict["shape"] = uint64VectorToPythonList(spec.dimensions);
        specDict["data_type"] = nb::cast(spec.dataType);
        specDict["offset_bytes"] = spec.offsetBytes;
        specDict["num_bytes"] = spec.numBytes;
        out[nb::str(spec.name.c_str())] = std::move(specDict);
    }
    return out;
}

std::vector<std::string> datasetLayoutTensorNames(const DatasetLayout& layout) {
    std::vector<std::string> names;
    names.reserve(layout.tensors().size() + layout.windowedTensors().size());
    for (const DatasetLayout::TensorSpec& spec : layout.tensors()) {
        names.push_back(spec.name);
    }
    for (const DatasetLayout::WindowedTensorSpec& spec : layout.windowedTensors()) {
        names.push_back(spec.name);
        if (spec.maskName.has_value()) {
            names.push_back(spec.maskName.value());
        }
    }
    return names;
}

std::map<std::string, std::vector<uint64_t>> datasetLayoutTensorShapes(const DatasetLayout& layout) {
    std::map<std::string, std::vector<uint64_t>> shapes;
    for (const DatasetLayout::TensorSpec& spec : layout.tensors()) {
        shapes.emplace(spec.name, spec.dimensions);
    }
    for (const DatasetLayout::WindowedTensorSpec& spec : layout.windowedTensors()) {
        shapes.emplace(spec.name, spec.dimensions);
        if (spec.maskName.has_value()) {
            shapes.emplace(spec.maskName.value(), std::vector<uint64_t>{spec.windowLength()});
        }
    }
    return shapes;
}

nb::dict datasetLayoutWindowSourceSpecsToPythonDict(const DatasetLayout& layout) {
    nb::dict out;
    for (const DatasetLayout::WindowedTensorSourceSpec& spec : layout.windowedTensorSources()) {
        nb::dict specDict;
        specDict["step_shape"] = uint64VectorToPythonList(spec.stepDimensions);
        specDict["data_type"] = nb::cast(spec.dataType);
        specDict["key_type"] = nb::cast(spec.keyDataType);
        specDict["step_num_bytes"] = spec.stepNumBytes();
        specDict["source_filename"] = spec.sourceFilename.has_value()
            ? nb::object(nb::str(spec.sourceFilename->c_str())) : nb::object(nb::none());
        specDict["source_num_bytes"] = spec.sourceNumBytes;
        nb::list sequences;
        for (const DatasetLayout::WindowedTensorSourceSequence& sequence : spec.sourceSequences) {
            nb::dict seq;
            seq["key_hex"] = nb::str(sequence.keyHex.c_str());
            seq["start_index"] = sequence.startIndex;
            seq["end_index_exclusive"] = sequence.endIndexExclusive;
            seq["offset_bytes"] = sequence.offsetBytes;
            seq["num_steps"] = sequence.numSteps;
            seq["num_bytes"] = sequence.numBytes;
            sequences.append(std::move(seq));
        }
        specDict["source_sequences"] = std::move(sequences);
        out[nb::str(spec.name.c_str())] = std::move(specDict);
    }
    return out;
}

nb::dict datasetLayoutWindowedTensorSpecsToPythonDict(const DatasetLayout& layout) {
    nb::dict out;
    for (const DatasetLayout::WindowedTensorSpec& spec : layout.windowedTensors()) {
        nb::dict specDict;
        specDict["shape"] = uint64VectorToPythonList(spec.dimensions);
        specDict["source"] = nb::str(spec.sourceName.c_str());
        specDict["data_type"] = nb::cast(spec.dataType);
        specDict["key_type"] = nb::cast(spec.keyDataType);
        specDict["index_type"] = nb::cast(spec.indexDataType);
        specDict["pad_value"] = spec.padValue;
        specDict["mask_name"] = spec.maskName.has_value()
            ? nb::object(nb::str(spec.maskName->c_str())) : nb::object(nb::none());
        specDict["reference_mode"] = nb::str(
            spec.referenceMode == DatasetLayout::WindowedTensorReferenceMode::AFFINE ? "affine" : "indexed");
        specDict["reference_offset_bytes"] = spec.referenceOffsetBytes;
        specDict["reference_num_bytes"] = spec.referenceNumBytes;
        specDict["num_bytes"] = spec.outputNumBytes();
        out[nb::str(spec.name.c_str())] = std::move(specDict);
    }
    return out;
}

template <typename T>
std::vector<uint8_t> scalarBytesFromPython(nb::handle value, const std::string& context, const std::string& expected) {
    T scalar = pybind::castOrTypeError<T>(value, context, expected, true);
    std::vector<uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &scalar, sizeof(T));
    return bytes;
}

std::vector<uint8_t> scalarBytesFromPython(nb::handle value, ThorImplementation::DataType dataType, const std::string& context) {
    switch (dataType) {
        case ThorImplementation::DataType::INT32:
            return scalarBytesFromPython<int32_t>(value, context, "int32-compatible integer");
        case ThorImplementation::DataType::UINT32:
            return scalarBytesFromPython<uint32_t>(value, context, "uint32-compatible integer");
        case ThorImplementation::DataType::INT64:
            return scalarBytesFromPython<int64_t>(value, context, "int64-compatible integer");
        case ThorImplementation::DataType::UINT64:
            return scalarBytesFromPython<uint64_t>(value, context, "uint64-compatible integer");
        default:
            throw nb::value_error((context + " supports only int32, uint32, int64, or uint64 metadata types").c_str());
    }
}

template <typename ArrayT>
uint64_t requireOneDimensionalCount(const ArrayT& array, const std::string& context) {
    if (array.ndim() != 1) {
        throw nb::value_error((context + " must be a 1D C-contiguous numpy array").c_str());
    }
    return static_cast<uint64_t>(array.shape(0));
}

struct TypedArrayPointer {
    const void* data = nullptr;
    uint64_t count = 0;
};

TypedArrayPointer typedOneDimensionalArrayPointer(nb::handle value,
                                                  ThorImplementation::DataType dataType,
                                                  const std::string& context,
                                                  std::vector<Int32Array>& int32Arrays,
                                                  std::vector<UInt32Array>& uint32Arrays,
                                                  std::vector<Int64Array>& int64Arrays,
                                                  std::vector<UInt64Array>& uint64Arrays) {
    switch (dataType) {
        case ThorImplementation::DataType::INT32: {
            Int32Array array = pybind::castOrTypeError<Int32Array>(value, context, "a C-contiguous numpy.int32 array", false);
            const uint64_t count = requireOneDimensionalCount(array, context);
            int32Arrays.push_back(array);
            return TypedArrayPointer{int32Arrays.back().data(), count};
        }
        case ThorImplementation::DataType::UINT32: {
            UInt32Array array = pybind::castOrTypeError<UInt32Array>(value, context, "a C-contiguous numpy.uint32 array", false);
            const uint64_t count = requireOneDimensionalCount(array, context);
            uint32Arrays.push_back(array);
            return TypedArrayPointer{uint32Arrays.back().data(), count};
        }
        case ThorImplementation::DataType::INT64: {
            Int64Array array = pybind::castOrTypeError<Int64Array>(value, context, "a C-contiguous numpy.int64 array", false);
            const uint64_t count = requireOneDimensionalCount(array, context);
            int64Arrays.push_back(array);
            return TypedArrayPointer{int64Arrays.back().data(), count};
        }
        case ThorImplementation::DataType::UINT64: {
            UInt64Array array = pybind::castOrTypeError<UInt64Array>(value, context, "a C-contiguous numpy.uint64 array", false);
            const uint64_t count = requireOneDimensionalCount(array, context);
            uint64Arrays.push_back(array);
            return TypedArrayPointer{uint64Arrays.back().data(), count};
        }
        default:
            throw nb::value_error((context + " supports only int32, uint32, int64, or uint64 metadata arrays").c_str());
    }
}

std::map<std::string, DatasetWriter::TensorView> datasetWriterTensorViewsFromPython(
    const nb::dict& tensors,
    const DatasetLayout& layout,
    const std::string& context,
    std::vector<nb::object>& ownedArrays) {
    if (nb::len(tensors) == 0) {
        throw nb::value_error((context + " tensors must contain at least one tensor").c_str());
    }

    std::set<std::string> expectedNames;
    for (const DatasetLayout::TensorSpec& spec : layout.tensors()) {
        expectedNames.insert(spec.name);
    }
    for (const DatasetLayout::WindowedTensorSpec& spec : layout.windowedTensors()) {
        expectedNames.insert(spec.name);
    }
    for (auto item : tensors) {
        const std::string name = tensorNameFromPythonKey(item.first, context);
        if (!expectedNames.contains(name)) {
            throw nb::value_error((context + " unexpected tensor name '" + name + "'").c_str());
        }
    }

    std::map<std::string, DatasetWriter::TensorView> views;
    ownedArrays.reserve(layout.tensors().size());
    for (const DatasetLayout::TensorSpec& spec : layout.tensors()) {
        const std::string& name = spec.name;
        if (!dictContainsString(tensors, name)) {
            throw std::runtime_error(context + " missing tensor '" + name + "'");
        }
        nb::object valueObject = dictGetString(tensors, name, context);
        const std::string tensorContext = context + "['" + name + "']";
        const pybind::CanonicalNumpyArrayView array =
            pybind::canonicalNumpyArrayViewNoCopy(valueObject, tensorContext, spec.dataType);
        const uint64_t numBytes = array.size * pybind::thorStorageDataTypeSizeBytes(spec.dataType);
        ownedArrays.push_back(std::move(valueObject));
        auto [it, inserted] = views.emplace(
            name,
            DatasetWriter::TensorView{
                spec.dataType,
                array.dimensions,
                array.data,
                numBytes,
            });
        (void)it;
        if (!inserted) {
            throw nb::value_error((context + " duplicate tensor name '" + name + "'").c_str());
        }
    }
    return views;
}

std::map<std::string, DatasetWriter::TensorBatchView> datasetWriterTensorBatchViewsFromPython(
    const nb::dict& tensors,
    const DatasetLayout& layout,
    const std::string& context,
    std::vector<nb::object>& ownedArrays) {
    if (nb::len(tensors) == 0) {
        throw nb::value_error((context + " tensors must contain at least one tensor").c_str());
    }

    std::set<std::string> expectedNames;
    for (const DatasetLayout::TensorSpec& spec : layout.tensors()) {
        expectedNames.insert(spec.name);
    }
    for (const DatasetLayout::WindowedTensorSpec& spec : layout.windowedTensors()) {
        expectedNames.insert(spec.name);
    }
    for (auto item : tensors) {
        const std::string name = tensorNameFromPythonKey(item.first, context);
        if (!expectedNames.contains(name)) {
            throw nb::value_error((context + " unexpected tensor name '" + name + "'").c_str());
        }
    }

    std::map<std::string, DatasetWriter::TensorBatchView> views;
    ownedArrays.reserve(layout.tensors().size());
    for (const DatasetLayout::TensorSpec& spec : layout.tensors()) {
        const std::string& name = spec.name;
        if (!dictContainsString(tensors, name)) {
            throw std::runtime_error(context + " missing tensor '" + name + "'");
        }
        nb::object valueObject = dictGetString(tensors, name, context);
        const std::string tensorContext = context + "['" + name + "']";
        const pybind::CanonicalNumpyArrayView array =
            pybind::canonicalNumpyArrayViewNoCopy(valueObject, tensorContext, spec.dataType);
        const uint64_t numBytes = array.size * pybind::thorStorageDataTypeSizeBytes(spec.dataType);
        ownedArrays.push_back(std::move(valueObject));
        auto [it, inserted] = views.emplace(
            name,
            DatasetWriter::TensorBatchView{
                spec.dataType,
                array.dimensions,
                array.data,
                numBytes,
            });
        (void)it;
        if (!inserted) {
            throw nb::value_error((context + " duplicate tensor name '" + name + "'").c_str());
        }
    }
    return views;
}

std::map<std::string, DatasetWriter::WindowedTensorReferenceView> datasetWriterWindowedTensorReferenceViewsFromPython(
    const nb::dict& tensors,
    const DatasetLayout& layout,
    const std::string& context,
    std::vector<std::vector<uint8_t>>& ownedScalars) {
    std::map<std::string, DatasetWriter::WindowedTensorReferenceView> views;
    ownedScalars.reserve(layout.windowedTensors().size() * 2);
    for (const DatasetLayout::WindowedTensorSpec& spec : layout.windowedTensors()) {
        if (!dictContainsString(tensors, spec.name)) {
            throw std::runtime_error(context + " missing windowed tensor reference '" + spec.name + "'");
        }
        nb::object chunkObject = dictGetString(tensors, spec.name, context);
        const std::string chunkContext = context + "['" + spec.name + "']";
        if (!pythonHasAttribute(chunkObject, "key") || !pythonHasAttribute(chunkObject, "start")) {
            throw nb::type_error((chunkContext + " must be thor.data.WindowedTensorChunk").c_str());
        }
        ownedScalars.push_back(scalarBytesFromPython(chunkObject.attr("key"), spec.keyDataType, chunkContext + ".key"));
        const uint8_t* keyBytes = ownedScalars.back().data();
        ownedScalars.push_back(scalarBytesFromPython(chunkObject.attr("start"), spec.indexDataType, chunkContext + ".start"));
        const uint8_t* startBytes = ownedScalars.back().data();
        views.emplace(spec.name,
                      DatasetWriter::WindowedTensorReferenceView{.keyDataType = spec.keyDataType,
                                                                                  .indexDataType = spec.indexDataType,
                                                                                  .key = keyBytes,
                                                                                  .start = startBytes});
    }
    return views;
}

std::map<std::string, DatasetWriter::WindowedTensorReferenceBatchView> datasetWriterWindowedTensorReferenceBatchViewsFromPython(
    const nb::dict& tensors,
    const DatasetLayout& layout,
    const std::string& context,
    std::vector<Int32Array>& int32Arrays,
    std::vector<UInt32Array>& uint32Arrays,
    std::vector<Int64Array>& int64Arrays,
    std::vector<UInt64Array>& uint64Arrays) {
    std::map<std::string, DatasetWriter::WindowedTensorReferenceBatchView> views;
    for (const DatasetLayout::WindowedTensorSpec& spec : layout.windowedTensors()) {
        if (!dictContainsString(tensors, spec.name)) {
            throw std::runtime_error(context + " missing windowed tensor reference batch '" + spec.name + "'");
        }
        nb::object chunkObject = dictGetString(tensors, spec.name, context);
        const std::string chunkContext = context + "['" + spec.name + "']";
        if (!pythonHasAttribute(chunkObject, "key") || !pythonHasAttribute(chunkObject, "start")) {
            throw nb::type_error((chunkContext + " must be thor.data.WindowedTensorChunk").c_str());
        }
        TypedArrayPointer keys = typedOneDimensionalArrayPointer(chunkObject.attr("key"),
                                                                spec.keyDataType,
                                                                chunkContext + ".key",
                                                                int32Arrays,
                                                                uint32Arrays,
                                                                int64Arrays,
                                                                uint64Arrays);
        TypedArrayPointer starts = typedOneDimensionalArrayPointer(chunkObject.attr("start"),
                                                                  spec.indexDataType,
                                                                  chunkContext + ".start",
                                                                  int32Arrays,
                                                                  uint32Arrays,
                                                                  int64Arrays,
                                                                  uint64Arrays);
        if (keys.count != starts.count) {
            throw nb::value_error((chunkContext + " key and start arrays must have the same length").c_str());
        }
        views.emplace(spec.name,
                      DatasetWriter::WindowedTensorReferenceBatchView{.keyDataType = spec.keyDataType,
                                                                                       .indexDataType = spec.indexDataType,
                                                                                       .keys = keys.data,
                                                                                       .starts = starts.data,
                                                                                       .count = keys.count});
    }
    return views;
}

std::map<std::string, DatasetWriter::AffineWindowedTensorReferenceView>
datasetWriterAffineWindowedTensorReferenceViewsFromPython(
    const nb::dict& tensors,
    const DatasetLayout& layout,
    const std::string& context,
    std::vector<std::vector<uint8_t>>& ownedKeys) {
    std::map<std::string, DatasetWriter::AffineWindowedTensorReferenceView> views;
    ownedKeys.reserve(layout.windowedTensors().size());
    for (const DatasetLayout::WindowedTensorSpec& spec : layout.windowedTensors()) {
        if (!dictContainsString(tensors, spec.name)) {
            throw std::runtime_error(context + " missing affine window reference '" + spec.name + "'");
        }
        nb::object chunkObject = dictGetString(tensors, spec.name, context);
        const std::string chunkContext = context + "['" + spec.name + "']";
        if (!pythonHasAttribute(chunkObject, "key") || !pythonHasAttribute(chunkObject, "base") ||
            !pythonHasAttribute(chunkObject, "stride") || !pythonHasAttribute(chunkObject, "field_offset")) {
            throw nb::type_error((chunkContext + " must be thor.data.AffineWindowedTensorChunk").c_str());
        }
        ownedKeys.push_back(scalarBytesFromPython(chunkObject.attr("key"), spec.keyDataType, chunkContext + ".key"));
        views.emplace(spec.name,
                      DatasetWriter::AffineWindowedTensorReferenceView{
                          .keyDataType = spec.keyDataType,
                          .key = ownedKeys.back().data(),
                          .base = pybind::castOrTypeError<int64_t>(chunkObject.attr("base"), chunkContext + ".base", "int", false),
                          .stride = pybind::castOrTypeError<int64_t>(chunkObject.attr("stride"), chunkContext + ".stride", "int", false),
                          .fieldOffset = pybind::castOrTypeError<int64_t>(
                              chunkObject.attr("field_offset"), chunkContext + ".field_offset", "int", false)});
    }
    return views;
}

DatasetWriter::WindowedTensorSourceView datasetWriterWindowedTensorSourceViewFromPython(
    const DatasetLayout& layout,
    const std::string& sourceName,
    nb::handle key,
    int64_t startIndex,
    nb::handle values,
    const std::string& context,
    std::vector<uint8_t>& ownedKey,
    std::vector<nb::object>& ownedArrays) {
    const DatasetLayout::WindowedTensorSourceSpec& spec = layout.windowedTensorSource(sourceName);
    ownedKey = scalarBytesFromPython(key, spec.keyDataType, context + " key");
    nb::object owner = nb::borrow<nb::object>(values);
    const pybind::CanonicalNumpyArrayView array =
        pybind::canonicalNumpyArrayViewNoCopy(owner, context + " values", spec.dataType);
    const uint64_t numBytes = array.size * pybind::thorStorageDataTypeSizeBytes(spec.dataType);
    ownedArrays.push_back(std::move(owner));
    return DatasetWriter::WindowedTensorSourceView{.dataType = spec.dataType,
                                                                    .key = ownedKey.data(),
                                                                    .startIndex = startIndex,
                                                                    .dimensions = array.dimensions,
                                                                    .data = array.data,
                                                                    .numBytes = numBytes};
}

Thor::ExampleIndexSet exampleIndexSetFromPython(
    nb::object value,
    const std::string& context) {
    if (nb::isinstance<Thor::ExampleIndexSet>(value)) {
        return nb::cast<Thor::ExampleIndexSet>(value);
    }
    constexpr uint64_t maxIndex = std::numeric_limits<uint64_t>::max();
    return Thor::ExampleIndexSet(uint64IndicesFromPython(std::move(value), context, maxIndex, true));
}

std::set<std::string> stringSetFromVector(std::vector<std::string> values) { return std::set<std::string>(values.begin(), values.end()); }

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

DeviceDatasetStorage deviceDatasetStorageFromPython(nb::object value, const std::string& argumentName) {
    DeviceDatasetStorage storage{};
    if (nb::try_cast<DeviceDatasetStorage>(value, storage, false)) {
        return storage;
    }
    if (nb::isinstance<nb::str>(value)) {
        const std::string text = pybind::castOrTypeError<std::string>(value, argumentName, "str", false);
        try {
            return deviceDatasetStorageFromName(text);
        } catch (const std::runtime_error& e) {
            throw nb::value_error(e.what());
        }
    }
    throw nb::type_error(
        (argumentName +
         " must be a thor.data.DeviceDatasetStorage value or one of: "
         "'best_effort', 'strict', 'strict_windowed_only', 'off'.")
            .c_str());
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

std::vector<nb::object> callbackRefsFromObject(nb::handle object);

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
    std::vector<TrainingRunsSpec> out;
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
        out.push_back(std::move(spec));
    }
    return out;
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


class PythonModelSelectionCallbackState : public TrainingModelSelectionScore::CallbackLifetimeAnchor {
   public:
    explicit PythonModelSelectionCallbackState(nb::handle callable) : callable(callable.ptr()) {
        nb::gil_scoped_acquire gil;
        Py_XINCREF(this->callable);
    }

    PythonModelSelectionCallbackState(const PythonModelSelectionCallbackState&) = delete;
    PythonModelSelectionCallbackState& operator=(const PythonModelSelectionCallbackState&) = delete;

    ~PythonModelSelectionCallbackState() override { clear(); }

    nb::handle get() const { return nb::handle(callable); }

    void clear() {
        if (callable == nullptr) {
            return;
        }
        nb::gil_scoped_acquire gil;
        PyObject* object = callable;
        callable = nullptr;
        Py_DECREF(object);
    }

   private:
    PyObject* callable = nullptr;
};

std::shared_ptr<PythonModelSelectionCallbackState> pythonModelSelectionCallbackState(
    const TrainingModelSelectionScore& score) {
    const std::shared_ptr<TrainingModelSelectionScore::CallbackLifetimeAnchor>& anchor = score.getCallbackLifetimeAnchor();
    if (!anchor) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<PythonModelSelectionCallbackState>(anchor);
}

int visitPythonModelSelectionCallback(const TrainingModelSelectionScore& score, visitproc visit, void* arg) {
    std::shared_ptr<PythonModelSelectionCallbackState> state = pythonModelSelectionCallbackState(score);
    if (state == nullptr) {
        return 0;
    }
    Py_VISIT(state->get().ptr());
    return 0;
}

int trainer_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));
    if (!nb::inst_ready(self)) {
        return 0;
    }
    Trainer* trainer = nb::inst_ptr<Trainer>(self);
    if (trainer == nullptr) {
        return 0;
    }
    return visitPythonModelSelectionCallback(trainer->getModelSelectionScore(), visit, arg);
}

int trainer_tp_clear(PyObject* self) {
    if (!nb::inst_ready(self)) {
        return 0;
    }
    Trainer* trainer = nb::inst_ptr<Trainer>(self);
    if (trainer != nullptr) {
        trainer->clearModelSelectionScoreCallbackLifetimeAnchorForPythonGc();
    }
    return 0;
}

int training_runs_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));
    if (!nb::inst_ready(self)) {
        return 0;
    }
    TrainingRuns* trainingRuns = nb::inst_ptr<TrainingRuns>(self);
    if (trainingRuns == nullptr) {
        return 0;
    }
    for (const TrainingRunsSpec& spec : trainingRuns->getRuns()) {
        if (spec.trainer == nullptr) {
            continue;
        }
        nb::handle trainerObject = nb::find(spec.trainer);
        if (trainerObject.is_valid()) {
            // If a Python Trainer wrapper exists, visit it and let Trainer.tp_traverse
            // expose the native-held Python callback. Visiting the callback here too
            // would double-count the same native Python reference in CPython's GC.
            Py_VISIT(trainerObject.ptr());
        } else {
            // A TrainingRuns instance can own a native Trainer after the original
            // Python Trainer wrapper has gone away. In that case, expose the
            // model-selection callback directly because releaseRunsForPythonGc()
            // can clear the native path that owns it.
            int result = visitPythonModelSelectionCallback(spec.trainer->getModelSelectionScore(), visit, arg);
            if (result != 0) {
                return result;
            }
        }
    }
    return 0;
}

int training_runs_tp_clear(PyObject* self) {
    if (!nb::inst_ready(self)) {
        return 0;
    }
    TrainingRuns* trainingRuns = nb::inst_ptr<TrainingRuns>(self);
    if (trainingRuns != nullptr) {
        trainingRuns->releaseRunsForPythonGc();
    }
    return 0;
}

PyType_Slot trainer_type_slots[] = {
    {Py_tp_traverse, reinterpret_cast<void*>(trainer_tp_traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(trainer_tp_clear)},
    {0, nullptr},
};

PyType_Slot training_runs_type_slots[] = {
    {Py_tp_traverse, reinterpret_cast<void*>(training_runs_tp_traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(training_runs_tp_clear)},
    {0, nullptr},
};

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

nb::dict doubleMapToPythonDict(const std::unordered_map<std::string, double>& values) {
    nb::dict out;
    for (const auto& [name, value] : values) {
        out[nb::str(name.c_str())] = value;
    }
    return out;
}

nb::dict modelSelectionPhaseStatsToPythonDict(const TrainingModelSelectionPhaseStats& stats) {
    nb::dict out;
    out["loss"] = optionalDouble(stats.loss);
    out["losses"] = doubleMapToPythonDict(stats.losses);
    out["metrics"] = doubleMapToPythonDict(stats.metrics);
    return out;
}

nb::dict modelSelectionContextToPythonDict(const TrainingModelSelectionContext& context) {
    nb::dict out;
    out["epoch"] = context.epoch;
    out["training_loss"] = optionalDouble(context.train.loss);
    out["train_loss"] = optionalDouble(context.train.loss);
    out["validation_loss"] = optionalDouble(context.validate.loss);
    out["validate_loss"] = optionalDouble(context.validate.loss);

    nb::dict train = modelSelectionPhaseStatsToPythonDict(context.train);
    nb::dict validate = modelSelectionPhaseStatsToPythonDict(context.validate);
    nb::dict test = modelSelectionPhaseStatsToPythonDict(context.test);
    out["train"] = train;
    out["validate"] = validate;
    out["validation"] = validate;
    out["test"] = test;
    return out;
}

std::optional<double> modelSelectionScoreFromPythonResult(nb::object result) {
    if (result.is_none()) {
        return std::nullopt;
    }
    return pybind::castOrTypeError<double>(result, "model_selection_score return value", "float or None", false);
}

TrainingModelSelectionScore trainingModelSelectionScoreFromPython(nb::object modelSelectionScore) {
    if (modelSelectionScore.is_none()) {
        return TrainingModelSelectionScore{};
    }
    if (!PyCallable_Check(modelSelectionScore.ptr())) {
        throw nb::type_error("model_selection_score must be callable");
    }

    auto callbackState = std::make_shared<PythonModelSelectionCallbackState>(modelSelectionScore);
    std::weak_ptr<PythonModelSelectionCallbackState> weakCallbackState = callbackState;

    TrainingModelSelectionScore::ContextScoreFunction scoreFunction =
        [weakCallbackState = std::move(weakCallbackState)](const TrainingModelSelectionContext& context) -> std::optional<double> {
        // Trainer::fit() / TrainingRuns::fit() release the GIL while native training runs.
        // The model-selection callback can also run from a native TrainingRuns worker thread.
        nb::gil_scoped_acquire acquire;
        std::shared_ptr<PythonModelSelectionCallbackState> callbackState = weakCallbackState.lock();
        if (callbackState == nullptr || callbackState->get().ptr() == nullptr) {
            throw std::runtime_error(
                "Python model-selection callback is no longer alive. This is an internal binding lifetime error: "
                "the owning Trainer/TrainingRuns object should retain the callback state while the C++ callback is active.");
        }
        nb::handle callableObject = callbackState->get();
        nb::dict contextObject = modelSelectionContextToPythonDict(context);

        PyObject* resultPtr = PyObject_CallFunctionObjArgs(callableObject.ptr(), contextObject.ptr(), nullptr);
        if (resultPtr == nullptr && PyErr_ExceptionMatches(PyExc_TypeError)) {
            // Backwards compatibility: pre-context callbacks accepted
            // (validation_loss, training_loss, epoch). Retry that contract
            // when the one-argument context call raises TypeError.
            PyErr_Clear();
            nb::object validationLoss = optionalDouble(context.validate.loss);
            nb::object trainingLoss = optionalDouble(context.train.loss);
            nb::object epoch = nb::cast(context.epoch);
            resultPtr = PyObject_CallFunctionObjArgs(
                callableObject.ptr(), validationLoss.ptr(), trainingLoss.ptr(), epoch.ptr(), nullptr);
        }
        if (resultPtr == nullptr) {
            throw nb::python_error();
        }

        nb::object result = nb::steal<nb::object>(nb::handle(resultPtr));
        return modelSelectionScoreFromPythonResult(std::move(result));
    };

    return TrainingModelSelectionScore{std::move(scoreFunction), std::move(callbackState)};
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

    auto batch_session = nb::class_<Thor::BatchSession>(training, "BatchSession");
    batch_session.attr("__module__") = "thor.data";
    batch_session.def("get_batch_size", &Thor::BatchSession::getBatchSize);
    batch_session.def("get_dataset_name", &Thor::BatchSession::getDatasetName);
    batch_session.def("cancel", &Thor::BatchSession::cancel);
    batch_session.def("get_num_train_examples",
                      [](Thor::BatchSession &self) { return self.getNumExamples(ExampleType::TRAIN); });
    batch_session.def("get_num_validate_examples",
                      [](Thor::BatchSession &self) { return self.getNumExamples(ExampleType::VALIDATE); });
    batch_session.def("get_num_test_examples",
                      [](Thor::BatchSession &self) { return self.getNumExamples(ExampleType::TEST); });
    batch_session.def("get_num_train_batches",
                      [](Thor::BatchSession &self) { return self.getNumBatchesPerEpoch(ExampleType::TRAIN); });
    batch_session.def("get_num_validate_batches",
                      [](Thor::BatchSession &self) { return self.getNumBatchesPerEpoch(ExampleType::VALIDATE); });
    batch_session.def("get_num_test_batches",
                      [](Thor::BatchSession &self) { return self.getNumBatchesPerEpoch(ExampleType::TEST); });

    auto constant_pad = nb::class_<PythonConstantPad>(training, "ConstantPad");
    constant_pad.attr("__module__") = "thor.data";
    constant_pad.def(nb::init<double>(), "value"_a = 0.0)
        .def_ro("value", &PythonConstantPad::value);

    auto tensor_layout = nb::class_<PythonTensorLayout>(training, "TensorLayout");
    tensor_layout.attr("__module__") = "thor.data";
    tensor_layout.def_static(
        "__new__",
        [](nb::handle cls, nb::object shape, ThorImplementation::DataType dataType) -> std::shared_ptr<PythonTensorLayout> {
            (void)cls;
            auto out = std::make_shared<PythonTensorLayout>();
            out->shape = uint64ShapeFromPython(shape, "TensorLayout shape");
            out->dataType = dataType;
            return out;
        },
        "cls"_a,
        "shape"_a,
        "data_type"_a = ThorImplementation::DataType::FP32);
    tensor_layout.def("__init__",
                      [](PythonTensorLayout*, nb::object, ThorImplementation::DataType) {},
                      "shape"_a,
                      "data_type"_a = ThorImplementation::DataType::FP32)
        .def_ro("shape", &PythonTensorLayout::shape)
        .def_ro("data_type", &PythonTensorLayout::dataType);

    auto windowed_tensor_source_layout =
        nb::class_<PythonWindowedTensorSourceLayout>(training, "WindowedTensorSourceLayout");
    windowed_tensor_source_layout.attr("__module__") = "thor.data";
    windowed_tensor_source_layout.def_static(
        "__new__",
        [](nb::handle cls,
           nb::object stepShape,
           ThorImplementation::DataType dataType,
           ThorImplementation::DataType keyType) -> std::shared_ptr<PythonWindowedTensorSourceLayout> {
            (void)cls;
            auto out = std::make_shared<PythonWindowedTensorSourceLayout>();
            out->stepShape = uint64OptionalShapeFromPython(stepShape, "WindowedTensorSourceLayout step_shape");
            out->dataType = dataType;
            out->keyType = keyType;
            return out;
        },
        "cls"_a,
        "step_shape"_a,
        "data_type"_a = ThorImplementation::DataType::FP32,
        "key_type"_a = ThorImplementation::DataType::UINT64,
        R"nbdoc(
Describe one immutable named source for windowed dataset fields.

``step_shape`` is the shape of one source step and may be empty for scalar
sources. Multiple ``WindowedTensorLayout`` fields may reference the same source
without duplicating its persisted bytes.
        )nbdoc");
    windowed_tensor_source_layout.def(
        "__init__",
        [](PythonWindowedTensorSourceLayout*, nb::object, ThorImplementation::DataType, ThorImplementation::DataType) {},
        "step_shape"_a,
        "data_type"_a = ThorImplementation::DataType::FP32,
        "key_type"_a = ThorImplementation::DataType::UINT64)
        .def_ro("step_shape", &PythonWindowedTensorSourceLayout::stepShape)
        .def_ro("data_type", &PythonWindowedTensorSourceLayout::dataType)
        .def_ro("key_type", &PythonWindowedTensorSourceLayout::keyType);

    auto windowed_tensor_layout = nb::class_<PythonWindowedTensorLayout>(training, "WindowedTensorLayout");
    windowed_tensor_layout.attr("__module__") = "thor.data";
    windowed_tensor_layout.def_static(
        "__new__",
        [](nb::handle cls,
           nb::object shape,
           std::string source,
           ThorImplementation::DataType indexType,
           nb::object pad,
           nb::object maskName,
           std::string referenceMode) -> std::shared_ptr<PythonWindowedTensorLayout> {
            (void)cls;
            if (source.empty()) {
                throw nb::value_error("WindowedTensorLayout source must not be empty");
            }
            auto out = std::make_shared<PythonWindowedTensorLayout>();
            out->shape = uint64ShapeFromPython(shape, "WindowedTensorLayout shape");
            out->source = std::move(source);
            out->indexType = indexType;
            out->padValue = padValueFromPython(pad, "WindowedTensorLayout pad");
            out->maskName = optionalStringFromPython(maskName, "WindowedTensorLayout mask_name");
            out->referenceMode = windowedReferenceModeFromPython(referenceMode, "WindowedTensorLayout reference_mode");
            return out;
        },
        "cls"_a,
        "shape"_a,
        "source"_a,
        "index_type"_a = ThorImplementation::DataType::INT32,
        "pad"_a = nb::none(),
        "mask_name"_a = nb::none(),
        "reference_mode"_a = "indexed",
        R"nbdoc(
Describe one field materialized as a fixed window over a named source.

The first dimension of ``shape`` is the window length. Its remaining dimensions
must equal the referenced source's ``step_shape``. ``reference_mode="indexed"``
uses per-row ``WindowedTensorChunk`` values. ``reference_mode="affine"`` uses
one ``AffineWindowedTensorChunk`` formula for each appended row segment.
        )nbdoc");
    windowed_tensor_layout.def(
        "__init__",
        [](PythonWindowedTensorLayout*, nb::object, std::string, ThorImplementation::DataType, nb::object, nb::object, std::string) {},
        "shape"_a,
        "source"_a,
        "index_type"_a = ThorImplementation::DataType::INT32,
        "pad"_a = nb::none(),
        "mask_name"_a = nb::none(),
        "reference_mode"_a = "indexed")
        .def_ro("shape", &PythonWindowedTensorLayout::shape)
        .def_ro("source", &PythonWindowedTensorLayout::source)
        .def_ro("index_type", &PythonWindowedTensorLayout::indexType)
        .def_ro("pad_value", &PythonWindowedTensorLayout::padValue)
        .def_ro("mask_name", &PythonWindowedTensorLayout::maskName)
        .def_prop_ro("reference_mode", [](const PythonWindowedTensorLayout& self) {
            return self.referenceMode == DatasetLayout::WindowedTensorReferenceMode::AFFINE ? "affine" : "indexed";
        });

    auto windowed_tensor_chunk = nb::class_<PythonWindowedTensorChunk>(training, "WindowedTensorChunk");
    windowed_tensor_chunk.attr("__module__") = "thor.data";
    windowed_tensor_chunk.def_static(
        "__new__",
        [](nb::handle cls, nb::object key, nb::object start) -> std::shared_ptr<PythonWindowedTensorChunk> {
            (void)cls;
            auto out = std::make_shared<PythonWindowedTensorChunk>();
            out->key = std::move(key);
            out->start = std::move(start);
            return out;
        },
        "cls"_a,
        "key"_a,
        "start"_a);
    windowed_tensor_chunk.def("__init__", [](PythonWindowedTensorChunk*, nb::object, nb::object) {}, "key"_a, "start"_a)
        .def_ro("key", &PythonWindowedTensorChunk::key)
        .def_ro("start", &PythonWindowedTensorChunk::start);

    auto affine_windowed_tensor_chunk =
        nb::class_<PythonAffineWindowedTensorChunk>(training, "AffineWindowedTensorChunk");
    affine_windowed_tensor_chunk.attr("__module__") = "thor.data";
    affine_windowed_tensor_chunk.def_static(
        "__new__",
        [](nb::handle cls, nb::object key, int64_t base, int64_t stride, int64_t fieldOffset)
            -> std::shared_ptr<PythonAffineWindowedTensorChunk> {
            (void)cls;
            if (stride <= 0) {
                throw nb::value_error("AffineWindowedTensorChunk stride must be >= 1");
            }
            auto out = std::make_shared<PythonAffineWindowedTensorChunk>();
            out->key = std::move(key);
            out->base = base;
            out->stride = stride;
            out->fieldOffset = fieldOffset;
            return out;
        },
        "cls"_a,
        "key"_a,
        "base"_a = 0,
        "stride"_a = 1,
        "field_offset"_a = 0);
    affine_windowed_tensor_chunk.def(
        "__init__",
        [](PythonAffineWindowedTensorChunk*, nb::object, int64_t, int64_t, int64_t) {},
        "key"_a,
        "base"_a = 0,
        "stride"_a = 1,
        "field_offset"_a = 0)
        .def_ro("key", &PythonAffineWindowedTensorChunk::key)
        .def_ro("base", &PythonAffineWindowedTensorChunk::base)
        .def_ro("stride", &PythonAffineWindowedTensorChunk::stride)
        .def_ro("field_offset", &PythonAffineWindowedTensorChunk::fieldOffset);

    auto dataset_layout = nb::class_<DatasetLayout>(training, "DatasetLayout");
    dataset_layout.attr("__module__") = "thor.data";
    dataset_layout.def_static(
        "__new__",
        [](nb::handle cls,
           nb::dict tensors,
           nb::object windowSources,
           nb::object windowedTensors) -> std::shared_ptr<DatasetLayout> {
            (void)cls;
            return std::make_shared<DatasetLayout>(DatasetLayout::fromTensorShapes(
                datasetTensorShapesFromPython(tensors, "DatasetLayout tensors"),
                datasetWindowedTensorSourceShapesFromPython(windowSources, "DatasetLayout"),
                datasetWindowedTensorShapesFromPython(windowedTensors, "DatasetLayout")));
        },
        "cls"_a,
        "tensors"_a,
        "window_sources"_a = nb::none(),
        "windowed_tensors"_a = nb::none(),
        R"nbdoc(
Define a persistent dataset record layout.

``tensors`` maps dense field names to ``TensorLayout``. ``window_sources`` maps
independent source names to ``WindowedTensorSourceLayout``. ``windowed_tensors``
maps output field names to ``WindowedTensorLayout`` and may point several fields
at the same source. Every dense field and every window source owns its dtype.
        )nbdoc");
    dataset_layout.def(
        "__init__",
        [](DatasetLayout*, nb::dict, nb::object, nb::object) {},
        "tensors"_a,
        "window_sources"_a = nb::none(),
        "windowed_tensors"_a = nb::none());
    dataset_layout.def_static(
        "read_manifest",
        [](nb::object path) { return DatasetLayout::readManifest(pathStringFromPython(path, "path")); },
        "path"_a);
    dataset_layout.def("write_manifest",
                       [](const DatasetLayout& self, nb::object path) {
                           self.writeManifest(pathStringFromPython(path, "path"));
                       },
                       "path"_a);
    dataset_layout.def("validate", &DatasetLayout::validate);
    dataset_layout.def("validate_requested_layout_exact", &DatasetLayout::validateRequestedLayoutExact, "requested"_a);
    dataset_layout.def("get_record_size_bytes", &DatasetLayout::recordSizeBytes);
    dataset_layout.def("get_tensor_names", &datasetLayoutTensorNames);
    dataset_layout.def("get_tensor_shapes", &datasetLayoutTensorShapes);
    dataset_layout.def("get_tensor_specs", &datasetLayoutTensorSpecsToPythonDict);
    dataset_layout.def("has_windowed_tensors", &DatasetLayout::hasWindowedTensors);
    dataset_layout.def("get_window_source_specs", &datasetLayoutWindowSourceSpecsToPythonDict);
    dataset_layout.def("get_windowed_tensor_specs", &datasetLayoutWindowedTensorSpecsToPythonDict);

    auto dataset_id = nb::class_<Thor::DatasetId>(training, "DatasetId");
    dataset_id.attr("__module__") = "thor.data";
    dataset_id.def_prop_ro("value", &Thor::DatasetId::str, nb::rv_policy::reference_internal);
    dataset_id.def("__str__", [](const Thor::DatasetId& self) { return self.str(); });
    dataset_id.def("__repr__", [](const Thor::DatasetId& self) { return "DatasetId('" + self.str() + "')"; });
    dataset_id.def("__eq__", [](const Thor::DatasetId& self, const Thor::DatasetId& other) { return self == other; });

    auto dataset_field_kind = nb::enum_<Thor::DatasetFieldKind>(training, "DatasetFieldKind")
                                  .value("DENSE", Thor::DatasetFieldKind::DENSE)
                                  .value("WINDOWED", Thor::DatasetFieldKind::WINDOWED)
                                  .value("WINDOW_MASK", Thor::DatasetFieldKind::WINDOW_MASK);
    dataset_field_kind.attr("__module__") = "thor.data";

    auto dataset_field = nb::class_<Thor::DatasetField>(training, "DatasetField");
    dataset_field.attr("__module__") = "thor.data";
    dataset_field.def_ro("id", &Thor::DatasetField::id);
    dataset_field.def_ro("name", &Thor::DatasetField::name);
    dataset_field.def_ro("data_type", &Thor::DatasetField::dataType);
    dataset_field.def_prop_ro("dtype", [](const Thor::DatasetField& self) { return self.dataType; });
    dataset_field.def_ro("dimensions", &Thor::DatasetField::dimensions);
    dataset_field.def_prop_ro("shape", [](const Thor::DatasetField& self) { return self.dimensions; });
    dataset_field.def_ro("kind", &Thor::DatasetField::kind);

    auto dataset_schema = nb::class_<Thor::DatasetSchema>(training, "DatasetSchema");
    dataset_schema.attr("__module__") = "thor.data";
    dataset_schema.def("__len__", &Thor::DatasetSchema::size);
    dataset_schema.def(
        "contains",
        [](const Thor::DatasetSchema& self, const std::string& name) { return self.contains(name); },
        "name"_a);
    dataset_schema.def(
        "field",
        [](const Thor::DatasetSchema& self, const std::string& name) -> const Thor::DatasetField& {
            return self.getField(name);
        },
        "name"_a,
        nb::rv_policy::reference_internal);
    dataset_schema.def(
        "__getitem__",
        [](const Thor::DatasetSchema& self, const std::string& name) -> const Thor::DatasetField& {
            return self.getField(name);
        },
        "name"_a,
        nb::rv_policy::reference_internal);
    dataset_schema.def_prop_ro("fields", [](const Thor::DatasetSchema& self) { return self.getFields(); });
    dataset_schema.def_prop_ro("names", [](const Thor::DatasetSchema& self) {
        std::vector<std::string> names;
        names.reserve(self.getFields().size());
        for (const Thor::DatasetField& field : self.getFields()) {
            names.push_back(field.name);
        }
        return names;
    });

    auto named_dataset = nb::class_<Thor::NamedDataset>(training, "NamedDataset");
    named_dataset.attr("__module__") = "thor.data";
    named_dataset.def_prop_ro("id", &Thor::NamedDataset::getId, nb::rv_policy::reference_internal);
    named_dataset.def_prop_ro("num_examples", &Thor::NamedDataset::getNumExamples);
    named_dataset.def_prop_ro("schema", &Thor::NamedDataset::getSchema, nb::rv_policy::reference_internal);
    named_dataset.def(
        "field",
        [](const Thor::NamedDataset& self, const std::string& name) -> const Thor::DatasetField& {
            return self.getField(name);
        },
        "name"_a,
        nb::rv_policy::reference_internal);

    Thor::PythonBindings::bindNumpyDataset(training);

    auto file_dataset = nb::class_<Thor::FileDataset, Thor::NamedDataset>(training, "FileDataset");
    file_dataset.attr("__module__") = "thor.data";
    file_dataset.def_static(
        "open",
        [](nb::object datasetPath) {
            return Thor::FileDataset::open(pathStringFromPython(datasetPath, "dataset_path"));
        },
        "dataset_path"_a);
    file_dataset.def_prop_ro("dataset_path", [](const Thor::FileDataset& self) { return self.getPath().string(); });
    file_dataset.def("assert_schema", &Thor::FileDataset::assertSchema, "expected_schema"_a);

    auto example_index_range = nb::class_<Thor::ExampleIndexRange>(training, "ExampleIndexRange");
    example_index_range.attr("__module__") = "thor.data";
    example_index_range.def_static(
        "__new__",
        [](nb::handle cls, uint64_t start, uint64_t count, uint64_t stride) -> std::shared_ptr<Thor::ExampleIndexRange> {
            (void)cls;
            if (count == 0) {
                throw nb::value_error("ExampleIndexRange count must be >= 1");
            }
            if (stride == 0) {
                throw nb::value_error("ExampleIndexRange stride must be >= 1");
            }
            Thor::ExampleIndexRange range{.start = start, .count = count, .stride = stride};
            (void)range.last();
            return std::make_shared<Thor::ExampleIndexRange>(std::move(range));
        },
        "cls"_a,
        "start"_a,
        "count"_a,
        "stride"_a = 1);
    example_index_range.def("__init__", [](Thor::ExampleIndexRange*, uint64_t, uint64_t, uint64_t) {},
                            "start"_a,
                            "count"_a,
                            "stride"_a = 1)
        .def_ro("start", &Thor::ExampleIndexRange::start)
        .def_ro("count", &Thor::ExampleIndexRange::count)
        .def_ro("stride", &Thor::ExampleIndexRange::stride)
        .def("__eq__", [](const Thor::ExampleIndexRange& self, const Thor::ExampleIndexRange& other) {
            return self == other;
        });

    auto example_index_set = nb::class_<Thor::ExampleIndexSet>(training, "ExampleIndexSet");
    example_index_set.attr("__module__") = "thor.data";
    example_index_set.def_static(
        "from_indices",
        [](nb::object indices) {
            constexpr uint64_t maxIndex = std::numeric_limits<uint64_t>::max();
            return Thor::ExampleIndexSet(
                uint64IndicesFromPython(std::move(indices), "ExampleIndexSet indices", maxIndex, true));
        },
        "indices"_a);
    example_index_set.def_static(
        "from_ranges",
        [](nb::iterable ranges) {
            std::vector<Thor::ExampleIndexRange> values;
            for (nb::handle range : ranges) {
                values.push_back(pybind::castOrTypeError<Thor::ExampleIndexRange>(
                    range, "ExampleIndexSet ranges", "thor.data.ExampleIndexRange", false));
            }
            return Thor::ExampleIndexSet(std::move(values));
        },
        "ranges"_a);
    example_index_set.def_static("contiguous", &Thor::ExampleIndexSet::contiguous, "start"_a, "count"_a);
    example_index_set.def_static("strided", &Thor::ExampleIndexSet::strided, "start"_a, "count"_a, "stride"_a);
    example_index_set.def("__len__", &Thor::ExampleIndexSet::size);
    example_index_set.def(
        "__getitem__",
        [](const Thor::ExampleIndexSet& self, int64_t index) {
            int64_t resolved = index;
            if (resolved < 0) {
                if (self.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    throw nb::index_error();
                }
                resolved += static_cast<int64_t>(self.size());
            }
            if (resolved < 0 || static_cast<uint64_t>(resolved) >= self.size()) {
                throw nb::index_error();
            }
            return self.at(static_cast<uint64_t>(resolved));
        },
        "index"_a);
    example_index_set.def_prop_ro("indices", &Thor::ExampleIndexSet::materialize);
    example_index_set.def_prop_ro("ranges", [](const Thor::ExampleIndexSet& self) {
        return self.isRangeBacked() ? self.getRanges() : std::vector<Thor::ExampleIndexRange>{};
    });
    example_index_set.def_prop_ro("is_range_backed", &Thor::ExampleIndexSet::isRangeBacked);
    example_index_set.def("__eq__", [](const Thor::ExampleIndexSet& self, const Thor::ExampleIndexSet& other) {
        return self == other;
    });

    auto dataset_split_manifest = nb::class_<Thor::DatasetSplitManifest>(training, "DatasetSplitManifest");
    dataset_split_manifest.attr("__module__") = "thor.data";
    dataset_split_manifest.def_static(
        "__new__",
        [](nb::handle cls,
           std::shared_ptr<Thor::NamedDataset> dataset,
           nb::object trainIndices,
           nb::object validateIndices,
           nb::object testIndices) -> std::shared_ptr<Thor::DatasetSplitManifest> {
            (void)cls;
            if (dataset == nullptr) {
                throw nb::value_error("DatasetSplitManifest dataset must not be None");
            }
            Thor::ExampleIndexSet train = exampleIndexSetFromPython(
                std::move(trainIndices), "DatasetSplitManifest train_indices");
            Thor::ExampleIndexSet validate = exampleIndexSetFromPython(
                std::move(validateIndices), "DatasetSplitManifest validate_indices");
            std::optional<Thor::ExampleIndexSet> test;
            if (!testIndices.is_none()) {
                test = exampleIndexSetFromPython(
                    std::move(testIndices), "DatasetSplitManifest test_indices");
            }
            return std::make_shared<Thor::DatasetSplitManifest>(
                *dataset, std::move(train), std::move(validate), std::move(test));
        },
        "cls"_a,
        "dataset"_a,
        "train_indices"_a,
        "validate_indices"_a,
        "test_indices"_a = nb::none());
    dataset_split_manifest.def(
        "__init__",
        [](Thor::DatasetSplitManifest*, std::shared_ptr<Thor::NamedDataset>, nb::object, nb::object, nb::object) {},
        "dataset"_a,
        "train_indices"_a,
        "validate_indices"_a,
        "test_indices"_a = nb::none());
    dataset_split_manifest.def_static(
        "load",
        [](nb::object path) { return Thor::DatasetSplitManifest::load(pathStringFromPython(path, "path")); },
        "path"_a);
    dataset_split_manifest.def(
        "save",
        [](const Thor::DatasetSplitManifest& self, nb::object path) {
            self.save(pathStringFromPython(path, "path"));
        },
        "path"_a);
    dataset_split_manifest.def(
        "validate_against",
        [](const Thor::DatasetSplitManifest& self, const Thor::NamedDataset& dataset) {
            self.validateAgainst(dataset);
        },
        "dataset"_a);
    dataset_split_manifest.def_prop_ro("dataset_id", &Thor::DatasetSplitManifest::getDatasetId,
                                       nb::rv_policy::reference_internal);
    dataset_split_manifest.def_prop_ro("num_examples", &Thor::DatasetSplitManifest::getNumExamples);
    dataset_split_manifest.def_prop_ro("train", &Thor::DatasetSplitManifest::getTrain,
                                       nb::rv_policy::reference_internal);
    dataset_split_manifest.def_prop_ro("validate", &Thor::DatasetSplitManifest::getValidate,
                                       nb::rv_policy::reference_internal);
    dataset_split_manifest.def_prop_ro("test", &Thor::DatasetSplitManifest::getTest,
                                       nb::rv_policy::reference_internal);
    dataset_split_manifest.def_prop_ro("has_explicit_test_split", &Thor::DatasetSplitManifest::hasExplicitTestSplit);
    dataset_split_manifest.def_prop_ro("test_aliases_validate", &Thor::DatasetSplitManifest::testAliasesValidate);
    dataset_split_manifest.def("__eq__", [](const Thor::DatasetSplitManifest& self,
                                             const Thor::DatasetSplitManifest& other) { return self == other; });

    auto batch_policy = nb::class_<Thor::BatchPolicy>(training, "BatchPolicy");
    batch_policy.attr("__module__") = "thor.data";
    batch_policy.def(nb::init<uint64_t, bool, std::optional<uint64_t>>(),
                     "batch_size"_a,
                     "randomize_train"_a = true,
                     "random_seed"_a = nb::none());
    batch_policy.def_prop_ro("batch_size", &Thor::BatchPolicy::getBatchSize);
    batch_policy.def_prop_ro("randomize_train", &Thor::BatchPolicy::getRandomizeTrain);
    batch_policy.def_prop_ro("random_seed", &Thor::BatchPolicy::getRandomSeed);

    auto dataset_input_bindings = nb::class_<Thor::DatasetInputBindings>(training, "DatasetInputBindings");
    dataset_input_bindings.attr("__module__") = "thor.training";
    dataset_input_bindings.def(nb::init<>());
    dataset_input_bindings.def(
        "bind",
        [](Thor::DatasetInputBindings& self,
           const Thor::NetworkInput& networkInput,
           const Thor::DatasetField& field) -> Thor::DatasetInputBindings& {
            return self.bind(networkInput, field);
        },
        "network_input"_a,
        "field"_a,
        nb::rv_policy::reference_internal);
    dataset_input_bindings.def_static(
        "by_exact_name",
        &Thor::DatasetInputBindings::byExactName,
        "network"_a,
        "dataset"_a);
    dataset_input_bindings.def("__len__", &Thor::DatasetInputBindings::size);
    dataset_input_bindings.def_prop_ro("empty", &Thor::DatasetInputBindings::empty);

    auto training_data = nb::class_<Thor::TrainingData>(training, "TrainingData", nb::is_weak_referenceable());
    training_data.attr("__module__") = "thor.data";
    training_data.def_static(
        "__new__",
        [](nb::handle cls,
           std::shared_ptr<Thor::NamedDataset> dataset,
           Thor::DatasetSplitManifest splits,
           Thor::BatchPolicy batching,
           std::string datasetName,
           nb::object deviceStorage) -> std::shared_ptr<Thor::TrainingData> {
            (void)cls;
            if (dataset == nullptr) {
                throw nb::value_error("TrainingData dataset must not be None");
            }
            Thor::DatasetAccessPolicy accessPolicy{
                .deviceStorage = deviceDatasetStorageFromPython(
                    std::move(deviceStorage), "device_storage")};
            return std::make_shared<Thor::TrainingData>(
                std::move(dataset),
                std::move(splits),
                std::move(batching),
                accessPolicy,
                std::move(datasetName));
        },
        "cls"_a,
        "dataset"_a,
        "splits"_a,
        "batching"_a,
        "dataset_name"_a = "dataset",
        "device_storage"_a = "off");
    training_data.def(
        "__init__",
        [](Thor::TrainingData*, std::shared_ptr<Thor::NamedDataset>, Thor::DatasetSplitManifest,
           Thor::BatchPolicy, std::string, nb::object) {},
        "dataset"_a,
        "splits"_a,
        "batching"_a,
        "dataset_name"_a = "dataset",
        "device_storage"_a = "off");
    training_data.def(
        "open_session",
        [](const Thor::TrainingData& self, uint64_t maxInFlightBatches) {
            return self.openSession(maxInFlightBatches);
        },
        "max_in_flight_batches"_a = 32);
    training_data.def_prop_ro("dataset", [](const Thor::TrainingData& self) {
        return std::const_pointer_cast<Thor::NamedDataset>(self.getDataset());
    });
    training_data.def_prop_ro("splits", &Thor::TrainingData::getSplits,
                              nb::rv_policy::reference_internal);
    training_data.def_prop_ro("batching", &Thor::TrainingData::getBatching,
                              nb::rv_policy::reference_internal);
    training_data.def_prop_ro("dataset_name", &Thor::TrainingData::getDatasetName,
                              nb::rv_policy::reference_internal);
    training_data.def_prop_ro(
        "device_storage",
        [](const Thor::TrainingData& self) {
            return self.getAccessPolicy().deviceStorage;
        });

    auto dataset_writer = nb::class_<DatasetWriter>(training, "DatasetWriter");
    dataset_writer.attr("__module__") = "thor.data";
    dataset_writer.def_static(
        "__new__",
        [](nb::handle cls,
           nb::object datasetPath,
           DatasetLayout layout,
           uint64_t examplesPerShard,
           nb::object expectedNumExamples,
           bool preallocate) -> std::shared_ptr<DatasetWriter> {
            (void)cls;
            if (examplesPerShard == 0) {
                throw nb::value_error("DatasetWriter examples_per_shard must be >= 1");
            }
            return std::make_shared<DatasetWriter>(pathStringFromPython(datasetPath, "dataset_path"),
                                                                    std::move(layout),
                                                                    examplesPerShard,
                                                                    optionalUint64FromPython(expectedNumExamples, "expected_num_examples"),
                                                                    preallocate);
        },
        "cls"_a,
        "dataset_path"_a,
        "layout"_a,
        "examples_per_shard"_a = 100000,
        "expected_num_examples"_a = nb::none(),
        "preallocate"_a = false);
    dataset_writer.def(
        "__init__",
        [](DatasetWriter*, nb::object, DatasetLayout, uint64_t, nb::object, bool) {},
        "dataset_path"_a,
        "layout"_a,
        "examples_per_shard"_a = 100000,
        "expected_num_examples"_a = nb::none(),
        "preallocate"_a = false);
    dataset_writer.def(
        "write_indexed_example",
        [](DatasetWriter& self, nb::dict tensors) {
            std::vector<nb::object> ownedArrays;
            std::map<std::string, DatasetWriter::TensorView> views =
                datasetWriterTensorViewsFromPython(tensors, self.getLayout(), "DatasetWriter.write_indexed_example", ownedArrays);
            if (self.getLayout().hasWindowedTensors()) {
                std::vector<std::vector<uint8_t>> ownedScalars;
                std::map<std::string, DatasetWriter::WindowedTensorReferenceView> windowedRefs =
                    datasetWriterWindowedTensorReferenceViewsFromPython(
                        tensors, self.getLayout(), "DatasetWriter.write_indexed_example", ownedScalars);
                self.writeIndexedExample(views, windowedRefs);
            } else {
                self.writeIndexedExample(views);
            }
        },
        "tensors"_a);
    dataset_writer.def(
        "write_indexed_examples",
        [](DatasetWriter& self, nb::dict tensors) {
            std::vector<nb::object> ownedArrays;
            std::map<std::string, DatasetWriter::TensorBatchView> views =
                datasetWriterTensorBatchViewsFromPython(tensors, self.getLayout(), "DatasetWriter.write_indexed_examples", ownedArrays);
            if (self.getLayout().hasWindowedTensors()) {
                std::vector<Int32Array> int32Arrays;
                std::vector<UInt32Array> uint32Arrays;
                std::vector<Int64Array> int64Arrays;
                std::vector<UInt64Array> uint64Arrays;
                std::map<std::string, DatasetWriter::WindowedTensorReferenceBatchView> windowedRefs =
                    datasetWriterWindowedTensorReferenceBatchViewsFromPython(tensors,
                                                                               self.getLayout(),
                                                                               "DatasetWriter.write_indexed_examples",
                                                                               int32Arrays,
                                                                               uint32Arrays,
                                                                               int64Arrays,
                                                                               uint64Arrays);
                self.writeIndexedExamples(views, windowedRefs);
            } else {
                self.writeIndexedExamples(views);
            }
        },
        "tensors"_a);
    dataset_writer.def(
        "write_affine_examples",
        [](DatasetWriter& self, uint64_t count, nb::dict tensors) {
            std::vector<nb::object> ownedArrays;
            std::map<std::string, DatasetWriter::TensorBatchView> denseViews =
                datasetWriterTensorBatchViewsFromPython(
                    tensors, self.getLayout(), "DatasetWriter.write_affine_examples", ownedArrays);
            std::vector<std::vector<uint8_t>> ownedKeys;
            std::map<std::string, DatasetWriter::AffineWindowedTensorReferenceView> affineViews =
                datasetWriterAffineWindowedTensorReferenceViewsFromPython(
                    tensors, self.getLayout(), "DatasetWriter.write_affine_examples", ownedKeys);
            self.writeAffineExamples(count, denseViews, affineViews);
        },
        "count"_a,
        "tensors"_a,
        R"nbdoc(
Append one compact affine row segment. For each affine window field,
``start(row) = base + row * stride + field_offset`` within the appended
segment. Dense tensor values, when present, still use arrays with leading
dimension ``count``.
        )nbdoc");
    dataset_writer.def(
        "write_window_source",
        [](DatasetWriter& self, const std::string& sourceName, nb::object key, int64_t startIndex, nb::object values) {
            std::vector<uint8_t> ownedKey;
            std::vector<nb::object> ownedArrays;
            DatasetWriter::WindowedTensorSourceView view = datasetWriterWindowedTensorSourceViewFromPython(
                self.getLayout(),
                sourceName,
                key,
                startIndex,
                values,
                "DatasetWriter.write_window_source",
                ownedKey,
                ownedArrays);
            self.writeWindowSource(sourceName, view);
        },
        "source_name"_a,
        "key"_a,
        "start_index"_a,
        "values"_a,
        R"nbdoc(
Write one keyed contiguous sequence into a named window source.

``values`` must have shape ``[num_steps, *step_shape]`` and use the exact
canonical NumPy/ml_dtypes representation of the source's Thor storage dtype.
Multiple windowed fields may reference the same persisted sequence.
        )nbdoc");
    dataset_writer.def("close", &DatasetWriter::close);
    dataset_writer.def("is_closed", &DatasetWriter::isClosed);
    dataset_writer.def("get_path", [](const DatasetWriter& self) { return self.path().string(); });
    dataset_writer.def("get_manifest_path", [](const DatasetWriter& self) { return self.manifestPath().string(); });
    dataset_writer.def("get_dataset_id", &DatasetWriter::getDatasetId,
                                           nb::rv_policy::reference_internal);
    dataset_writer.def("get_layout", &DatasetWriter::getLayout, nb::rv_policy::copy);
    dataset_writer.def("get_expected_num_examples", [](const DatasetWriter& self) -> nb::object {
        std::optional<uint64_t> expected = self.getExpectedNumExamples();
        if (!expected.has_value()) {
            return nb::none();
        }
        return nb::int_(expected.value());
    });
    dataset_writer.def("get_preallocate", &DatasetWriter::getPreallocate);
    dataset_writer.def("get_num_examples", [](const DatasetWriter& self) { return self.numExamples(); });
    dataset_writer.def("__enter__", [](DatasetWriter& self) -> DatasetWriter& { return self; }, nb::rv_policy::reference_internal);
    dataset_writer.def("__exit__",
                                           [](DatasetWriter& self, nb::object, nb::object, nb::object) {
                                               self.close();
                                               return false;
                                           });

    auto device_dataset_storage = nb::enum_<DeviceDatasetStorage>(training, "DeviceDatasetStorage")
                                      .value("OFF", DeviceDatasetStorage::OFF)
                                      .value("BEST_EFFORT", DeviceDatasetStorage::BEST_EFFORT)
                                      .value("STRICT", DeviceDatasetStorage::STRICT)
                                      .value("STRICT_WINDOWED_ONLY",
                                             DeviceDatasetStorage::STRICT_WINDOWED_ONLY);
    device_dataset_storage.attr("__module__") = "thor.data";

    auto dataset_access_policy = nb::class_<Thor::DatasetAccessPolicy>(training, "DatasetAccessPolicy");
    dataset_access_policy.attr("__module__") = "thor.data";
    dataset_access_policy.def(nb::init<>());
    dataset_access_policy.def_prop_rw(
        "device_storage",
        [](const Thor::DatasetAccessPolicy& self) { return self.deviceStorage; },
        [](Thor::DatasetAccessPolicy& self, nb::object value) {
            self.deviceStorage = deviceDatasetStorageFromPython(std::move(value), "device_storage");
        });

    auto device_dataset_storage_report = nb::class_<DeviceDatasetStorageReport>(training, "DeviceDatasetStorageReport");
    device_dataset_storage_report.attr("__module__") = "thor.training";
    device_dataset_storage_report.def_ro("requested", &DeviceDatasetStorageReport::requested);
    device_dataset_storage_report.def_ro("attempted", &DeviceDatasetStorageReport::attempted);
    device_dataset_storage_report.def_ro("used", &DeviceDatasetStorageReport::used);
    device_dataset_storage_report.def_ro("reason", &DeviceDatasetStorageReport::reason);
    device_dataset_storage_report.def_ro("examples", &DeviceDatasetStorageReport::examples);
    device_dataset_storage_report.def_ro("required_bytes", &DeviceDatasetStorageReport::requiredBytes);
    device_dataset_storage_report.def_ro("available_bytes_after_model_placement",
                                         &DeviceDatasetStorageReport::availableBytesAfterPlacement);
    device_dataset_storage_report.def_ro("resident_bytes", &DeviceDatasetStorageReport::residentBytes);
    device_dataset_storage_report.def_ro("resident_cache_hit", &DeviceDatasetStorageReport::residentCacheHit);
    device_dataset_storage_report.def_ro("resident_construction_joined",
                                         &DeviceDatasetStorageReport::residentConstructionJoined);
    device_dataset_storage_report.def_ro("resident_construction_started",
                                         &DeviceDatasetStorageReport::residentConstructionStarted);
    device_dataset_storage_report.def_ro("materialization_seconds", &DeviceDatasetStorageReport::materializationSeconds);

    auto trainer_fit_options = nb::class_<TrainerFitOptions>(training, "TrainerFitOptions");
    trainer_fit_options.attr("__module__") = "thor.training";
    trainer_fit_options.def(nb::init<>())
        .def_rw("epochs", &TrainerFitOptions::epochs)
        .def_rw("check_best_model_every_epochs", &TrainerFitOptions::checkBestModelEveryEpochs)
        .def_rw("first_model_selection_epoch", &TrainerFitOptions::firstModelSelectionEpoch)
        .def_rw("max_training_batches_per_epoch", &TrainerFitOptions::maxTrainingBatchesPerEpoch);

    auto trainer = nb::class_<Trainer>(training, "Trainer", nb::type_slots(trainer_type_slots));
    trainer.attr("__module__") = "thor.training";
    trainer.def_static(
        "__new__",
        [](nb::handle cls,
           std::shared_ptr<Network> network,
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
           nb::object model_selection_score,
           std::shared_ptr<Thor::TrainingData> data,
           Thor::DatasetInputBindings* input_bindings) -> nb::object {
            (void)cls;
            if (data == nullptr) {
                throw nb::value_error("Trainer requires data");
            }
            TrainingModelSelectionScore modelSelectionScore = trainingModelSelectionScoreFromPython(std::move(model_selection_score));
            Trainer::Builder builder;
            if (network != nullptr) {
                builder.network(std::move(network));
            }
            builder.statsIntervalSeconds(stats_interval_s)
                .statsStderrAlso(stats_stderr_also)
                .statsColorMode(lineStatsColorModeFromString(stats_color))
                .maxInFlightBatches(max_in_flight_batches)
                .scalarTensorsToReport(stringSetFromVector(std::move(scalar_tensors_to_report)))
                .saveModelDirectory(optionalPathStringFromPython(save_model_dir, "save_model_dir"))
                .saveModelOverwrite(save_model_overwrite)
                .modelSelectionScore(std::move(modelSelectionScore));
            builder.data(std::move(data));
            if (input_bindings != nullptr) {
                builder.inputBindings(*input_bindings);
            }
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
            return nb::cast(std::make_shared<Trainer>(builder.build()));
        },
        "cls"_a,
        "network"_a.none() = nb::none(),
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
        "model_selection_score"_a.none() = nb::none(),
        "data"_a.none() = nb::none(),
        "input_bindings"_a.none() = nb::none());
    trainer.def(
        "__init__",
        [](Trainer*,
           std::shared_ptr<Network>,
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
           nb::object,
           std::shared_ptr<Thor::TrainingData>,
           Thor::DatasetInputBindings*) {},
        "network"_a.none() = nb::none(),
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
        "model_selection_score"_a.none() = nb::none(),
        "data"_a.none() = nb::none(),
        "input_bindings"_a.none() = nb::none());
    trainer.def(
        "fit",
        [](Trainer& self,
           uint32_t epochs,
           uint32_t check_best_model_every_epochs,
           uint64_t first_model_selection_epoch,
           nb::object restart_conditions,
           nb::object early_completion_policies,
           nb::object max_training_batches_per_epoch) -> nb::object {
            TrainerFitOptions options;
            options.epochs = epochs;
            options.checkBestModelEveryEpochs = check_best_model_every_epochs;
            options.firstModelSelectionEpoch = first_model_selection_epoch;
            options.maxTrainingBatchesPerEpoch = optionalUint64FromPython(std::move(max_training_batches_per_epoch),
                                                                          "max_training_batches_per_epoch");
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
        "first_model_selection_epoch"_a = 0,
        "restart_conditions"_a.none() = nb::none(),
        "early_completion_policies"_a.none() = nb::none(),
        "max_training_batches_per_epoch"_a.none() = nb::none());
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
    trainer.def_prop_ro("completed_training_elapsed_seconds", &Trainer::getCompletedTrainingElapsedSeconds);

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
    training_stats_snapshot.def_ro("losses", &TrainingStatsSnapshot::losses);
    training_stats_snapshot.def_ro("metrics", &TrainingStatsSnapshot::metrics);
    training_stats_snapshot.def_ro("device_dataset_storage", &TrainingStatsSnapshot::deviceDatasetStorage);

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
        "max_restarts"_a = 5,
        nb::sig("def __init__(self, "
                "run_name: str | None = None, "
                "ensemble_group: str | None = None, "
                "progress_check_epochs: int = 3, "
                "progress_improvement_min_percentage: float = 5.0, "
                "max_restarts: int = 5"
                ") -> None"));
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
        "completion_condition"_a,
        nb::sig("def __new__(cls, "
                "completion_condition: object"
                ") -> thor.training.TrainingEarlyCompletionPolicy"));
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
        "ensemble_group"_a.none() = nb::none(),
        nb::sig("def __new__(cls, "
                "completion_condition: object, "
                "run_name: str | None = None, "
                "ensemble_group: str | None = None"
                ") -> thor.training.TrainingRunsEarlyCompletionRule"));
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

    auto training_runs = nb::class_<TrainingRuns>(training, "TrainingRuns", nb::type_slots(training_runs_type_slots));
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
            std::vector<TrainingRunsSpec> boundRuns = trainingRunsSpecsFromPython(runs);
            auto self = std::make_shared<TrainingRuns>(std::move(boundRuns),
                                                       trainingRunsFailurePolicyFromString(failure_policy),
                                                       max_summary_logs_per_second,
                                                       max_parallel_runs,
                                                       trainingRunsMinSuccessfulModelsFromPython(min_successful_models));
            return nb::cast(std::move(self));
        },
        "cls"_a,
        "runs"_a,
        "failure_policy"_a = "cancel_siblings",
        "max_summary_logs_per_second"_a = 2.0,
        "max_parallel_runs"_a.none() = size_t{3},
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
        "max_parallel_runs"_a.none() = size_t{3},
        "min_successful_models"_a.none() = nb::none());
    training_runs.def_prop_ro("max_parallel_runs", [](const TrainingRuns& self) -> nb::object {
        std::optional<size_t> maxParallelRuns = self.getMaxParallelRuns();
        if (!maxParallelRuns.has_value()) {
            return nb::none();
        }
        return nb::cast(maxParallelRuns.value());
    });
    training_runs.def_prop_ro("effective_max_parallel_runs", &TrainingRuns::getEffectiveMaxParallelRuns);
    training_runs.def_prop_ro("reports", [](const TrainingRuns& self) { return self.getReports(); });
    training_runs.def(
        "fit",
        [](TrainingRuns& self,
           uint32_t epochs,
           std::shared_ptr<Thor::TrainingData> test_data,
           uint32_t check_best_model_every_epochs,
           uint64_t first_model_selection_epoch,
           nb::object restart_conditions,
           nb::object early_completion_rules,
           nb::object reports,
           bool evaluate_training_population,
           nb::object max_training_batches_per_epoch) {
            TrainerFitOptions options;
            options.epochs = epochs;
            options.checkBestModelEveryEpochs = check_best_model_every_epochs;
            options.firstModelSelectionEpoch = first_model_selection_epoch;
            options.maxTrainingBatchesPerEpoch = optionalUint64FromPython(std::move(max_training_batches_per_epoch),
                                                                          "max_training_batches_per_epoch");
            TrainingRunsSessionOptions sessionOptions;
            sessionOptions.restartConditions = trainingRestartPoliciesFromPython(restart_conditions, /*trainerScope=*/false);
            TrainingRunsEarlyCompletionRulesBinding earlyRules = trainingRunsEarlyCompletionRulesFromPython(early_completion_rules);
            sessionOptions.earlyCompletionRules = std::move(earlyRules.rules);
            sessionOptions.reports = trainingRunsReportsFromPython(reports, self.getRuns());
            sessionOptions.evaluation.testData = std::move(test_data);
            sessionOptions.evaluation.evaluateTrainingPopulation = evaluate_training_population;
            nb::gil_scoped_release release;
            return self.fit(options, sessionOptions);
        },
        "epochs"_a,
        "test_data"_a.none() = nb::none(),
        "check_best_model_every_epochs"_a = 0,
        "first_model_selection_epoch"_a = 0,
        "restart_conditions"_a.none() = nb::none(),
        "early_completion_rules"_a.none() = nb::none(),
        "reports"_a.none() = nb::none(),
        "evaluate_training_population"_a = true,
        "max_training_batches_per_epoch"_a.none() = nb::none());

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
