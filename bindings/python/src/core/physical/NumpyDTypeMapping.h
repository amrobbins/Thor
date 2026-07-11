#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "DeepLearning/Implementation/Tensor/DataType.h"

namespace Thor::PythonBindings {

namespace nb = nanobind;

inline std::string thorStorageDataTypeName(ThorImplementation::DataType dataType) {
    using ThorImplementation::DataType;
    switch (dataType) {
        case DataType::BOOLEAN: return "bool";
        case DataType::INT8: return "int8";
        case DataType::UINT8: return "uint8";
        case DataType::INT16: return "int16";
        case DataType::UINT16: return "uint16";
        case DataType::INT32: return "int32";
        case DataType::UINT32: return "uint32";
        case DataType::INT64: return "int64";
        case DataType::UINT64: return "uint64";
        case DataType::FP16: return "fp16";
        case DataType::FP32: return "fp32";
        case DataType::FP64: return "fp64";
        case DataType::BF16: return "bf16";
        case DataType::FP8_E4M3: return "fp8_e4m3";
        case DataType::FP8_E5M2: return "fp8_e5m2";
        case DataType::TF32: return "tf32";
    }
    throw std::invalid_argument("Unsupported Thor data type");
}

inline nb::object numpyScalarTypeForThorStorageDataType(ThorImplementation::DataType dataType) {
    using ThorImplementation::DataType;
    nb::object numpy = nb::module_::import_("numpy");
    switch (dataType) {
        case DataType::BOOLEAN: return numpy.attr("bool_");
        case DataType::INT8: return numpy.attr("int8");
        case DataType::UINT8: return numpy.attr("uint8");
        case DataType::INT16: return numpy.attr("int16");
        case DataType::UINT16: return numpy.attr("uint16");
        case DataType::INT32: return numpy.attr("int32");
        case DataType::UINT32: return numpy.attr("uint32");
        case DataType::INT64: return numpy.attr("int64");
        case DataType::UINT64: return numpy.attr("uint64");
        case DataType::FP16: return numpy.attr("float16");
        case DataType::FP32: return numpy.attr("float32");
        case DataType::FP64: return numpy.attr("float64");
        case DataType::BF16: return nb::module_::import_("ml_dtypes").attr("bfloat16");
        case DataType::FP8_E4M3: return nb::module_::import_("ml_dtypes").attr("float8_e4m3fn");
        case DataType::FP8_E5M2: return nb::module_::import_("ml_dtypes").attr("float8_e5m2");
        case DataType::TF32:
            throw std::invalid_argument("thor.DataType.tf32 is compute-only and cannot be used as a stored dataset dtype");
    }
    throw std::invalid_argument("Unsupported Thor dataset storage dtype");
}

inline bool pythonObjectsEqual(nb::handle left, nb::handle right) {
    const int result = PyObject_RichCompareBool(left.ptr(), right.ptr(), Py_EQ);
    if (result < 0) {
        throw nb::python_error();
    }
    return result != 0;
}

inline nb::object numpyDTypeObjectForThorStorageDataType(ThorImplementation::DataType dataType) {
    nb::object numpy = nb::module_::import_("numpy");
    return numpy.attr("dtype")(numpyScalarTypeForThorStorageDataType(dataType));
}

inline std::optional<ThorImplementation::DataType> thorStorageDataTypeFromNumpyDTypeObject(nb::handle numpyDataType) {
    using ThorImplementation::DataType;
    const std::string name = nb::cast<std::string>(nb::borrow<nb::object>(numpyDataType).attr("name"));
    std::optional<DataType> candidate;
    if (name == "bool") candidate = DataType::BOOLEAN;
    else if (name == "int8") candidate = DataType::INT8;
    else if (name == "uint8") candidate = DataType::UINT8;
    else if (name == "int16") candidate = DataType::INT16;
    else if (name == "uint16") candidate = DataType::UINT16;
    else if (name == "int32") candidate = DataType::INT32;
    else if (name == "uint32") candidate = DataType::UINT32;
    else if (name == "int64") candidate = DataType::INT64;
    else if (name == "uint64") candidate = DataType::UINT64;
    else if (name == "float16") candidate = DataType::FP16;
    else if (name == "float32") candidate = DataType::FP32;
    else if (name == "float64") candidate = DataType::FP64;
    else if (name == "bfloat16") candidate = DataType::BF16;
    else if (name == "float8_e4m3fn") candidate = DataType::FP8_E4M3;
    else if (name == "float8_e5m2") candidate = DataType::FP8_E5M2;

    if (!candidate.has_value()) {
        return std::nullopt;
    }
    nb::object canonicalDType = numpyDTypeObjectForThorStorageDataType(*candidate);
    if (!pythonObjectsEqual(numpyDataType, canonicalDType)) {
        return std::nullopt;
    }
    return candidate;
}

inline uint64_t thorStorageDataTypeSizeBytes(ThorImplementation::DataType dataType) {
    using ThorImplementation::DataType;
    switch (dataType) {
        case DataType::BOOLEAN:
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return 1;
        case DataType::INT16:
        case DataType::UINT16:
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::INT32:
        case DataType::UINT32:
        case DataType::FP32:
            return 4;
        case DataType::INT64:
        case DataType::UINT64:
        case DataType::FP64:
            return 8;
        case DataType::TF32:
            throw std::invalid_argument("thor.DataType.tf32 is compute-only and cannot be used as a stored dataset dtype");
    }
    throw std::invalid_argument("Unsupported Thor dataset storage dtype");
}

struct CanonicalNumpyArrayView {
    const void *data = nullptr;
    std::vector<uint64_t> dimensions;
    uint64_t size = 0;
    ThorImplementation::DataType dataType = ThorImplementation::DataType::FP32;
};

inline CanonicalNumpyArrayView canonicalNumpyArrayViewNoCopy(
    nb::handle value,
    const std::string &context,
    std::optional<ThorImplementation::DataType> expectedDataType = std::nullopt) {
    nb::object numpy = nb::module_::import_("numpy");
    if (!nb::isinstance(value, numpy.attr("ndarray"))) {
        throw nb::type_error((context + " must be a numpy.ndarray").c_str());
    }

    nb::object owner = nb::borrow<nb::object>(value);
    if (!nb::cast<bool>(owner.attr("flags").attr("c_contiguous"))) {
        throw nb::type_error((context + " must be a C-contiguous numpy.ndarray").c_str());
    }

    const std::optional<ThorImplementation::DataType> actualDataType =
        thorStorageDataTypeFromNumpyDTypeObject(owner.attr("dtype"));
    if (!actualDataType.has_value()) {
        throw nb::type_error(
            (context + " must use a canonical NumPy/ml_dtypes representation of a storable Thor dtype").c_str());
    }
    if (expectedDataType.has_value() && *actualDataType != *expectedDataType) {
        throw nb::type_error((context + " must have the canonical numpy dtype for thor.DataType." +
                              thorStorageDataTypeName(*expectedDataType)).c_str());
    }

    nb::tuple shape = nb::cast<nb::tuple>(owner.attr("shape"));
    std::vector<uint64_t> dimensions;
    dimensions.reserve(nb::len(shape));
    for (nb::handle dimension : shape) {
        dimensions.push_back(nb::cast<uint64_t>(dimension));
    }

    const uint64_t size = nb::cast<uint64_t>(owner.attr("size"));
    const uintptr_t dataAddress = nb::cast<uintptr_t>(owner.attr("ctypes").attr("data"));
    return CanonicalNumpyArrayView{
        .data = reinterpret_cast<const void *>(dataAddress),
        .dimensions = std::move(dimensions),
        .size = size,
        .dataType = *actualDataType,
    };
}

}  // namespace Thor::PythonBindings
