#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <Python.h>

#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace Thor::PythonBindings {

namespace nb = nanobind;

inline std::string pythonTypeName(nb::handle obj) {
    if (!obj.is_valid() || obj.ptr() == nullptr) {
        return "<invalid>";
    }

    PyTypeObject* type = Py_TYPE(obj.ptr());
    if (type == nullptr || type->tp_name == nullptr) {
        return "<unknown>";
    }

    return type->tp_name;
}

inline std::string makeCastErrorMessage(std::string_view context, std::string_view expected, nb::handle actual) {
    std::ostringstream oss;
    oss << context << ": expected " << expected << ", got " << pythonTypeName(actual);
    return oss.str();
}

[[noreturn]] inline void raiseCastTypeError(std::string_view context, std::string_view expected, nb::handle actual) {
    const std::string message = makeCastErrorMessage(context, expected, actual);
    throw nb::type_error(message.c_str());
}

template <typename T>
T castOrTypeError(nb::handle obj, std::string_view context, std::string_view expected, bool convert = false) {
    static_assert(!std::is_reference_v<T>, "castOrTypeError<T> returns an owned value; do not instantiate it with a reference type.");

    using ValueT = std::remove_cv_t<T>;

    if constexpr (std::is_default_constructible_v<ValueT>) {
        ValueT value{};
        if (!nb::try_cast<ValueT>(obj, value, convert)) {
            raiseCastTypeError(context, expected, obj);
        }
        return value;
    } else {
        std::optional<ValueT> value;
        if (!nb::try_cast<std::optional<ValueT>>(obj, value, convert) || !value.has_value()) {
            raiseCastTypeError(context, expected, obj);
        }
        return std::move(*value);
    }
}

template <typename T>
T castArgument(
    nb::handle obj, std::string_view functionName, std::string_view argumentName, std::string_view expected, bool convert = false) {
    std::ostringstream context;
    context << functionName << "() argument '" << argumentName << "'";
    return castOrTypeError<T>(obj, context.str(), expected, convert);
}

template <typename T>
bool tryCast(nb::handle obj, T& out, bool convert = false) noexcept {
    return nb::try_cast<T>(obj, out, convert);
}

}  // namespace Thor::PythonBindings
