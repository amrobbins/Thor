#include "DeepLearning/Implementation/Parameter/ParameterConstraint.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace ThorImplementation {

void launchClampParameter(void* values,
                          DataType dataType,
                          uint64_t numElements,
                          bool hasMinValue,
                          double minValue,
                          bool hasMaxValue,
                          double maxValue,
                          int gpuNum,
                          Stream stream);

namespace {

void validateMinMax(double minValue, double maxValue) {
    if (minValue > maxValue) {
        throw std::runtime_error("MinMaxParameterConstraint requires min_value <= max_value.");
    }
}

void validateSupportedParameterStorage(const Tensor& parameterStorage, const Stream& stream, const std::string& constraintName) {
    if (!parameterStorage.isInitialized()) {
        throw std::runtime_error(constraintName + " cannot apply to uninitialized parameter storage.");
    }

    const DataType dataType = parameterStorage.getDataType();
    switch (dataType) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
        case DataType::FP64:
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            break;
        default:
            throw std::runtime_error(constraintName +
                                     " supports fp16, bf16, fp32, fp64, fp8_e4m3, and fp8_e5m2 parameter tensors. Got " +
                                     TensorDescriptor::getElementTypeName(dataType) + ".");
    }

    if (parameterStorage.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::runtime_error(constraintName + " currently requires GPU parameter storage.");
    }
    if (!stream.isInitialized()) {
        throw std::runtime_error(constraintName + " requires an initialized stream.");
    }
    if (stream.getGpuNum() != parameterStorage.getPlacement().getDeviceNum()) {
        throw std::runtime_error(constraintName + " stream GPU does not match parameter storage GPU.");
    }
}

void applyScalarBounds(Tensor& parameterStorage,
                       Stream& stream,
                       bool hasMinValue,
                       double minValue,
                       bool hasMaxValue,
                       double maxValue,
                       const std::string& constraintName) {
    validateSupportedParameterStorage(parameterStorage, stream, constraintName);
    launchClampParameter(parameterStorage.getMemPtr(),
                         parameterStorage.getDataType(),
                         parameterStorage.getTotalNumElements(),
                         hasMinValue,
                         minValue,
                         hasMaxValue,
                         maxValue,
                         parameterStorage.getPlacement().getDeviceNum(),
                         stream);
}

}  // namespace

void NonNegativeParameterConstraint::apply(Tensor& parameterStorage, Stream& stream) const {
    applyScalarBounds(parameterStorage, stream, true, 0.0, false, 0.0, "NonNegativeParameterConstraint");
}

bool NonNegativeParameterConstraint::supportsDenseExpressionFusion() const { return true; }

Expression NonNegativeParameterConstraint::applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                                         const std::string& namePrefix) const {
    (void)namePrefix;
    return unconstrainedParameterUpdate.max(Expression::constantScalar(0.0));
}

std::shared_ptr<ParameterConstraint> NonNegativeParameterConstraint::clone() const {
    return std::make_shared<NonNegativeParameterConstraint>(*this);
}

std::string NonNegativeParameterConstraint::getConstraintType() const { return "non_negative"; }

void NonPositiveParameterConstraint::apply(Tensor& parameterStorage, Stream& stream) const {
    applyScalarBounds(parameterStorage, stream, false, 0.0, true, 0.0, "NonPositiveParameterConstraint");
}

bool NonPositiveParameterConstraint::supportsDenseExpressionFusion() const { return true; }

Expression NonPositiveParameterConstraint::applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                                         const std::string& namePrefix) const {
    (void)namePrefix;
    return unconstrainedParameterUpdate.min(Expression::constantScalar(0.0));
}

std::shared_ptr<ParameterConstraint> NonPositiveParameterConstraint::clone() const {
    return std::make_shared<NonPositiveParameterConstraint>(*this);
}

std::string NonPositiveParameterConstraint::getConstraintType() const { return "non_positive"; }

MinParameterConstraint::MinParameterConstraint(double minValue) : minValue(minValue) {}

void MinParameterConstraint::apply(Tensor& parameterStorage, Stream& stream) const {
    applyScalarBounds(parameterStorage, stream, true, minValue, false, 0.0, "MinParameterConstraint");
}

bool MinParameterConstraint::supportsDenseExpressionFusion() const { return true; }

Expression MinParameterConstraint::applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                                  const std::string& namePrefix) const {
    (void)namePrefix;
    return unconstrainedParameterUpdate.max(Expression::constantScalar(minValue));
}

std::shared_ptr<ParameterConstraint> MinParameterConstraint::clone() const { return std::make_shared<MinParameterConstraint>(*this); }

std::string MinParameterConstraint::getConstraintType() const { return "min"; }

double MinParameterConstraint::getMinValue() const { return minValue; }

MaxParameterConstraint::MaxParameterConstraint(double maxValue) : maxValue(maxValue) {}

void MaxParameterConstraint::apply(Tensor& parameterStorage, Stream& stream) const {
    applyScalarBounds(parameterStorage, stream, false, 0.0, true, maxValue, "MaxParameterConstraint");
}

bool MaxParameterConstraint::supportsDenseExpressionFusion() const { return true; }

Expression MaxParameterConstraint::applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                                  const std::string& namePrefix) const {
    (void)namePrefix;
    return unconstrainedParameterUpdate.min(Expression::constantScalar(maxValue));
}

std::shared_ptr<ParameterConstraint> MaxParameterConstraint::clone() const { return std::make_shared<MaxParameterConstraint>(*this); }

std::string MaxParameterConstraint::getConstraintType() const { return "max"; }

double MaxParameterConstraint::getMaxValue() const { return maxValue; }

MinMaxParameterConstraint::MinMaxParameterConstraint(double minValue, double maxValue) : minValue(minValue), maxValue(maxValue) {
    validateMinMax(minValue, maxValue);
}

void MinMaxParameterConstraint::apply(Tensor& parameterStorage, Stream& stream) const {
    applyScalarBounds(parameterStorage, stream, true, minValue, true, maxValue, "MinMaxParameterConstraint");
}

bool MinMaxParameterConstraint::supportsDenseExpressionFusion() const { return true; }

Expression MinMaxParameterConstraint::applyDenseExpressionConstraint(const Expression& unconstrainedParameterUpdate,
                                                                     const std::string& namePrefix) const {
    (void)namePrefix;
    return unconstrainedParameterUpdate.max(Expression::constantScalar(minValue)).min(Expression::constantScalar(maxValue));
}

std::shared_ptr<ParameterConstraint> MinMaxParameterConstraint::clone() const {
    return std::make_shared<MinMaxParameterConstraint>(*this);
}

std::string MinMaxParameterConstraint::getConstraintType() const { return "min_max"; }

double MinMaxParameterConstraint::getMinValue() const { return minValue; }

double MinMaxParameterConstraint::getMaxValue() const { return maxValue; }

}  // namespace ThorImplementation
