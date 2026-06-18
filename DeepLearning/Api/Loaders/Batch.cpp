#include "DeepLearning/Api/Loaders/Batch.h"


namespace {

std::runtime_error missingBatchValueError(const std::string& name) {
    return std::runtime_error("Batch is missing input '" + name + "'.");
}

std::runtime_error wrongBatchValueTypeError(const std::string& name,
                                           const std::string& expected,
                                           const std::string& actual) {
    return std::runtime_error("Batch input '" + name + "' is a " + actual + " value, not a " + expected + " value.");
}

std::string batchValueTypeName(const BatchValue& value) {
    if (std::holds_alternative<ThorImplementation::Tensor>(value)) {
        return "dense tensor";
    }
    if (std::holds_alternative<ThorImplementation::RaggedTensor>(value)) {
        return "ragged tensor";
    }
    return "unknown";
}

}  // namespace

Batch::Batch(std::map<std::string, ThorImplementation::Tensor> tensors) {
    for (auto& [name, tensor] : tensors) {
        insert(std::move(name), std::move(tensor));
    }
}

void Batch::insert(std::string name, ThorImplementation::Tensor tensor) {
    THOR_THROW_IF_FALSE(tensor.isInitialized());
    auto [it, inserted] = values_.emplace(std::move(name), std::move(tensor));
    (void)it;
    THOR_THROW_IF_FALSE(inserted);
}

void Batch::insert(std::string name, ThorImplementation::RaggedTensor raggedTensor) {
    THOR_THROW_IF_FALSE(raggedTensor.isInitialized());
    auto [it, inserted] = values_.emplace(std::move(name), std::move(raggedTensor));
    (void)it;
    THOR_THROW_IF_FALSE(inserted);
}

BatchValue& Batch::at(const std::string& name) {
    auto it = values_.find(name);
    if (it == values_.end()) {
        throw missingBatchValueError(name);
    }
    return it->second;
}

const BatchValue& Batch::at(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        throw missingBatchValueError(name);
    }
    return it->second;
}

bool Batch::isTensor(const std::string& name) const {
    auto it = values_.find(name);
    return it != values_.end() && std::holds_alternative<ThorImplementation::Tensor>(it->second);
}

bool Batch::isRaggedTensor(const std::string& name) const {
    auto it = values_.find(name);
    return it != values_.end() && std::holds_alternative<ThorImplementation::RaggedTensor>(it->second);
}

bool Batch::isDenseOnly() const {
    for (const auto& [name, value] : values_) {
        (void)name;
        if (!std::holds_alternative<ThorImplementation::Tensor>(value)) {
            return false;
        }
    }
    return true;
}

ThorImplementation::Tensor& Batch::getTensor(const std::string& name) {
    BatchValue& value = at(name);
    if (!std::holds_alternative<ThorImplementation::Tensor>(value)) {
        throw wrongBatchValueTypeError(name, "dense tensor", batchValueTypeName(value));
    }
    return std::get<ThorImplementation::Tensor>(value);
}

const ThorImplementation::Tensor& Batch::getTensor(const std::string& name) const {
    const BatchValue& value = at(name);
    if (!std::holds_alternative<ThorImplementation::Tensor>(value)) {
        throw wrongBatchValueTypeError(name, "dense tensor", batchValueTypeName(value));
    }
    return std::get<ThorImplementation::Tensor>(value);
}

ThorImplementation::RaggedTensor& Batch::getRaggedTensor(const std::string& name) {
    BatchValue& value = at(name);
    if (!std::holds_alternative<ThorImplementation::RaggedTensor>(value)) {
        throw wrongBatchValueTypeError(name, "ragged tensor", batchValueTypeName(value));
    }
    return std::get<ThorImplementation::RaggedTensor>(value);
}

const ThorImplementation::RaggedTensor& Batch::getRaggedTensor(const std::string& name) const {
    const BatchValue& value = at(name);
    if (!std::holds_alternative<ThorImplementation::RaggedTensor>(value)) {
        throw wrongBatchValueTypeError(name, "ragged tensor", batchValueTypeName(value));
    }
    return std::get<ThorImplementation::RaggedTensor>(value);
}

Batch batchFromTensorMap(std::map<std::string, ThorImplementation::Tensor> tensors) { return Batch(std::move(tensors)); }

std::map<std::string, ThorImplementation::Tensor> denseTensorMapFromBatchOrThrow(const Batch& batch, const std::string& context) {
    std::map<std::string, ThorImplementation::Tensor> tensors;
    for (const auto& [name, value] : batch.values()) {
        if (!std::holds_alternative<ThorImplementation::Tensor>(value)) {
            throw std::runtime_error(context + " contains ragged batch input '" + name +
                                     "', but this execution path currently accepts only dense tensor inputs.");
        }
        tensors.emplace(name, std::get<ThorImplementation::Tensor>(value));
    }
    return tensors;
}
