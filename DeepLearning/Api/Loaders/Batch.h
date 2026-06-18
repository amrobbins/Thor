#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

using BatchValue = std::variant<ThorImplementation::Tensor, ThorImplementation::RaggedTensor>;

class Batch {
   public:
    using Storage = std::map<std::string, BatchValue>;
    using iterator = Storage::iterator;
    using const_iterator = Storage::const_iterator;

    Batch() = default;
    explicit Batch(Storage values) : values_(std::move(values)) {}
    explicit Batch(std::map<std::string, ThorImplementation::Tensor> tensors);

    void insert(std::string name, ThorImplementation::Tensor tensor);
    void insert(std::string name, ThorImplementation::RaggedTensor raggedTensor);

    bool contains(const std::string& name) const { return values_.count(name) != 0; }
    std::size_t count(const std::string& name) const { return values_.count(name); }
    bool empty() const { return values_.empty(); }
    std::size_t size() const { return values_.size(); }
    void clear() { values_.clear(); }

    BatchValue& at(const std::string& name);
    const BatchValue& at(const std::string& name) const;

    bool isTensor(const std::string& name) const;
    bool isRaggedTensor(const std::string& name) const;
    bool isDenseOnly() const;

    ThorImplementation::Tensor& getTensor(const std::string& name);
    const ThorImplementation::Tensor& getTensor(const std::string& name) const;
    ThorImplementation::RaggedTensor& getRaggedTensor(const std::string& name);
    const ThorImplementation::RaggedTensor& getRaggedTensor(const std::string& name) const;

    Storage& values() { return values_; }
    const Storage& values() const { return values_; }

    iterator begin() { return values_.begin(); }
    iterator end() { return values_.end(); }
    const_iterator begin() const { return values_.begin(); }
    const_iterator end() const { return values_.end(); }
    const_iterator cbegin() const { return values_.cbegin(); }
    const_iterator cend() const { return values_.cend(); }

   private:
    Storage values_;
};

Batch batchFromTensorMap(std::map<std::string, ThorImplementation::Tensor> tensors);
std::map<std::string, ThorImplementation::Tensor> denseTensorMapFromBatchOrThrow(
    const Batch& batch,
    const std::string& context = "Batch");
