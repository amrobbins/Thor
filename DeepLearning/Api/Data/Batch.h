#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Data/DeviceBatchReference.h"
#include "DeepLearning/Api/Data/BatchSourceResource.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using BatchValue = std::variant<ThorImplementation::Tensor, ThorImplementation::RaggedTensor, Thor::DeviceBatchReference>;

namespace Thor {
class BatchLease;
class BatchSession;
}

class Batch {
   public:
    using Storage = std::map<std::string, BatchValue>;
    using iterator = Storage::iterator;
    using const_iterator = Storage::const_iterator;

    Batch() = default;
    Batch(const Batch& other)
        : values_(other.values_), sourceReferences_(other.sourceReferences_) {}
    Batch& operator=(const Batch& other) {
        if (this != &other) {
            releaseAllSourceResources();
            values_ = other.values_;
            sourceReferences_ = other.sourceReferences_;
            ownsSourceResourceLifecycle_ = false;
            recycleToken_.reset();
        }
        return *this;
    }
    Batch(Batch&&) noexcept = default;
    Batch& operator=(Batch&&) noexcept = default;
    explicit Batch(Storage values) : values_(std::move(values)) {}
    explicit Batch(std::map<std::string, ThorImplementation::Tensor> tensors);

    void insert(std::string name, ThorImplementation::Tensor tensor);
    void insert(std::string name, ThorImplementation::RaggedTensor raggedTensor);
    void insert(std::string name, Thor::DeviceBatchReference deviceBatchReference);

    bool contains(const std::string& name) const { return values_.count(name) != 0; }
    std::size_t count(const std::string& name) const { return values_.count(name); }
    bool empty() const { return values_.empty(); }
    std::size_t size() const { return values_.size(); }
    void clear() {
        releaseAllSourceResources();
        values_.clear();
        sourceReferences_.clear();
        ownsSourceResourceLifecycle_ = false;
        recycleToken_.reset();
    }

    BatchValue& at(const std::string& name);
    const BatchValue& at(const std::string& name) const;

    bool isTensor(const std::string& name) const;
    bool isRaggedTensor(const std::string& name) const;
    bool isDeviceBatchReference(const std::string& name) const;
    bool isDenseOnly() const;

    ThorImplementation::Tensor& getTensor(const std::string& name);
    const ThorImplementation::Tensor& getTensor(const std::string& name) const;
    ThorImplementation::RaggedTensor& getRaggedTensor(const std::string& name);
    const ThorImplementation::RaggedTensor& getRaggedTensor(const std::string& name) const;
    Thor::DeviceBatchReference& getDeviceBatchReference(const std::string& name);
    const Thor::DeviceBatchReference& getDeviceBatchReference(const std::string& name) const;

    void setSourceReference(
        const std::string& name,
        Thor::BatchSourceReference sourceReference) {
        THOR_THROW_IF_FALSE(contains(name));
        THOR_THROW_IF_FALSE(sourceReference.isInitialized());
        THOR_THROW_IF_FALSE(sourceReferences_.count(name) == 0);
        sourceReferences_.emplace(name, std::move(sourceReference));
    }

    [[nodiscard]] std::optional<Thor::BatchSourceReference> getSourceReference(
        const std::string& name) const {
        auto found = sourceReferences_.find(name);
        if (found == sourceReferences_.end()) {
            return std::nullopt;
        }
        return found->second;
    }

    /**
     * True when every logical field has independently tracked source storage.
     * Such a batch may release its BatchLease after all input submissions have
     * registered their source-consumed events; downstream network execution no
     * longer depends on the original Batch object.
     */
    [[nodiscard]] bool allFieldsHaveSourceReferences() const {
        if (values_.empty() || sourceReferences_.size() != values_.size()) {
            return false;
        }
        for (const auto& [name, value] : values_) {
            (void)value;
            if (sourceReferences_.count(name) == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * True only for the original Batch instance that owns the producer side of
     * its source-resource lifecycle. Batch copies retain consumer references
     * for input binding, but intentionally do not gain authority to recycle the
     * originating session's reusable buffers.
     */
    [[nodiscard]] bool ownsSourceResourceLifecycle() const {
        return ownsSourceResourceLifecycle_;
    }

    Storage& values() { return values_; }
    const Storage& values() const { return values_; }

    iterator begin() { return values_.begin(); }
    iterator end() { return values_.end(); }
    const_iterator begin() const { return values_.begin(); }
    const_iterator end() const { return values_.end(); }
    const_iterator cbegin() const { return values_.cbegin(); }
    const_iterator cend() const { return values_.cend(); }

   private:
    struct OwnedSourceResource {
        std::set<std::string> fieldNames;
        Thor::BatchSourceOwner owner;
    };

    void addSourceResource(
        std::set<std::string> fieldNames,
        Thor::BatchSourceOwner owner);
    void releaseSourceResourcesExcept(const std::set<std::string>& retainedFields);
    void releaseAllSourceResources() { releaseSourceResourcesExcept({}); }

    void setRecycleToken(std::shared_ptr<void> token) { recycleToken_ = std::move(token); }
    std::shared_ptr<void> takeRecycleToken() {
        std::shared_ptr<void> token = std::move(recycleToken_);
        recycleToken_.reset();
        return token;
    }

    Storage values_;
    std::map<std::string, Thor::BatchSourceReference> sourceReferences_;
    std::vector<OwnedSourceResource> ownedSourceResources_;
    // Set only on the original producer-owned Batch. Intentionally omitted
    // from copy construction/assignment while moving with the leased Batch.
    bool ownsSourceResourceLifecycle_ = false;
    // Session-private ownership identity for reusable storage. It moves with
    // the leased Batch but is intentionally not copied into bound/user copies.
    std::shared_ptr<void> recycleToken_;

    friend class Thor::BatchLease;
    friend class Thor::BatchSession;
};

Batch batchFromTensorMap(std::map<std::string, ThorImplementation::Tensor> tensors);
std::map<std::string, ThorImplementation::Tensor> denseTensorMapFromBatchOrThrow(
    const Batch& batch,
    const std::string& context = "Batch");
