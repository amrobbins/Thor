#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <cstdint>
#include <memory>
#include <utility>

namespace Thor {

/**
 * Type-erased description of one device-resident batch field.
 *
 * The materializer owns or shares every object needed to enqueue the field's
 * device work.  Implementations may capture a resident field, a shared batch
 * selection, and readiness events.  enqueueMaterialization() must only enqueue
 * work on the supplied NetworkInput processing stream; it must not synchronize
 * the host.
 */
class DeviceBatchMaterializer {
   public:
    virtual ~DeviceBatchMaterializer() = default;

    [[nodiscard]] virtual ThorImplementation::TensorDescriptor getOutputDescriptor() const = 0;
    [[nodiscard]] virtual ThorImplementation::TensorPlacement getOutputPlacement() const = 0;

    virtual void enqueueMaterialization(
        ThorImplementation::Tensor& destination,
        Stream& destinationStream) const = 0;
};

/**
 * Small copyable batch value used in place of a materialized tensor.
 *
 * NetworkInput stores this value in its normal queue slot.  The referenced
 * materializer then writes the selected field directly into NetworkInput's
 * statically connected featureOutput tensor.
 */
class DeviceBatchReference {
   public:
    DeviceBatchReference() = default;

    DeviceBatchReference(std::shared_ptr<const DeviceBatchMaterializer> materializer,
                         uint32_t batchSize)
        : materializer(std::move(materializer)), batchSize(batchSize) {
        THOR_THROW_IF_FALSE(this->materializer != nullptr);
        THOR_THROW_IF_FALSE(batchSize >= 1);
    }

    [[nodiscard]] bool isInitialized() const { return materializer != nullptr; }
    [[nodiscard]] uint32_t getBatchSize() const {
        THOR_THROW_IF_FALSE(isInitialized());
        return batchSize;
    }
    [[nodiscard]] ThorImplementation::TensorDescriptor getOutputDescriptor() const {
        THOR_THROW_IF_FALSE(isInitialized());
        return materializer->getOutputDescriptor();
    }
    [[nodiscard]] ThorImplementation::TensorPlacement getOutputPlacement() const {
        THOR_THROW_IF_FALSE(isInitialized());
        return materializer->getOutputPlacement();
    }

    void enqueueMaterialization(ThorImplementation::Tensor& destination,
                                Stream& destinationStream) const {
        THOR_THROW_IF_FALSE(isInitialized());
        materializer->enqueueMaterialization(destination, destinationStream);
    }

   private:
    std::shared_ptr<const DeviceBatchMaterializer> materializer;
    uint32_t batchSize = 0;
};

}  // namespace Thor
