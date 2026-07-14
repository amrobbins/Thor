#pragma once

#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"

#include <optional>

namespace Thor {

enum class BatchFieldSourceKind {
    MATERIALIZED_TENSOR,
    DEVICE_REFERENCE,
};

/**
 * Describes the value kind a BatchSession will produce for one named field.
 *
 * For MATERIALIZED_TENSOR, placement may be unknown.  For DEVICE_REFERENCE,
 * placement is the destination placement supported by the reference
 * materializer and must be known before input-slot preallocation.
 */
struct BatchFieldSourceDescription {
    BatchFieldSourceKind kind = BatchFieldSourceKind::MATERIALIZED_TENSOR;
    std::optional<ThorImplementation::TensorPlacement> placement;

    static BatchFieldSourceDescription materialized(
        std::optional<ThorImplementation::TensorPlacement> placement = std::nullopt) {
        return BatchFieldSourceDescription{BatchFieldSourceKind::MATERIALIZED_TENSOR, placement};
    }

    static BatchFieldSourceDescription deviceReference(
        ThorImplementation::TensorPlacement placement) {
        return BatchFieldSourceDescription{BatchFieldSourceKind::DEVICE_REFERENCE, placement};
    }
};

}  // namespace Thor
