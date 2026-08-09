#pragma once

#include <cstdio>
#include <exception>
#include <utility>

namespace ThorImplementation::SharedOwnership {

/**
 * Concurrency and cleanup contract for Thor resource-owning handles.
 *
 * Thor delegates shared lifetime management to std::shared_ptr rather than
 * maintaining a framework-specific reference-count implementation. Resource
 * handles store shared_ptr-backed state (or, for Tensor, shared_ptr-backed
 * allocation state) and follow the standard library ownership rules below.
 *
 * - Distinct handle objects may share one resource state and may be copied,
 *   moved, assigned, reset, and destroyed concurrently.
 * - A single handle object must not be read while another thread mutates that
 *   same handle object, and must not be mutated concurrently by multiple
 *   threads, unless the owning code provides external synchronization.
 * - Resource-state destructors are noexcept. Cleanup failures must be reported
 *   rather than escaping destruction.
 *
 * This deliberately matches the ordinary std::shared_ptr concurrency model.
 * Thor does not add a stronger same-handle synchronization contract on top of
 * the standard library ownership primitive.
 */

inline void reportCleanupFailure(const char *resourceType, const char *operation, const char *detail) noexcept {
    const char *safeResourceType = resourceType != nullptr ? resourceType : "<unknown resource>";
    const char *safeOperation = operation != nullptr ? operation : "<unknown cleanup operation>";
    const char *safeDetail = detail != nullptr ? detail : "<no detail>";

    std::fprintf(stderr,
                 "Thor resource cleanup failure [%s] %s: %s\n",
                 safeResourceType,
                 safeOperation,
                 safeDetail);
    std::fflush(stderr);
}

template <typename Cleanup>
void cleanupNoThrow(const char *resourceType, const char *operation, Cleanup &&cleanup) noexcept {
    try {
        std::forward<Cleanup>(cleanup)();
    } catch (const std::exception &error) {
        reportCleanupFailure(resourceType, operation, error.what());
    } catch (...) {
        reportCleanupFailure(resourceType, operation, "unknown exception");
    }
}

}  // namespace ThorImplementation::SharedOwnership
