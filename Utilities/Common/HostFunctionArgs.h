#pragma once

#include "cuda_runtime.h"

class Stream;

// Base class for state passed to cudaLaunchHostFunc callbacks. CUDA host
// callbacks must not call CUDA APIs, directly or through destructors. Thor
// therefore owns callback state until the callback has completed and destroys
// it on a normal host worker thread. The callback must not delete its args.
struct HostFunctionArgsBase {
    virtual ~HostFunctionArgsBase() = default;

   private:
    // Populated by Stream::enqueueHostFunction(). Keeping the dispatch metadata
    // in the already-allocated callback state lets every host function pass
    // through one noexcept trampoline without adding another allocation.
    cudaHostFn_t function = nullptr;
    void *failureState = nullptr;

    friend class Stream;
};
