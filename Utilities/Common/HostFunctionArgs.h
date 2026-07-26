#pragma once

// Base class for state passed to cudaLaunchHostFunc callbacks. CUDA host
// callbacks must not call CUDA APIs, directly or through destructors. Thor
// therefore keeps callback state alive until the callback has completed and
// destroys it on a normal host worker thread.
struct HostFunctionArgsBase {
    virtual ~HostFunctionArgsBase() = default;
};
