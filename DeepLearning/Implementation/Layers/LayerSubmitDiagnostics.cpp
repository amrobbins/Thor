#include "DeepLearning/Implementation/Layers/LayerSubmitDiagnostics.h"

#if THOR_ENABLE_LAYER_SUBMIT_DIAGNOSTICS

#include <cstdio>
#include <cstdlib>

namespace ThorImplementation {
namespace {

thread_local LayerSubmitDiagnosticContext currentContext;

bool truthyEnv(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}

}  // namespace

bool layerSubmitDiagnosticsEnabled() { return truthyEnv("THOR_TRAINING_LAYER_SUBMIT_DIAGNOSTICS"); }

bool layerSubmitDiagnosticsActive() { return currentContext.active && layerSubmitDiagnosticsEnabled(); }

LayerSubmitDiagnosticTimePoint layerSubmitDiagnosticNow() { return std::chrono::high_resolution_clock::now(); }

uint64_t layerSubmitDiagnosticElapsedMicros(LayerSubmitDiagnosticTimePoint start, LayerSubmitDiagnosticTimePoint finish) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count());
}

std::string layerSubmitDiagnosticLabel(const char* layerType, uint64_t layerId, const std::string& layerName) {
    std::string label = std::string(layerType) + "#" + std::to_string(layerId);
    if (!layerName.empty()) {
        label += "(" + layerName + ")";
    }
    return label;
}

void setLayerSubmitDiagnosticContext(const std::string& phase,
                                     uint64_t epoch,
                                     uint64_t batch,
                                     uint64_t slot,
                                     uint64_t inFlight,
                                     uint64_t done,
                                     uint64_t total,
                                     bool validationPass) {
    currentContext.active = true;
    currentContext.phase = phase;
    currentContext.epoch = epoch;
    currentContext.batch = batch;
    currentContext.slot = slot;
    currentContext.inFlight = inFlight;
    currentContext.done = done;
    currentContext.total = total;
    currentContext.validationPass = validationPass;
}

void clearLayerSubmitDiagnosticContext() { currentContext = LayerSubmitDiagnosticContext(); }

ScopedLayerSubmitDiagnosticContext::ScopedLayerSubmitDiagnosticContext(const std::string& phase,
                                                                       uint64_t epoch,
                                                                       uint64_t batch,
                                                                       uint64_t slot,
                                                                       uint64_t inFlight,
                                                                       uint64_t done,
                                                                       uint64_t total,
                                                                       bool validationPass,
                                                                       bool enabled)
    : active(enabled && layerSubmitDiagnosticsEnabled()) {
    if (active) {
        setLayerSubmitDiagnosticContext(phase, epoch, batch, slot, inFlight, done, total, validationPass);
    }
}

ScopedLayerSubmitDiagnosticContext::~ScopedLayerSubmitDiagnosticContext() {
    if (active) {
        clearLayerSubmitDiagnosticContext();
    }
}

void emitLayerSubmitDiagnostic(const char* event,
                               const std::string& layer,
                               uint64_t layerId,
                               uint64_t totalMicros,
                               std::initializer_list<std::pair<const char*, uint64_t>> fields) {
    if (!layerSubmitDiagnosticsActive()) {
        return;
    }

    std::fprintf(stderr,
                 "THOR_TRAINING_LAYER_SUBMIT_DIAGNOSTIC event=%s phase=%s epoch=%lu batch=%lu slot=%lu "
                 "in_flight=%lu done=%lu/%lu validation=%d layer=%s layer_id=%lu total_us=%lu",
                 event,
                 currentContext.phase.c_str(),
                 currentContext.epoch + 1,
                 currentContext.batch + 1,
                 currentContext.slot,
                 currentContext.inFlight,
                 currentContext.done,
                 currentContext.total,
                 currentContext.validationPass ? 1 : 0,
                 layer.c_str(),
                 layerId,
                 totalMicros);
    for (const auto& [name, value] : fields) {
        std::fprintf(stderr, " %s=%lu", name, value);
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

}  // namespace ThorImplementation

#endif  // THOR_ENABLE_LAYER_SUBMIT_DIAGNOSTICS
