#pragma once

#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

namespace ThorImplementation {

using LayerSubmitDiagnosticTimePoint = std::chrono::high_resolution_clock::time_point;

struct LayerSubmitDiagnosticContext {
    bool active = false;
    std::string phase;
    uint64_t epoch = 0;
    uint64_t batch = 0;
    uint64_t slot = 0;
    uint64_t inFlight = 0;
    uint64_t done = 0;
    uint64_t total = 0;
    bool validationPass = false;
};

bool layerSubmitDiagnosticsEnabled();
bool layerSubmitDiagnosticsActive();
LayerSubmitDiagnosticTimePoint layerSubmitDiagnosticNow();
uint64_t layerSubmitDiagnosticElapsedMicros(LayerSubmitDiagnosticTimePoint start, LayerSubmitDiagnosticTimePoint finish);
std::string layerSubmitDiagnosticLabel(const char* layerType, uint64_t layerId, const std::string& layerName = std::string());

void setLayerSubmitDiagnosticContext(const std::string& phase,
                                     uint64_t epoch,
                                     uint64_t batch,
                                     uint64_t slot,
                                     uint64_t inFlight,
                                     uint64_t done,
                                     uint64_t total,
                                     bool validationPass);
void clearLayerSubmitDiagnosticContext();

class ScopedLayerSubmitDiagnosticContext {
   public:
    ScopedLayerSubmitDiagnosticContext(const std::string& phase,
                                       uint64_t epoch,
                                       uint64_t batch,
                                       uint64_t slot,
                                       uint64_t inFlight,
                                       uint64_t done,
                                       uint64_t total,
                                       bool validationPass,
                                       bool enabled);
    ~ScopedLayerSubmitDiagnosticContext();

    ScopedLayerSubmitDiagnosticContext(const ScopedLayerSubmitDiagnosticContext&) = delete;
    ScopedLayerSubmitDiagnosticContext& operator=(const ScopedLayerSubmitDiagnosticContext&) = delete;

   private:
    bool active = false;
};

void emitLayerSubmitDiagnostic(const char* event,
                               const std::string& layer,
                               uint64_t layerId,
                               uint64_t totalMicros,
                               std::initializer_list<std::pair<const char*, uint64_t>> fields = {});

}  // namespace ThorImplementation
