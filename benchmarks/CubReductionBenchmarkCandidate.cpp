#include "benchmarks/CubReductionBenchmarkCandidate.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation::CubReductionBenchmarking {
namespace {

std::vector<std::unique_ptr<ReductionCandidate>>& mutableCandidates() {
    static std::vector<std::unique_ptr<ReductionCandidate>> candidates;
    return candidates;
}

}  // namespace

void registerReductionCandidate(std::unique_ptr<ReductionCandidate> candidate) {
    if (!candidate) {
        throw std::invalid_argument("Reduction benchmark candidate registration requires a non-null candidate.");
    }
    const std::string_view name = candidate->getName();
    if (name.empty()) {
        throw std::invalid_argument("Reduction benchmark candidate names must be non-empty.");
    }

    for (const std::unique_ptr<ReductionCandidate>& existing : mutableCandidates()) {
        if (existing->getName() == name) {
            throw std::invalid_argument("Duplicate reduction benchmark candidate name: " + std::string(name));
        }
    }
    mutableCandidates().push_back(std::move(candidate));
}

const std::vector<std::unique_ptr<ReductionCandidate>>& getReductionCandidates() { return mutableCandidates(); }

}  // namespace ThorImplementation::CubReductionBenchmarking
