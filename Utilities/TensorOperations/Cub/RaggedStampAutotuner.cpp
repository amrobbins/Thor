#include "Utilities/TensorOperations/Cub/RaggedStampAutotuner.h"

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/TensorOperations/Cub/ReductionAutotuner.h"
#include "Utilities/TensorOperations/Cub/ReductionGpuMeasurementLock.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ThorImplementation::CubReductionInternal {
namespace {

// The site of a kind-RaggedKR / RaggedRK key: a layout version, then the site's fields.
constexpr uint64_t RaggedSiteVersion = 1;

uint32_t elementBytes(DataType dtype) { return raggedKRInputBytes(dtype); }  // 1, 2, 4, or 0 for any other dtype

bool valueOpRuns(CubReductionOp op) {
    return op == CubReductionOp::Sum || op == CubReductionOp::Mean || op == CubReductionOp::Min ||
           op == CubReductionOp::Max;
}

bool valueOutputRuns(DataType dtype) {
    return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2 || dtype == DataType::FP16 ||
           dtype == DataType::BF16 || dtype == DataType::FP32;
}

RaggedFamily checkedFamily(const RaggedSite& site) {
    const std::optional<RaggedFamily> family = raggedFamilyFor(site.arg, site.input_dtype, site.components);
    if (!family) throw std::invalid_argument("Ragged selection: no ragged family runs this input dtype and D");
    return *family;
}

void validateSite(const RaggedSite& site) {
    const auto fail = [](const std::string& what) { throw std::invalid_argument("Ragged selection: " + what); };
    (void)checkedFamily(site);
    if (site.offsets_dtype != DataType::UINT32 && site.offsets_dtype != DataType::UINT64)
        fail("the offsets must be UINT32 or UINT64");
    if (site.arg) {
        if (site.arg_op != CubArgReductionOp::ArgMin && site.arg_op != CubArgReductionOp::ArgMax)
            fail("the ARG operation must be ARGMIN or ARGMAX");
        if (site.output_dtype != DataType::UINT32 && site.output_dtype != DataType::UINT64)
            fail("the index output must be UINT32 or UINT64");
        // The launchers' rule, checked before the contest allocates an input of the capacity.
        if (site.output_dtype == DataType::UINT32 && site.components != 0 &&
            site.capacity_rows > uint64_t{0xFFFFFFFFU} / site.components)
            fail("a UINT32 index output needs capacity x D <= 2^32 - 1");
    } else {
        if (!valueOpRuns(site.value_op)) fail("the operation must be SUM, MEAN, MIN or MAX");
        if (!valueOutputRuns(site.output_dtype)) fail("the output must be FP8, FP16, BF16 or FP32");
    }
    const uint32_t bytes = elementBytes(site.input_dtype);
    if (site.input_head >= 16 || site.input_head % bytes != 0)
        fail("the input head must be a multiple of the element's bytes below 16");
}

std::vector<uint64_t> inputDims(const RaggedSite& site) {
    return site.components == 1 ? std::vector<uint64_t>{site.capacity_rows}
                                : std::vector<uint64_t>{site.capacity_rows, site.components};
}

std::vector<uint64_t> outputDims(const RaggedSite& site) {
    return site.components == 1 ? std::vector<uint64_t>{site.segments}
                                : std::vector<uint64_t>{site.segments, site.components};
}

// Decision 4: B segments of uniform length over all N capacity rows, offsets[i] = floor(i N / B) (clamped to the
// offset dtype's range). 1324: built incrementally, with N = q B + r: each step adds q rows and carries r / B, so no
// division per segment (1323: a 128-bit divide each, 39-55 ms of a stamp at 7-10M segments).
template <typename OffsetT>
void writeUniformOffsets(OffsetT* out, uint64_t capacity_rows, uint64_t segments) {
    const uint64_t cap = static_cast<uint64_t>(std::numeric_limits<OffsetT>::max());
    const uint64_t q = capacity_rows / segments, r = capacity_rows % segments;
    uint64_t row = 0, carry = 0;  // row = i q + floor(i r / B), carry = i r mod B
    for (uint64_t i = 0; i <= segments; ++i) {
        out[i] = static_cast<OffsetT>(std::min(row, cap));
        row += q;
        carry += r;
        if (carry >= segments) {
            carry -= segments;
            ++row;
        }
    }
}

Tensor uploadUniformOffsets(uint64_t capacity_rows, uint64_t segments, DataType offsets_dtype,
                            const TensorPlacement& placement, Stream& stream) {
    const TensorDescriptor descriptor(offsets_dtype, {segments + 1});
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), descriptor);
    if (offsets_dtype == DataType::UINT64) {
        writeUniformOffsets(cpu.getMemPtr<uint64_t>(), capacity_rows, segments);
    } else {
        writeUniformOffsets(cpu.getMemPtr<uint32_t>(), capacity_rows, segments);
    }
    Tensor gpu(placement, descriptor);
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();  // the upload source is `cpu`
    return gpu;
}

// A legal candidate: its selection, its launch for the site.
struct LegalCandidate {
    RaggedSelection selection;
    RaggedSelectedLaunch launch;
};

RaggedSelection tuneRagged(const RaggedSite& site, RaggedFamily family, const Stream& stream) {
    const int gpu = static_cast<int>(stream.getGpuNum());
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    const uint32_t bytes = elementBytes(site.input_dtype);
    const std::vector<uint64_t> input_dims = inputDims(site);
    const uint64_t input_elements = site.capacity_rows * site.components + site.input_head / bytes;
    const TensorDescriptor output_descriptor(site.output_dtype, outputDims(site));

    std::vector<LegalCandidate> legal;
    std::exception_ptr first_refusal;
    Tensor representative, offsets, output, workspace;
    const auto release = [&] {
        ScopedReductionGpuMeasurement cleanup(gpu);
        representative.dropReference();
        offsets.dropReference();
        output.dropReference();
        workspace.dropReference();
    };
    try {
        // Under the measurement lock: the representative input, the synthetic offsets, and one launch of each
        // candidate on them (a candidate the launcher refuses for this site is dropped).
        {
            ScopedReductionGpuMeasurement preparation(gpu);
            Stream prepare_stream = stream;
            std::vector<std::pair<RaggedSelection, RaggedSelectedLaunch>> planned;
            uint64_t workspace_bytes = 1;
            for (const RaggedSelection& candidate : raggedCandidateSelections(family)) {
                try {
                    const RaggedSelectedLaunch launch = raggedSelectedLaunch(candidate, site, gpu);
                    planned.emplace_back(candidate, launch);
                    workspace_bytes = std::max(workspace_bytes, launch.workspace_bytes);
                } catch (const std::invalid_argument&) {
                    if (!first_refusal) first_refusal = std::current_exception();
                }
            }
            // A flat UINT8 tensor (so every cold slot can be viewed at the site's head), 1.0 in every element.
            representative =
                Tensor(placement, TensorDescriptor(DataType::UINT8, {input_elements * uint64_t{bytes}}));
            representative.aliasByteStorageAs(site.input_dtype, {input_elements}, 0).fill(1.0, prepare_stream);
            offsets = uploadUniformOffsets(site.capacity_rows, site.segments, site.offsets_dtype, placement,
                                           prepare_stream);
            output = Tensor(placement, output_descriptor);
            workspace = Tensor(placement, TensorDescriptor(DataType::UINT8, {workspace_bytes}));
            const Tensor input = representative.aliasByteStorageAs(site.input_dtype, input_dims, site.input_head);
            for (const auto& [candidate, launch] : planned) {
                try {
                    launchRaggedSelection(candidate, site, launch, input, offsets, output, workspace,
                                          prepare_stream);
                    legal.push_back({candidate, launch});
                } catch (const std::invalid_argument&) {
                    if (!first_refusal) first_refusal = std::current_exception();
                }
            }
            prepare_stream.synchronize();
        }
        if (legal.empty()) {
            if (first_refusal) std::rethrow_exception(first_refusal);
            throw std::logic_error("Ragged selection has no candidate");
        }

        std::vector<ReductionAutotuning::Candidate> candidates;
        for (const LegalCandidate& c : legal) {
            ReductionAutotuning::Candidate candidate;
            candidate.name = c.selection.winner_name;
            candidate.scratch_bytes = c.launch.workspace_bytes;
            const RaggedSelection selection = c.selection;
            const RaggedSelectedLaunch launch = c.launch;
            const Tensor site_offsets = offsets;
            candidate.launch = [site, selection, launch, site_offsets, input_dims](
                                   const Tensor& slot, Tensor& out, Tensor& scratch, Stream& run_stream) {
                const Tensor in = slot.aliasByteStorageAs(site.input_dtype, input_dims, site.input_head);
                launchRaggedSelection(selection, site, launch, in, site_offsets, out, scratch, run_stream);
            };
            candidates.push_back(std::move(candidate));
        }

        ReductionAutotuning::RunResult result;
        {
            ScopedReductionGpuMeasurement measurement(gpu);
            ReductionAutotuning::Options options;
            options.measurement_count = RAGGED_CONTEST_ROUNDS;
            options.trim_count = RAGGED_CONTEST_TRIMMED;
            Stream tuning_stream = stream;
            result = ReductionAutotuning::autotune(candidates, representative, output_descriptor, tuning_stream,
                                                   options);
        }
        if (result.winner_index >= legal.size())
            throw std::logic_error("Ragged selection: the contest returned an out-of-range winner");
        // 1324: the first candidate in list order within RAGGED_TIE_MARGIN of the contest's best (the 1323 audit:
        // near-ties on the uniform offsets go to the list's order, the more frequent winner on real lengths).
        size_t pick = result.winner_index;
        const double best = result.candidates[pick].trimmed_mean_us;
        for (size_t i = 0; i < legal.size() && i < result.candidates.size(); ++i) {
            const double us = result.candidates[i].trimmed_mean_us;
            if (us > 0.0 && us <= best * (1.0 + RAGGED_TIE_MARGIN)) {
                pick = i;
                break;
            }
        }
        RaggedSelection selection = legal[pick].selection;
        selection.candidate_count = legal.size();
        for (size_t i = 0; i < legal.size(); ++i) {
            selection.contest_names.push_back(legal[i].selection.winner_name);
            selection.contest_us.push_back(i < result.candidates.size() ? result.candidates[i].trimmed_mean_us : 0.0);
        }
        release();
        return selection;
    } catch (...) {
        release();
        throw;
    }
}

}  // namespace

std::optional<RaggedFamily> raggedFamilyFor(bool arg, DataType input_dtype, uint64_t components) {
    const uint32_t bytes = elementBytes(input_dtype);
    if (bytes == 0 || components == 0) return std::nullopt;
    if (arg) {
        if (raggedKRArgSupported(input_dtype, components)) return RaggedFamily::KR;
        if (raggedRKArgSupported(input_dtype, components)) return RaggedFamily::RK;
        return std::nullopt;
    }
    if (components <= RaggedMergePath::raggedKRMaxComponents(bytes)) return RaggedFamily::KR;
    if (components <= RAGGED_FINISH_MAX_COMPONENTS && components * bytes >= RaggedMergePath::RAGGED_RK_MIN_ROW_BYTES &&
        components * bytes <= RaggedMergePath::RAGGED_RK_MAX_ROW_BYTES)
        return RaggedFamily::RK;
    return std::nullopt;
}

std::vector<uint64_t> raggedUniformOffsets(uint64_t capacity_rows, uint64_t segments, DataType offsets_dtype) {
    if (segments == 0) throw std::invalid_argument("raggedUniformOffsets: at least one segment");
    std::vector<uint64_t> out(segments + 1);
    if (offsets_dtype == DataType::UINT64) {
        writeUniformOffsets(out.data(), capacity_rows, segments);
    } else {
        std::vector<uint32_t> narrow(segments + 1);
        writeUniformOffsets(narrow.data(), capacity_rows, segments);
        std::copy(narrow.begin(), narrow.end(), out.begin());
    }
    return out;
}

std::vector<RaggedKRShape> raggedKRCandidateShapes() {
    // The audit's KR shapes (1275 on), the most frequent winner first (KR ARG 1301 / 1311 / 1313: e12 262-295 cells of
    // 600, e4 148-208, c8 132-173; c8, 8 CTAs per SM, wins the small cells).
    return {RaggedKRShape{16, 0, true, 12}, RaggedKRShape{16, 0, true, 4}, RaggedKRShape{16, 8, true, 4}};
}

std::vector<RaggedRKShape> raggedRKCandidateShapes() {
    // The audit's RK shapes (1285 on): the auto K tile cap, then the caps 64 and 512 (1318-1321: ta and t64 won most
    // cells, t512 120-175 of 600).
    return {RaggedRKShape{0, 0, true, 4}, RaggedRKShape{64, 0, true, 4}, RaggedRKShape{512, 0, true, 4}};
}

std::vector<RaggedSelection> raggedCandidateSelections(RaggedFamily family) {
    std::vector<RaggedSelection> out;
    if (family == RaggedFamily::KR) {
        for (const RaggedKRShape& shape : raggedKRCandidateShapes()) {
            RaggedSelection s;
            s.family = RaggedFamily::KR;
            s.kr_shape = shape;
            s.winner_name = raggedKRShapeName(shape);
            out.push_back(std::move(s));
        }
    } else {
        for (const RaggedRKShape& shape : raggedRKCandidateShapes()) {
            RaggedSelection s;
            s.family = RaggedFamily::RK;
            s.rk_shape = shape;
            s.winner_name = raggedRKShapeName(shape);
            out.push_back(std::move(s));
        }
    }
    return out;
}

std::string raggedSelectionShapeName(const RaggedSelection& selection) {
    return selection.family == RaggedFamily::KR ? raggedKRShapeName(selection.kr_shape)
                                                : raggedRKShapeName(selection.rk_shape);
}

RaggedSelectedLaunch raggedSelectedLaunch(const RaggedSelection& selection, const RaggedSite& site, int gpu) {
    if (raggedFamilyFor(site.arg, site.input_dtype, site.components) != selection.family)
        throw std::invalid_argument("Ragged selection: the selection's family does not run this site");
    RaggedSelectedLaunch launch;
    const uint32_t d32 = static_cast<uint32_t>(site.components);  // KR: D below 16
    if (selection.family == RaggedFamily::KR) {
        if (site.arg) {
            launch.grid_or_ctas = raggedKRArgGrid(selection.kr_shape, site.input_dtype, gpu, d32);
            launch.workspace_bytes = raggedArgWorkspaceBytes(site.input_dtype, launch.grid_or_ctas, d32);
        } else {
            launch.grid_or_ctas = raggedKRGrid(selection.kr_shape, site.input_dtype, gpu, d32);
            launch.workspace_bytes = raggedValueWorkspaceBytes(launch.grid_or_ctas, d32);
        }
    } else if (site.arg) {
        launch.grid_or_ctas = raggedRKArgCtas(selection.rk_shape, site.input_dtype, gpu);
        launch.workspace_bytes = raggedRKArgLaunch(selection.rk_shape, launch.grid_or_ctas, site.input_dtype,
                                                   site.components, site.capacity_rows)
                                     .workspace_bytes;
    } else {
        launch.grid_or_ctas = raggedRKCtas(selection.rk_shape, site.input_dtype, gpu);
        launch.workspace_bytes = raggedRKLaunch(selection.rk_shape, launch.grid_or_ctas, site.input_dtype,
                                                site.components, site.capacity_rows)
                                     .workspace_bytes;
    }
    return launch;
}

void launchRaggedSelection(const RaggedSelection& selection,
                           const RaggedSite& site,
                           const RaggedSelectedLaunch& launch,
                           const Tensor& input,
                           const Tensor& offsets,
                           Tensor& output,
                           Tensor& workspace,
                           Stream& stream) {
    if (selection.family == RaggedFamily::KR) {
        if (site.arg) {
            launchRaggedKRArg(site.arg_op, selection.kr_shape, launch.grid_or_ctas, input, offsets, output, workspace,
                              stream);
        } else {
            launchRaggedKRValue(site.value_op, selection.kr_shape, launch.grid_or_ctas, input, offsets, output,
                                workspace, stream);
        }
    } else if (site.arg) {
        launchRaggedRKArg(site.arg_op, selection.rk_shape, launch.grid_or_ctas, input, offsets, output, workspace,
                          stream);
    } else {
        launchRaggedRKValue(site.value_op, selection.rk_shape, launch.grid_or_ctas, input, offsets, output, workspace,
                            stream);
    }
}

DenseReductionSelectionKey raggedSelectionKey(const RaggedSite& site, const Stream& stream) {
    const RaggedFamily family = checkedFamily(site);
    DenseReductionSelectionKey key;
    key.kind =
        family == RaggedFamily::KR ? DenseReductionSelectionKind::RaggedKR : DenseReductionSelectionKind::RaggedRK;
    key.device = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    key.tuning_depth = CubReductionTuningDepth::Regular;  // no depth: every legal candidate is timed
    key.site = {RaggedSiteVersion,
                static_cast<uint64_t>(site.arg),
                site.arg ? static_cast<uint64_t>(site.arg_op) : static_cast<uint64_t>(site.value_op),
                static_cast<uint64_t>(site.input_dtype),
                static_cast<uint64_t>(site.offsets_dtype),
                static_cast<uint64_t>(site.output_dtype),
                site.components,
                site.capacity_rows,
                site.segments,
                site.input_head};
    return key;
}

bool raggedSiteRoutable(const RaggedSite& site) {
    try {
        validateSite(site);
        return true;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

std::shared_ptr<const RaggedRoutedStamp> raggedRouteSegmented(bool arg,
                                                              CubReductionOp value_op,
                                                              CubArgReductionOp arg_op,
                                                              const Tensor& input,
                                                              const Tensor& output,
                                                              const Tensor& offsets,
                                                              const Stream& stream) {
    const auto fail = [](const std::string& what) {
        throw std::invalid_argument("CUB segmented reduction: no ragged kernel runs this site: " + what);
    };
    const std::vector<uint64_t> input_dims = input.getDimensions();
    const std::vector<uint64_t> offsets_dims = offsets.getDimensions();
    if (input_dims.empty() || input_dims.front() == 0) fail("the input needs at least one row");
    if (offsets_dims.size() != 1 || offsets_dims.front() < 2) fail("the offsets must be [B + 1] with B >= 1");
    RaggedSite site;
    site.arg = arg;
    site.value_op = value_op;
    site.arg_op = arg_op;
    site.input_dtype = input.getDataType();
    site.offsets_dtype = offsets.getDataType();
    site.output_dtype = output.getDataType();
    site.capacity_rows = input_dims.front();
    site.components = input.getTotalNumElements() / site.capacity_rows;
    site.segments = offsets_dims.front() - 1;
    site.input_head = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(input.getMemPtr<void>()) & 15U);
    // 1330 (phase 9): an unroutable site is an error (1325-1329: the legacy kernels, deleted).
    if (!raggedFamilyFor(site.arg, site.input_dtype, site.components)) {
        const uint32_t bytes = elementBytes(site.input_dtype);
        fail(bytes == 0 ? "the input must be FP8, FP16, BF16 or FP32"
                        : "rows of " + std::to_string(site.components) + " components (" +
                              std::to_string(site.components * bytes) +
                              " bytes); ragged KR takes rows under 16 bytes, ragged RK rows of 16 bytes to 2 GiB of "
                              "at most 2^30 components");
    }
    try {
        validateSite(site);
    } catch (const std::invalid_argument& error) {
        fail(error.what());
    }
    auto routed = std::make_shared<RaggedRoutedStamp>();
    routed->site = site;
    routed->selection = selectCachedRagged(site, stream);
    routed->launch = raggedSelectedLaunch(routed->selection, site, static_cast<int>(stream.getGpuNum()));
    return routed;
}

void runRaggedRouted(const RaggedRoutedStamp& routed,
                     const Tensor& input,
                     const Tensor& offsets,
                     Tensor& output,
                     Tensor& workspace,
                     Stream& stream) {
    const RaggedSite& site = routed.site;
    Tensor view = input;
    view.reshape(site.components == 1 ? std::vector<uint64_t>{site.capacity_rows}
                                      : std::vector<uint64_t>{site.capacity_rows, site.components});
    launchRaggedSelection(routed.selection, site, routed.launch, view, offsets, output, workspace, stream);
}

std::vector<std::string> raggedRoutedPassNames(const RaggedRoutedStamp& routed) {
    return {routed.selection.winner_name, routed.site.arg ? "ragged_finish_arg" : "ragged_finish"};
}

RaggedSelection selectCachedRagged(const RaggedSite& site, const Stream& stream) {
    validateSite(site);
    const RaggedFamily family = checkedFamily(site);
    if (site.capacity_rows == 0 || site.segments == 0) return raggedCandidateSelections(family).front();
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const DenseReductionSelectionKey key = raggedSelectionKey(site, stream);
    return denseReductionSelectionCache().getOrTune<RaggedSelection>(
        key, [&] { return tuneRagged(site, family, stream); });
}

}  // namespace ThorImplementation::CubReductionInternal
