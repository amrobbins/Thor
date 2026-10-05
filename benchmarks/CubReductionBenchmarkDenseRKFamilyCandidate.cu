#include "benchmarks/CubReductionBenchmarkCandidate.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"
#include "Utilities/Common/ReusableEventPool.h"

#include <cuda/std/functional>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionBenchmarking {
namespace {

using namespace CubReductionInternal;

[[nodiscard]] DenseRKValueOperation valueOperation(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
            return DenseRKValueOperation::Sum;
        case CubReductionOp::Mean:
            return DenseRKValueOperation::Mean;
        case CubReductionOp::L1Norm:
            return DenseRKValueOperation::L1;
        case CubReductionOp::L2Norm:
            return DenseRKValueOperation::L2;
        case CubReductionOp::SumSquares:
            return DenseRKValueOperation::SumSquares;
        case CubReductionOp::Product:
            return DenseRKValueOperation::Product;
        case CubReductionOp::Min:
            return DenseRKValueOperation::Minimum;
        case CubReductionOp::Max:
            return DenseRKValueOperation::Maximum;
    }
    throw std::logic_error("Dense RK family benchmark received an invalid value operation.");
}

[[nodiscard]] std::vector<uint64_t> runDimensions(const DenseReductionProblem& problem) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(problem.runs.size());
    for (const DenseReductionRun& run : problem.runs) {
        dimensions.push_back(run.extent);
    }
    return dimensions;
}

[[nodiscard]] DenseRKOccupancyInfo queryDenseRKFamilyExactOccupancy(const DenseRKFamilyPhysicalPlan& plan,
                                                                     void* user_context);

[[nodiscard]] int queryMultiprocessors(int gpu_num) {
    int multiprocessors = 0;
    const cudaError_t status = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, gpu_num);
    if (status != cudaSuccess || multiprocessors <= 0) {
        throw std::runtime_error(std::string("Dense RK family census could not query GPU multiprocessor count: ")
                                 + cudaGetErrorString(status));
    }
    return multiprocessors;
}

[[nodiscard]] const char* strategyName(DenseRKStrategy strategy) {
    switch (strategy) {
        case DenseRKStrategy::KParallel:
            return "k_parallel";
        case DenseRKStrategy::RCooperative:
            return "r_cooperative";
    }
    return "unknown";
}

[[nodiscard]] const char* accessName(DenseRKAccess access) {
    switch (access) {
        case DenseRKAccess::Aligned:
            return "aligned";
        case DenseRKAccess::Rotated:
            return "rotated";
        case DenseRKAccess::FlatRows:
            return "flat_rows";
    }
    return "unknown";
}


struct ExecutableRKStage {
    std::shared_ptr<const DenseRKFamilyPhysicalPlan> plan;
    Tensor input;
    mutable Tensor output;
    CubReductionGeometry geometry;
};

template <typename InputT, typename InputTransformT>
void launchPacketAdaptiveAdditiveForTransform(const CubReductionStageSemantics& semantics,
                                              const DenseRKFamilyPhysicalPlan& plan,
                                              const Tensor& input,
                                              Tensor& output,
                                              const CubReductionGeometry& geometry,
                                              InputTransformT input_transform,
                                              Stream& stream) {
    if (semantics.combine != CubReductionStageCombine::Sum) {
        throw std::logic_error("Packet-adaptive R-cooperative family pass requires additive combine semantics.");
    }
    const float divisor = semantics.finalize == CubReductionStageFinalize::Divide
                              ? static_cast<float>(semantics.finalize_divisor)
                              : 1.0f;
    const bool square_root = semantics.finalize == CubReductionStageFinalize::SquareRoot;
    if (semantics.finalize != CubReductionStageFinalize::Identity
        && semantics.finalize != CubReductionStageFinalize::Divide
        && semantics.finalize != CubReductionStageFinalize::SquareRoot) {
        throw std::logic_error("Packet-adaptive R-cooperative family pass received unsupported finalization.");
    }

    const auto launch_packet = [&]<size_t PacketBytes>() {
        if (plan.packet_bytes != PacketBytes) {
            return false;
        }
        const int expected_warps = chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, PacketBytes>(geometry);
        if (plan.cooperative_warps_per_tile != static_cast<uint32_t>(expected_warps)) {
            throw std::logic_error("Dense RK family packet-adaptive plan disagrees with the production warp policy.");
        }
        const uint32_t expected_block_warps =
            std::max<uint32_t>(static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK),
                               static_cast<uint32_t>(expected_warps));
        if (plan.block_threads != expected_block_warps * static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS)) {
            throw std::logic_error("Dense RK family packet-adaptive plan disagrees with the production CTA policy.");
        }
        launchAdaptiveCooperativeTiledFixedSegmentReductionForPacketBytes<InputT,
                                                                          cuda::std::plus<float>,
                                                                          InputTransformT,
                                                                          AdditiveFinalizeFp32,
                                                                          PacketBytes>(input.getMemPtr<InputT>(),
                                                                                       output.getMemPtr<void>(),
                                                                                       output.getDataType(),
                                                                                       geometry,
                                                                                       cuda::std::plus<float>{},
                                                                                       0.0f,
                                                                                       input_transform,
                                                                                       AdditiveFinalizeFp32{divisor, square_root},
                                                                                       1.0f,
                                                                                       stream.getStream());
        return true;
    };

    if (launch_packet.template operator()<16>() || launch_packet.template operator()<8>()
        || launch_packet.template operator()<4>()) {
        return;
    }
    throw std::logic_error("Dense RK family packet-adaptive plan selected an invalid packet width.");
}

template <typename InputT>
void launchPacketAdaptiveAdditive(const CubReductionStageSemantics& semantics,
                                  const DenseRKFamilyPhysicalPlan& plan,
                                  const Tensor& input,
                                  Tensor& output,
                                  const CubReductionGeometry& geometry,
                                  Stream& stream) {
    switch (semantics.input_transform) {
        case CubReductionStageInputTransform::Identity:
            launchPacketAdaptiveAdditiveForTransform<InputT>(
                semantics, plan, input, output, geometry, IdentityFp32{}, stream);
            return;
        case CubReductionStageInputTransform::AbsoluteValue:
            launchPacketAdaptiveAdditiveForTransform<InputT>(
                semantics, plan, input, output, geometry, AbsoluteValueFp32{}, stream);
            return;
        case CubReductionStageInputTransform::Square:
            launchPacketAdaptiveAdditiveForTransform<InputT>(
                semantics, plan, input, output, geometry, SquareFp32{}, stream);
            return;
    }
    throw std::logic_error("Packet-adaptive R-cooperative family pass received an invalid input transform.");
}

void launchPacketAdaptiveComplete(const CubReductionStageSemantics& semantics,
                                  const DenseRKFamilyPhysicalPlan& plan,
                                  const Tensor& input,
                                  Tensor& output,
                                  const CubReductionGeometry& geometry,
                                  Stream& stream) {
    switch (input.getDataType()) {
        case DataType::FP16:
            launchPacketAdaptiveAdditive<__half>(semantics, plan, input, output, geometry, stream);
            return;
        case DataType::BF16:
            launchPacketAdaptiveAdditive<__nv_bfloat16>(semantics, plan, input, output, geometry, stream);
            return;
        case DataType::FP32:
            launchPacketAdaptiveAdditive<float>(semantics, plan, input, output, geometry, stream);
            return;
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            launchPacketAdaptiveAdditive<__nv_fp8_e4m3>(semantics, plan, input, output, geometry, stream);
            return;
        case DataType::FP8_E5M2:
            launchPacketAdaptiveAdditive<__nv_fp8_e5m2>(semantics, plan, input, output, geometry, stream);
            return;
#endif
        default:
            throw std::logic_error("Dense RK family packet-adaptive pass received an unsupported dtype.");
    }
}

class StampedDenseRKFamilyCandidate final : public StampedReductionCandidate {
   public:
    StampedDenseRKFamilyCandidate(CubReductionOp op,
                                  Tensor input,
                                  std::vector<uint32_t> axes,
                                  const Stream& stream)
        : op_(op),
          original_input_(std::move(input)),
          public_geometry_(CubReduction::analyzeValueGeometry(op_, original_input_.getDimensions(), axes)),
          output_(original_input_.getPlacement(),
                  TensorDescriptor(original_input_.getDataType(), public_geometry_.output_dimensions)) {
        if (public_geometry_.path != CubReductionPath::TiledFixedSegment || !original_input_.isDenseContiguous()) {
            throw std::invalid_argument(
                "Dense RK family benchmark requires ordinary dense TiledFixedSegment geometry.");
        }

        DenseReductionOutputSpec output_spec;
        output_spec.produce_value = true;
        output_spec.value_dtype = original_input_.getDataType();
        output_spec.produce_index = false;
        DenseReductionProblem problem = makeInitialDenseReductionProblem(original_input_.getDimensions(),
                                                                         axes,
                                                                         DenseReductionAggregateKind::Value,
                                                                         output_spec);
        std::vector<DenseReductionSite> sites = enumerateDenseReductionSites(problem);
        if (sites.size() != 1 || classifyDenseReducerFamily(problem, sites.front()) != DenseReducerFamily::RK) {
            throw std::invalid_argument("Dense RK family benchmark requires exactly one RK reduction site.");
        }

        DenseRKPlanningContext context;
        context.original_input_dtype = original_input_.getDataType();
        context.value_operation = valueOperation(op_);
        context.multiprocessors = static_cast<uint32_t>(queryMultiprocessors(stream.getGpuNum()));
        context.occupancy_query = queryDenseRKFamilyExactOccupancy;
        context.occupancy_query_context = nullptr;
        context.allow_generic_rk_fallback = false;

        Tensor current_input = original_input_;
        current_input.reshape(runDimensions(problem));
        const uint64_t original_reduction_size = public_geometry_.reduction_size;

        while (hasDenseReductionSites(problem)) {
            sites = enumerateDenseReductionSites(problem);
            if (sites.size() != 1 || classifyDenseReducerFamily(problem, sites.front()) != DenseReducerFamily::RK) {
                throw std::logic_error(
                    "Dense RK family census expected every residual pass to remain a single RK problem.");
            }

            DenseReductionCandidate candidate;
            try {
                candidate = ReducersDenseRK::propose(problem, sites.front(), context);
            } catch (const std::logic_error& error) {
                const std::string_view message(error.what());
                if (message.find("no calibrated/feasible KParallel or RCooperative") != std::string_view::npos) {
                    throw ReductionCandidateCoverageGap(error.what());
                }
                throw;
            }
            auto plan = std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(candidate.physical_plan);
            if (!plan) {
                throw std::logic_error("Dense RK family census received a non-modern RK physical plan.");
            }
            if (!denseRKIsFamilyImplementation(plan->implementation)) {
                throw std::logic_error("Dense RK family census selected legacy overlap.");
            }

            CubReductionGeometry geometry = CubReduction::analyzeGeometry(
                plan->input_dimensions, std::vector<uint32_t>{plan->reduction_axis});
            if (geometry.path != CubReductionPath::TiledFixedSegment) {
                throw std::logic_error("Dense RK family census stage did not analyze as TiledFixedSegment.");
            }

            Tensor stage_output;
            if (plan->progress == DenseRKProgress::Complete) {
                stage_output = output_;
            } else {
                stage_output = Tensor(original_input_.getPlacement(),
                                      TensorDescriptor(DataType::FP32,
                                                       {geometry.outer_size,
                                                        plan->shards_per_output,
                                                        geometry.inner_size}));
                intermediate_storage_.push_back(stage_output);
            }

            stages_.push_back(ExecutableRKStage{plan, current_input, stage_output, geometry});
            if (plan_chain_.empty()) {
                plan_chain_ = strategyName(plan->strategy);
            } else {
                plan_chain_ += ">";
                plan_chain_ += strategyName(plan->strategy);
            }
            plan_chain_ += ":";
            plan_chain_ += accessName(plan->access);
            plan_chain_ += plan->progress == DenseRKProgress::Complete ? ":complete" : ":staged";

            problem = candidate.next_problem;
            if (plan->progress == DenseRKProgress::Staged) {
                Tensor next_input = stage_output;
                next_input.reshape(runDimensions(problem));
                current_input = next_input;
            }
        }

        if (stages_.empty() || stages_.back().plan->progress != DenseRKProgress::Complete) {
            throw std::logic_error("Dense RK family census plan did not terminate in one Complete pass.");
        }
        total_reduction_size_ = original_reduction_size;
    }

    void runOn(Stream& stream) const override {
        for (const ExecutableRKStage& stage : stages_) {
            const CubReductionStageSemantics semantics =
                makeValueReductionStageSemantics(op_, stage.plan->role, total_reduction_size_);
            switch (stage.plan->implementation) {
                case DenseRKProductionImplementation::KParallelPass: {
                    const KParallelTiledStagePlan kplan{
                        stage.plan->progress == DenseRKProgress::Complete ? TiledRKStageTopology::Complete
                                                                         : TiledRKStageTopology::Staged,
                        stage.plan->packet_bytes,
                        stage.plan->block_threads,
                        stage.plan->shards_per_output};
                    launchKParallelTiledStage(
                        semantics, stage.input, stage.output, stage.geometry, kplan, 1.0f, stream);
                    break;
                }
                case DenseRKProductionImplementation::RCooperativeAlignedComplete:
                    if (stage.plan->progress != DenseRKProgress::Complete) {
                        throw std::logic_error("Packet-adaptive R-cooperative family pass must be Complete.");
                    }
                    launchPacketAdaptiveComplete(
                        semantics, *stage.plan, stage.input, stage.output, stage.geometry, stream);
                    break;
                case DenseRKProductionImplementation::RCooperativeAlignedStaged:
                    if (stage.plan->progress != DenseRKProgress::Staged) {
                        throw std::logic_error("Aligned sharded R-cooperative family pass must be Staged.");
                    }
                    launchCooperativeShardedTiledFirstStage(semantics,
                                                            stage.input,
                                                            stage.output,
                                                            stage.geometry,
                                                            stage.plan->rows_per_shard,
                                                            stage.plan->shards_per_output,
                                                            stream);
                    break;
                case DenseRKProductionImplementation::RCooperativeRotatedStaged:
                    if (stage.plan->progress != DenseRKProgress::Staged) {
                        throw std::logic_error("Rotated R-cooperative family pass must be Staged.");
                    }
                    launchAwkwardAlignmentRotatedShardedFirstStage(semantics,
                                                                   stage.input,
                                                                   stage.output,
                                                                   stage.geometry,
                                                                   stage.plan->rows_per_shard,
                                                                   stage.plan->shards_per_output,
                                                                   stream);
                    break;
                case DenseRKProductionImplementation::RCooperativeFlatRowsStaged:
                    if (stage.plan->progress != DenseRKProgress::Staged) {
                        throw std::logic_error("Narrow-flat R-cooperative family pass must be Staged.");
                    }
                    launchNarrowLowPrecisionFlatRCooperativeFirstStage(semantics,
                                                                        stage.input,
                                                                        stage.output,
                                                                        stage.geometry,
                                                                        stage.plan->rows_per_shard,
                                                                        stage.plan->shards_per_output,
                                                                        stream);
                    break;
                default:
                    throw std::logic_error(
                        "Dense RK family census selected an implementation that is not executable by the modern gate.");
            }
        }
    }

    [[nodiscard]] const Tensor& getOutputTensor() const override { return output_; }

    [[nodiscard]] size_t getWorkspaceSizeInBytes() const override {
        size_t bytes = 0;
        for (const Tensor& tensor : intermediate_storage_) {
            bytes += tensor.getArraySizeInBytes();
        }
        return bytes;
    }

    [[nodiscard]] CandidateLaunchMetadata getLaunchMetadata() const override {
        CandidateLaunchMetadata metadata;
        metadata.implementation = "dense_rk_family";
        metadata.strategy = plan_chain_;
        if (!stages_.empty()) {
            const DenseRKFamilyPhysicalPlan& first = *stages_.front().plan;
            if (first.packet_bytes != 0) {
                metadata.vector_elements_per_load = static_cast<uint32_t>(
                    first.packet_bytes / TensorDescriptor::getElementSizeInBytes(first.input_dtype));
            }
            if (first.block_threads != 0) {
                metadata.block_threads = first.block_threads;
            }
            if (first.first_stage_blocks != 0) {
                metadata.first_stage_blocks = first.first_stage_blocks;
            }
            metadata.shards_per_output = first.shards_per_output;
        }
        return metadata;
    }

   private:
    CubReductionOp op_;
    Tensor original_input_;
    CubReductionGeometry public_geometry_;
    mutable Tensor output_;
    std::vector<Tensor> intermediate_storage_;
    std::vector<ExecutableRKStage> stages_;
    std::string plan_chain_;
    uint64_t total_reduction_size_ = 1;
};

class DenseRKFamilyCandidate final : public ReductionCandidate {
   public:
    [[nodiscard]] std::string_view getName() const override { return "dense_rk_family"; }

    [[nodiscard]] bool supports(CubReductionOp,
                                const TensorDescriptor& input_descriptor,
                                const std::vector<uint32_t>& axes) const override {
        const DataType dtype = input_descriptor.getDataType();
        if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
            return false;
        }
        const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input_descriptor.getDimensions(), axes);
        if (geometry.path != CubReductionPath::TiledFixedSegment) {
            return false;
        }
        DenseReductionOutputSpec output_spec;
        output_spec.produce_value = true;
        output_spec.value_dtype = dtype;
        DenseReductionProblem problem = makeInitialDenseReductionProblem(input_descriptor.getDimensions(),
                                                                         axes,
                                                                         DenseReductionAggregateKind::Value,
                                                                         output_spec);
        const std::vector<DenseReductionSite> sites = enumerateDenseReductionSites(problem);
        return sites.size() == 1 && classifyDenseReducerFamily(problem, sites.front()) == DenseReducerFamily::RK;
    }

    [[nodiscard]] std::unique_ptr<StampedReductionCandidate> stamp(CubReductionOp op,
                                                                    const Tensor& input,
                                                                    const std::vector<uint32_t>& axes,
                                                                    const Stream& stream) const override {
        if (!supports(op, input.getDescriptor(), axes) || !input.isDenseContiguous()) {
            throw std::invalid_argument("Dense RK family benchmark candidate requires one dense contiguous RK site.");
        }
        return std::make_unique<StampedDenseRKFamilyCandidate>(op, input, axes, stream);
    }
};



constexpr int RK_SINGLE_PASS_WARMUPS = 2;
constexpr int RK_SINGLE_PASS_SAMPLES = 3;
// The FP8 staged-layout sweep is a policy-calibration benchmark, not a broad census. Its rows are close enough in
// absolute time that three samples occasionally produce a bimodal median and a false 3-5% policy miss. Use a wider
// odd sample count here so the median remains the reported statistic while being materially harder for one transient
// scheduling/cache event to move.
constexpr int FP8_STAGED_LAYOUT_TIMING_SAMPLES = 7;
constexpr uint64_t RK_SINGLE_PASS_MAX_INPUT_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr double RK_SINGLE_PASS_FILL_VALUE = 1.0 / 1024.0;

struct RKSinglePassSweepCase {
    std::string family;
    std::string name;
    uint64_t outer = 1;
    uint64_t reduction = 1;
    uint64_t inner = 1;
};

struct RKSinglePassTiming {
    double best_ms = 0.0;
    double median_ms = 0.0;
    double worst_ms = 0.0;
};

struct RKSinglePassDeviceGeometry {
    int device = 0;
    uint32_t sm_count = 0;
    uint32_t warp_size = 0;
    uint32_t max_threads_per_sm = 0;
    uint32_t max_threads_per_block = 0;
    uint32_t max_blocks_per_sm = 0;
    uint64_t shared_mem_per_sm = 0;
    uint32_t registers_per_sm = 0;
};

void checkSinglePassCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
    }
}

[[nodiscard]] RKSinglePassDeviceGeometry querySinglePassDeviceGeometry() {
    int device = 0;
    checkSinglePassCuda(cudaGetDevice(&device), "cudaGetDevice(rk_single_pass_focus)");
    cudaDeviceProp properties{};
    checkSinglePassCuda(cudaGetDeviceProperties(&properties, device),
                        "cudaGetDeviceProperties(rk_single_pass_focus)");
    if (properties.multiProcessorCount <= 0 || properties.warpSize <= 0
        || properties.maxThreadsPerMultiProcessor <= 0 || properties.maxThreadsPerBlock <= 0
        || properties.maxBlocksPerMultiProcessor <= 0 || properties.sharedMemPerMultiprocessor == 0
        || properties.regsPerMultiprocessor <= 0) {
        throw std::runtime_error("RK focused single-pass sweep received invalid CUDA device geometry.");
    }
    return RKSinglePassDeviceGeometry{device,
                                      static_cast<uint32_t>(properties.multiProcessorCount),
                                      static_cast<uint32_t>(properties.warpSize),
                                      static_cast<uint32_t>(properties.maxThreadsPerMultiProcessor),
                                      static_cast<uint32_t>(properties.maxThreadsPerBlock),
                                      static_cast<uint32_t>(properties.maxBlocksPerMultiProcessor),
                                      static_cast<uint64_t>(properties.sharedMemPerMultiprocessor),
                                      static_cast<uint32_t>(properties.regsPerMultiprocessor)};
}

[[nodiscard]] uint64_t singlePassCeilDiv(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0) {
        throw std::logic_error("RK single-pass sweep denominator must be non-zero.");
    }
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] const char* singlePassDtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
            return "fp8_e4m3";
        case DataType::FP8_E5M2:
            return "fp8_e5m2";
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        case DataType::FP32:
            return "fp32";
        default:
            return "unsupported";
    }
}

[[nodiscard]] uint64_t checkedSinglePassElements(uint64_t outer, uint64_t reduction, uint64_t inner) {
    if (outer == 0 || reduction == 0 || inner == 0
        || outer > std::numeric_limits<uint64_t>::max() / reduction
        || outer * reduction > std::numeric_limits<uint64_t>::max() / inner) {
        throw std::overflow_error("RK single-pass sweep input element count overflowed.");
    }
    return outer * reduction * inner;
}

[[nodiscard]] std::vector<RKSinglePassSweepCase> makeRKSinglePassSweepCases() {
    std::vector<RKSinglePassSweepCase> cases;
    const auto add = [&](std::string family, std::string name, uint64_t outer, uint64_t reduction, uint64_t inner) {
        for (const RKSinglePassSweepCase& existing : cases) {
            if (existing.outer == outer && existing.reduction == reduction && existing.inner == inner) {
                return;
            }
        }
        cases.push_back(RKSinglePassSweepCase{std::move(family), std::move(name), outer, reduction, inner});
    };

    // First-pass retained-width sweep: dense around every packet/tile boundary implicated by the prior RK censuses.
    for (uint64_t inner : {15ULL, 16ULL, 17ULL, 31ULL, 32ULL, 33ULL, 63ULL, 64ULL, 65ULL,
                           127ULL, 128ULL, 129ULL, 255ULL, 256ULL, 257ULL, 383ULL, 384ULL, 385ULL,
                           511ULL, 512ULL, 513ULL, 767ULL, 768ULL, 769ULL, 1023ULL, 1024ULL, 1025ULL,
                           1535ULL, 1536ULL, 1537ULL, 2047ULL, 2048ULL, 2049ULL, 4095ULL, 4096ULL, 4097ULL}) {
        add("retained_width", "o1_r32768_i" + std::to_string(inner), 1, 32768, inner);
    }

    // Reduction-depth sweep on one naturally aligned and one awkward retained width.
    for (uint64_t reduction : {32ULL, 64ULL, 128ULL, 256ULL, 512ULL, 1024ULL, 2048ULL, 4096ULL,
                               8192ULL, 16384ULL, 32768ULL, 65536ULL, 104832ULL, 131072ULL}) {
        add("reduction_depth", "o1_r" + std::to_string(reduction) + "_i256", 1, reduction, 256);
        add("reduction_depth", "o1_r" + std::to_string(reduction) + "_i257", 1, reduction, 257);
    }

    // Output-count sweep catches the point at which a Complete kernel has enough naturally independent work.
    for (uint64_t outer : {1ULL, 2ULL, 4ULL, 8ULL, 16ULL, 32ULL, 64ULL}) {
        for (uint64_t inner : {17ULL, 256ULL, 513ULL, 4097ULL}) {
            add("output_count",
                "o" + std::to_string(outer) + "_r32768_i" + std::to_string(inner),
                outer,
                32768,
                inner);
        }
    }

    // Wide-K controls use a smaller R so the exact shape stays comfortably below the first-pass memory ceiling.
    for (uint64_t inner : {4096ULL, 4097ULL, 8192ULL, 8193ULL, 16384ULL, 16385ULL,
                           32768ULL, 32769ULL, 65536ULL, 65537ULL}) {
        add("wide_retained", "o1_r2048_i" + std::to_string(inner), 1, 2048, inner);
    }
    return cases;
}

[[nodiscard]] uint64_t focusScaledCount(uint32_t sm_count, uint32_t numerator, uint32_t denominator) {
    return std::max<uint64_t>(1, singlePassCeilDiv(static_cast<uint64_t>(sm_count) * numerator, denominator));
}

[[nodiscard]] std::vector<RKSinglePassSweepCase> makeRKSinglePassFocusCases(uint32_t sm_count) {
    std::vector<RKSinglePassSweepCase> cases;
    const auto add = [&](std::string family, std::string name, uint64_t outer, uint64_t reduction, uint64_t inner) {
        for (const RKSinglePassSweepCase& existing : cases) {
            if (existing.outer == outer && existing.reduction == reduction && existing.inner == inner) {
                return;
            }
        }
        cases.push_back(RKSinglePassSweepCase{std::move(family), std::move(name), outer, reduction, inner});
    };

    // Fill in the retained-width midpoints between the broad sweep's power-of-two/packet boundaries. These rows are
    // where K-parallel versus aligned/rotated R-cooperative winners changed most often in the first-pass evidence.
    constexpr std::array<uint64_t, 13> crossover_widths = {
        24, 48, 96, 160, 192, 224, 320, 448, 640, 896, 1280, 1792, 3072};
    for (uint64_t reduction : {2048ULL, 8192ULL, 32768ULL}) {
        for (uint64_t inner : crossover_widths) {
            add("crossover_width",
                "o1_r" + std::to_string(reduction) + "_i" + std::to_string(inner),
                1,
                reduction,
                inner);
        }
    }

    // Device-relative Complete supply. K=16 keeps deep-R inputs bounded enough to probe several CTA/SM supply points;
    // K=256 is a representative wider retained domain. These target CTAs are deliberately named as supply, not waves:
    // actual resident CTAs per SM remain specialization/resource dependent.
    constexpr std::array<std::pair<uint32_t, uint32_t>, 8> supply_scales = {{{1, 4}, {1, 2}, {3, 4}, {1, 1},
                                                                              {3, 2}, {2, 1}, {3, 1}, {4, 1}}};
    for (uint64_t reduction : {512ULL, 2048ULL, 8192ULL, 32768ULL}) {
        for (uint64_t inner : {16ULL, 256ULL}) {
            for (const auto& [num, den] : supply_scales) {
                const uint64_t outer = focusScaledCount(sm_count, num, den);
                add("complete_supply",
                    "sm" + std::to_string(num) + "of" + std::to_string(den) + "_r"
                        + std::to_string(reduction) + "_i" + std::to_string(inner),
                    outer,
                    reduction,
                    inner);
            }
        }
    }

    // Staged supply boundary. With rows128 and O=1, R approximately equals 128 * first-stage shard count. Generate
    // R from the current device SM count so the sweep directly surrounds sub-wave, one-wave, and multi-wave supply.
    // Aligned and awkward K pairs let the same cases compare K-parallel, aligned R-cooperative and rotated R-cooperative.
    constexpr std::array<std::pair<uint32_t, uint32_t>, 7> staged_scales = {{{1, 2}, {1, 1}, {2, 1}, {3, 1},
                                                                              {4, 1}, {6, 1}, {8, 1}}};
    for (uint64_t inner : {16ULL, 17ULL, 256ULL, 257ULL}) {
        for (const auto& [num, den] : staged_scales) {
            const uint64_t target_shards = focusScaledCount(sm_count, num, den);
            const uint64_t reduction = std::max<uint64_t>(128, target_shards * 128ULL);
            add("staged_supply",
                "sm" + std::to_string(num) + "of" + std::to_string(den) + "_r"
                    + std::to_string(reduction) + "_i" + std::to_string(inner),
                1,
                reduction,
                inner);
        }
    }

    return cases;
}

template <typename InputT>
[[nodiscard]] bool supportsSinglePassKParallelPacket(const CubReductionGeometry& geometry, size_t packet_bytes) {
    const bool supported_packet = packet_bytes == 4 || packet_bytes == 8 || packet_bytes == 16
                                  || (sizeof(InputT) == 1 && packet_bytes == 2);
    if (!supported_packet || packet_bytes < sizeof(InputT) || packet_bytes % sizeof(InputT) != 0) {
        return false;
    }
    return geometry.inner_size % static_cast<uint64_t>(packet_bytes / sizeof(InputT)) == 0;
}

template <typename InputT>
[[nodiscard]] uint32_t singlePassKParallelBlockThreads(const CubReductionGeometry& geometry, size_t packet_bytes) {
    const uint64_t items_per_lane = packet_bytes / sizeof(InputT);
    const uint64_t packets = singlePassCeilDiv(geometry.inner_size, items_per_lane);
    const uint64_t required_warps = singlePassCeilDiv(packets, TILED_REDUCTION_WARP_THREADS);
    uint32_t warps = 1;
    while (warps < static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK)
           && static_cast<uint64_t>(warps) < required_warps) {
        warps <<= 1;
    }
    return warps * static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS);
}

template <typename InputT>
[[nodiscard]] uint64_t singlePassKParallelComponentTiles(const CubReductionGeometry& geometry,
                                                         size_t packet_bytes,
                                                         uint32_t block_threads) {
    const uint64_t components_per_cta =
        static_cast<uint64_t>(block_threads) * static_cast<uint64_t>(packet_bytes / sizeof(InputT));
    return singlePassCeilDiv(geometry.inner_size, components_per_cta);
}

[[nodiscard]] uint64_t singlePassAlignedRCoopTiles(const CubReductionGeometry& geometry, DataType dtype) {
    const uint64_t items_per_lane = TensorDescriptor::getElementSizeInBytes(dtype) == 1 ? 16ULL
                                    : dtype == DataType::FP32 ? 4ULL
                                                            : 8ULL;
    return singlePassCeilDiv(geometry.inner_size, 32ULL * items_per_lane);
}

[[nodiscard]] bool singlePassAlignedRCoopSupports(const CubReductionGeometry& geometry, DataType dtype) {
    if (TensorDescriptor::getElementSizeInBytes(dtype) == 1) {
        return false;  // FP8 RCooperative calibration intentionally starts with FlatRows/Rotated, not aligned staging.
    }
    const uint64_t items_per_lane = dtype == DataType::FP32 ? 4ULL : 8ULL;
    return geometry.inner_size % items_per_lane == 0;
}

[[nodiscard]] uint64_t singlePassRotatedRCoopTiles(const CubReductionGeometry& geometry,
                                                        DataType dtype,
                                                        size_t packet_bytes,
                                                        bool exact_fp8_tile = true) {
    const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    if (packet_bytes < element_bytes || packet_bytes % element_bytes != 0) {
        throw std::logic_error("Rotated R-cooperative packet width is not divisible by the input element size.");
    }
    const uint64_t items_per_lane = packet_bytes / element_bytes;
    const uint64_t packet_span = 32ULL * items_per_lane;
    const uint64_t logical_tile = element_bytes == 1 && exact_fp8_tile
                                      ? packet_span
                                      : packet_span - (items_per_lane - 1ULL);
    return singlePassCeilDiv(geometry.inner_size, logical_tile);
}

[[nodiscard]] uint64_t singlePassRotatedRCoopTiles(const CubReductionGeometry& geometry, DataType dtype) {
    return singlePassRotatedRCoopTiles(geometry, dtype, 16, true);
}

[[nodiscard]] size_t singlePassPacketAdaptivePacketBytes(const CubReductionGeometry& geometry, DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return chooseAdaptiveCooperativeTiledPacketBytes<__half>(geometry);
        case DataType::BF16:
            return chooseAdaptiveCooperativeTiledPacketBytes<__nv_bfloat16>(geometry);
        case DataType::FP32:
            return chooseAdaptiveCooperativeTiledPacketBytes<float>(geometry);
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return chooseAdaptiveCooperativeTiledPacketBytes<__nv_fp8_e4m3>(geometry);
        case DataType::FP8_E5M2:
            return chooseAdaptiveCooperativeTiledPacketBytes<__nv_fp8_e5m2>(geometry);
#endif
        default:
            throw std::logic_error("RK single-pass packet-adaptive sweep received unsupported dtype.");
    }
}

[[nodiscard]] uint32_t singlePassPacketAdaptiveWarps(const CubReductionGeometry& geometry,
                                                     DataType dtype,
                                                     size_t packet_bytes) {
    const auto warps_for = [&]<typename InputT>() -> uint32_t {
        switch (packet_bytes) {
            case 16:
                return static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 16>(geometry));
            case 8:
                return static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 8>(geometry));
            case 4:
                return static_cast<uint32_t>(chooseAdaptiveCooperativeTiledWarpsPerTile<InputT, 4>(geometry));
            default:
                throw std::logic_error("RK single-pass packet-adaptive sweep selected invalid packet width.");
        }
    };
    switch (dtype) {
        case DataType::FP16:
            return warps_for.template operator()<__half>();
        case DataType::BF16:
            return warps_for.template operator()<__nv_bfloat16>();
        case DataType::FP32:
            return warps_for.template operator()<float>();
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return warps_for.template operator()<__nv_fp8_e4m3>();
        case DataType::FP8_E5M2:
            return warps_for.template operator()<__nv_fp8_e5m2>();
#endif
        default:
            throw std::logic_error("RK single-pass packet-adaptive sweep received unsupported dtype.");
    }
}

[[nodiscard]] uint64_t singlePassPacketAdaptiveTiles(const CubReductionGeometry& geometry,
                                                      DataType dtype,
                                                      size_t packet_bytes) {
    const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    const uint64_t items_per_lane = packet_bytes / element_bytes;
    return singlePassCeilDiv(geometry.inner_size, 32ULL * items_per_lane);
}

template <typename InputT, size_t PacketBytes, int WarpsPerTile, bool NaturallyAlignedRows>
[[nodiscard]] CubKernelOccupancyInfo querySinglePassPacketAdaptiveOccupancyForWarps() {
    constexpr int ItemsPerLane = adaptiveCooperativeTiledItemsPerLane<InputT, PacketBytes>();
    constexpr int BlockWarps = WarpsPerTile < TILED_REDUCTION_WARPS_PER_BLOCK
                                   ? TILED_REDUCTION_WARPS_PER_BLOCK
                                   : WarpsPerTile;
    constexpr int BlockThreads = BlockWarps * TILED_REDUCTION_WARP_THREADS;
    const auto kernel = adaptiveCooperativeTiledFixedSegmentReductionKernel<InputT,
                                                                             cuda::std::plus<float>,
                                                                             IdentityFp32,
                                                                             AdditiveFinalizeFp32,
                                                                             WarpsPerTile,
                                                                             ItemsPerLane,
                                                                             NaturallyAlignedRows>;
    cudaFuncAttributes attributes{};
    checkSinglePassCuda(cudaFuncGetAttributes(&attributes, kernel),
                        "cudaFuncGetAttributes(rk_packet_adaptive)");
    int active_blocks = 0;
    checkSinglePassCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &active_blocks, kernel, BlockThreads, 0),
                        "cudaOccupancyMaxActiveBlocksPerMultiprocessor(rk_packet_adaptive)");
    return CubKernelOccupancyInfo{attributes.numRegs,
                                  attributes.sharedSizeBytes,
                                  0,
                                  active_blocks,
                                  active_blocks * BlockWarps};
}

template <typename InputT, size_t PacketBytes, bool NaturallyAlignedRows>
[[nodiscard]] CubKernelOccupancyInfo querySinglePassPacketAdaptiveOccupancyForAlignment(int warps) {
    switch (warps) {
        case 1:
            return querySinglePassPacketAdaptiveOccupancyForWarps<InputT, PacketBytes, 1, NaturallyAlignedRows>();
        case 2:
            return querySinglePassPacketAdaptiveOccupancyForWarps<InputT, PacketBytes, 2, NaturallyAlignedRows>();
        case 4:
            return querySinglePassPacketAdaptiveOccupancyForWarps<InputT, PacketBytes, 4, NaturallyAlignedRows>();
        case 8:
            return querySinglePassPacketAdaptiveOccupancyForWarps<InputT, PacketBytes, 8, NaturallyAlignedRows>();
        case 16:
            return querySinglePassPacketAdaptiveOccupancyForWarps<InputT, PacketBytes, 16, NaturallyAlignedRows>();
        case 32:
            if constexpr (sizeof(InputT) == 1) {
                throw std::logic_error("FP8 packet-adaptive occupancy is capped at 16 cooperative warps.");
            } else {
                return querySinglePassPacketAdaptiveOccupancyForWarps<InputT, PacketBytes, 32, NaturallyAlignedRows>();
            }
        default:
            throw std::logic_error("RK packet-adaptive occupancy query received invalid warps-per-tile.");
    }
}

template <typename InputT, size_t PacketBytes>
[[nodiscard]] CubKernelOccupancyInfo querySinglePassPacketAdaptiveOccupancyForPacket(
    const CubReductionGeometry& geometry, int warps) {
    const bool naturally_aligned_rows =
        (geometry.inner_size * static_cast<uint64_t>(sizeof(InputT))) % PacketBytes == 0;
    if constexpr (PacketBytes < 16) {
        if (!naturally_aligned_rows) {
            throw std::logic_error("RK packet-adaptive occupancy query found an illegal sub-16-byte row alignment.");
        }
        return querySinglePassPacketAdaptiveOccupancyForAlignment<InputT, PacketBytes, true>(warps);
    } else {
        if (naturally_aligned_rows) {
            return querySinglePassPacketAdaptiveOccupancyForAlignment<InputT, PacketBytes, true>(warps);
        }
        return querySinglePassPacketAdaptiveOccupancyForAlignment<InputT, PacketBytes, false>(warps);
    }
}

template <typename InputT>
[[nodiscard]] CubKernelOccupancyInfo querySinglePassPacketAdaptiveOccupancyForInput(
    const CubReductionGeometry& geometry, size_t packet_bytes, int warps) {
    switch (packet_bytes) {
        case 4:
            return querySinglePassPacketAdaptiveOccupancyForPacket<InputT, 4>(geometry, warps);
        case 8:
            return querySinglePassPacketAdaptiveOccupancyForPacket<InputT, 8>(geometry, warps);
        case 16:
            return querySinglePassPacketAdaptiveOccupancyForPacket<InputT, 16>(geometry, warps);
        default:
            throw std::logic_error("RK packet-adaptive occupancy query received an invalid packet width.");
    }
}

[[nodiscard]] CubKernelOccupancyInfo querySinglePassPacketAdaptiveOccupancy(
    const CubReductionGeometry& geometry, DataType dtype, size_t packet_bytes, int warps) {
    switch (dtype) {
        case DataType::FP16:
            return querySinglePassPacketAdaptiveOccupancyForInput<__half>(geometry, packet_bytes, warps);
        case DataType::BF16:
            return querySinglePassPacketAdaptiveOccupancyForInput<__nv_bfloat16>(geometry, packet_bytes, warps);
        case DataType::FP32:
            return querySinglePassPacketAdaptiveOccupancyForInput<float>(geometry, packet_bytes, warps);
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return querySinglePassPacketAdaptiveOccupancyForInput<__nv_fp8_e4m3>(geometry, packet_bytes, warps);
        case DataType::FP8_E5M2:
            return querySinglePassPacketAdaptiveOccupancyForInput<__nv_fp8_e5m2>(geometry, packet_bytes, warps);
#endif
        default:
            throw std::logic_error("RK packet-adaptive occupancy query received an unsupported dtype.");
    }
}

[[nodiscard]] DenseRKOccupancyInfo queryDenseRKFamilyExactOccupancy(const DenseRKFamilyPhysicalPlan& plan,
                                                                     void*) {
    CubKernelOccupancyInfo info;
    switch (plan.implementation) {
        case DenseRKProductionImplementation::KParallelPass: {
            const KParallelTiledStagePlan physical{
                plan.progress == DenseRKProgress::Complete ? TiledRKStageTopology::Complete
                                                          : TiledRKStageTopology::Staged,
                plan.packet_bytes,
                plan.block_threads,
                plan.shards_per_output};
            info = queryKParallelTiledStageOccupancy(plan.input_dtype, physical);
            break;
        }
        case DenseRKProductionImplementation::RCooperativeAlignedComplete: {
            CubReductionGeometry geometry;
            geometry.outer_size = plan.outer_size;
            geometry.reduction_size = plan.reduction_size;
            geometry.inner_size = plan.inner_size;
            info = querySinglePassPacketAdaptiveOccupancy(geometry,
                                                          plan.input_dtype,
                                                          plan.packet_bytes,
                                                          static_cast<int>(plan.cooperative_warps_per_tile));
            break;
        }
        case DenseRKProductionImplementation::RCooperativeAlignedStaged:
            info = queryCooperativeShardedTiledFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        case DenseRKProductionImplementation::RCooperativeRotatedStaged:
            info = queryAwkwardAlignmentRotatedShardedFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        case DenseRKProductionImplementation::RCooperativeFlatRowsStaged:
            info = queryNarrowLowPrecisionFlatRCooperativeFirstStageOccupancy(plan.input_dtype, plan.rows_per_shard);
            break;
        default:
            throw std::logic_error("Dense RK family exact occupancy query received a non-family implementation.");
    }
    if (info.max_active_blocks_per_sm <= 0 || info.max_active_warps_per_sm <= 0) {
        throw std::logic_error("Dense RK family exact occupancy query returned zero residency.");
    }
    return DenseRKOccupancyInfo{static_cast<uint32_t>(info.max_active_blocks_per_sm),
                                static_cast<uint32_t>(info.max_active_warps_per_sm)};
}

[[nodiscard]] RKSinglePassTiming timeRKSinglePass(Tensor& cache_flush,
                                                       Stream& stream,
                                                       const auto& run,
                                                       int timing_samples = RK_SINGLE_PASS_SAMPLES) {
    if (timing_samples <= 0 || timing_samples % 2 == 0) {
        throw std::invalid_argument("Dense RK single-pass timing requires a positive odd sample count.");
    }
    for (int i = 0; i < RK_SINGLE_PASS_WARMUPS; ++i) {
        run();
    }
    stream.synchronize();

    // Cache-cold timing does not require a host barrier between the eviction memset and the start event: both are
    // ordered on the same CUDA stream. Queue every cold sample back-to-back and wait once on the final stop event.
    // Reusing pooled timing events removes per-candidate cudaEventCreate/cudaEventDestroy churn without changing the
    // measured interval or the >=8x-L2 eviction contract.
    ReusableEventLeases event_leases(static_cast<size_t>(timing_samples) * 2);
    std::vector<Event> starts;
    std::vector<Event> stops;
    starts.reserve(static_cast<size_t>(timing_samples));
    stops.reserve(static_cast<size_t>(timing_samples));
    for (int sample = 0; sample < timing_samples; ++sample) {
        starts.emplace_back(event_leases.acquire(stream.getGpuNum(), true, false));
        stops.emplace_back(event_leases.acquire(stream.getGpuNum(), true, false));
    }

    for (int sample = 0; sample < timing_samples; ++sample) {
        checkSinglePassCuda(cudaMemsetAsync(cache_flush.getMemPtr<void>(),
                                            0x41 + sample,
                                            cache_flush.getArraySizeInBytes(),
                                            stream.getStream()),
                            "cudaMemsetAsync(rk_single_pass_cache_flush)");
        stream.putEvent(starts[static_cast<size_t>(sample)], true, false);
        run();
        stream.putEvent(stops[static_cast<size_t>(sample)], true, false);
    }
    stops.back().synchronize();

    std::vector<double> samples(static_cast<size_t>(timing_samples));
    for (int sample = 0; sample < timing_samples; ++sample) {
        float elapsed_ms = 0.0f;
        checkSinglePassCuda(cudaEventElapsedTime(&elapsed_ms,
                                                 starts[static_cast<size_t>(sample)].getEvent(),
                                                 stops[static_cast<size_t>(sample)].getEvent()),
                            "cudaEventElapsedTime(rk_single_pass)");
        samples[static_cast<size_t>(sample)] = elapsed_ms;
    }
    std::sort(samples.begin(), samples.end());
    return RKSinglePassTiming{samples.front(), samples[samples.size() / 2], samples.back()};
}

class RKRotatingTensorSlots {
   public:
    RKRotatingTensorSlots() = default;

    RKRotatingTensorSlots(TensorPlacement placement, const TensorDescriptor& descriptor, uint32_t slot_count)
        : slot_count_(slot_count) {
        if (slot_count_ == 0) {
            throw std::invalid_argument("Dense RK rotating working set requires at least one slot.");
        }
        const uint64_t elements_per_slot = descriptor.getTotalNumElements();
        const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(descriptor.getDataType());
        constexpr uint64_t SLOT_ALIGNMENT_BYTES = 16;
        if (elements_per_slot == 0 || element_bytes == 0 || SLOT_ALIGNMENT_BYTES % element_bytes != 0) {
            throw std::invalid_argument("Dense RK rotating working set requires a non-zero element size dividing 16 bytes.");
        }
        const uint64_t slot_alignment_elements = SLOT_ALIGNMENT_BYTES / element_bytes;
        if (elements_per_slot > std::numeric_limits<uint64_t>::max() - (slot_alignment_elements - 1)) {
            throw std::overflow_error("Dense RK rotating working-set slot stride overflowed.");
        }
        const uint64_t slot_stride_elements =
            ((elements_per_slot + slot_alignment_elements - 1) / slot_alignment_elements) * slot_alignment_elements;
        if (slot_stride_elements > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(slot_count_)) {
            throw std::overflow_error("Dense RK rotating working-set allocation overflowed.");
        }
        const uint64_t storage_elements = slot_stride_elements * static_cast<uint64_t>(slot_count_);
        storage_ = Tensor(placement, TensorDescriptor(descriptor.getDataType(), {storage_elements}));

        const std::vector<uint64_t> dimensions = descriptor.getDimensions();
        std::vector<uint64_t> dense_strides(dimensions.size(), 1);
        for (size_t i = dimensions.size(); i-- > 1;) {
            dense_strides[i - 1] = dense_strides[i] * dimensions[i];
        }

        slots_.reserve(slot_count_);
        for (uint32_t slot = 0; slot < slot_count_; ++slot) {
            slots_.push_back(storage_.aliasView(
                dimensions, dense_strides, static_cast<uint64_t>(slot) * slot_stride_elements));
        }
    }

    [[nodiscard]] uint32_t size() const { return slot_count_; }
    [[nodiscard]] Tensor& slot(uint32_t index) { return slots_.at(index); }
    [[nodiscard]] Tensor& storage() { return storage_; }

   private:
    Tensor storage_;
    std::vector<Tensor> slots_;
    uint32_t slot_count_ = 0;
};

[[nodiscard]] uint32_t rkRotatingColdSlotCount(uint64_t cold_reuse_bytes, uint64_t input_bytes) {
    if (cold_reuse_bytes == 0 || input_bytes == 0) {
        throw std::invalid_argument("Dense RK rotating working set requires non-zero byte counts.");
    }
    const uint64_t slots = (cold_reuse_bytes + input_bytes - 1) / input_bytes + 1;
    if (slots > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("Dense RK rotating working-set slot count overflowed uint32_t.");
    }
    return static_cast<uint32_t>(slots);
}

[[nodiscard]] RKSinglePassTiming timeRKSinglePassRotatingWorkingSet(
    Tensor& input,
    Tensor& output,
    RKRotatingTensorSlots& input_slots,
    RKRotatingTensorSlots& output_slots,
    uint64_t& rotation_cursor,
    Tensor& cache_flush,
    bool& needs_bootstrap_flush,
    Stream& stream,
    const auto& run,
    int timing_samples = RK_SINGLE_PASS_SAMPLES) {
    if (timing_samples <= 0 || timing_samples % 2 == 0) {
        throw std::invalid_argument("Dense RK single-pass timing requires a positive odd sample count.");
    }
    if (input_slots.size() != output_slots.size()) {
        throw std::logic_error("Dense RK rotating input/output working sets must have the same slot count.");
    }

    if (needs_bootstrap_flush) {
        // Establish one known-cold starting point. After this single bootstrap eviction, the continuously rotating
        // input/output tensors provide the reuse distance; there is no per-sample cache-flush traffic.
        checkSinglePassCuda(cudaMemsetAsync(cache_flush.getMemPtr<void>(),
                                            0x41,
                                            cache_flush.getArraySizeInBytes(),
                                            stream.getStream()),
                            "cudaMemsetAsync(rk_rotating_working_set_bootstrap)");
        needs_bootstrap_flush = false;
    }

    const auto bind_next_slot = [&]() {
        const uint32_t slot = static_cast<uint32_t>(rotation_cursor % input_slots.size());
        ++rotation_cursor;
        input = input_slots.slot(slot);
        output = output_slots.slot(slot);
    };

    for (int i = 0; i < RK_SINGLE_PASS_WARMUPS; ++i) {
        bind_next_slot();
        run();
    }
    stream.synchronize();

    ReusableEventLeases event_leases(static_cast<size_t>(timing_samples) * 2);
    std::vector<Event> starts;
    std::vector<Event> stops;
    starts.reserve(static_cast<size_t>(timing_samples));
    stops.reserve(static_cast<size_t>(timing_samples));
    for (int sample = 0; sample < timing_samples; ++sample) {
        starts.emplace_back(event_leases.acquire(stream.getGpuNum(), true, false));
        stops.emplace_back(event_leases.acquire(stream.getGpuNum(), true, false));
    }

    for (int sample = 0; sample < timing_samples; ++sample) {
        bind_next_slot();
        stream.putEvent(starts[static_cast<size_t>(sample)], true, false);
        run();
        stream.putEvent(stops[static_cast<size_t>(sample)], true, false);
    }
    stops.back().synchronize();

    std::vector<double> samples(static_cast<size_t>(timing_samples));
    for (int sample = 0; sample < timing_samples; ++sample) {
        float elapsed_ms = 0.0f;
        checkSinglePassCuda(cudaEventElapsedTime(&elapsed_ms,
                                                 starts[static_cast<size_t>(sample)].getEvent(),
                                                 stops[static_cast<size_t>(sample)].getEvent()),
                            "cudaEventElapsedTime(rk_single_pass_rotating_working_set)");
        samples[static_cast<size_t>(sample)] = elapsed_ms;
    }
    std::sort(samples.begin(), samples.end());
    return RKSinglePassTiming{samples.front(), samples[samples.size() / 2], samples.back()};
}

template <typename T>
[[nodiscard]] float singlePassDeviceValue(const Tensor& tensor, uint64_t index, Stream& stream) {
    T value{};
    checkSinglePassCuda(cudaMemcpyAsync(&value,
                                        tensor.getMemPtr<T>() + index,
                                        sizeof(T),
                                        cudaMemcpyDeviceToHost,
                                        stream.getStream()),
                        "cudaMemcpyAsync(rk_single_pass_validation)");
    stream.synchronize();
    return static_cast<float>(value);
}

[[nodiscard]] float singlePassDeviceValue(const Tensor& tensor, uint64_t index, Stream& stream) {
    switch (tensor.getDataType()) {
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
            return singlePassDeviceValue<__nv_fp8_e4m3>(tensor, index, stream);
        case DataType::FP8_E5M2:
            return singlePassDeviceValue<__nv_fp8_e5m2>(tensor, index, stream);
#endif
        case DataType::FP16:
            return singlePassDeviceValue<__half>(tensor, index, stream);
        case DataType::BF16:
            return singlePassDeviceValue<__nv_bfloat16>(tensor, index, stream);
        case DataType::FP32:
            return singlePassDeviceValue<float>(tensor, index, stream);
        default:
            throw std::logic_error("RK single-pass validation received unsupported output dtype.");
    }
}

void validateSinglePassSample(const Tensor& output,
                              uint64_t index,
                              double expected,
                              Stream& stream,
                              std::string_view implementation) {
    const float actual = singlePassDeviceValue(output, index, stream);
    const double tolerance = output.getDataType() == DataType::FP32
                                 ? 1.0e-5
                                 : std::max(0.02, std::abs(expected) * 0.01);
    if (!std::isfinite(actual) || std::abs(static_cast<double>(actual) - expected) > tolerance) {
        throw std::runtime_error("RK single-pass validation failed for " + std::string(implementation)
                                 + " at output index " + std::to_string(index) + ": expected "
                                 + std::to_string(expected) + ", got " + std::to_string(actual)
                                 + ", tolerance " + std::to_string(tolerance) + ".");
    }
}

void validateSinglePassComplete(const Tensor& output,
                                const CubReductionGeometry& geometry,
                                Stream& stream,
                                std::string_view implementation,
                                double fill_value = RK_SINGLE_PASS_FILL_VALUE) {
    const double expected = static_cast<double>(geometry.reduction_size) * fill_value;
    validateSinglePassSample(output, 0, expected, stream, implementation);
    validateSinglePassSample(output, output.getTotalNumElements() - 1, expected, stream, implementation);
}

enum class SinglePassShardPolicy : uint8_t { Balanced, FixedRows };

void validateSinglePassPartial(const Tensor& output,
                               const CubReductionGeometry& geometry,
                               uint64_t shards_per_output,
                               uint64_t rows_per_shard,
                               SinglePassShardPolicy policy,
                               Stream& stream,
                               std::string_view implementation,
                               double fill_value = RK_SINGLE_PASS_FILL_VALUE) {
    if (shards_per_output <= 1) {
        throw std::logic_error("RK single-pass partial validation requires multiple shards.");
    }
    uint64_t first_rows = 0;
    uint64_t last_rows = 0;
    if (policy == SinglePassShardPolicy::Balanced) {
        const uint64_t base = geometry.reduction_size / shards_per_output;
        const uint64_t extra = geometry.reduction_size % shards_per_output;
        first_rows = base + static_cast<uint64_t>(extra != 0);
        last_rows = base + static_cast<uint64_t>(shards_per_output - 1 < extra);
    } else {
        first_rows = std::min<uint64_t>(rows_per_shard, geometry.reduction_size);
        const uint64_t last_begin = (shards_per_output - 1) * rows_per_shard;
        last_rows = geometry.reduction_size - last_begin;
    }
    validateSinglePassSample(output,
                             0,
                             static_cast<double>(first_rows) * fill_value,
                             stream,
                             implementation);
    validateSinglePassSample(output,
                             output.getTotalNumElements() - 1,
                             static_cast<double>(last_rows) * fill_value,
                             stream,
                             implementation);
}

void printSinglePassRow(const RKSinglePassSweepCase& benchmark_case,
                        DataType dtype,
                        std::string_view strategy,
                        std::string_view access,
                        std::string_view progress,
                        std::string_view implementation,
                        size_t packet_bytes,
                        uint32_t block_threads,
                        uint64_t rows_per_shard,
                        uint64_t shards_per_output,
                        uint64_t first_stage_blocks,
                        const Tensor& input,
                        const Tensor& output,
                        const RKSinglePassTiming& timing,
                        const RKSinglePassDeviceGeometry* device_geometry = nullptr,
                        const CubKernelOccupancyInfo* occupancy_info = nullptr,
                        std::string_view operation = "sum") {
    const uint64_t logical_bytes = input.getArraySizeInBytes() + output.getArraySizeInBytes();
    const double gbps = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
    std::cout << benchmark_case.family << ',' << benchmark_case.name << ',' << benchmark_case.outer << ','
              << benchmark_case.reduction << ',' << benchmark_case.inner << ',' << singlePassDtypeName(dtype)
              << ',' << operation << ',' << strategy << ',' << access << ',' << progress << ',' << implementation << ','
              << packet_bytes << ',' << block_threads << ',' << rows_per_shard << ',' << shards_per_output << ','
              << first_stage_blocks;
    if (device_geometry != nullptr) {
        if (occupancy_info == nullptr || occupancy_info->max_active_blocks_per_sm <= 0
            || occupancy_info->max_active_warps_per_sm <= 0) {
            throw std::logic_error("RK focused single-pass row is missing exact kernel occupancy metadata.");
        }
        const uint32_t thread_limited_blocks_per_sm = std::max<uint32_t>(
            1,
            std::min<uint32_t>(device_geometry->max_blocks_per_sm,
                               device_geometry->max_threads_per_sm / block_threads));
        const double ctas_per_sm = static_cast<double>(first_stage_blocks) / device_geometry->sm_count;
        const double thread_capacity_waves =
            static_cast<double>(first_stage_blocks)
            / (static_cast<double>(device_geometry->sm_count) * thread_limited_blocks_per_sm);
        const double thread_capacity_occupancy =
            static_cast<double>(thread_limited_blocks_per_sm) * block_threads / device_geometry->max_threads_per_sm;
        const uint32_t warps_per_cta = block_threads / device_geometry->warp_size;
        const double launch_warps_per_sm = std::min(
            static_cast<double>(occupancy_info->max_active_warps_per_sm),
            static_cast<double>(first_stage_blocks) * warps_per_cta / device_geometry->sm_count);
        const double occupancy_waves =
            static_cast<double>(first_stage_blocks)
            / (static_cast<double>(device_geometry->sm_count) * occupancy_info->max_active_blocks_per_sm);
        std::cout << ',' << device_geometry->sm_count << ',' << device_geometry->warp_size << ','
                  << device_geometry->max_threads_per_sm << ',' << device_geometry->max_threads_per_block << ','
                  << device_geometry->max_blocks_per_sm << ',' << device_geometry->shared_mem_per_sm << ','
                  << device_geometry->registers_per_sm << ',' << std::fixed << std::setprecision(4) << ctas_per_sm
                  << ',' << thread_limited_blocks_per_sm << ',' << thread_capacity_waves << ','
                  << thread_capacity_occupancy << ',' << occupancy_info->registers_per_thread << ','
                  << occupancy_info->static_shared_bytes << ',' << occupancy_info->dynamic_shared_bytes << ','
                  << occupancy_info->max_active_blocks_per_sm << ',' << occupancy_info->max_active_warps_per_sm << ','
                  << launch_warps_per_sm << ',' << occupancy_waves;
    }
    std::cout << ',' << input.getArraySizeInBytes() << ',' << output.getArraySizeInBytes() << ','
              << std::fixed << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ','
              << timing.worst_ms << ',' << std::setprecision(2) << gbps << '\n';
}

template <typename InputT>
void runSinglePassKParallelRows(const RKSinglePassSweepCase& benchmark_case,
                                DataType dtype,
                                const Tensor& input,
                                const CubReductionGeometry& geometry,
                                const CubReductionStageSemantics& complete_semantics,
                                const CubReductionStageSemantics& first_semantics,
                                Tensor& cache_flush,
                                Stream& stream,
                                const RKSinglePassDeviceGeometry* device_geometry = nullptr) {
    constexpr std::array<size_t, 3> packets = {4, 8, 16};
    constexpr std::array<uint64_t, 6> rows_values = {32, 64, 128, 256, 512, 1024};
    for (size_t packet_bytes : packets) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            continue;
        }
        const uint32_t block_threads = singlePassKParallelBlockThreads<InputT>(geometry, packet_bytes);
        const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(geometry, packet_bytes, block_threads);

        Tensor complete_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const KParallelTiledStagePlan complete_plan{
            TiledRKStageTopology::Complete, packet_bytes, block_threads, 1};
        const auto run_complete = [&]() {
            launchKParallelTiledStage(
                complete_semantics, input, complete_output, geometry, complete_plan, 1.0f, stream);
        };
        run_complete();
        stream.synchronize();
        validateSinglePassComplete(complete_output, geometry, stream, "k_parallel_complete");
        const RKSinglePassTiming complete_timing = timeRKSinglePass(cache_flush, stream, run_complete);
        const CubKernelOccupancyInfo complete_occupancy =
            queryKParallelTiledStageOccupancy(dtype, complete_plan);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "k_parallel",
                           "aligned",
                           "complete",
                           "k_parallel_tiled_stage",
                           packet_bytes,
                           block_threads,
                           geometry.reduction_size,
                           1,
                           geometry.outer_size * component_tiles,
                           input,
                           complete_output,
                           complete_timing,
                           device_geometry,
                           &complete_occupancy);

        for (uint64_t target_rows : rows_values) {
            const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, target_rows);
            if (shards <= 1 || shards > geometry.reduction_size) {
                continue;
            }
            Tensor partials(input.getPlacement(),
                            TensorDescriptor(DataType::FP32,
                                             {geometry.outer_size, shards, geometry.inner_size}));
            const KParallelTiledStagePlan staged_plan{
                TiledRKStageTopology::Staged, packet_bytes, block_threads, shards};
            const auto run_staged = [&]() {
                launchKParallelTiledStage(
                    first_semantics, input, partials, geometry, staged_plan, 1.0f, stream);
            };
            run_staged();
            stream.synchronize();
            validateSinglePassPartial(partials,
                                      geometry,
                                      shards,
                                      target_rows,
                                      SinglePassShardPolicy::Balanced,
                                      stream,
                                      "k_parallel_staged");
            const RKSinglePassTiming staged_timing = timeRKSinglePass(cache_flush, stream, run_staged);
            const CubKernelOccupancyInfo staged_occupancy =
                queryKParallelTiledStageOccupancy(dtype, staged_plan);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "k_parallel",
                               "aligned",
                               "staged",
                               "k_parallel_tiled_stage",
                               packet_bytes,
                               block_threads,
                               target_rows,
                               shards,
                               geometry.outer_size * component_tiles * shards,
                               input,
                               partials,
                               staged_timing,
                               device_geometry,
                               &staged_occupancy);
        }
    }
}

void runSinglePassRCooperativeRows(const RKSinglePassSweepCase& benchmark_case,
                                   DataType dtype,
                                   const Tensor& input,
                                   const CubReductionGeometry& geometry,
                                   const CubReductionStageSemantics& complete_semantics,
                                   const CubReductionStageSemantics& first_semantics,
                                   Tensor& cache_flush,
                                   Stream& stream,
                                   const RKSinglePassDeviceGeometry* device_geometry = nullptr) {
    // Packet-adaptive Complete is one modern aligned R-cooperative kernel family. Let its own packet policy operate;
    // this sweep is mapping the kernel's envelope, not using the RK family selector.
    const size_t packet_bytes = singlePassPacketAdaptivePacketBytes(geometry, dtype);
    const uint32_t cooperative_warps = singlePassPacketAdaptiveWarps(geometry, dtype, packet_bytes);
    const uint32_t block_warps = std::max<uint32_t>(static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK),
                                                    cooperative_warps);
    DenseRKFamilyPhysicalPlan packet_plan;
    packet_plan.packet_bytes = packet_bytes;
    packet_plan.cooperative_warps_per_tile = cooperative_warps;
    packet_plan.block_threads = block_warps * static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS);
    Tensor complete_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
    const auto run_complete = [&]() {
        launchPacketAdaptiveComplete(
            complete_semantics, packet_plan, input, complete_output, geometry, stream);
    };
    run_complete();
    stream.synchronize();
    validateSinglePassComplete(complete_output, geometry, stream, "r_cooperative_aligned_complete");
    const RKSinglePassTiming complete_timing = timeRKSinglePass(cache_flush, stream, run_complete);
    const CubKernelOccupancyInfo complete_occupancy = querySinglePassPacketAdaptiveOccupancy(
        geometry, dtype, packet_bytes, static_cast<int>(cooperative_warps));
    const uint64_t packet_tiles = singlePassPacketAdaptiveTiles(geometry, dtype, packet_bytes);
    const uint64_t groups_per_block = block_warps / cooperative_warps;
    printSinglePassRow(benchmark_case,
                       dtype,
                       "r_cooperative",
                       "aligned",
                       "complete",
                       "packet_adaptive_cooperative_tiled",
                       packet_bytes,
                       packet_plan.block_threads,
                       geometry.reduction_size,
                       1,
                       singlePassCeilDiv(geometry.outer_size * packet_tiles, groups_per_block),
                       input,
                       complete_output,
                       complete_timing,
                       device_geometry,
                       &complete_occupancy);

    constexpr std::array<uint64_t, 4> rows_values = {128, 256, 512, 1024};
    if (singlePassAlignedRCoopSupports(geometry, dtype)) {
        const uint64_t component_tiles = singlePassAlignedRCoopTiles(geometry, dtype);
        for (uint64_t rows : rows_values) {
            const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, rows);
            if (shards <= 1) {
                continue;
            }
            Tensor partials(input.getPlacement(),
                            TensorDescriptor(DataType::FP32,
                                             {geometry.outer_size, shards, geometry.inner_size}));
            const auto run_aligned = [&]() {
                launchCooperativeShardedTiledFirstStage(
                    first_semantics, input, partials, geometry, rows, shards, stream);
            };
            run_aligned();
            stream.synchronize();
            validateSinglePassPartial(partials,
                                      geometry,
                                      shards,
                                      rows,
                                      SinglePassShardPolicy::FixedRows,
                                      stream,
                                      "r_cooperative_aligned_staged");
            const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run_aligned);
            const CubKernelOccupancyInfo occupancy =
                queryCooperativeShardedTiledFirstStageOccupancy(dtype, rows);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "aligned",
                               "staged",
                               "cooperative_sharded_tiled_first_stage",
                               16,
                               256,
                               rows,
                               shards,
                               geometry.outer_size * component_tiles * shards,
                               input,
                               partials,
                               timing,
                               device_geometry,
                               &occupancy);
        }
    }

    const uint64_t rotated_tiles = singlePassRotatedRCoopTiles(geometry, dtype);
    for (uint64_t rows : rows_values) {
        const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, rows);
        if (shards <= 1) {
            continue;
        }
        Tensor partials(input.getPlacement(),
                        TensorDescriptor(DataType::FP32,
                                         {geometry.outer_size, shards, geometry.inner_size}));
        const auto run_rotated = [&]() {
            launchAwkwardAlignmentRotatedShardedFirstStage(
                first_semantics, input, partials, geometry, rows, shards, stream);
        };
        run_rotated();
        stream.synchronize();
        validateSinglePassPartial(partials,
                                  geometry,
                                  shards,
                                  rows,
                                  SinglePassShardPolicy::FixedRows,
                                  stream,
                                  "r_cooperative_rotated_staged");
        const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run_rotated);
        const CubKernelOccupancyInfo occupancy =
            queryAwkwardAlignmentRotatedShardedFirstStageOccupancy(dtype, rows);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "r_cooperative",
                           "rotated",
                           "staged",
                           "awkward_alignment_rotated_sharded_first_stage",
                           16,
                           256,
                           rows,
                           shards,
                           geometry.outer_size * rotated_tiles * shards,
                           input,
                           partials,
                           timing,
                           device_geometry,
                           &occupancy);
    }
}


[[nodiscard]] const char* gapOperationName(CubReductionOp op) {
    switch (op) {
        case CubReductionOp::Sum:
            return "sum";
        case CubReductionOp::Min:
            return "min";
        case CubReductionOp::Max:
            return "max";
        default:
            throw std::logic_error("RK gap census received unsupported operation.");
    }
}

[[nodiscard]] double gapExpectedValue(CubReductionOp op, uint64_t rows) {
    switch (op) {
        case CubReductionOp::Sum:
            return static_cast<double>(rows) * RK_SINGLE_PASS_FILL_VALUE;
        case CubReductionOp::Min:
        case CubReductionOp::Max:
            return RK_SINGLE_PASS_FILL_VALUE;
        default:
            throw std::logic_error("RK gap census received unsupported operation.");
    }
}

void validateGapComplete(const Tensor& output,
                         const CubReductionGeometry& geometry,
                         CubReductionOp op,
                         Stream& stream,
                         std::string_view implementation) {
    const double expected = gapExpectedValue(op, geometry.reduction_size);
    validateSinglePassSample(output, 0, expected, stream, implementation);
    validateSinglePassSample(output, output.getTotalNumElements() - 1, expected, stream, implementation);
}

void validateGapPartial(const Tensor& output,
                        const CubReductionGeometry& geometry,
                        uint64_t shards_per_output,
                        uint64_t rows_per_shard,
                        CubReductionOp op,
                        Stream& stream,
                        std::string_view implementation) {
    const uint64_t first_rows = std::min<uint64_t>(rows_per_shard, geometry.reduction_size);
    const uint64_t last_begin = (shards_per_output - 1) * rows_per_shard;
    const uint64_t last_rows = geometry.reduction_size - last_begin;
    validateSinglePassSample(output, 0, gapExpectedValue(op, first_rows), stream, implementation);
    validateSinglePassSample(output,
                             output.getTotalNumElements() - 1,
                             gapExpectedValue(op, last_rows),
                             stream,
                             implementation);
}

template <typename InputT>
void runGapKParallel(const RKSinglePassSweepCase& benchmark_case,
                     DataType dtype,
                     CubReductionOp op,
                     const Tensor& input,
                     const CubReductionGeometry& geometry,
                     Tensor& cache_flush,
                     Stream& stream,
                     const RKSinglePassDeviceGeometry& device_geometry,
                     bool include_complete,
                     bool include_staged) {
    constexpr std::array<size_t, 3> packets = {4, 8, 16};
    constexpr std::array<uint64_t, 2> staged_rows = {128, 256};
    const CubReductionStageSemantics complete_semantics = makeValueReductionStageSemantics(
        op, CubReductionStageRole::Complete, geometry.reduction_size);
    const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
        op, CubReductionStageRole::First, geometry.reduction_size);

    for (size_t packet_bytes : packets) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            continue;
        }
        const uint32_t block_threads = singlePassKParallelBlockThreads<InputT>(geometry, packet_bytes);
        const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(
            geometry, packet_bytes, block_threads);

        if (include_complete) {
            Tensor output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const KParallelTiledStagePlan plan{TiledRKStageTopology::Complete, packet_bytes, block_threads, 1};
            const auto run = [&]() {
                launchKParallelTiledStage(complete_semantics, input, output, geometry, plan, 1.0f, stream);
            };
            run();
            stream.synchronize();
            validateGapComplete(output, geometry, op, stream, "gap_k_parallel_complete");
            const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run);
            const CubKernelOccupancyInfo occupancy = queryKParallelTiledStageOccupancy(dtype, plan);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "k_parallel",
                               "aligned",
                               "complete",
                               "k_parallel_tiled_stage",
                               packet_bytes,
                               block_threads,
                               geometry.reduction_size,
                               1,
                               geometry.outer_size * component_tiles,
                               input,
                               output,
                               timing,
                               &device_geometry,
                               &occupancy,
                               gapOperationName(op));
        }

        if (include_staged) {
            for (uint64_t rows : staged_rows) {
                const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, rows);
                if (shards <= 1) {
                    continue;
                }
                Tensor partials(input.getPlacement(),
                                TensorDescriptor(DataType::FP32,
                                                 {geometry.outer_size, shards, geometry.inner_size}));
                const KParallelTiledStagePlan plan{
                    TiledRKStageTopology::Staged, packet_bytes, block_threads, shards};
                const auto run = [&]() {
                    launchKParallelTiledStage(first_semantics, input, partials, geometry, plan, 1.0f, stream);
                };
                run();
                stream.synchronize();
                // K-parallel distributes rows evenly across shards rather than fixed rows-per-shard.
                const uint64_t base = geometry.reduction_size / shards;
                const uint64_t extra = geometry.reduction_size % shards;
                validateSinglePassSample(partials,
                                         0,
                                         gapExpectedValue(op, base + static_cast<uint64_t>(extra != 0)),
                                         stream,
                                         "gap_k_parallel_staged");
                validateSinglePassSample(partials,
                                         partials.getTotalNumElements() - 1,
                                         gapExpectedValue(op, base + static_cast<uint64_t>(shards - 1 < extra)),
                                         stream,
                                         "gap_k_parallel_staged");
                const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run);
                const CubKernelOccupancyInfo occupancy = queryKParallelTiledStageOccupancy(dtype, plan);
                printSinglePassRow(benchmark_case,
                                   dtype,
                                   "k_parallel",
                                   "aligned",
                                   "staged",
                                   "k_parallel_tiled_stage",
                                   packet_bytes,
                                   block_threads,
                                   rows,
                                   shards,
                                   geometry.outer_size * component_tiles * shards,
                                   input,
                                   partials,
                                   timing,
                                   &device_geometry,
                                   &occupancy,
                                   gapOperationName(op));
            }
        }
    }
}

void runGapRotatedAndNarrowFlat(const RKSinglePassSweepCase& benchmark_case,
                                DataType dtype,
                                CubReductionOp op,
                                const Tensor& input,
                                const CubReductionGeometry& geometry,
                                Tensor& cache_flush,
                                Stream& stream,
                                const RKSinglePassDeviceGeometry& device_geometry) {
    if (dtype != DataType::FP16 && dtype != DataType::BF16) {
        return;
    }
    const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
        op, CubReductionStageRole::First, geometry.reduction_size);
    constexpr std::array<uint64_t, 4> rows_values = {128, 256, 512, 1024};
    const uint64_t rotated_tiles = singlePassRotatedRCoopTiles(geometry, dtype);

    for (uint64_t rows : rows_values) {
        const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, rows);
        if (shards <= 1) {
            continue;
        }

        Tensor rotated_partials(input.getPlacement(),
                                TensorDescriptor(DataType::FP32,
                                                 {geometry.outer_size, shards, geometry.inner_size}));
        const auto run_rotated = [&]() {
            launchAwkwardAlignmentRotatedShardedFirstStage(
                first_semantics, input, rotated_partials, geometry, rows, shards, stream);
        };
        run_rotated();
        stream.synchronize();
        validateGapPartial(rotated_partials,
                           geometry,
                           shards,
                           rows,
                           op,
                           stream,
                           "gap_rotated_r_cooperative");
        const RKSinglePassTiming rotated_timing = timeRKSinglePass(cache_flush, stream, run_rotated);
        const CubKernelOccupancyInfo rotated_occupancy =
            queryAwkwardAlignmentRotatedShardedFirstStageOccupancy(dtype, rows);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "r_cooperative",
                           "rotated",
                           "staged",
                           "awkward_alignment_rotated_sharded_first_stage",
                           16,
                           256,
                           rows,
                           shards,
                           geometry.outer_size * rotated_tiles * shards,
                           input,
                           rotated_partials,
                           rotated_timing,
                           &device_geometry,
                           &rotated_occupancy,
                           gapOperationName(op));

        if (geometry.inner_size <= 32) {
            Tensor flat_partials(input.getPlacement(),
                                 TensorDescriptor(DataType::FP32,
                                                  {geometry.outer_size, shards, geometry.inner_size}));
            const auto run_flat = [&]() {
                launchNarrowLowPrecisionFlatRCooperativeFirstStage(
                    first_semantics, input, flat_partials, geometry, rows, shards, stream);
            };
            run_flat();
            stream.synchronize();
            validateGapPartial(flat_partials,
                               geometry,
                               shards,
                               rows,
                               op,
                               stream,
                               "gap_narrow_flat_r_cooperative");
            const RKSinglePassTiming flat_timing = timeRKSinglePass(cache_flush, stream, run_flat);
            const CubKernelOccupancyInfo flat_occupancy =
                queryNarrowLowPrecisionFlatRCooperativeFirstStageOccupancy(dtype, rows);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "flat_rows",
                               "staged",
                               "narrow_low_precision_flat_r_cooperative_first_stage",
                               16,
                               256,
                               rows,
                               shards,
                               geometry.outer_size * shards,
                               input,
                               flat_partials,
                               flat_timing,
                               &device_geometry,
                               &flat_occupancy,
                               gapOperationName(op));
        }
    }
}

void printGapCensusHeader(const RKSinglePassDeviceGeometry& device_geometry) {
    std::cout << "# mode=rk_modern_gap_census operations=sum|min|max\n";
    std::cout << "# Every measured row is exactly one physical pass. The census targets the two remaining modern RK gaps: "
                 "MIN/MAX combine throughput and narrow awkward FP16/BF16 K.\n";
    std::cout << "# narrow_low_precision_flat_r_cooperative_first_stage is admitted beneath RCooperative for odd "
                 "FP16/BF16 K<=31; aligned K=16/32 remain KParallel controls for selector-envelope calibration.\n";
    std::cout << "family,case,outer,R,K,dtype,operation,strategy,access,progress,implementation,packet_bytes,block_threads,"
                 "rows_per_shard,shards_per_output,first_stage_blocks,device_sms,device_warp_size,"
                 "device_max_threads_per_sm,device_max_threads_per_block,device_max_blocks_per_sm,"
                 "device_shared_mem_per_sm,device_registers_per_sm,ctas_per_sm,thread_capacity_blocks_per_sm,"
                 "thread_capacity_waves,thread_capacity_occupancy,registers_per_thread,static_shared_bytes,"
                 "dynamic_shared_bytes,occupancy_limit_ctas_per_sm,occupancy_limit_warps_per_sm,launch_warps_per_sm,"
                 "occupancy_waves,input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";
    (void)device_geometry;
}

void runModernRKGapCensusImpl(Tensor& cache_flush, Stream& stream) {
    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    printGapCensusHeader(device_geometry);

    // MIN/MAX diagnostic: hold the memory geometry in regions where SUM has already demonstrated near-line-rate
    // throughput, then force the same modern K-parallel Complete/Staged specializations for SUM/MIN/MAX.
    const std::array<RKSinglePassSweepCase, 4> minmax_cases = {{
        {"minmax", "o2048_r256_i64", 2048, 256, 64},
        {"minmax", "o1024_r256_i128", 1024, 256, 128},
        {"minmax", "o256_r1024_i128", 256, 1024, 128},
        {"minmax", "o64_r4096_i256", 64, 4096, 256},
    }};
    constexpr std::array<DataType, 3> all_dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};
    constexpr std::array<CubReductionOp, 3> ops = {CubReductionOp::Sum, CubReductionOp::Min, CubReductionOp::Max};

    for (const RKSinglePassSweepCase& benchmark_case : minmax_cases) {
        for (DataType dtype : all_dtypes) {
            Tensor input(cache_flush.getPlacement(),
                         TensorDescriptor(dtype,
                                          {benchmark_case.outer, benchmark_case.reduction, benchmark_case.inner}));
            input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
            stream.synchronize();
            for (CubReductionOp op : ops) {
                const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(op, input.getDimensions(), {1});
                switch (dtype) {
                    case DataType::FP16:
                        runGapKParallel<__half>(benchmark_case,
                                                dtype,
                                                op,
                                                input,
                                                geometry,
                                                cache_flush,
                                                stream,
                                                device_geometry,
                                                true,
                                                true);
                        break;
                    case DataType::BF16:
                        runGapKParallel<__nv_bfloat16>(benchmark_case,
                                                       dtype,
                                                       op,
                                                       input,
                                                       geometry,
                                                       cache_flush,
                                                       stream,
                                                       device_geometry,
                                                       true,
                                                       true);
                        break;
                    case DataType::FP32:
                        runGapKParallel<float>(benchmark_case,
                                               dtype,
                                               op,
                                               input,
                                               geometry,
                                               cache_flush,
                                               stream,
                                               device_geometry,
                                               true,
                                               true);
                        break;
                    default:
                        throw std::logic_error("RK gap census received unsupported dtype.");
                }
            }
        }
    }

    // Narrow awkward low-precision diagnostic: enough input bytes / outer work to be bandwidth-bound. Aligned K=16/32
    // are controls; odd K=15/17/31 are the 2-byte row-stride cases that cannot use a legal 4-byte K-parallel packet.
    const std::array<uint64_t, 5> narrow_k = {15, 16, 17, 31, 32};
    constexpr std::array<DataType, 2> low_dtypes = {DataType::FP16, DataType::BF16};
    for (uint64_t inner : narrow_k) {
        RKSinglePassSweepCase benchmark_case{"narrow_awkward",
                                              "o64_r32768_i" + std::to_string(inner),
                                              64,
                                              32768,
                                              inner};
        for (DataType dtype : low_dtypes) {
            Tensor input(cache_flush.getPlacement(),
                         TensorDescriptor(dtype,
                                          {benchmark_case.outer, benchmark_case.reduction, benchmark_case.inner}));
            input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
            stream.synchronize();
            for (CubReductionOp op : ops) {
                const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(op, input.getDimensions(), {1});
                runGapRotatedAndNarrowFlat(benchmark_case,
                                           dtype,
                                           op,
                                           input,
                                           geometry,
                                           cache_flush,
                                           stream,
                                           device_geometry);
                // Aligned controls also show the modern K-parallel ceiling on the same O/R scale.
                if (inner % 2 == 0) {
                    if (dtype == DataType::FP16) {
                        runGapKParallel<__half>(benchmark_case,
                                                dtype,
                                                op,
                                                input,
                                                geometry,
                                                cache_flush,
                                                stream,
                                                device_geometry,
                                                false,
                                                true);
                    } else {
                        runGapKParallel<__nv_bfloat16>(benchmark_case,
                                                       dtype,
                                                       op,
                                                       input,
                                                       geometry,
                                                       cache_flush,
                                                       stream,
                                                       device_geometry,
                                                       false,
                                                       true);
                    }
                }
            }
        }
    }
}

void printNarrowLowPrecisionCalibrationHeader(const RKSinglePassDeviceGeometry& device_geometry) {
    std::cout << "# mode=rk_narrow_low_precision_calibration operations=sum|min|max dtypes=fp16|bf16 K=1..32 O=64 R=32768\n";
    std::cout << "# Every measured row is exactly one physical staged pass. Flat-row and rotated R-cooperative rows/shard "
                 "128/256/512/1024 are measured for every K; all legal staged K-parallel packet geometries are measured "
                 "when K admits a >=4-byte low-precision packet. No continuation or production selector is timed.\n";
    std::cout << "family,case,outer,R,K,dtype,operation,strategy,access,progress,implementation,packet_bytes,block_threads,"
                 "rows_per_shard,shards_per_output,first_stage_blocks,device_sms,device_warp_size,"
                 "device_max_threads_per_sm,device_max_threads_per_block,device_max_blocks_per_sm,"
                 "device_shared_mem_per_sm,device_registers_per_sm,ctas_per_sm,thread_capacity_blocks_per_sm,"
                 "thread_capacity_waves,thread_capacity_occupancy,registers_per_thread,static_shared_bytes,"
                 "dynamic_shared_bytes,occupancy_limit_ctas_per_sm,occupancy_limit_warps_per_sm,launch_warps_per_sm,"
                 "occupancy_waves,input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";
    (void)device_geometry;
}

void runRKNarrowLowPrecisionCalibrationImpl(Tensor& cache_flush, Stream& stream) {
    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    printNarrowLowPrecisionCalibrationHeader(device_geometry);

    constexpr uint64_t outer = 64;
    constexpr uint64_t reduction = 32768;
    constexpr std::array<DataType, 2> dtypes = {DataType::FP16, DataType::BF16};
    constexpr std::array<CubReductionOp, 3> ops = {CubReductionOp::Sum, CubReductionOp::Min, CubReductionOp::Max};

    for (uint64_t inner = 1; inner <= 32; ++inner) {
        RKSinglePassSweepCase benchmark_case{"narrow_low_precision_calibration",
                                              "o64_r32768_i" + std::to_string(inner),
                                              outer,
                                              reduction,
                                              inner};
        for (DataType dtype : dtypes) {
            Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, reduction, inner}));
            input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
            stream.synchronize();

            for (CubReductionOp op : ops) {
                const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(op, input.getDimensions(), {1});
                runGapRotatedAndNarrowFlat(benchmark_case,
                                           dtype,
                                           op,
                                           input,
                                           geometry,
                                           cache_flush,
                                           stream,
                                           device_geometry);

                // A low-precision K-parallel lane packet must be at least 4 bytes (two elements), so odd K has no legal
                // packet geometry. For even K, runGapKParallel enumerates all legal staged packet widths/CTA geometries.
                if ((inner % 2) == 0) {
                    if (dtype == DataType::FP16) {
                        runGapKParallel<__half>(benchmark_case,
                                                dtype,
                                                op,
                                                input,
                                                geometry,
                                                cache_flush,
                                                stream,
                                                device_geometry,
                                                false,
                                                true);
                    } else {
                        runGapKParallel<__nv_bfloat16>(benchmark_case,
                                                       dtype,
                                                       op,
                                                       input,
                                                       geometry,
                                                       cache_flush,
                                                       stream,
                                                       device_geometry,
                                                       false,
                                                       true);
                    }
                }
            }
        }
    }
}

void runModernRKSinglePassCases(const std::vector<RKSinglePassSweepCase>& cases,
                                std::string_view mode,
                                const RKSinglePassDeviceGeometry* device_geometry,
                                Tensor& cache_flush,
                                Stream& stream) {
    constexpr std::array<DataType, 3> dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};

    std::cout << "# mode=" << mode << " operation=sum dtypes=fp16|bf16|fp32\n";
    std::cout << "# Every measured row launches exactly one modern production RK physical pass. No remainder, terminal, "
                 "CubReduction continuation, or family replanning is included in timing.\n";
    std::cout << "# K-parallel: Complete packet4/8/16 plus Staged packet4/8/16 x target_rows32/64/128/256/512/1024 "
                 "where legal; CTA width is the existing compile-time-specialized K-coverage rule.\n";
    std::cout << "# R-cooperative: packet-adaptive aligned Complete plus aligned and rotated Staged "
                 "rows128/256/512/1024. Aligned staged rows are emitted only when the full 16-byte packet divides K.\n";
    std::cout << "# Inputs are filled with exactly 1/1024 so Complete and partial outputs can be sampled against analytic "
                 "expected sums before timing. Cache eviction is outside the timed interval.\n";
    if (device_geometry != nullptr) {
        std::cout << "# device_geometry sm_count=" << device_geometry->sm_count
                  << " warp_size=" << device_geometry->warp_size
                  << " max_threads_per_sm=" << device_geometry->max_threads_per_sm
                  << " max_threads_per_block=" << device_geometry->max_threads_per_block
                  << " max_blocks_per_sm=" << device_geometry->max_blocks_per_sm
                  << " shared_mem_per_sm=" << device_geometry->shared_mem_per_sm
                  << " registers_per_sm=" << device_geometry->registers_per_sm << '\n';
        std::cout << "# ctas_per_sm is launch supply. thread_capacity_* remain simple thread/block upper bounds for comparison. "
                     "registers_per_thread/static_shared_bytes/dynamic_shared_bytes and occupancy_limit_* are queried from "
                     "the exact CUDA kernel specialization with cudaFuncGetAttributes and "
                     "cudaOccupancyMaxActiveBlocksPerMultiprocessor. launch_warps_per_sm is supplied resident warp capacity "
                     "capped by exact occupancy; occupancy_waves is total CTAs divided by exact resident CTA capacity across "
                     "all SMs.\n";
    }
    std::cout << "# max_input_bytes=" << RK_SINGLE_PASS_MAX_INPUT_BYTES
              << " warmups=" << RK_SINGLE_PASS_WARMUPS
              << " timing_samples=" << RK_SINGLE_PASS_SAMPLES << " reported_time=median\n";
    std::cout << "family,case,outer,R,K,dtype,operation,strategy,access,progress,implementation,packet_bytes,block_threads,"
                 "rows_per_shard,shards_per_output,first_stage_blocks";
    if (device_geometry != nullptr) {
        std::cout << ",device_sms,device_warp_size,device_max_threads_per_sm,device_max_threads_per_block,"
                     "device_max_blocks_per_sm,device_shared_mem_per_sm,device_registers_per_sm,ctas_per_sm,"
                     "thread_capacity_blocks_per_sm,thread_capacity_waves,thread_capacity_occupancy,"
                     "registers_per_thread,static_shared_bytes,dynamic_shared_bytes,occupancy_limit_ctas_per_sm,"
                     "occupancy_limit_warps_per_sm,launch_warps_per_sm,occupancy_waves";
    }
    std::cout << ",input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";

    uint64_t measured_shape_dtypes = 0;
    uint64_t skipped_shape_dtypes = 0;
    for (const RKSinglePassSweepCase& benchmark_case : cases) {
        const uint64_t elements = checkedSinglePassElements(
            benchmark_case.outer, benchmark_case.reduction, benchmark_case.inner);
        for (DataType dtype : dtypes) {
            const uint64_t input_bytes = elements * TensorDescriptor::getElementSizeInBytes(dtype);
            if (input_bytes > RK_SINGLE_PASS_MAX_INPUT_BYTES) {
                ++skipped_shape_dtypes;
                continue;
            }
            Tensor input(cache_flush.getPlacement(),
                         TensorDescriptor(dtype,
                                          {benchmark_case.outer,
                                           benchmark_case.reduction,
                                           benchmark_case.inner}));
            input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
            stream.synchronize();
            const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
                CubReductionOp::Sum, input.getDimensions(), {1});
            if (geometry.path != CubReductionPath::TiledFixedSegment) {
                throw std::logic_error("RK single-pass sweep case did not analyze as TiledFixedSegment.");
            }
            const CubReductionStageSemantics complete_semantics = makeValueReductionStageSemantics(
                CubReductionOp::Sum, CubReductionStageRole::Complete, geometry.reduction_size);
            const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
                CubReductionOp::Sum, CubReductionStageRole::First, geometry.reduction_size);

            switch (dtype) {
                case DataType::FP16:
                    runSinglePassKParallelRows<__half>(benchmark_case,
                                                       dtype,
                                                       input,
                                                       geometry,
                                                       complete_semantics,
                                                       first_semantics,
                                                       cache_flush,
                                                       stream,
                                                       device_geometry);
                    break;
                case DataType::BF16:
                    runSinglePassKParallelRows<__nv_bfloat16>(benchmark_case,
                                                              dtype,
                                                              input,
                                                              geometry,
                                                              complete_semantics,
                                                              first_semantics,
                                                              cache_flush,
                                                              stream,
                                                              device_geometry);
                    break;
                case DataType::FP32:
                    runSinglePassKParallelRows<float>(benchmark_case,
                                                      dtype,
                                                      input,
                                                      geometry,
                                                      complete_semantics,
                                                      first_semantics,
                                                      cache_flush,
                                                      stream,
                                                      device_geometry);
                    break;
                default:
                    throw std::logic_error("RK single-pass sweep received unsupported dtype.");
            }
            runSinglePassRCooperativeRows(benchmark_case,
                                          dtype,
                                          input,
                                          geometry,
                                          complete_semantics,
                                          first_semantics,
                                          cache_flush,
                                          stream,
                                          device_geometry);
            ++measured_shape_dtypes;
        }
    }
    std::cout << "# " << mode << "_complete measured_shape_dtypes=" << measured_shape_dtypes
              << " skipped_shape_dtypes_over_input_ceiling=" << skipped_shape_dtypes << '\n';
}

void runModernRKSinglePassSweepImpl(Tensor& cache_flush, Stream& stream) {
    runModernRKSinglePassCases(
        makeRKSinglePassSweepCases(), "rk_modern_single_pass_sweep", nullptr, cache_flush, stream);
}

void runModernRKSinglePassFocusSweepImpl(Tensor& cache_flush, Stream& stream) {
    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    runModernRKSinglePassCases(makeRKSinglePassFocusCases(device_geometry.sm_count),
                               "rk_modern_single_pass_focus",
                               &device_geometry,
                               cache_flush,
                               stream);
}



template <typename T>
__device__ __forceinline__ T leanCompleteFromFp32(float value);

template <>
__device__ __forceinline__ float leanCompleteFromFp32<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __half leanCompleteFromFp32<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 leanCompleteFromFp32<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T, int ItemsPerLane>
struct alignas(sizeof(T) * ItemsPerLane) LeanCompletePacket {
    T values[ItemsPerLane];
};

template <typename InputT, int ItemsPerLane, int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void leanKParallelCompleteSumKernel(const InputT* input,
                                                                              InputT* output,
                                                                              uint64_t reduction_size,
                                                                              uint64_t inner_size,
                                                                              uint64_t component_tiles) {
    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(ItemsPerLane);
    const uint64_t outer_component = static_cast<uint64_t>(blockIdx.x);
    const uint64_t outer_index = outer_component / component_tiles;
    const uint64_t component_tile = outer_component - outer_index * component_tiles;
    const uint64_t packet_component_begin =
        component_tile * components_per_cta
        + static_cast<uint64_t>(threadIdx.x) * static_cast<uint64_t>(ItemsPerLane);
    if (packet_component_begin >= inner_size) {
        return;
    }

    float local[ItemsPerLane];
#pragma unroll
    for (int item = 0; item < ItemsPerLane; ++item) {
        local[item] = 0.0f;
    }

    uint64_t input_index = outer_index * reduction_size * inner_size + packet_component_begin;
    for (uint64_t row = 0; row < reduction_size; ++row) {
        const auto packet = *reinterpret_cast<const LeanCompletePacket<InputT, ItemsPerLane>*>(input + input_index);
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            local[item] += ToFp32<InputT>{}(packet.values[item]);
        }
        input_index += inner_size;
    }

    const uint64_t remaining = inner_size - packet_component_begin;
    if (remaining >= static_cast<uint64_t>(ItemsPerLane)) {
        LeanCompletePacket<InputT, ItemsPerLane> packet{};
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            packet.values[item] = leanCompleteFromFp32<InputT>(local[item]);
        }
        *reinterpret_cast<LeanCompletePacket<InputT, ItemsPerLane>*>(
            output + outer_index * inner_size + packet_component_begin) = packet;
    } else {
#pragma unroll
        for (int item = 0; item < ItemsPerLane; ++item) {
            if (static_cast<uint64_t>(item) < remaining) {
                output[outer_index * inner_size + packet_component_begin + static_cast<uint64_t>(item)] =
                    leanCompleteFromFp32<InputT>(local[item]);
            }
        }
    }
}

template <typename InputT, size_t PacketBytes, int BlockThreads>
void launchLeanKParallelCompleteSumForPacketAndBlock(const Tensor& input,
                                                      Tensor& output,
                                                      const CubReductionGeometry& geometry,
                                                      Stream& stream) {
    static_assert(PacketBytes == 4 || PacketBytes == 8 || PacketBytes == 16);
    static_assert(PacketBytes % sizeof(InputT) == 0);
    constexpr int items_per_lane = static_cast<int>(PacketBytes / sizeof(InputT));
    constexpr uint64_t components_per_cta =
        static_cast<uint64_t>(BlockThreads) * static_cast<uint64_t>(items_per_lane);
    const uint64_t component_tiles = singlePassCeilDiv(geometry.inner_size, components_per_cta);
    const uint64_t blocks = geometry.outer_size * component_tiles;
    if (blocks == 0 || blocks > static_cast<uint64_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::logic_error("Lean KParallel Complete benchmark received an invalid grid size.");
    }
    leanKParallelCompleteSumKernel<InputT, items_per_lane, BlockThreads>
        <<<static_cast<unsigned int>(blocks), BlockThreads, 0, stream.getStream()>>>(
            input.getMemPtr<InputT>(),
            output.getMemPtr<InputT>(),
            geometry.reduction_size,
            geometry.inner_size,
            component_tiles);
    checkSinglePassCuda(cudaGetLastError(), "leanKParallelCompleteSumKernel");
}

template <typename InputT, size_t PacketBytes>
void launchLeanKParallelCompleteSumForPacket(const Tensor& input,
                                             Tensor& output,
                                             const CubReductionGeometry& geometry,
                                             uint32_t block_threads,
                                             Stream& stream) {
    switch (block_threads) {
        case 32:
            launchLeanKParallelCompleteSumForPacketAndBlock<InputT, PacketBytes, 32>(input, output, geometry, stream);
            return;
        case 64:
            launchLeanKParallelCompleteSumForPacketAndBlock<InputT, PacketBytes, 64>(input, output, geometry, stream);
            return;
        case 128:
            launchLeanKParallelCompleteSumForPacketAndBlock<InputT, PacketBytes, 128>(input, output, geometry, stream);
            return;
        case 256:
            launchLeanKParallelCompleteSumForPacketAndBlock<InputT, PacketBytes, 256>(input, output, geometry, stream);
            return;
        default:
            throw std::logic_error("Lean KParallel Complete benchmark requires a supported CTA width.");
    }
}

template <typename InputT>
void launchLeanKParallelCompleteSum(const Tensor& input,
                                    Tensor& output,
                                    const CubReductionGeometry& geometry,
                                    size_t packet_bytes,
                                    uint32_t block_threads,
                                    Stream& stream) {
    switch (packet_bytes) {
        case 4:
            launchLeanKParallelCompleteSumForPacket<InputT, 4>(input, output, geometry, block_threads, stream);
            return;
        case 8:
            launchLeanKParallelCompleteSumForPacket<InputT, 8>(input, output, geometry, block_threads, stream);
            return;
        case 16:
            launchLeanKParallelCompleteSumForPacket<InputT, 16>(input, output, geometry, block_threads, stream);
            return;
        default:
            throw std::logic_error("Lean KParallel Complete benchmark requires a 4-, 8-, or 16-byte packet.");
    }
}

void printLeanCompleteRow(const RKSinglePassSweepCase& benchmark_case,
                          DataType dtype,
                          std::string_view implementation,
                          size_t packet_bytes,
                          uint32_t block_threads,
                          uint64_t blocks,
                          const Tensor& input,
                          const Tensor& output,
                          const RKSinglePassTiming& timing) {
    const uint64_t logical_bytes = input.getArraySizeInBytes() + output.getArraySizeInBytes();
    const double gbps = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
    std::cout << benchmark_case.family << ',' << benchmark_case.name << ',' << benchmark_case.outer << ','
              << benchmark_case.reduction << ',' << benchmark_case.inner << ',' << singlePassDtypeName(dtype)
              << ",sum," << implementation << ',' << packet_bytes << ',' << block_threads << ',' << blocks << ','
              << input.getArraySizeInBytes() << ',' << output.getArraySizeInBytes() << ',' << std::fixed
              << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ',' << timing.worst_ms << ','
              << std::setprecision(2) << gbps << '\n';
}

template <typename InputT>
void runLeanCompleteCase(const RKSinglePassSweepCase& benchmark_case,
                         DataType dtype,
                         const Tensor& input,
                         const CubReductionGeometry& geometry,
                         Tensor& cache_flush,
                         Stream& stream) {
    constexpr std::array<size_t, 3> packets = {4, 8, 16};
    const CubReductionStageSemantics semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::Complete, geometry.reduction_size);

    for (size_t packet_bytes : packets) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            continue;
        }
        const uint32_t block_threads = singlePassKParallelBlockThreads<InputT>(geometry, packet_bytes);
        const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(geometry, packet_bytes, block_threads);
        const uint64_t blocks = geometry.outer_size * component_tiles;

        Tensor current_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const KParallelTiledStagePlan current_plan{TiledRKStageTopology::Complete, packet_bytes, block_threads, 1};
        const auto run_current = [&]() {
            launchKParallelTiledStage(semantics, input, current_output, geometry, current_plan, 1.0f, stream);
        };
        run_current();
        stream.synchronize();
        validateSinglePassComplete(current_output, geometry, stream, "lean_complete_current");
        const RKSinglePassTiming current_timing = timeRKSinglePass(cache_flush, stream, run_current);
        printLeanCompleteRow(benchmark_case,
                             dtype,
                             "current_kparallel_complete",
                             packet_bytes,
                             block_threads,
                             blocks,
                             input,
                             current_output,
                             current_timing);

        Tensor lean_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_lean = [&]() {
            launchLeanKParallelCompleteSum<InputT>(
                input, lean_output, geometry, packet_bytes, block_threads, stream);
        };
        run_lean();
        stream.synchronize();
        validateSinglePassComplete(lean_output, geometry, stream, "lean_complete_specialized");
        const RKSinglePassTiming lean_timing = timeRKSinglePass(cache_flush, stream, run_lean);
        printLeanCompleteRow(benchmark_case,
                             dtype,
                             "lean_kparallel_complete",
                             packet_bytes,
                             block_threads,
                             blocks,
                             input,
                             lean_output,
                             lean_timing);
    }
}

void runRKLeanCompleteCalibrationImpl(Tensor& cache_flush, Stream& stream) {
    std::cout << "# mode=rk_lean_complete_calibration operation=sum dtypes=fp16|bf16|fp32\n";
    std::cout << "# current_kparallel_complete and lean_kparallel_complete use identical packet/CTA geometry. "
                 "The lean kernel removes shard decoding, generic work-index decoding, and grid-stride iteration. "
                 "production_selector is timed once per shape/dtype as the current production reference.\n";
    std::cout << "family,case,outer,R,K,dtype,operation,implementation,packet_bytes,block_threads,blocks,input_bytes,"
                 "output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";

    constexpr std::array<uint64_t, 3> outers = {1, 2, 4};
    constexpr std::array<uint64_t, 6> reductions = {64, 103, 127, 205, 256, 410};
    constexpr std::array<uint64_t, 8> inners = {32, 64, 128, 256, 384, 512, 768, 1024};
    constexpr std::array<DataType, 3> dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};

    for (uint64_t outer : outers) {
        for (uint64_t reduction : reductions) {
            for (uint64_t inner : inners) {
                RKSinglePassSweepCase benchmark_case{
                    "lean_complete", "o" + std::to_string(outer) + "_r" + std::to_string(reduction)
                                         + "_i" + std::to_string(inner), outer, reduction, inner};
                for (DataType dtype : dtypes) {
                    Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, reduction, inner}));
                    input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
                    stream.synchronize();
                    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
                        CubReductionOp::Sum, input.getDimensions(), {1});
                    if (geometry.path != CubReductionPath::TiledFixedSegment) {
                        throw std::logic_error("Lean Complete calibration case did not analyze as TiledFixedSegment.");
                    }

                    std::shared_ptr<StampedCubReduction> production =
                        CubReduction(CubReductionOp::Sum, {1}).stamp(input, stream);
                    const auto run_production = [&]() { production->runOn(stream); };
                    run_production();
                    stream.synchronize();
                    const Tensor production_output = production->getOutputTensor();
                    validateSinglePassComplete(production_output, geometry, stream, "lean_complete_production");
                    const RKSinglePassTiming production_timing = timeRKSinglePass(cache_flush, stream, run_production);
                    printLeanCompleteRow(benchmark_case,
                                         dtype,
                                         "production_selector",
                                         0,
                                         0,
                                         0,
                                         input,
                                         production_output,
                                         production_timing);

                    switch (dtype) {
                        case DataType::FP16:
                            runLeanCompleteCase<__half>(benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        case DataType::BF16:
                            runLeanCompleteCase<__nv_bfloat16>(benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        case DataType::FP32:
                            runLeanCompleteCase<float>(benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        default:
                            throw std::logic_error("Lean Complete calibration received unsupported dtype.");
                    }
                }
            }
        }
    }
}



void printStageCrossoverRow(const RKSinglePassSweepCase& benchmark_case,
                            DataType dtype,
                            std::string_view implementation,
                            size_t packet_bytes,
                            uint32_t block_threads,
                            uint64_t rows_per_shard,
                            uint64_t shards_per_output,
                            uint64_t blocks,
                            const Tensor& input,
                            const Tensor& output,
                            const RKSinglePassTiming& timing) {
    const uint64_t logical_bytes = input.getArraySizeInBytes() + output.getArraySizeInBytes();
    const double gbps = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
    std::cout << benchmark_case.family << ',' << benchmark_case.name << ',' << benchmark_case.outer << ','
              << benchmark_case.reduction << ',' << benchmark_case.inner << ',' << singlePassDtypeName(dtype)
              << ",sum," << implementation << ',' << packet_bytes << ',' << block_threads << ',' << rows_per_shard
              << ',' << shards_per_output << ',' << blocks << ',' << input.getArraySizeInBytes() << ','
              << output.getArraySizeInBytes() << ',' << std::fixed << std::setprecision(4) << timing.median_ms << ','
              << timing.best_ms << ',' << timing.worst_ms << ',' << std::setprecision(2) << gbps << '\n';
}

template <typename InputT>
void runKParallelStageCrossoverCase(const RKSinglePassSweepCase& benchmark_case,
                                    DataType dtype,
                                    const Tensor& input,
                                    const CubReductionGeometry& geometry,
                                    Tensor& cache_flush,
                                    Stream& stream) {
    constexpr std::array<size_t, 3> packets = {4, 8, 16};
    constexpr std::array<uint64_t, 4> staged_rows = {16, 32, 64, 128};
    const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::First, geometry.reduction_size);

    for (size_t packet_bytes : packets) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            continue;
        }
        const uint32_t block_threads = singlePassKParallelBlockThreads<InputT>(geometry, packet_bytes);
        const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(geometry, packet_bytes, block_threads);
        const uint64_t complete_blocks = geometry.outer_size * component_tiles;

        Tensor lean_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_lean = [&]() {
            launchLeanKParallelCompleteSum<InputT>(input, lean_output, geometry, packet_bytes, block_threads, stream);
        };
        run_lean();
        stream.synchronize();
        validateSinglePassComplete(lean_output, geometry, stream, "stage_crossover_lean_complete");
        const RKSinglePassTiming lean_timing = timeRKSinglePass(cache_flush, stream, run_lean);
        printStageCrossoverRow(benchmark_case,
                               dtype,
                               "lean_kparallel_complete",
                               packet_bytes,
                               block_threads,
                               geometry.reduction_size,
                               1,
                               complete_blocks,
                               input,
                               lean_output,
                               lean_timing);

        for (uint64_t target_rows : staged_rows) {
            const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, target_rows);
            if (shards <= 1 || shards > geometry.reduction_size) {
                continue;
            }
            Tensor partials(input.getPlacement(),
                            TensorDescriptor(DataType::FP32,
                                             {geometry.outer_size, shards, geometry.inner_size}));
            const KParallelTiledStagePlan staged_plan{
                TiledRKStageTopology::Staged, packet_bytes, block_threads, shards};
            const auto run_staged = [&]() {
                launchKParallelTiledStage(first_semantics, input, partials, geometry, staged_plan, 1.0f, stream);
            };
            run_staged();
            stream.synchronize();
            validateSinglePassPartial(partials,
                                      geometry,
                                      shards,
                                      target_rows,
                                      SinglePassShardPolicy::Balanced,
                                      stream,
                                      "stage_crossover_kparallel_staged");
            const RKSinglePassTiming staged_timing = timeRKSinglePass(cache_flush, stream, run_staged);
            printStageCrossoverRow(benchmark_case,
                                   dtype,
                                   "kparallel_staged",
                                   packet_bytes,
                                   block_threads,
                                   target_rows,
                                   shards,
                                   geometry.outer_size * component_tiles * shards,
                                   input,
                                   partials,
                                   staged_timing);
        }
    }
}

void runRKKParallelStageCrossoverCalibrationImpl(Tensor& cache_flush, Stream& stream) {
    std::cout << "# mode=rk_kparallel_stage_crossover_calibration operation=sum dtypes=fp16|bf16|fp32\n";
    std::cout << "# Compares benchmark-only lean KParallel Complete against one physical modern staged KParallel pass. "
                 "Staged rows sweep target rows/shard 16/32/64/128; output_bytes is the FP32 intermediate size. "
                 "No staged continuation is included. production_selector is retained only as a reference.\n";
    std::cout << "family,case,outer,R,K,dtype,operation,implementation,packet_bytes,block_threads,rows_per_shard,"
                 "shards_per_output,blocks,input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";

    constexpr std::array<uint64_t, 3> outers = {1, 2, 4};
    constexpr std::array<uint64_t, 6> reductions = {64, 103, 127, 205, 256, 410};
    constexpr std::array<uint64_t, 8> inners = {32, 64, 128, 256, 384, 512, 768, 1024};
    constexpr std::array<DataType, 3> dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};

    for (uint64_t outer : outers) {
        for (uint64_t reduction : reductions) {
            for (uint64_t inner : inners) {
                RKSinglePassSweepCase benchmark_case{
                    "stage_crossover", "o" + std::to_string(outer) + "_r" + std::to_string(reduction)
                                           + "_i" + std::to_string(inner), outer, reduction, inner};
                for (DataType dtype : dtypes) {
                    Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, reduction, inner}));
                    input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
                    stream.synchronize();
                    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
                        CubReductionOp::Sum, input.getDimensions(), {1});
                    if (geometry.path != CubReductionPath::TiledFixedSegment) {
                        throw std::logic_error("KParallel stage-crossover case did not analyze as TiledFixedSegment.");
                    }

                    std::shared_ptr<StampedCubReduction> production =
                        CubReduction(CubReductionOp::Sum, {1}).stamp(input, stream);
                    const auto run_production = [&]() { production->runOn(stream); };
                    run_production();
                    stream.synchronize();
                    const Tensor production_output = production->getOutputTensor();
                    validateSinglePassComplete(production_output, geometry, stream, "stage_crossover_production");
                    const RKSinglePassTiming production_timing = timeRKSinglePass(cache_flush, stream, run_production);
                    printStageCrossoverRow(benchmark_case,
                                           dtype,
                                           "production_selector",
                                           0,
                                           0,
                                           0,
                                           0,
                                           0,
                                           input,
                                           production_output,
                                           production_timing);

                    switch (dtype) {
                        case DataType::FP16:
                            runKParallelStageCrossoverCase<__half>(
                                benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        case DataType::BF16:
                            runKParallelStageCrossoverCase<__nv_bfloat16>(
                                benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        case DataType::FP32:
                            runKParallelStageCrossoverCase<float>(
                                benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        default:
                            throw std::logic_error("KParallel stage-crossover calibration received unsupported dtype.");
                    }
                }
            }
        }
    }
}


template <typename InputT>
void runKParallelStagedGeometryCase(const RKSinglePassSweepCase& benchmark_case,
                                    DataType dtype,
                                    const Tensor& input,
                                    const CubReductionGeometry& geometry,
                                    Tensor& cache_flush,
                                    Stream& stream) {
    constexpr std::array<size_t, 3> packets = {4, 8, 16};
    constexpr std::array<uint32_t, 4> cta_threads = {32, 64, 128, 256};
    constexpr uint64_t target_rows = 16;
    const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, target_rows);
    if (shards <= 1 || shards > geometry.reduction_size) {
        throw std::logic_error("Staged-geometry calibration requires more than one shard.");
    }
    const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::First, geometry.reduction_size);

    for (size_t packet_bytes : packets) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            continue;
        }
        const uint64_t items_per_lane = packet_bytes / sizeof(InputT);
        const uint64_t packet_owners = singlePassCeilDiv(geometry.inner_size, items_per_lane);
        for (uint32_t block_threads : cta_threads) {
            const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(
                geometry, packet_bytes, block_threads);
            Tensor partials(input.getPlacement(),
                            TensorDescriptor(DataType::FP32,
                                             {geometry.outer_size, shards, geometry.inner_size}));
            const KParallelTiledStagePlan staged_plan{
                TiledRKStageTopology::Staged, packet_bytes, block_threads, shards};
            const auto run_staged = [&]() {
                launchKParallelTiledStage(first_semantics, input, partials, geometry, staged_plan, 1.0f, stream);
            };
            run_staged();
            stream.synchronize();
            validateSinglePassPartial(partials,
                                      geometry,
                                      shards,
                                      target_rows,
                                      SinglePassShardPolicy::Balanced,
                                      stream,
                                      "staged_geometry_kparallel");
            const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run_staged);
            const uint64_t blocks = geometry.outer_size * component_tiles * shards;
            const uint64_t warps_per_cta = block_threads / TILED_REDUCTION_WARP_THREADS;
            const uint64_t logical_bytes = input.getArraySizeInBytes() + partials.getArraySizeInBytes();
            const double gbps = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
            std::cout << benchmark_case.family << ',' << benchmark_case.name << ',' << benchmark_case.outer << ','
                      << benchmark_case.reduction << ',' << benchmark_case.inner << ',' << singlePassDtypeName(dtype)
                      << ",sum,kparallel_staged," << packet_bytes << ',' << block_threads << ',' << warps_per_cta << ','
                      << packet_owners << ',' << component_tiles << ',' << target_rows << ',' << shards << ',' << blocks
                      << ',' << input.getArraySizeInBytes() << ',' << partials.getArraySizeInBytes() << ',' << std::fixed
                      << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ',' << timing.worst_ms << ','
                      << std::setprecision(2) << gbps << '\n';
        }
    }
}

void runRKKParallelStagedGeometryCalibrationImpl(Tensor& cache_flush, Stream& stream) {
    std::cout << "# mode=rk_kparallel_staged_geometry_calibration operation=sum dtypes=fp16|bf16|fp32 rows_per_shard=16\n";
    std::cout << "# Holds the established staged topology fixed at 16 target rows/shard and sweeps every legal "
                 "4/8/16-byte packet across CTA widths 32/64/128/256. Every timed row is one physical first pass; "
                 "no continuation is included and production selection is unchanged.\n";
    std::cout << "family,case,outer,R,K,dtype,operation,implementation,packet_bytes,block_threads,warps_per_cta,"
                 "packet_owners,component_tiles,rows_per_shard,shards_per_output,blocks,input_bytes,output_bytes,"
                 "median_ms,best_ms,worst_ms,logical_GBps\n";

    constexpr std::array<uint64_t, 3> outers = {1, 2, 4};
    constexpr std::array<uint64_t, 5> reductions = {103, 127, 205, 256, 410};
    constexpr std::array<uint64_t, 8> inners = {32, 64, 128, 256, 384, 512, 768, 1024};
    constexpr std::array<DataType, 3> dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};

    for (uint64_t outer : outers) {
        for (uint64_t reduction : reductions) {
            for (uint64_t inner : inners) {
                RKSinglePassSweepCase benchmark_case{
                    "staged_geometry", "o" + std::to_string(outer) + "_r" + std::to_string(reduction)
                                           + "_i" + std::to_string(inner), outer, reduction, inner};
                for (DataType dtype : dtypes) {
                    Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, reduction, inner}));
                    input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
                    stream.synchronize();
                    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
                        CubReductionOp::Sum, input.getDimensions(), {1});
                    if (geometry.path != CubReductionPath::TiledFixedSegment) {
                        throw std::logic_error("KParallel staged-geometry case did not analyze as TiledFixedSegment.");
                    }
                    switch (dtype) {
                        case DataType::FP16:
                            runKParallelStagedGeometryCase<__half>(
                                benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        case DataType::BF16:
                            runKParallelStagedGeometryCase<__nv_bfloat16>(
                                benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        case DataType::FP32:
                            runKParallelStagedGeometryCase<float>(
                                benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        default:
                            throw std::logic_error("KParallel staged-geometry calibration received unsupported dtype.");
                    }
                }
            }
        }
    }
}


constexpr double FP8_PACKET_AB_FILL_VALUE = 0.125;
constexpr uint32_t FP8_PACKET_AB_BLOCK_THREADS = 32;
constexpr uint64_t FP8_PACKET_AB_STAGED_ROWS_PER_SHARD = 64;

struct Fp8PacketABPhase {
    const char* name;
    TiledRKStageTopology topology;
    uint64_t reduction;
    uint64_t target_input_bytes;
};

template <typename InputT>
void runFp8PacketABShape(const Fp8PacketABPhase& phase,
                         DataType dtype,
                         uint64_t outer,
                         uint64_t inner,
                         const Tensor& input,
                         Tensor& cache_flush,
                         Stream& stream) {
    static_assert(sizeof(InputT) == 1, "FP8 packet A/B census requires one-byte input storage.");
    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, input.getDimensions(), {1});
    if (geometry.path != CubReductionPath::TiledFixedSegment) {
        throw std::logic_error("FP8 packet A/B census case was not TiledFixedSegment.");
    }
    if (geometry.outer_size != outer || geometry.reduction_size != phase.reduction || geometry.inner_size != inner) {
        throw std::logic_error("FP8 packet A/B census geometry disagrees with the requested exact shape.");
    }

    const bool staged = phase.topology == TiledRKStageTopology::Staged;
    const uint64_t shards = staged ? singlePassCeilDiv(phase.reduction, FP8_PACKET_AB_STAGED_ROWS_PER_SHARD) : 1ULL;
    if (staged && (shards <= 1 || phase.reduction % FP8_PACKET_AB_STAGED_ROWS_PER_SHARD != 0)) {
        throw std::logic_error("FP8 packet A/B staged census requires an exact fixed rows-per-shard geometry.");
    }

    Tensor output(input.getPlacement(),
                  staged
                      ? TensorDescriptor(DataType::FP32, {outer, shards, inner})
                      : TensorDescriptor(dtype, geometry.output_dimensions));
    const CubReductionStageSemantics semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum,
        staged ? CubReductionStageRole::First : CubReductionStageRole::Complete,
        geometry.reduction_size);

    constexpr std::array<size_t, 4> PACKETS = {16, 8, 4, 2};
    for (size_t packet_bytes : PACKETS) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            throw std::logic_error("FP8 packet A/B exact shape does not support every requested packet width.");
        }
        const KParallelTiledStagePlan plan{
            phase.topology, packet_bytes, FP8_PACKET_AB_BLOCK_THREADS, shards};
        const auto run = [&]() {
            launchKParallelTiledStage(semantics, input, output, geometry, plan, 1.0f, stream);
        };

        run();
        stream.synchronize();
        if (staged) {
            validateSinglePassPartial(output,
                                      geometry,
                                      shards,
                                      FP8_PACKET_AB_STAGED_ROWS_PER_SHARD,
                                      SinglePassShardPolicy::Balanced,
                                      stream,
                                      "fp8_packet_ab_staged",
                                      FP8_PACKET_AB_FILL_VALUE);
        } else {
            validateSinglePassComplete(
                output, geometry, stream, "fp8_packet_ab_complete", FP8_PACKET_AB_FILL_VALUE);
        }

        const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run);
        const uint64_t items_per_lane = packet_bytes / sizeof(InputT);
        const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(
            geometry, packet_bytes, FP8_PACKET_AB_BLOCK_THREADS);
        const uint64_t blocks = outer * component_tiles * shards;
        const uint64_t input_bytes = input.getArraySizeInBytes();
        const uint64_t output_bytes = output.getArraySizeInBytes();
        const double logical_gbps =
            static_cast<double>(input_bytes + output_bytes) / (timing.median_ms * 1.0e6);

        std::cout << phase.name << ",o" << outer << "_r" << phase.reduction << "_k" << inner << ','
                  << singlePassDtypeName(dtype) << ',' << (staged ? "staged" : "complete") << ',' << outer << ','
                  << phase.reduction << ',' << inner << ',' << packet_bytes << ',' << items_per_lane << ','
                  << FP8_PACKET_AB_BLOCK_THREADS << ',' << shards << ','
                  << (staged ? FP8_PACKET_AB_STAGED_ROWS_PER_SHARD : phase.reduction) << ',' << blocks << ','
                  << input_bytes << ',' << output_bytes << ',' << std::fixed << std::setprecision(4)
                  << timing.median_ms << ',' << timing.best_ms << ',' << timing.worst_ms << ','
                  << std::setprecision(2) << logical_gbps << '\n';
    }
}

void runFp8KParallelPacketABCensusImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr uint64_t MIB = 1024ULL * 1024ULL;
    constexpr std::array<Fp8PacketABPhase, 3> PHASES = {{{"complete_64mib",
                                                           TiledRKStageTopology::Complete,
                                                           64,
                                                           64ULL * MIB},
                                                          {"staged_64mib",
                                                           TiledRKStageTopology::Staged,
                                                           4096,
                                                           64ULL * MIB},
                                                          {"deep_staged_512mib",
                                                           TiledRKStageTopology::Staged,
                                                           32768,
                                                           512ULL * MIB}}};
    constexpr std::array<uint64_t, 4> INNER_WIDTHS = {512, 1024, 2048, 8192};
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};

    std::cout << "# mode=rk_fp8_kparallel_packet_census comparison=exact_shape_physical_kernel_ab\n";
    std::cout << "# purpose=compare FP8 KParallel packet16/8/4/2 on literally identical reductions\n";
    std::cout << "# invariant=same dtype,O,R,K,input buffer,output buffer,topology,block_threads,shards; only packet_bytes changes\n";
    std::cout << "# complete topology: block_threads=32, shards=1, output dtype=input dtype\n";
    std::cout << "# staged topology: block_threads=32, rows_per_shard=64, output dtype=fp32\n";
    std::cout << "# production planner is not used to choose packet width in this benchmark\n";
    std::cout << "# warmups=" << RK_SINGLE_PASS_WARMUPS << " timing_samples=" << RK_SINGLE_PASS_SAMPLES << '\n';
    std::cout << "phase,case,dtype,topology,outer,R,K,packet_bytes,items_per_lane,block_threads,shards_per_output,"
                 "rows_per_shard,blocks,input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";

    uint64_t rows = 0;
    for (const Fp8PacketABPhase& phase : PHASES) {
        for (uint64_t inner : INNER_WIDTHS) {
            const uint64_t elements_per_outer = checkedSinglePassElements(1, phase.reduction, inner);
            if (phase.target_input_bytes % elements_per_outer != 0) {
                throw std::logic_error("FP8 packet A/B target bytes must map to one exact shared outer count.");
            }
            const uint64_t outer = phase.target_input_bytes / elements_per_outer;
            if (outer == 0) {
                throw std::logic_error("FP8 packet A/B exact shape requires non-zero outer count.");
            }
            for (DataType dtype : DTYPES) {
                Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, phase.reduction, inner}));
                input.fill(FP8_PACKET_AB_FILL_VALUE, stream);
                stream.synchronize();
                if (input.getArraySizeInBytes() != phase.target_input_bytes) {
                    throw std::logic_error("FP8 packet A/B input bytes drifted from the exact target size.");
                }
                switch (dtype) {
                    case DataType::FP8_E4M3:
                        runFp8PacketABShape<__nv_fp8_e4m3>(
                            phase, dtype, outer, inner, input, cache_flush, stream);
                        break;
                    case DataType::FP8_E5M2:
                        runFp8PacketABShape<__nv_fp8_e5m2>(
                            phase, dtype, outer, inner, input, cache_flush, stream);
                        break;
                    default:
                        throw std::logic_error("FP8 packet A/B census received an unsupported dtype.");
                }
                rows += 4;
            }
        }
    }
    std::cout << "# rk_fp8_kparallel_packet_census_complete measured_rows=" << rows << '\n';
#else
    (void)cache_flush;
    (void)stream;
    throw std::logic_error("FP8 packet A/B census requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


constexpr uint64_t FP8_SMALL_PACKET_SWEEP_MAX_INPUT_BYTES = 64ULL * 1024ULL * 1024ULL;
constexpr uint32_t FP8_SMALL_PACKET_SWEEP_BLOCK_THREADS = 32;
constexpr double FP8_SMALL_PACKET_SWEEP_FILL_VALUE = 1.0 / 512.0;

struct Fp8SmallPacketShape {
    const char* family;
    uint64_t outer;
    uint64_t reduction;
    uint64_t inner;
};

template <typename InputT>
void runFp8SmallPacketShape(const Fp8SmallPacketShape& benchmark_case,
                            DataType dtype,
                            const Tensor& input,
                            Tensor& cache_flush,
                            Stream& stream) {
    static_assert(sizeof(InputT) == 1, "FP8 small-packet sweep requires one-byte input storage.");
    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, input.getDimensions(), {1});
    if (geometry.path != CubReductionPath::TiledFixedSegment) {
        throw std::logic_error("FP8 small-packet sweep case was not TiledFixedSegment.");
    }
    if (geometry.outer_size != benchmark_case.outer || geometry.reduction_size != benchmark_case.reduction
        || geometry.inner_size != benchmark_case.inner) {
        throw std::logic_error("FP8 small-packet sweep geometry disagrees with the requested exact shape.");
    }

    Tensor output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
    const CubReductionStageSemantics semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::Complete, geometry.reduction_size);

    constexpr std::array<size_t, 4> PACKETS = {16, 8, 4, 2};
    std::array<size_t, 4> legal_packets{};
    size_t legal_count = 0;
    for (size_t packet_bytes : PACKETS) {
        if (supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            legal_packets[legal_count++] = packet_bytes;
        }
    }
    if (legal_count < 2) {
        throw std::logic_error("FP8 small-packet sweep requires at least two legal packet widths per exact shape.");
    }
    const size_t widest_legal_packet = legal_packets[0];

    for (size_t packet_index = 0; packet_index < legal_count; ++packet_index) {
        const size_t packet_bytes = legal_packets[packet_index];
        const KParallelTiledStagePlan plan{
            TiledRKStageTopology::Complete, packet_bytes, FP8_SMALL_PACKET_SWEEP_BLOCK_THREADS, 1};
        const auto run = [&]() {
            launchKParallelTiledStage(semantics, input, output, geometry, plan, 1.0f, stream);
        };

        run();
        stream.synchronize();
        validateSinglePassComplete(
            output, geometry, stream, "fp8_small_packet_complete", FP8_SMALL_PACKET_SWEEP_FILL_VALUE);

        const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run);
        const uint64_t items_per_lane = packet_bytes / sizeof(InputT);
        const uint64_t packet_owners = geometry.inner_size / items_per_lane;
        const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(
            geometry, packet_bytes, FP8_SMALL_PACKET_SWEEP_BLOCK_THREADS);
        const uint64_t blocks = geometry.outer_size * component_tiles;
        const uint64_t input_bytes = input.getArraySizeInBytes();
        const uint64_t output_bytes = output.getArraySizeInBytes();
        const double logical_gbps =
            static_cast<double>(input_bytes + output_bytes) / (timing.median_ms * 1.0e6);

        std::cout << benchmark_case.family << ",o" << benchmark_case.outer << "_r" << benchmark_case.reduction
                  << "_k" << benchmark_case.inner << ',' << singlePassDtypeName(dtype) << ",complete,"
                  << benchmark_case.outer << ',' << benchmark_case.reduction << ',' << benchmark_case.inner << ','
                  << widest_legal_packet << ',' << legal_count << ',' << packet_bytes << ',' << items_per_lane << ','
                  << FP8_SMALL_PACKET_SWEEP_BLOCK_THREADS << ",1," << packet_owners << ',' << component_tiles << ','
                  << blocks << ',' << input_bytes << ',' << output_bytes << ',' << std::fixed << std::setprecision(4)
                  << timing.median_ms << ',' << timing.best_ms << ',' << timing.worst_ms << ',' << std::setprecision(2)
                  << logical_gbps << '\n';
    }
}

void runFp8KParallelSmallPacketSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    constexpr std::array<uint64_t, 6> SUPPLY_OUTERS = {1, 4, 16, 64, 256, 1024};
    constexpr std::array<uint64_t, 3> SUPPLY_REDUCTIONS = {32, 128, 512};
    constexpr std::array<uint64_t, 4> SUPPLY_INNERS = {16, 64, 256, 1024};

    constexpr std::array<uint64_t, 4> AWKWARD_OUTERS = {1, 8, 64, 512};
    constexpr std::array<uint64_t, 11> AWKWARD_REDUCTIONS = {17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513};
    constexpr std::array<uint64_t, 3> AWKWARD_R_INNERS = {64, 256, 1024};

    // These K values deliberately exercise three widest-legal packet classes while retaining at least one narrower
    // packet for exact-shape A/B. Multiples of 16 are awkward-but-four-way; 8-only and 4-only groups test the natural
    // fallback envelope without inventing benchmark-only padding. Odd K is intentionally outside this packet sweep:
    // none of the 16/8/4/2-byte KParallel specializations can represent it exactly.
    constexpr std::array<uint64_t, 7> AWKWARD_K_P16 = {48, 80, 112, 144, 240, 496, 1008};
    constexpr std::array<uint64_t, 7> AWKWARD_K_P8 = {24, 40, 72, 120, 248, 504, 1016};
    constexpr std::array<uint64_t, 7> AWKWARD_K_P4 = {20, 36, 68, 124, 252, 508, 1020};
    constexpr std::array<uint64_t, 3> AWKWARD_K_REDUCTIONS = {32, 128, 512};

    constexpr std::array<std::pair<uint64_t, uint64_t>, 6> AWKWARD_BOTH = {{{33, 48},
                                                                            {65, 120},
                                                                            {129, 124},
                                                                            {257, 240},
                                                                            {511, 504},
                                                                            {513, 508}}};

    std::cout << "# mode=rk_fp8_kparallel_small_packet_sweep comparison=exact_shape_complete_kernel_ab\n";
    std::cout << "# purpose=find small-reduction packet-width crossovers caused by KParallel work/parallelism supply\n";
    std::cout << "# invariant=same dtype,O,R,K,input buffer,output buffer,Complete topology,block_threads=32,shards=1; only packet_bytes changes\n";
    std::cout << "# packet policy=measure every legal width among 16/8/4/2; baseline is widest legal packet for that exact K\n";
    std::cout << "# families=supply|awkward_r|awkward_k|awkward_both max_input_bytes="
              << FP8_SMALL_PACKET_SWEEP_MAX_INPUT_BYTES << '\n';
    std::cout << "# warmups=" << RK_SINGLE_PASS_WARMUPS << " timing_samples=" << RK_SINGLE_PASS_SAMPLES << '\n';
    std::cout << "family,case,dtype,topology,outer,R,K,widest_legal_packet,legal_packet_count,packet_bytes,items_per_lane,"
                 "block_threads,shards_per_output,packet_owners,component_tiles,blocks,input_bytes,output_bytes,median_ms,"
                 "best_ms,worst_ms,logical_GBps\n";

    uint64_t groups = 0;
    uint64_t rows = 0;
    uint64_t skipped_over_size = 0;

    const auto run_shape = [&](const char* family, uint64_t outer, uint64_t reduction, uint64_t inner) {
        const uint64_t elements = checkedSinglePassElements(outer, reduction, inner);
        if (elements > FP8_SMALL_PACKET_SWEEP_MAX_INPUT_BYTES) {
            ++skipped_over_size;
            return;
        }
        Fp8SmallPacketShape benchmark_case{family, outer, reduction, inner};
        for (DataType dtype : DTYPES) {
            Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, reduction, inner}));
            input.fill(FP8_SMALL_PACKET_SWEEP_FILL_VALUE, stream);
            stream.synchronize();
            switch (dtype) {
                case DataType::FP8_E4M3:
                    runFp8SmallPacketShape<__nv_fp8_e4m3>(benchmark_case, dtype, input, cache_flush, stream);
                    break;
                case DataType::FP8_E5M2:
                    runFp8SmallPacketShape<__nv_fp8_e5m2>(benchmark_case, dtype, input, cache_flush, stream);
                    break;
                default:
                    throw std::logic_error("FP8 small-packet sweep received an unsupported dtype.");
            }
            size_t legal_count = 0;
            for (size_t packet_bytes : std::array<size_t, 4>{16, 8, 4, 2}) {
                if (inner % packet_bytes == 0) {
                    ++legal_count;
                }
            }
            rows += legal_count;
            ++groups;
        }
    };

    for (uint64_t outer : SUPPLY_OUTERS) {
        for (uint64_t reduction : SUPPLY_REDUCTIONS) {
            for (uint64_t inner : SUPPLY_INNERS) {
                run_shape("supply", outer, reduction, inner);
            }
        }
    }
    for (uint64_t outer : AWKWARD_OUTERS) {
        for (uint64_t reduction : AWKWARD_REDUCTIONS) {
            for (uint64_t inner : AWKWARD_R_INNERS) {
                run_shape("awkward_r", outer, reduction, inner);
            }
        }
    }
    for (uint64_t outer : AWKWARD_OUTERS) {
        for (uint64_t reduction : AWKWARD_K_REDUCTIONS) {
            for (uint64_t inner : AWKWARD_K_P16) {
                run_shape("awkward_k", outer, reduction, inner);
            }
            for (uint64_t inner : AWKWARD_K_P8) {
                run_shape("awkward_k", outer, reduction, inner);
            }
            for (uint64_t inner : AWKWARD_K_P4) {
                run_shape("awkward_k", outer, reduction, inner);
            }
        }
    }
    for (uint64_t outer : AWKWARD_OUTERS) {
        for (const auto& [reduction, inner] : AWKWARD_BOTH) {
            run_shape("awkward_both", outer, reduction, inner);
        }
    }

    std::cout << "# rk_fp8_kparallel_small_packet_sweep_complete measured_groups=" << groups
              << " measured_rows=" << rows << " skipped_shapes_over_input_ceiling=" << skipped_over_size << '\n';
#else
    (void)cache_flush;
    (void)stream;
    throw std::logic_error("FP8 small-packet sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

constexpr uint64_t FP8_STRADDLED_SWEEP_MAX_INPUT_BYTES = 128ULL * 1024ULL * 1024ULL;
constexpr uint64_t FP8_STRADDLED_SWEEP_TARGET_INPUT_BYTES = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t FP8_STRADDLED_SWEEP_TARGET_ROWS_PER_SHARD = 128;
constexpr double FP8_STRADDLED_SWEEP_FILL_VALUE = 1.0 / 512.0;

struct Fp8StraddledShape {
    const char* family;
    uint64_t outer;
    uint64_t reduction;
    uint64_t inner;
};

void runFp8KParallelStraddledShape(const Fp8StraddledShape& benchmark_case,
                                   DataType dtype,
                                   const Tensor& input,
                                   Tensor& cache_flush,
                                   Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, input.getDimensions(), {1});
    if (geometry.path != CubReductionPath::TiledFixedSegment || geometry.outer_size != benchmark_case.outer
        || geometry.reduction_size != benchmark_case.reduction || geometry.inner_size != benchmark_case.inner) {
        throw std::logic_error("FP8 KParallel Straddled sweep geometry disagrees with the requested exact shape.");
    }

    const CubReductionStageSemantics complete_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::Complete, geometry.reduction_size);
    const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::First, geometry.reduction_size);
    constexpr std::array<uint32_t, 4> BLOCK_THREADS = {32, 64, 128, 256};

    const uint64_t gcd = std::gcd(geometry.inner_size, uint64_t{16});
    const uint64_t row_period = 16 / gcd;
    const uint64_t packets_per_supertile = geometry.inner_size / gcd;

    for (uint32_t block_threads : BLOCK_THREADS) {
        const uint64_t warps_per_block = block_threads / TILED_REDUCTION_WARP_THREADS;

        Tensor complete_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const KParallelStraddledStagePlan complete_plan{
            TiledRKStageTopology::Complete, block_threads, 1};
        const auto run_complete = [&]() {
            launchFp8KParallelStraddledStage(
                complete_semantics, input, complete_output, geometry, complete_plan, 1.0f, stream);
        };
        run_complete();
        stream.synchronize();
        validateSinglePassComplete(
            complete_output, geometry, stream, "fp8_kparallel_straddled_complete", FP8_STRADDLED_SWEEP_FILL_VALUE);
        const RKSinglePassTiming complete_timing = timeRKSinglePass(cache_flush, stream, run_complete);
        const CubKernelOccupancyInfo complete_occupancy =
            queryFp8KParallelStraddledStageOccupancy(dtype, complete_plan);
        const uint64_t complete_blocks = singlePassCeilDiv(geometry.outer_size, warps_per_block);
        const double complete_gbps = static_cast<double>(input.getArraySizeInBytes() + complete_output.getArraySizeInBytes())
                                     / (complete_timing.median_ms * 1.0e6);
        std::cout << benchmark_case.family << ",o" << geometry.outer_size << "_r" << geometry.reduction_size << "_k"
                  << geometry.inner_size << ',' << singlePassDtypeName(dtype)
                  << ",complete," << geometry.outer_size << ',' << geometry.reduction_size << ',' << geometry.inner_size
                  << ",16," << row_period << ',' << packets_per_supertile << ',' << block_threads << ",1,"
                  << geometry.reduction_size << ',' << complete_blocks << ',' << complete_occupancy.registers_per_thread
                  << ',' << complete_occupancy.dynamic_shared_bytes << ',' << complete_occupancy.max_active_blocks_per_sm
                  << ',' << complete_occupancy.max_active_warps_per_sm << ',' << input.getArraySizeInBytes() << ','
                  << complete_output.getArraySizeInBytes() << ',' << std::fixed << std::setprecision(4)
                  << complete_timing.median_ms << ',' << complete_timing.best_ms << ',' << complete_timing.worst_ms << ','
                  << std::setprecision(2) << complete_gbps << '\n';

        const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, FP8_STRADDLED_SWEEP_TARGET_ROWS_PER_SHARD);
        if (shards <= 1 || shards > geometry.reduction_size) {
            continue;
        }
        Tensor partials(input.getPlacement(),
                        TensorDescriptor(DataType::FP32, {geometry.outer_size, shards, geometry.inner_size}));
        const KParallelStraddledStagePlan staged_plan{
            TiledRKStageTopology::Staged, block_threads, shards};
        const auto run_staged = [&]() {
            launchFp8KParallelStraddledStage(
                first_semantics, input, partials, geometry, staged_plan, 1.0f, stream);
        };
        run_staged();
        stream.synchronize();
        validateSinglePassPartial(partials,
                                  geometry,
                                  shards,
                                  FP8_STRADDLED_SWEEP_TARGET_ROWS_PER_SHARD,
                                  SinglePassShardPolicy::Balanced,
                                  stream,
                                  "fp8_kparallel_straddled_staged",
                                  FP8_STRADDLED_SWEEP_FILL_VALUE);
        const RKSinglePassTiming staged_timing = timeRKSinglePass(cache_flush, stream, run_staged);
        const CubKernelOccupancyInfo staged_occupancy =
            queryFp8KParallelStraddledStageOccupancy(dtype, staged_plan);
        const uint64_t staged_work = geometry.outer_size * shards;
        const uint64_t staged_blocks = singlePassCeilDiv(staged_work, warps_per_block);
        const double staged_gbps = static_cast<double>(input.getArraySizeInBytes() + partials.getArraySizeInBytes())
                                   / (staged_timing.median_ms * 1.0e6);
        std::cout << benchmark_case.family << ",o" << geometry.outer_size << "_r" << geometry.reduction_size << "_k"
                  << geometry.inner_size << ',' << singlePassDtypeName(dtype)
                  << ",staged," << geometry.outer_size << ',' << geometry.reduction_size << ',' << geometry.inner_size
                  << ",16," << row_period << ',' << packets_per_supertile << ',' << block_threads << ',' << shards << ','
                  << FP8_STRADDLED_SWEEP_TARGET_ROWS_PER_SHARD << ',' << staged_blocks << ','
                  << staged_occupancy.registers_per_thread << ',' << staged_occupancy.dynamic_shared_bytes << ','
                  << staged_occupancy.max_active_blocks_per_sm << ',' << staged_occupancy.max_active_warps_per_sm << ','
                  << input.getArraySizeInBytes() << ',' << partials.getArraySizeInBytes() << ',' << std::fixed
                  << std::setprecision(4) << staged_timing.median_ms << ',' << staged_timing.best_ms << ','
                  << staged_timing.worst_ms << ',' << std::setprecision(2) << staged_gbps << '\n';
    }
#else
    static_cast<void>(benchmark_case);
    static_cast<void>(dtype);
    static_cast<void>(input);
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 KParallel Straddled sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void runFp8KParallelStraddledSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    constexpr std::array<uint64_t, 6> K17_OUTERS = {1, 4, 16, 64, 256, 1024};
    constexpr std::array<uint64_t, 5> K17_REDUCTIONS = {16, 64, 256, 1024, 4096};
    // K=1 canonicalizes to a suffix/R reduction in analyzeValueGeometry(), so it is not an RK
    // exact shape and must not be included in this KParallel Straddled calibration.
    constexpr std::array<uint64_t, 15> ODD_K = {
        3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    constexpr std::array<uint64_t, 3> SATURATION_REDUCTIONS = {128, 1024, 8192};

    std::cout << "# mode=rk_fp8_kparallel_straddled_sweep operation=sum comparison=physical_layout_calibration\n";
    std::cout << "# candidate=KParallel_straddled invariant=aligned p16 flattened loads; packets may cross logical rows; production selector unchanged\n";
    std::cout << "# support=FP8 K=1..31 with K%16!=0; Complete and balanced Staged; block_threads=32|64|128|256\n";
    std::cout << "# saturation_odd_k=3..31; K=1 canonicalizes to suffix/R geometry and is intentionally excluded\n";
    std::cout << "# staged_target_rows_per_shard=" << FP8_STRADDLED_SWEEP_TARGET_ROWS_PER_SHARD
              << " max_input_bytes=" << FP8_STRADDLED_SWEEP_MAX_INPUT_BYTES
              << " saturation_target_input_bytes=" << FP8_STRADDLED_SWEEP_TARGET_INPUT_BYTES << '\n';
    std::cout << "family,case,dtype,topology,outer,R,K,packet_bytes,row_period,packets_per_supertile,block_threads,"
                 "shards_per_output,target_rows_per_shard,blocks,regs_per_thread,dynamic_shared_bytes,max_active_blocks_per_sm,"
                 "max_active_warps_per_sm,input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";

    uint64_t shape_dtypes = 0;
    uint64_t skipped_over_size = 0;
    const auto run_shape = [&](const char* family, uint64_t outer, uint64_t reduction, uint64_t inner) {
        const uint64_t elements = checkedSinglePassElements(outer, reduction, inner);
        if (elements > FP8_STRADDLED_SWEEP_MAX_INPUT_BYTES) {
            ++skipped_over_size;
            return;
        }
        const Fp8StraddledShape shape{family, outer, reduction, inner};
        for (DataType dtype : DTYPES) {
            Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, reduction, inner}));
            input.fill(FP8_STRADDLED_SWEEP_FILL_VALUE, stream);
            stream.synchronize();
            runFp8KParallelStraddledShape(shape, dtype, input, cache_flush, stream);
            ++shape_dtypes;
        }
    };

    for (uint64_t outer : K17_OUTERS) {
        for (uint64_t reduction : K17_REDUCTIONS) {
            run_shape("k17_supply", outer, reduction, 17);
        }
    }

    for (uint64_t reduction : SATURATION_REDUCTIONS) {
        for (uint64_t inner : ODD_K) {
            const uint64_t row_bytes = reduction * inner;
            const uint64_t outer = std::max<uint64_t>(1, FP8_STRADDLED_SWEEP_TARGET_INPUT_BYTES / row_bytes);
            run_shape("odd_k_saturation", outer, reduction, inner);
        }
    }

    std::cout << "# rk_fp8_kparallel_straddled_sweep_complete shape_dtypes=" << shape_dtypes
              << " skipped_shapes_over_input_ceiling=" << skipped_over_size << '\n';
#else
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 KParallel Straddled sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void executeModernRKStage(const ExecutableRKStage& stage,
                          CubReductionOp op,
                          uint64_t total_reduction_size,
                          Stream& stream) {
    const CubReductionStageSemantics semantics =
        makeValueReductionStageSemantics(op, stage.plan->role, total_reduction_size);
    switch (stage.plan->implementation) {
        case DenseRKProductionImplementation::KParallelPass: {
            const KParallelTiledStagePlan kplan{
                stage.plan->progress == DenseRKProgress::Complete ? TiledRKStageTopology::Complete
                                                                 : TiledRKStageTopology::Staged,
                stage.plan->packet_bytes,
                stage.plan->block_threads,
                stage.plan->shards_per_output};
            launchKParallelTiledStage(semantics, stage.input, stage.output, stage.geometry, kplan, 1.0f, stream);
            return;
        }
        case DenseRKProductionImplementation::RCooperativeAlignedComplete:
            launchPacketAdaptiveComplete(semantics, *stage.plan, stage.input, stage.output, stage.geometry, stream);
            return;
        case DenseRKProductionImplementation::RCooperativeAlignedStaged:
            launchCooperativeShardedTiledFirstStage(semantics,
                                                    stage.input,
                                                    stage.output,
                                                    stage.geometry,
                                                    stage.plan->rows_per_shard,
                                                    stage.plan->shards_per_output,
                                                    stream);
            return;
        case DenseRKProductionImplementation::RCooperativeRotatedStaged:
            launchAwkwardAlignmentRotatedShardedFirstStage(semantics,
                                                           stage.input,
                                                           stage.output,
                                                           stage.geometry,
                                                           stage.plan->rows_per_shard,
                                                           stage.plan->shards_per_output,
                                                           stream);
            return;
        case DenseRKProductionImplementation::RCooperativeFlatRowsStaged:
            launchNarrowLowPrecisionFlatRCooperativeFirstStage(semantics,
                                                               stage.input,
                                                               stage.output,
                                                               stage.geometry,
                                                               stage.plan->rows_per_shard,
                                                               stage.plan->shards_per_output,
                                                               stream);
            return;
        default:
            throw std::logic_error("End-to-end RK calibration selected a non-modern implementation.");
    }
}

struct EndToEndStagedSequence {
    Tensor first_partials;
    Tensor final_output;
    std::vector<Tensor> intermediate_storage;
    std::vector<ExecutableRKStage> continuation_stages;
    std::string continuation_chain;
};

[[nodiscard]] EndToEndStagedSequence buildEndToEndStagedSequence(const Tensor& original_input,
                                                                  const CubReductionGeometry& original_geometry,
                                                                  uint64_t first_shards,
                                                                  DataType original_dtype,
                                                                  const Stream& stream) {
    if (first_shards <= 1 || first_shards >= original_geometry.reduction_size) {
        throw std::invalid_argument("End-to-end staged sequence requires a proper first-stage shard count.");
    }

    DenseReductionOutputSpec output_spec;
    output_spec.produce_value = true;
    output_spec.value_dtype = original_dtype;
    output_spec.produce_index = false;
    DenseReductionProblem problem = makeInitialDenseReductionProblem(original_input.getDimensions(),
                                                                     {1},
                                                                     DenseReductionAggregateKind::Value,
                                                                     output_spec);
    std::vector<DenseReductionSite> sites = enumerateDenseReductionSites(problem);
    if (sites.size() != 1 || classifyDenseReducerFamily(problem, sites.front()) != DenseReducerFamily::RK) {
        throw std::logic_error("End-to-end calibration expected one initial RK site.");
    }
    problem = advanceDenseReductionProblem(problem, sites.front(), first_shards);

    EndToEndStagedSequence sequence{
        Tensor(original_input.getPlacement(),
               TensorDescriptor(DataType::FP32,
                                {original_geometry.outer_size, first_shards, original_geometry.inner_size})),
        Tensor(original_input.getPlacement(), TensorDescriptor(original_dtype, original_geometry.output_dimensions)),
        {},
        {},
        {}};

    DenseRKPlanningContext context;
    context.original_input_dtype = original_dtype;
    context.value_operation = DenseRKValueOperation::Sum;
    context.multiprocessors = static_cast<uint32_t>(queryMultiprocessors(stream.getGpuNum()));
    context.occupancy_query = queryDenseRKFamilyExactOccupancy;
    context.occupancy_query_context = nullptr;
    context.allow_generic_rk_fallback = false;

    Tensor current_input = sequence.first_partials;
    current_input.reshape(runDimensions(problem));
    while (hasDenseReductionSites(problem)) {
        sites = enumerateDenseReductionSites(problem);
        if (sites.size() != 1 || classifyDenseReducerFamily(problem, sites.front()) != DenseReducerFamily::RK) {
            throw std::logic_error("End-to-end calibration continuation escaped the RK family.");
        }
        DenseReductionCandidate candidate = ReducersDenseRK::propose(problem, sites.front(), context);
        auto plan = std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(candidate.physical_plan);
        if (!plan || !denseRKIsFamilyImplementation(plan->implementation)) {
            throw std::logic_error("End-to-end calibration continuation selected a non-modern RK plan.");
        }
        CubReductionGeometry geometry = CubReduction::analyzeGeometry(
            plan->input_dimensions, std::vector<uint32_t>{plan->reduction_axis});

        Tensor stage_output;
        if (plan->progress == DenseRKProgress::Complete) {
            stage_output = sequence.final_output;
        } else {
            stage_output = Tensor(original_input.getPlacement(),
                                  TensorDescriptor(DataType::FP32,
                                                   {geometry.outer_size,
                                                    plan->shards_per_output,
                                                    geometry.inner_size}));
            sequence.intermediate_storage.push_back(stage_output);
        }
        sequence.continuation_stages.push_back(ExecutableRKStage{plan, current_input, stage_output, geometry});
        if (!sequence.continuation_chain.empty()) {
            sequence.continuation_chain += ">";
        }
        sequence.continuation_chain += strategyName(plan->strategy);
        sequence.continuation_chain += ":";
        sequence.continuation_chain += accessName(plan->access);
        sequence.continuation_chain += plan->progress == DenseRKProgress::Complete ? ":complete" : ":staged";

        problem = candidate.next_problem;
        if (plan->progress == DenseRKProgress::Staged) {
            Tensor next_input = stage_output;
            next_input.reshape(runDimensions(problem));
            current_input = next_input;
        }
    }
    if (sequence.continuation_stages.empty()
        || sequence.continuation_stages.back().plan->progress != DenseRKProgress::Complete) {
        throw std::logic_error("End-to-end calibration continuation did not terminate in Complete.");
    }
    return sequence;
}

void printEndToEndRow(const RKSinglePassSweepCase& benchmark_case,
                      DataType dtype,
                      std::string_view implementation,
                      size_t packet_bytes,
                      uint32_t block_threads,
                      uint64_t first_rows_per_shard,
                      uint64_t first_shards,
                      uint32_t total_stages,
                      uint64_t intermediate_bytes,
                      std::string_view continuation_chain,
                      const Tensor& input,
                      const Tensor& output,
                      const RKSinglePassTiming& timing) {
    const uint64_t logical_bytes = input.getArraySizeInBytes() + output.getArraySizeInBytes();
    const double gbps = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
    std::cout << benchmark_case.family << ',' << benchmark_case.name << ',' << benchmark_case.outer << ','
              << benchmark_case.reduction << ',' << benchmark_case.inner << ',' << singlePassDtypeName(dtype)
              << ",sum," << implementation << ',' << packet_bytes << ',' << block_threads << ','
              << first_rows_per_shard << ',' << first_shards << ',' << total_stages << ',' << intermediate_bytes << ','
              << continuation_chain << ',' << input.getArraySizeInBytes() << ',' << output.getArraySizeInBytes() << ','
              << std::fixed << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ','
              << timing.worst_ms << ',' << std::setprecision(2) << gbps << '\n';
}

template <typename InputT>
void runEndToEndCrossoverCase(const RKSinglePassSweepCase& benchmark_case,
                              DataType dtype,
                              const Tensor& input,
                              const CubReductionGeometry& geometry,
                              Tensor& cache_flush,
                              Stream& stream) {
    constexpr std::array<size_t, 3> packets = {4, 8, 16};
    constexpr std::array<uint64_t, 2> first_rows = {16, 32};
    const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::First, geometry.reduction_size);

    for (size_t packet_bytes : packets) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            continue;
        }
        const uint32_t block_threads = singlePassKParallelBlockThreads<InputT>(geometry, packet_bytes);

        Tensor lean_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_lean = [&]() {
            launchLeanKParallelCompleteSum<InputT>(input, lean_output, geometry, packet_bytes, block_threads, stream);
        };
        run_lean();
        stream.synchronize();
        validateSinglePassComplete(lean_output, geometry, stream, "end_to_end_lean_complete");
        const RKSinglePassTiming lean_timing = timeRKSinglePass(cache_flush, stream, run_lean);
        printEndToEndRow(benchmark_case,
                         dtype,
                         "lean_kparallel_complete",
                         packet_bytes,
                         block_threads,
                         geometry.reduction_size,
                         1,
                         1,
                         0,
                         "complete",
                         input,
                         lean_output,
                         lean_timing);

        for (uint64_t target_rows : first_rows) {
            const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, target_rows);
            if (shards <= 1 || shards >= geometry.reduction_size) {
                continue;
            }
            EndToEndStagedSequence sequence =
                buildEndToEndStagedSequence(input, geometry, shards, dtype, stream);
            const KParallelTiledStagePlan first_plan{
                TiledRKStageTopology::Staged, packet_bytes, block_threads, shards};
            const auto run_sequence = [&]() {
                launchKParallelTiledStage(first_semantics,
                                          input,
                                          sequence.first_partials,
                                          geometry,
                                          first_plan,
                                          1.0f,
                                          stream);
                for (const ExecutableRKStage& stage : sequence.continuation_stages) {
                    executeModernRKStage(stage, CubReductionOp::Sum, geometry.reduction_size, stream);
                }
            };
            run_sequence();
            stream.synchronize();
            validateSinglePassComplete(sequence.final_output, geometry, stream, "end_to_end_staged_replan");
            const RKSinglePassTiming sequence_timing = timeRKSinglePass(cache_flush, stream, run_sequence);
            uint64_t intermediate_bytes = sequence.first_partials.getArraySizeInBytes();
            for (const Tensor& tensor : sequence.intermediate_storage) {
                intermediate_bytes += tensor.getArraySizeInBytes();
            }
            std::string chain = "k_parallel:aligned:staged>" + sequence.continuation_chain;
            printEndToEndRow(benchmark_case,
                             dtype,
                             target_rows == 16 ? "staged16_then_replan" : "staged32_then_replan",
                             packet_bytes,
                             block_threads,
                             target_rows,
                             shards,
                             static_cast<uint32_t>(1 + sequence.continuation_stages.size()),
                             intermediate_bytes,
                             chain,
                             input,
                             sequence.final_output,
                             sequence_timing);
        }
    }
}

void runRKKParallelEndToEndCrossoverCalibrationImpl(Tensor& cache_flush, Stream& stream) {
    std::cout << "# mode=rk_kparallel_end_to_end_crossover_calibration operation=sum dtypes=fp16|bf16|fp32\n";
    std::cout << "# Measures total reduction latency. staged16/staged32 force only the first physical KParallel staged pass; "
                 "the emitted FP32 intermediate is then replanned from scratch through ReducersDenseRK until Complete. "
                 "No production selector changes are made.\n";
    std::cout << "family,case,outer,R,K,dtype,operation,implementation,packet_bytes,block_threads,first_rows_per_shard,"
                 "first_shards,total_stages,intermediate_bytes,continuation_chain,input_bytes,output_bytes,median_ms,best_ms,"
                 "worst_ms,logical_GBps\n";

    constexpr std::array<uint64_t, 3> outers = {1, 2, 4};
    constexpr std::array<uint64_t, 6> reductions = {64, 103, 127, 205, 256, 410};
    constexpr std::array<uint64_t, 8> inners = {32, 64, 128, 256, 384, 512, 768, 1024};
    constexpr std::array<DataType, 3> dtypes = {DataType::FP16, DataType::BF16, DataType::FP32};

    for (uint64_t outer : outers) {
        for (uint64_t reduction : reductions) {
            for (uint64_t inner : inners) {
                RKSinglePassSweepCase benchmark_case{
                    "end_to_end_crossover", "o" + std::to_string(outer) + "_r" + std::to_string(reduction)
                                                 + "_i" + std::to_string(inner), outer, reduction, inner};
                for (DataType dtype : dtypes) {
                    Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, reduction, inner}));
                    input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
                    stream.synchronize();
                    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
                        CubReductionOp::Sum, input.getDimensions(), {1});
                    if (geometry.path != CubReductionPath::TiledFixedSegment) {
                        throw std::logic_error("End-to-end KParallel crossover case was not TiledFixedSegment.");
                    }

                    std::shared_ptr<StampedCubReduction> production =
                        CubReduction(CubReductionOp::Sum, {1}).stamp(input, stream);
                    const auto run_production = [&]() { production->runOn(stream); };
                    run_production();
                    stream.synchronize();
                    const Tensor production_output = production->getOutputTensor();
                    validateSinglePassComplete(production_output, geometry, stream, "end_to_end_production");
                    const RKSinglePassTiming production_timing = timeRKSinglePass(cache_flush, stream, run_production);
                    printEndToEndRow(benchmark_case,
                                     dtype,
                                     "production_selector",
                                     0,
                                     0,
                                     0,
                                     0,
                                     0,
                                     production->getWorkspaceSizeInBytes(),
                                     "production",
                                     input,
                                     production_output,
                                     production_timing);

                    switch (dtype) {
                        case DataType::FP16:
                            runEndToEndCrossoverCase<__half>(benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        case DataType::BF16:
                            runEndToEndCrossoverCase<__nv_bfloat16>(benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        case DataType::FP32:
                            runEndToEndCrossoverCase<float>(benchmark_case, dtype, input, geometry, cache_flush, stream);
                            break;
                        default:
                            throw std::logic_error("End-to-end KParallel crossover received unsupported dtype.");
                    }
                }
            }
        }
    }
}

struct RKRealisticProgressCase {
    const char* name;
    uint64_t outer;
    uint64_t reduction;
    uint64_t inner;
    DataType dtype;
};

void printRealisticEndToEndRow(const RKRealisticProgressCase& benchmark_case,
                               std::string_view implementation,
                               std::string_view first_strategy,
                               uint64_t first_rows_per_shard,
                               uint64_t first_shards,
                               uint32_t total_stages,
                               uint64_t intermediate_bytes,
                               std::string_view continuation_chain,
                               const Tensor& input,
                               const Tensor& output,
                               const RKSinglePassTiming& timing) {
    const uint64_t logical_bytes = input.getArraySizeInBytes() + output.getArraySizeInBytes();
    const double logical_gb_per_second = static_cast<double>(logical_bytes) / (timing.median_ms * 1.0e6);
    std::cout << "realistic_end_to_end," << benchmark_case.name << ',' << benchmark_case.outer << ','
              << benchmark_case.reduction << ',' << benchmark_case.inner << ','
              << singlePassDtypeName(benchmark_case.dtype) << ",sum," << implementation << ',' << first_strategy << ','
              << first_rows_per_shard << ',' << first_shards << ',' << total_stages << ',' << intermediate_bytes << ','
              << continuation_chain << ',' << input.getArraySizeInBytes() << ',' << output.getArraySizeInBytes() << ','
              << std::fixed << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ','
              << timing.worst_ms << ',' << std::setprecision(2) << logical_gb_per_second << '\n';
}

template <typename InputT>
void runRealisticForcedKParallelProgress(const RKRealisticProgressCase& benchmark_case,
                                         const Tensor& input,
                                         const CubReductionGeometry& geometry,
                                         Tensor& cache_flush,
                                         Stream& stream) {
    constexpr std::array<uint64_t, 8> target_rows_values = {16, 64, 128, 256, 512, 1024, 2048, 4096};
    const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::First, geometry.reduction_size);

    size_t packet_bytes = 0;
    for (size_t candidate : std::array<size_t, 3>{16, 8, 4}) {
        if (supportsSinglePassKParallelPacket<InputT>(geometry, candidate)) {
            packet_bytes = candidate;
            break;
        }
    }
    if (packet_bytes == 0) {
        return;
    }
    constexpr uint32_t block_threads = 32;

    for (uint64_t target_rows : target_rows_values) {
        const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, target_rows);
        if (shards <= 1 || shards >= geometry.reduction_size) {
            continue;
        }
        EndToEndStagedSequence sequence =
            buildEndToEndStagedSequence(input, geometry, shards, benchmark_case.dtype, stream);
        const KParallelTiledStagePlan first_plan{
            TiledRKStageTopology::Staged, packet_bytes, block_threads, shards};
        const auto run_sequence = [&]() {
            launchKParallelTiledStage(first_semantics, input, sequence.first_partials, geometry, first_plan, 1.0f, stream);
            for (const ExecutableRKStage& stage : sequence.continuation_stages) {
                executeModernRKStage(stage, CubReductionOp::Sum, geometry.reduction_size, stream);
            }
        };
        run_sequence();
        stream.synchronize();
        validateSinglePassComplete(sequence.final_output, geometry, stream, "realistic_kparallel_progress");
        const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run_sequence);
        uint64_t intermediate_bytes = sequence.first_partials.getArraySizeInBytes();
        for (const Tensor& tensor : sequence.intermediate_storage) {
            intermediate_bytes += tensor.getArraySizeInBytes();
        }
        printRealisticEndToEndRow(benchmark_case,
                                  "forced_first_pass",
                                  "k_parallel",
                                  target_rows,
                                  shards,
                                  static_cast<uint32_t>(1 + sequence.continuation_stages.size()),
                                  intermediate_bytes,
                                  "k_parallel:staged>" + sequence.continuation_chain,
                                  input,
                                  sequence.final_output,
                                  timing);
    }
}

void runRealisticForcedRCooperativeProgress(const RKRealisticProgressCase& benchmark_case,
                                             const Tensor& input,
                                             const CubReductionGeometry& geometry,
                                             Tensor& cache_flush,
                                             Stream& stream) {
    constexpr std::array<uint64_t, 4> target_rows_values = {128, 256, 512, 1024};
    const CubReductionStageSemantics first_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::First, geometry.reduction_size);

    for (uint64_t target_rows : target_rows_values) {
        const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, target_rows);
        if (shards <= 1 || shards >= geometry.reduction_size) {
            continue;
        }
        if (singlePassAlignedRCoopSupports(geometry, benchmark_case.dtype)) {
            EndToEndStagedSequence sequence =
                buildEndToEndStagedSequence(input, geometry, shards, benchmark_case.dtype, stream);
            const auto run_sequence = [&]() {
                launchCooperativeShardedTiledFirstStage(
                    first_semantics, input, sequence.first_partials, geometry, target_rows, shards, stream);
                for (const ExecutableRKStage& stage : sequence.continuation_stages) {
                    executeModernRKStage(stage, CubReductionOp::Sum, geometry.reduction_size, stream);
                }
            };
            run_sequence();
            stream.synchronize();
            validateSinglePassComplete(sequence.final_output, geometry, stream, "realistic_rcoop_aligned_progress");
            const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run_sequence);
            uint64_t intermediate_bytes = sequence.first_partials.getArraySizeInBytes();
            for (const Tensor& tensor : sequence.intermediate_storage) {
                intermediate_bytes += tensor.getArraySizeInBytes();
            }
            printRealisticEndToEndRow(benchmark_case,
                                      "forced_first_pass",
                                      "r_cooperative_aligned",
                                      target_rows,
                                      shards,
                                      static_cast<uint32_t>(1 + sequence.continuation_stages.size()),
                                      intermediate_bytes,
                                      "r_cooperative:aligned:staged>" + sequence.continuation_chain,
                                      input,
                                      sequence.final_output,
                                      timing);
        }

        EndToEndStagedSequence sequence =
            buildEndToEndStagedSequence(input, geometry, shards, benchmark_case.dtype, stream);
        const auto run_sequence = [&]() {
            launchAwkwardAlignmentRotatedShardedFirstStage(
                first_semantics, input, sequence.first_partials, geometry, target_rows, shards, stream);
            for (const ExecutableRKStage& stage : sequence.continuation_stages) {
                executeModernRKStage(stage, CubReductionOp::Sum, geometry.reduction_size, stream);
            }
        };
        run_sequence();
        stream.synchronize();
        validateSinglePassComplete(sequence.final_output, geometry, stream, "realistic_rcoop_rotated_progress");
        const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run_sequence);
        uint64_t intermediate_bytes = sequence.first_partials.getArraySizeInBytes();
        for (const Tensor& tensor : sequence.intermediate_storage) {
            intermediate_bytes += tensor.getArraySizeInBytes();
        }
        printRealisticEndToEndRow(benchmark_case,
                                  "forced_first_pass",
                                  "r_cooperative_rotated",
                                  target_rows,
                                  shards,
                                  static_cast<uint32_t>(1 + sequence.continuation_stages.size()),
                                  intermediate_bytes,
                                  "r_cooperative:rotated:staged>" + sequence.continuation_chain,
                                  input,
                                  sequence.final_output,
                                  timing);
    }
}

void runRKRealisticEndToEndProgressCalibrationImpl(Tensor& cache_flush, Stream& stream) {
    std::cout << "# mode=rk_realistic_end_to_end_progress_calibration operation=sum\n";
    std::cout << "# Realistic expensive RK cases only. Every row measures the complete reduction. Forced rows change only "
                 "the first physical pass; the emitted FP32 intermediate is then replanned from scratch through "
                 "ReducersDenseRK until Complete. Optimize absolute full-reduction microseconds, not tiny-case ratios.\n";
    std::cout << "family,case,outer,R,K,dtype,operation,implementation,first_strategy,first_rows_per_shard,first_shards,"
                 "total_stages,intermediate_bytes,continuation_chain,input_bytes,output_bytes,median_ms,best_ms,worst_ms,"
                 "logical_GBps\n";

    // Keep this population intentionally small and realistic. The benchmark objective is full-reduction wall time and
    // absolute microseconds saved, not the number of tiny cases improved. The last four cases cover the expensive
    // wide-K family that remains the largest absolute gap in the RK census.
    constexpr std::array<RKRealisticProgressCase, 18> cases = {{
        {"moderate_o2048_r256_k128_fp32", 2048, 256, 128, DataType::FP32},
        {"moderate_o128_r2048_k128_fp32", 128, 2048, 128, DataType::FP32},
        {"moderate_o64_r2048_k256_fp16", 64, 2048, 256, DataType::FP16},
        {"moderate_o8_r2048_k256_fp32", 8, 2048, 256, DataType::FP32},
        {"long_o16_r32768_k128_fp32", 16, 32768, 128, DataType::FP32},
        {"long_o16_r32768_k512_fp16", 16, 32768, 512, DataType::FP16},
        {"long_o16_r32768_k512_bf16", 16, 32768, 512, DataType::BF16},
        {"long_o8_r32768_k512_fp32", 8, 32768, 512, DataType::FP32},
        {"wide_o1_r32768_k2813_fp32", 1, 32768, 2813, DataType::FP32},
        {"awkward_o2_r32768_k2049_fp32", 2, 32768, 2049, DataType::FP32},
        {"awkward_o1_r104832_k191_fp32", 1, 104832, 191, DataType::FP32},
        {"awkward_o1_r104832_k255_fp32", 1, 104832, 255, DataType::FP32},
        {"transformer_o1_r104832_k768_fp32", 1, 104832, 768, DataType::FP32},
        {"transformer_o1_r104832_k769_fp32", 1, 104832, 769, DataType::FP32},
        {"wide_o1_r104832_k2049_fp32", 1, 104832, 2049, DataType::FP32},
        {"wide_o1_r104832_k3071_fp32", 1, 104832, 3071, DataType::FP32},
        {"wide_o1_r104832_k3073_fp32", 1, 104832, 3073, DataType::FP32},
        {"wide_o1_r104832_k4095_fp32", 1, 104832, 4095, DataType::FP32},
    }};

    for (const RKRealisticProgressCase& benchmark_case : cases) {
        Tensor input(cache_flush.getPlacement(),
                     TensorDescriptor(benchmark_case.dtype,
                                      {benchmark_case.outer, benchmark_case.reduction, benchmark_case.inner}));
        input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
        stream.synchronize();
        const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
            CubReductionOp::Sum, input.getDimensions(), {1});
        if (geometry.path != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error("Realistic end-to-end progress case was not TiledFixedSegment.");
        }

        std::shared_ptr<StampedCubReduction> production = CubReduction(CubReductionOp::Sum, {1}).stamp(input, stream);
        const auto run_production = [&]() { production->runOn(stream); };
        run_production();
        stream.synchronize();
        const Tensor production_output = production->getOutputTensor();
        validateSinglePassComplete(production_output, geometry, stream, "realistic_production");
        const RKSinglePassTiming production_timing = timeRKSinglePass(cache_flush, stream, run_production);
        printRealisticEndToEndRow(benchmark_case,
                                  "production_selector",
                                  "production",
                                  0,
                                  0,
                                  0,
                                  production->getWorkspaceSizeInBytes(),
                                  "production",
                                  input,
                                  production_output,
                                  production_timing);

        StampedDenseRKFamilyCandidate modern(CubReductionOp::Sum, input, {1}, stream);
        const auto run_modern = [&]() { modern.runOn(stream); };
        run_modern();
        stream.synchronize();
        validateSinglePassComplete(modern.getOutputTensor(), geometry, stream, "realistic_current_modern");
        const RKSinglePassTiming modern_timing = timeRKSinglePass(cache_flush, stream, run_modern);
        const CandidateLaunchMetadata modern_metadata = modern.getLaunchMetadata();
        printRealisticEndToEndRow(benchmark_case,
                                  "current_modern",
                                  modern_metadata.strategy,
                                  0,
                                  modern_metadata.shards_per_output.value_or(0),
                                  0,
                                  modern.getWorkspaceSizeInBytes(),
                                  modern_metadata.strategy,
                                  input,
                                  modern.getOutputTensor(),
                                  modern_timing);

        switch (benchmark_case.dtype) {
            case DataType::FP16:
                runRealisticForcedKParallelProgress<__half>(benchmark_case, input, geometry, cache_flush, stream);
                break;
            case DataType::BF16:
                runRealisticForcedKParallelProgress<__nv_bfloat16>(benchmark_case, input, geometry, cache_flush, stream);
                break;
            case DataType::FP32:
                runRealisticForcedKParallelProgress<float>(benchmark_case, input, geometry, cache_flush, stream);
                break;
            default:
                throw std::logic_error("Realistic end-to-end progress received unsupported dtype.");
        }
        runRealisticForcedRCooperativeProgress(benchmark_case, input, geometry, cache_flush, stream);
    }
}


void runRKFamilyBoundaryCalibrationImpl(Tensor& cache_flush, Stream& stream) {
    std::cout << "# mode=rk_family_boundary_calibration operation=sum dtype=fp32\n";
    std::cout << "# Focused DenseRK family calibration. Sweep packet alignment class against serial R depth; every row is a complete reduction.\n";
    std::cout << "family,case,outer,R,K,dtype,operation,implementation,first_strategy,first_rows_per_shard,first_shards,"
                 "total_stages,intermediate_bytes,continuation_chain,input_bytes,output_bytes,median_ms,best_ms,worst_ms,"
                 "logical_GBps\n";

    // Three K scales, each represented by a 16-byte-aligned width, an 8-byte-only width, and a 4-byte-only width.
    // Three R depths expose the serial-depth crossover without turning this into another broad census.
    constexpr std::array<RKRealisticProgressCase, 27> cases = {{
        {"boundary_o1_r32768_k512_fp32", 1, 32768, 512, DataType::FP32},
        {"boundary_o1_r32768_k514_fp32", 1, 32768, 514, DataType::FP32},
        {"boundary_o1_r32768_k513_fp32", 1, 32768, 513, DataType::FP32},
        {"boundary_o1_r65536_k512_fp32", 1, 65536, 512, DataType::FP32},
        {"boundary_o1_r65536_k514_fp32", 1, 65536, 514, DataType::FP32},
        {"boundary_o1_r65536_k513_fp32", 1, 65536, 513, DataType::FP32},
        {"boundary_o1_r104832_k512_fp32", 1, 104832, 512, DataType::FP32},
        {"boundary_o1_r104832_k514_fp32", 1, 104832, 514, DataType::FP32},
        {"boundary_o1_r104832_k513_fp32", 1, 104832, 513, DataType::FP32},

        {"boundary_o1_r32768_k2048_fp32", 1, 32768, 2048, DataType::FP32},
        {"boundary_o1_r32768_k2050_fp32", 1, 32768, 2050, DataType::FP32},
        {"boundary_o1_r32768_k2049_fp32", 1, 32768, 2049, DataType::FP32},
        {"boundary_o1_r65536_k2048_fp32", 1, 65536, 2048, DataType::FP32},
        {"boundary_o1_r65536_k2050_fp32", 1, 65536, 2050, DataType::FP32},
        {"boundary_o1_r65536_k2049_fp32", 1, 65536, 2049, DataType::FP32},
        {"boundary_o1_r104832_k2048_fp32", 1, 104832, 2048, DataType::FP32},
        {"boundary_o1_r104832_k2050_fp32", 1, 104832, 2050, DataType::FP32},
        {"boundary_o1_r104832_k2049_fp32", 1, 104832, 2049, DataType::FP32},

        {"boundary_o1_r32768_k4096_fp32", 1, 32768, 4096, DataType::FP32},
        {"boundary_o1_r32768_k4098_fp32", 1, 32768, 4098, DataType::FP32},
        {"boundary_o1_r32768_k4095_fp32", 1, 32768, 4095, DataType::FP32},
        {"boundary_o1_r65536_k4096_fp32", 1, 65536, 4096, DataType::FP32},
        {"boundary_o1_r65536_k4098_fp32", 1, 65536, 4098, DataType::FP32},
        {"boundary_o1_r65536_k4095_fp32", 1, 65536, 4095, DataType::FP32},
        {"boundary_o1_r104832_k4096_fp32", 1, 104832, 4096, DataType::FP32},
        {"boundary_o1_r104832_k4098_fp32", 1, 104832, 4098, DataType::FP32},
        {"boundary_o1_r104832_k4095_fp32", 1, 104832, 4095, DataType::FP32},
    }};

    for (const RKRealisticProgressCase& benchmark_case : cases) {
        Tensor input(cache_flush.getPlacement(),
                     TensorDescriptor(benchmark_case.dtype,
                                      {benchmark_case.outer, benchmark_case.reduction, benchmark_case.inner}));
        input.fill(RK_SINGLE_PASS_FILL_VALUE, stream);
        stream.synchronize();
        const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
            CubReductionOp::Sum, input.getDimensions(), {1});
        if (geometry.path != CubReductionPath::TiledFixedSegment) {
            throw std::logic_error("RK family-boundary case was not TiledFixedSegment.");
        }

        StampedDenseRKFamilyCandidate modern(CubReductionOp::Sum, input, {1}, stream);
        const auto run_modern = [&]() { modern.runOn(stream); };
        run_modern();
        stream.synchronize();
        validateSinglePassComplete(modern.getOutputTensor(), geometry, stream, "boundary_current_modern");
        const RKSinglePassTiming modern_timing = timeRKSinglePass(cache_flush, stream, run_modern);
        const CandidateLaunchMetadata modern_metadata = modern.getLaunchMetadata();
        printRealisticEndToEndRow(benchmark_case,
                                  "current_modern",
                                  modern_metadata.strategy,
                                  0,
                                  modern_metadata.shards_per_output.value_or(0),
                                  0,
                                  modern.getWorkspaceSizeInBytes(),
                                  modern_metadata.strategy,
                                  input,
                                  modern.getOutputTensor(),
                                  modern_timing);

        runRealisticForcedKParallelProgress<float>(benchmark_case, input, geometry, cache_flush, stream);
        runRealisticForcedRCooperativeProgress(benchmark_case, input, geometry, cache_flush, stream);
    }
}


constexpr uint32_t FP8_COMPLETE_LAYOUT_COMFORT_WARPS_PER_SM = 8;
constexpr double FP8_COMPLETE_LAYOUT_FILL_VALUE = 1.0 / 512.0;

struct Fp8CompleteLayoutShape {
    const char* family;
    uint64_t outer;
    uint64_t reduction;
    uint64_t inner;
};

template <typename InputT>
void runFp8CompleteLayoutShape(const Fp8CompleteLayoutShape& benchmark_case,
                               DataType dtype,
                               const Tensor& input,
                               Tensor& cache_flush,
                               Stream& stream,
                               const RKSinglePassDeviceGeometry& device_geometry) {
    static_assert(sizeof(InputT) == 1, "FP8 Complete layout sweep requires one-byte input storage.");
    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, input.getDimensions(), {1});
    if (geometry.path != CubReductionPath::TiledFixedSegment) {
        throw std::logic_error("FP8 Complete layout sweep case was not TiledFixedSegment.");
    }
    if (geometry.outer_size != benchmark_case.outer || geometry.reduction_size != benchmark_case.reduction
        || geometry.inner_size != benchmark_case.inner) {
        throw std::logic_error("FP8 Complete layout sweep geometry disagrees with requested exact shape.");
    }

    Tensor output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
    const CubReductionStageSemantics semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::Complete, geometry.reduction_size);

    constexpr std::array<size_t, 4> PACKETS = {16, 8, 4, 2};
    constexpr std::array<uint32_t, 4> CTA_WIDTHS = {32, 64, 128, 256};
    size_t widest_legal_packet = 0;
    size_t legal_packet_count = 0;
    for (size_t packet_bytes : PACKETS) {
        if (supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            if (widest_legal_packet == 0) {
                widest_legal_packet = packet_bytes;
            }
            ++legal_packet_count;
        }
    }
    if (legal_packet_count < 2) {
        throw std::logic_error("FP8 Complete layout sweep requires at least two legal packet widths per exact shape.");
    }

    for (size_t packet_bytes : PACKETS) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            continue;
        }
        const uint64_t items_per_lane = packet_bytes / sizeof(InputT);
        const uint64_t packet_owners = geometry.inner_size / items_per_lane;

        for (uint32_t block_threads : CTA_WIDTHS) {
            const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(
                geometry, packet_bytes, block_threads);
            const uint64_t blocks = geometry.outer_size * component_tiles;
            const uint64_t warps_per_cta = block_threads / device_geometry.warp_size;
            const uint64_t physical_warps = blocks * warps_per_cta;

            const KParallelTiledStagePlan plan{
                TiledRKStageTopology::Complete, packet_bytes, block_threads, 1};
            const CubKernelOccupancyInfo occupancy = queryKParallelTiledStageOccupancy(dtype, plan);
            if (occupancy.max_active_blocks_per_sm <= 0 || occupancy.max_active_warps_per_sm <= 0) {
                throw std::logic_error("FP8 Complete layout sweep received invalid exact kernel occupancy.");
            }
            const uint64_t useful_warps_per_output =
                (packet_owners + static_cast<uint64_t>(device_geometry.warp_size) - 1)
                / static_cast<uint64_t>(device_geometry.warp_size);
            const uint64_t useful_warps = geometry.outer_size * useful_warps_per_output;
            const uint64_t resident_device_blocks =
                static_cast<uint64_t>(device_geometry.sm_count)
                * static_cast<uint64_t>(occupancy.max_active_blocks_per_sm);
            const uint64_t resident_device_warps =
                static_cast<uint64_t>(device_geometry.sm_count)
                * static_cast<uint64_t>(occupancy.max_active_warps_per_sm);
            const unsigned __int128 useful_capacity_numerator =
                static_cast<unsigned __int128>(resident_device_blocks)
                * static_cast<unsigned __int128>(useful_warps);
            const uint64_t useful_capacity_by_blocks = static_cast<uint64_t>(
                useful_capacity_numerator / static_cast<unsigned __int128>(blocks));
            const uint64_t useful_resident_warps =
                std::min(resident_device_warps, useful_capacity_by_blocks);
            const uint64_t launch_warp_supply = std::min(useful_warps, useful_resident_warps);
            const uint64_t comfort_target =
                static_cast<uint64_t>(device_geometry.sm_count)
                * static_cast<uint64_t>(std::min<int>(FP8_COMPLETE_LAYOUT_COMFORT_WARPS_PER_SM,
                                                      occupancy.max_active_warps_per_sm));
            const double comfort_fraction = comfort_target == 0
                                                ? 0.0
                                                : static_cast<double>(launch_warp_supply)
                                                      / static_cast<double>(comfort_target);
            const uint64_t lane_slots = component_tiles * static_cast<uint64_t>(block_threads);
            const double lane_fill = lane_slots == 0
                                         ? 0.0
                                         : static_cast<double>(packet_owners) / static_cast<double>(lane_slots);

            const auto run = [&]() {
                launchKParallelTiledStage(semantics, input, output, geometry, plan, 1.0f, stream);
            };
            run();
            stream.synchronize();
            validateSinglePassComplete(
                output, geometry, stream, "fp8_complete_layout", FP8_COMPLETE_LAYOUT_FILL_VALUE);
            const RKSinglePassTiming timing = timeRKSinglePass(cache_flush, stream, run);
            const uint64_t input_bytes = input.getArraySizeInBytes();
            const uint64_t output_bytes = output.getArraySizeInBytes();
            const double logical_gbps =
                static_cast<double>(input_bytes + output_bytes) / (timing.median_ms * 1.0e6);

            std::cout << benchmark_case.family << ",o" << benchmark_case.outer << "_r"
                      << benchmark_case.reduction << "_k" << benchmark_case.inner << ','
                      << singlePassDtypeName(dtype) << ",complete," << benchmark_case.outer << ','
                      << benchmark_case.reduction << ',' << benchmark_case.inner << ',' << widest_legal_packet << ','
                      << legal_packet_count << ',' << packet_bytes << ',' << items_per_lane << ',' << block_threads << ','
                      << packet_owners << ',' << component_tiles << ',' << blocks << ',' << warps_per_cta << ','
                      << physical_warps << ',' << useful_warps << ',' << occupancy.registers_per_thread << ','
                      << occupancy.static_shared_bytes << ',' << occupancy.dynamic_shared_bytes << ','
                      << occupancy.max_active_blocks_per_sm << ',' << occupancy.max_active_warps_per_sm << ','
                      << resident_device_blocks << ',' << resident_device_warps << ',' << useful_resident_warps << ','
                      << launch_warp_supply << ',' << comfort_target << ','
                      << std::fixed << std::setprecision(6) << comfort_fraction << ',' << lane_fill << ','
                      << input_bytes << ',' << output_bytes << ',' << std::setprecision(4) << timing.median_ms << ','
                      << timing.best_ms << ',' << timing.worst_ms << ',' << std::setprecision(2) << logical_gbps << '\n';
        }
    }
}

void runFp8KParallelCompleteLayoutSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr uint64_t MAX_INPUT_BYTES = 64ULL * 1024ULL * 1024ULL;
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    constexpr std::array<uint64_t, 6> SUPPLY_OUTERS = {1, 4, 16, 64, 256, 1024};
    constexpr std::array<uint64_t, 3> SUPPLY_REDUCTIONS = {32, 128, 512};
    constexpr std::array<uint64_t, 4> SUPPLY_INNERS = {16, 64, 256, 1024};

    constexpr std::array<uint64_t, 4> AWKWARD_OUTERS = {1, 8, 64, 512};
    constexpr std::array<uint64_t, 11> AWKWARD_REDUCTIONS = {17, 31, 33, 63, 65, 127, 129, 255, 257, 511, 513};
    constexpr std::array<uint64_t, 3> AWKWARD_R_INNERS = {64, 256, 1024};

    constexpr std::array<uint64_t, 7> AWKWARD_K_P16 = {48, 80, 112, 144, 240, 496, 1008};
    constexpr std::array<uint64_t, 7> AWKWARD_K_P8 = {24, 40, 72, 120, 248, 504, 1016};
    constexpr std::array<uint64_t, 7> AWKWARD_K_P4 = {20, 36, 68, 124, 252, 508, 1020};
    constexpr std::array<uint64_t, 3> AWKWARD_K_REDUCTIONS = {32, 128, 512};
    constexpr std::array<std::pair<uint64_t, uint64_t>, 6> AWKWARD_BOTH = {{{33, 48},
                                                                            {65, 120},
                                                                            {129, 124},
                                                                            {257, 240},
                                                                            {511, 504},
                                                                            {513, 508}}};

    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    std::cout << "# mode=rk_fp8_kparallel_complete_layout_sweep comparison=exact_shape_packet_and_cta_optimization\n";
    std::cout << "# purpose=measure each FP8 KParallel packet reducer at its best Complete CTA layout on small/awkward reductions\n";
    std::cout << "# invariant=within one exact shape and packet, dtype/O/R/K/input/output/topology are fixed; only block_threads=32/64/128/256 changes\n";
    std::cout << "# packet comparison=for each exact shape, compare the measured-best CTA layout independently found for every legal packet width\n";
    std::cout << "# occupancy_policy_reference=Complete comfort uses useful warps; idle physical warps consume exact CTA/warp residency but receive zero progress credit\n";
    std::cout << "# families=supply|awkward_r|awkward_k|awkward_both max_input_bytes=" << MAX_INPUT_BYTES
              << " sm_count=" << device_geometry.sm_count << " warp_size=" << device_geometry.warp_size << '\n';
    std::cout << "# warmups=" << RK_SINGLE_PASS_WARMUPS << " timing_samples=" << RK_SINGLE_PASS_SAMPLES << '\n';
    std::cout << "family,case,dtype,topology,outer,R,K,widest_legal_packet,legal_packet_count,packet_bytes,items_per_lane,"
                 "block_threads,packet_owners,component_tiles,blocks,warps_per_cta,physical_warps,useful_warps,registers_per_thread,"
                 "static_shared_bytes,dynamic_shared_bytes,max_active_blocks_per_sm,max_active_warps_per_sm,"
                 "resident_device_blocks,resident_device_warps,useful_resident_warps,launch_warp_supply,comfort_target,comfort_fraction,lane_fill,input_bytes,"
                 "output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";

    uint64_t shape_dtypes = 0;
    uint64_t rows = 0;
    uint64_t skipped_over_size = 0;
    const auto run_shape = [&](const char* family, uint64_t outer, uint64_t reduction, uint64_t inner) {
        const uint64_t elements = checkedSinglePassElements(outer, reduction, inner);
        if (elements > MAX_INPUT_BYTES) {
            ++skipped_over_size;
            return;
        }
        Fp8CompleteLayoutShape benchmark_case{family, outer, reduction, inner};
        size_t legal_packet_count = 0;
        for (size_t packet_bytes : std::array<size_t, 4>{16, 8, 4, 2}) {
            if (inner % packet_bytes == 0) {
                ++legal_packet_count;
            }
        }
        if (legal_packet_count < 2) {
            return;
        }
        for (DataType dtype : DTYPES) {
            Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, reduction, inner}));
            input.fill(FP8_COMPLETE_LAYOUT_FILL_VALUE, stream);
            stream.synchronize();
            switch (dtype) {
                case DataType::FP8_E4M3:
                    runFp8CompleteLayoutShape<__nv_fp8_e4m3>(
                        benchmark_case, dtype, input, cache_flush, stream, device_geometry);
                    break;
                case DataType::FP8_E5M2:
                    runFp8CompleteLayoutShape<__nv_fp8_e5m2>(
                        benchmark_case, dtype, input, cache_flush, stream, device_geometry);
                    break;
                default:
                    throw std::logic_error("FP8 Complete layout sweep received unsupported dtype.");
            }
            rows += legal_packet_count * 4;
            ++shape_dtypes;
        }
    };

    for (uint64_t outer : SUPPLY_OUTERS) {
        for (uint64_t reduction : SUPPLY_REDUCTIONS) {
            for (uint64_t inner : SUPPLY_INNERS) {
                run_shape("supply", outer, reduction, inner);
            }
        }
    }
    for (uint64_t outer : AWKWARD_OUTERS) {
        for (uint64_t reduction : AWKWARD_REDUCTIONS) {
            for (uint64_t inner : AWKWARD_R_INNERS) {
                run_shape("awkward_r", outer, reduction, inner);
            }
        }
    }
    for (uint64_t outer : AWKWARD_OUTERS) {
        for (uint64_t reduction : AWKWARD_K_REDUCTIONS) {
            for (uint64_t inner : AWKWARD_K_P16) {
                run_shape("awkward_k", outer, reduction, inner);
            }
            for (uint64_t inner : AWKWARD_K_P8) {
                run_shape("awkward_k", outer, reduction, inner);
            }
            for (uint64_t inner : AWKWARD_K_P4) {
                run_shape("awkward_k", outer, reduction, inner);
            }
        }
    }
    for (uint64_t outer : AWKWARD_OUTERS) {
        for (const auto& [reduction, inner] : AWKWARD_BOTH) {
            run_shape("awkward_both", outer, reduction, inner);
        }
    }

    std::cout << "# rk_fp8_kparallel_complete_layout_sweep_complete shape_dtypes=" << shape_dtypes
              << " timed_rows=" << rows << " skipped_shapes_over_size=" << skipped_over_size << '\n';
#else
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 Complete layout sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}



constexpr uint32_t FP8_STAGED_LAYOUT_COMFORT_WARPS_PER_SM = 6;
constexpr uint64_t FP8_STAGED_LAYOUT_MIN_ROWS_PER_SHARD = 16;
constexpr double FP8_STAGED_LAYOUT_FILL_VALUE = 1.0 / 512.0;

struct Fp8StagedLayoutShape {
    const char* family;
    uint64_t outer;
    uint64_t reduction;
    uint64_t inner;
};

void addFp8StagedShardCandidate(std::vector<uint64_t>& shards, uint64_t reduction, uint64_t candidate) {
    if (candidate <= 1 || candidate > reduction) {
        return;
    }
    if (reduction / candidate < FP8_STAGED_LAYOUT_MIN_ROWS_PER_SHARD) {
        return;
    }
    if (std::find(shards.begin(), shards.end(), candidate) == shards.end()) {
        shards.push_back(candidate);
    }
}

void addFp8StagedShardCandidateNeighborhood(std::vector<uint64_t>& shards,
                                            uint64_t reduction,
                                            uint64_t candidate) {
    if (candidate > 2) {
        addFp8StagedShardCandidate(shards, reduction, candidate - 1);
    }
    addFp8StagedShardCandidate(shards, reduction, candidate);
    if (candidate < std::numeric_limits<uint64_t>::max()) {
        addFp8StagedShardCandidate(shards, reduction, candidate + 1);
    }
}

[[nodiscard]] uint64_t scaledCeil(uint64_t value, uint32_t numerator, uint32_t denominator) {
    if (denominator == 0) {
        throw std::logic_error("FP8 staged layout sweep received a zero scale denominator.");
    }
    const unsigned __int128 wide = static_cast<unsigned __int128>(value) * numerator;
    return static_cast<uint64_t>((wide + denominator - 1) / denominator);
}

template <typename InputT>
void runFp8StagedLayoutShape(const Fp8StagedLayoutShape& benchmark_case,
                             DataType dtype,
                             const Tensor& input,
                             Tensor& cache_flush,
                             Stream& stream,
                             const RKSinglePassDeviceGeometry& device_geometry) {
    static_assert(sizeof(InputT) == 1, "FP8 staged layout sweep requires one-byte input storage.");
    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, input.getDimensions(), {1});
    if (geometry.path != CubReductionPath::TiledFixedSegment) {
        throw std::logic_error("FP8 staged layout sweep case was not TiledFixedSegment.");
    }
    if (geometry.outer_size != benchmark_case.outer || geometry.reduction_size != benchmark_case.reduction
        || geometry.inner_size != benchmark_case.inner) {
        throw std::logic_error("FP8 staged layout sweep geometry disagrees with requested exact shape.");
    }

    const CubReductionStageSemantics semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::First, geometry.reduction_size);

    constexpr std::array<size_t, 4> PACKETS = {16, 8, 4, 2};
    constexpr std::array<uint32_t, 4> CTA_WIDTHS = {32, 64, 128, 256};
    constexpr std::array<uint64_t, 9> ROW_DEPTHS = {4096, 2048, 1024, 512, 256, 128, 64, 32, 16};
    constexpr std::array<std::pair<uint32_t, uint32_t>, 4> SUPPLY_SCALES = {{{1, 2}, {1, 1}, {2, 1}, {4, 1}}};

    for (size_t packet_bytes : PACKETS) {
        if (!supportsSinglePassKParallelPacket<InputT>(geometry, packet_bytes)) {
            continue;
        }
        const uint64_t items_per_lane = packet_bytes / sizeof(InputT);
        const uint64_t packet_owners = geometry.inner_size / items_per_lane;
        const uint64_t useful_warps_per_output_shard = singlePassCeilDiv(
            packet_owners, static_cast<uint64_t>(device_geometry.warp_size));

        for (uint32_t block_threads : CTA_WIDTHS) {
            const uint64_t component_tiles = singlePassKParallelComponentTiles<InputT>(
                geometry, packet_bytes, block_threads);
            const uint64_t base_blocks_per_shard = geometry.outer_size * component_tiles;
            const uint64_t base_useful_warps_per_shard = geometry.outer_size * useful_warps_per_output_shard;
            const KParallelTiledStagePlan occupancy_plan{
                TiledRKStageTopology::Staged, packet_bytes, block_threads, 2};
            const CubKernelOccupancyInfo occupancy = queryKParallelTiledStageOccupancy(dtype, occupancy_plan);
            if (occupancy.max_active_blocks_per_sm <= 0 || occupancy.max_active_warps_per_sm <= 0) {
                throw std::logic_error("FP8 staged layout sweep received invalid exact kernel occupancy.");
            }

            const uint64_t resident_device_blocks =
                static_cast<uint64_t>(device_geometry.sm_count)
                * static_cast<uint64_t>(occupancy.max_active_blocks_per_sm);
            const uint64_t resident_device_warps =
                static_cast<uint64_t>(device_geometry.sm_count)
                * static_cast<uint64_t>(occupancy.max_active_warps_per_sm);
            const uint64_t comfort_target =
                static_cast<uint64_t>(device_geometry.sm_count)
                * static_cast<uint64_t>(std::min<int>(FP8_STAGED_LAYOUT_COMFORT_WARPS_PER_SM,
                                                      occupancy.max_active_warps_per_sm));

            std::vector<uint64_t> shard_candidates;
            shard_candidates.reserve(40);

            // Retain the established rows/shard ladder as broad coverage over per-CTA serial R work.
            for (uint64_t requested_rows : ROW_DEPTHS) {
                if (geometry.reduction_size <= requested_rows) {
                    continue;
                }
                addFp8StagedShardCandidate(
                    shard_candidates, geometry.reduction_size, singlePassCeilDiv(geometry.reduction_size, requested_rows));
            }

            // Hardware-relative useful-warp targets. These are based on useful reduction work only; idle physical
            // warps inside a CTA receive no progress credit.
            for (const auto& [num, den] : SUPPLY_SCALES) {
                const uint64_t target_useful_warps = scaledCeil(comfort_target, num, den);
                if (base_useful_warps_per_shard != 0) {
                    const uint64_t target_shards = singlePassCeilDiv(target_useful_warps, base_useful_warps_per_shard);
                    addFp8StagedShardCandidateNeighborhood(shard_candidates, geometry.reduction_size, target_shards);
                }
            }

            // Physical CTA waves remain a separate constraint: idle warps may not do useful work, but their CTAs and
            // warp slots still consume real GPU residency. Probe around half/one/two/four exact resident CTA waves.
            for (const auto& [num, den] : SUPPLY_SCALES) {
                const uint64_t target_blocks = scaledCeil(resident_device_blocks, num, den);
                if (base_blocks_per_shard != 0) {
                    const uint64_t target_shards = singlePassCeilDiv(target_blocks, base_blocks_per_shard);
                    addFp8StagedShardCandidateNeighborhood(shard_candidates, geometry.reduction_size, target_shards);
                }
            }

            std::sort(shard_candidates.begin(), shard_candidates.end());
            if (shard_candidates.empty()) {
                throw std::logic_error("FP8 staged layout sweep produced no legal shard candidates.");
            }

            for (uint64_t shards : shard_candidates) {
                const uint64_t rows_per_shard = singlePassCeilDiv(geometry.reduction_size, shards);
                Tensor partials(input.getPlacement(),
                                TensorDescriptor(DataType::FP32,
                                                 {geometry.outer_size, shards, geometry.inner_size}));
                const KParallelTiledStagePlan plan{
                    TiledRKStageTopology::Staged, packet_bytes, block_threads, shards};
                const auto run = [&]() {
                    launchKParallelTiledStage(semantics, input, partials, geometry, plan, 1.0f, stream);
                };

                run();
                stream.synchronize();
                validateSinglePassPartial(partials,
                                          geometry,
                                          shards,
                                          rows_per_shard,
                                          SinglePassShardPolicy::Balanced,
                                          stream,
                                          "fp8_staged_layout",
                                          FP8_STAGED_LAYOUT_FILL_VALUE);

                const RKSinglePassTiming timing =
                    timeRKSinglePass(cache_flush, stream, run, FP8_STAGED_LAYOUT_TIMING_SAMPLES);
                const uint64_t blocks = base_blocks_per_shard * shards;
                const uint64_t warps_per_cta = block_threads / device_geometry.warp_size;
                const uint64_t physical_warps = blocks * warps_per_cta;
                const uint64_t useful_warps = base_useful_warps_per_shard * shards;
                const unsigned __int128 useful_capacity_numerator =
                    static_cast<unsigned __int128>(resident_device_blocks)
                    * static_cast<unsigned __int128>(useful_warps);
                const uint64_t useful_capacity_by_blocks = static_cast<uint64_t>(
                    useful_capacity_numerator / static_cast<unsigned __int128>(blocks));
                const uint64_t useful_resident_warps =
                    std::min(resident_device_warps, useful_capacity_by_blocks);
                const uint64_t launch_warp_supply = std::min(useful_warps, useful_resident_warps);
                const double comfort_fraction = comfort_target == 0
                                                    ? 0.0
                                                    : static_cast<double>(launch_warp_supply)
                                                          / static_cast<double>(comfort_target);
                const double physical_block_waves = resident_device_blocks == 0
                                                        ? 0.0
                                                        : static_cast<double>(blocks)
                                                              / static_cast<double>(resident_device_blocks);
                const uint64_t lane_slots = component_tiles * static_cast<uint64_t>(block_threads);
                const double lane_fill = lane_slots == 0
                                             ? 0.0
                                             : static_cast<double>(packet_owners) / static_cast<double>(lane_slots);
                const uint64_t input_bytes = input.getArraySizeInBytes();
                const uint64_t output_bytes = partials.getArraySizeInBytes();
                const double input_gbps = static_cast<double>(input_bytes) / (timing.median_ms * 1.0e6);
                const double logical_gbps =
                    static_cast<double>(input_bytes + output_bytes) / (timing.median_ms * 1.0e6);

                std::cout << benchmark_case.family << ",o" << benchmark_case.outer << "_r"
                          << benchmark_case.reduction << "_k" << benchmark_case.inner << ','
                          << singlePassDtypeName(dtype) << ",staged," << benchmark_case.outer << ','
                          << benchmark_case.reduction << ',' << benchmark_case.inner << ',' << packet_bytes << ','
                          << items_per_lane << ',' << block_threads << ',' << shards << ',' << rows_per_shard << ','
                          << packet_owners << ',' << component_tiles << ',' << blocks << ',' << warps_per_cta << ','
                          << physical_warps << ',' << useful_warps << ',' << occupancy.registers_per_thread << ','
                          << occupancy.static_shared_bytes << ',' << occupancy.dynamic_shared_bytes << ','
                          << occupancy.max_active_blocks_per_sm << ',' << occupancy.max_active_warps_per_sm << ','
                          << resident_device_blocks << ',' << resident_device_warps << ',' << useful_resident_warps << ','
                          << launch_warp_supply << ',' << comfort_target << ',' << std::fixed << std::setprecision(6)
                          << comfort_fraction << ',' << physical_block_waves << ',' << lane_fill << ',' << input_bytes << ','
                          << output_bytes << ',' << std::setprecision(4) << timing.median_ms << ',' << timing.best_ms << ','
                          << timing.worst_ms << ',' << std::setprecision(2) << input_gbps << ',' << logical_gbps << '\n';
            }
        }
    }
}

void runFp8KParallelStagedLayoutSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    constexpr std::array<Fp8StagedLayoutShape, 15> SHAPES = {{{"clean_512mib", 32, 32768, 512},
                                                               {"clean_512mib", 16, 32768, 1024},
                                                               {"clean_512mib", 8, 32768, 2048},
                                                               {"clean_512mib", 4, 32768, 4096},
                                                               {"clean_512mib", 2, 32768, 8192},
                                                               {"clean_512mib", 1, 32768, 16384},
                                                               {"depth_512mib", 16, 16384, 2048},
                                                               {"depth_512mib", 4, 65536, 2048},
                                                               {"depth_512mib", 2, 131072, 2048},
                                                               {"awkward_512mib", 8, 32771, 2032},
                                                               {"awkward_512mib", 2, 32771, 8176},
                                                               {"awkward_512mib", 4, 65521, 2032},
                                                               {"awkward_512mib", 1, 65521, 8176},
                                                               {"clean_1gib", 16, 32768, 2048},
                                                               {"clean_1gib", 4, 32768, 8192}}};

    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    std::cout << "# mode=rk_fp8_kparallel_staged_layout_sweep comparison=exact_shape_packet_cta_shard_optimization\n";
    std::cout << "# purpose=find the maximum large/saturating FP8 KParallel staged throughput before calibrating packet selection policy\n";
    std::cout << "# invariant=within one exact shape all packet/CTA/shard layouts read the same dtype/O/R/K input; only physical first-stage layout changes\n";
    std::cout << "# shard candidates=existing rows/shard ladder plus neighborhoods around useful-warp comfort and exact resident-CTA wave targets\n";
    std::cout << "# occupancy model=useful warps receive progress credit; idle physical warps still consume exact compiled-kernel block/warp residency\n";
    std::cout << "# throughput target=large enough cases should approach logical 1600 GB/s before FP8 packet-selection policy is considered calibrated\n";
    std::cout << "# sm_count=" << device_geometry.sm_count << " warp_size=" << device_geometry.warp_size
              << " max_threads_per_sm=" << device_geometry.max_threads_per_sm
              << " warmups=" << RK_SINGLE_PASS_WARMUPS
              << " timing_samples=" << FP8_STAGED_LAYOUT_TIMING_SAMPLES << '\n';
    std::cout << "family,case,dtype,topology,outer,R,K,packet_bytes,items_per_lane,block_threads,shards_per_output,"
                 "rows_per_shard,packet_owners,component_tiles,blocks,warps_per_cta,physical_warps,useful_warps,"
                 "registers_per_thread,static_shared_bytes,dynamic_shared_bytes,max_active_blocks_per_sm,"
                 "max_active_warps_per_sm,resident_device_blocks,resident_device_warps,useful_resident_warps,"
                 "launch_warp_supply,comfort_target,comfort_fraction,physical_block_waves,lane_fill,input_bytes,"
                 "output_bytes,median_ms,best_ms,worst_ms,input_GBps,logical_GBps\n";

    uint64_t shape_dtypes = 0;
    for (const Fp8StagedLayoutShape& benchmark_case : SHAPES) {
        const uint64_t elements = checkedSinglePassElements(
            benchmark_case.outer, benchmark_case.reduction, benchmark_case.inner);
        if (elements < 480ULL * 1024ULL * 1024ULL) {
            throw std::logic_error("FP8 staged layout sweep unexpectedly generated a sub-saturation input shape.");
        }
        for (DataType dtype : DTYPES) {
            Tensor input(cache_flush.getPlacement(),
                         TensorDescriptor(dtype,
                                          {benchmark_case.outer, benchmark_case.reduction, benchmark_case.inner}));
            input.fill(FP8_STAGED_LAYOUT_FILL_VALUE, stream);
            stream.synchronize();
            switch (dtype) {
                case DataType::FP8_E4M3:
                    runFp8StagedLayoutShape<__nv_fp8_e4m3>(
                        benchmark_case, dtype, input, cache_flush, stream, device_geometry);
                    break;
                case DataType::FP8_E5M2:
                    runFp8StagedLayoutShape<__nv_fp8_e5m2>(
                        benchmark_case, dtype, input, cache_flush, stream, device_geometry);
                    break;
                default:
                    throw std::logic_error("FP8 staged layout sweep received unsupported dtype.");
            }
            ++shape_dtypes;
        }
    }
    std::cout << "# rk_fp8_kparallel_staged_layout_sweep_complete shape_dtypes=" << shape_dtypes << '\n';
#else
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 staged layout sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


constexpr double FP8_RCOOP_SINGLE_PASS_FILL_VALUE = 1.0 / 512.0;
constexpr uint64_t FP8_RCOOP_SINGLE_PASS_TARGET_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr int FP8_RCOOP_SINGLE_PASS_TIMING_SAMPLES = 3;
constexpr double FP8_RCOOP_LINE_RATE_FLOOR_GBPS = 1500.0;
constexpr double FP8_RCOOP_PEAK_REFERENCE_GBPS = 1600.0;
constexpr uint64_t FP8_RCOOP_MAX_GRID_BLOCKS = 65535;

struct Fp8RCooperativeSinglePassShape {
    uint64_t inner;
};

[[nodiscard]] uint64_t fp8RCooperativeSinglePassOuter(uint64_t reduction, uint64_t inner) {
    if (reduction == 0 || inner < 2) {
        throw std::logic_error("FP8 RCooperative single-pass sweep requires R>0 and K>=2 RK geometry.");
    }
    const unsigned __int128 elements_per_outer =
        static_cast<unsigned __int128>(reduction) * static_cast<unsigned __int128>(inner);
    const unsigned __int128 target_elements = FP8_RCOOP_SINGLE_PASS_TARGET_BYTES;  // FP8 is one byte/element.
    const uint64_t outer = static_cast<uint64_t>(
        (target_elements + elements_per_outer - 1) / elements_per_outer);
    return std::max<uint64_t>(1, outer);
}

void printFp8RCooperativeSinglePassHeader(const RKSinglePassDeviceGeometry& device_geometry) {
    std::cout << "# mode=rk_fp8_rcooperative_single_pass_sweep operation=sum comparison=one_physical_pass\n";
    std::cout << "# objective=for each physical RCooperative pattern consume as much R as possible while sustaining the "
                 "calibrated line-rate floor; deeper rows/shard wins whenever it remains above that floor\n";
    std::cout << "# production policy is unchanged. FP8 FlatRows calibration remains p16 but now adaptively fills toward a "
                 "512-byte warp window (497 logical bytes after worst-case alignment reserve), preserves eight-warp shard "
                 "participation, and supports K<=256 with multiple components/lane. Rotated compares PrefixReservedP16 "
                 "and ExactP16. Rows above the production 128..1024 ladder are benchmark-only\n";
    std::cout << "# every timed row is exactly one CUDA reduction pass; there is no generic fallback, no production selector, "
                 "and no continuation timing\n";
    std::cout << "# streaming target input bytes >=" << FP8_RCOOP_SINGLE_PASS_TARGET_BYTES
              << " line_rate_floor_input_GBps=" << std::fixed << std::setprecision(1)
              << FP8_RCOOP_LINE_RATE_FLOOR_GBPS << " peak_reference_physical_GBps="
              << FP8_RCOOP_PEAK_REFERENCE_GBPS << " warmups=" << RK_SINGLE_PASS_WARMUPS
              << " timing_samples=" << FP8_RCOOP_SINGLE_PASS_TIMING_SAMPLES << '\n';
    std::cout << "# sm_count=" << device_geometry.sm_count << " warp_size=" << device_geometry.warp_size
              << " max_threads_per_sm=" << device_geometry.max_threads_per_sm << '\n';
    std::cout << "family,case,outer,R,K,dtype,operation,strategy,access,progress,implementation,packet_bytes,block_threads,"
                 "rows_per_shard,shards_per_output,first_stage_blocks,device_sms,device_warp_size,"
                 "device_max_threads_per_sm,device_max_threads_per_block,device_max_blocks_per_sm,"
                 "device_shared_mem_per_sm,device_registers_per_sm,ctas_per_sm,thread_capacity_blocks_per_sm,"
                 "thread_capacity_waves,thread_capacity_occupancy,registers_per_thread,static_shared_bytes,"
                 "dynamic_shared_bytes,occupancy_limit_ctas_per_sm,occupancy_limit_warps_per_sm,launch_warps_per_sm,"
                 "occupancy_waves,input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";
}

void runFp8RCooperativeSinglePassShape(const Fp8RCooperativeSinglePassShape& shape,
                                       DataType dtype,
                                       Tensor& cache_flush,
                                       Stream& stream,
                                       const RKSinglePassDeviceGeometry& device_geometry) {
    constexpr uint64_t REDUCTION = 32768;
    constexpr std::array<uint64_t, 8> ROW_DEPTHS = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    const uint64_t outer = fp8RCooperativeSinglePassOuter(REDUCTION, shape.inner);
    RKSinglePassSweepCase benchmark_case{
        "streaming_512mib",
        "o" + std::to_string(outer) + "_r" + std::to_string(REDUCTION) + "_k" + std::to_string(shape.inner),
        outer,
        REDUCTION,
        shape.inner};

    Tensor input(cache_flush.getPlacement(), TensorDescriptor(dtype, {outer, REDUCTION, shape.inner}));
    if (input.getArraySizeInBytes() < FP8_RCOOP_SINGLE_PASS_TARGET_BYTES) {
        throw std::logic_error("FP8 RCooperative single-pass sweep generated a sub-target streaming input.");
    }
    input.fill(FP8_RCOOP_SINGLE_PASS_FILL_VALUE, stream);
    stream.synchronize();
    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, input.getDimensions(), {1});
    if (geometry.path != CubReductionPath::TiledFixedSegment) {
        throw std::logic_error("FP8 RCooperative single-pass sweep expected ordinary dense RK geometry.");
    }

    // Keep Complete as a deep-R reference endpoint, but do not use it to tune the staged packet/access patterns.
    const CubReductionStageSemantics complete_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::Complete, geometry.reduction_size);
    const size_t complete_packet_bytes = singlePassPacketAdaptivePacketBytes(geometry, dtype);
    const uint32_t complete_warps = singlePassPacketAdaptiveWarps(geometry, dtype, complete_packet_bytes);
    const uint32_t complete_block_warps = std::max<uint32_t>(
        static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK), complete_warps);
    DenseRKFamilyPhysicalPlan complete_plan;
    complete_plan.packet_bytes = complete_packet_bytes;
    complete_plan.cooperative_warps_per_tile = complete_warps;
    complete_plan.block_threads = complete_block_warps * static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS);
    Tensor complete_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
    const auto run_complete = [&]() {
        launchPacketAdaptiveComplete(complete_semantics, complete_plan, input, complete_output, geometry, stream);
    };
    run_complete();
    stream.synchronize();
    validateSinglePassComplete(complete_output,
                               geometry,
                               stream,
                               "fp8_rcoop_single_pass_complete",
                               FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
    const RKSinglePassTiming complete_timing = timeRKSinglePass(
        cache_flush, stream, run_complete, FP8_RCOOP_SINGLE_PASS_TIMING_SAMPLES);
    const CubKernelOccupancyInfo complete_occupancy = querySinglePassPacketAdaptiveOccupancy(
        geometry, dtype, complete_packet_bytes, static_cast<int>(complete_warps));
    const uint64_t complete_tiles = singlePassPacketAdaptiveTiles(geometry, dtype, complete_packet_bytes);
    const uint64_t complete_groups_per_block = complete_block_warps / complete_warps;
    const uint64_t complete_blocks = std::min<uint64_t>(
        singlePassCeilDiv(geometry.outer_size * complete_tiles, complete_groups_per_block),
        FP8_RCOOP_MAX_GRID_BLOCKS);
    printSinglePassRow(benchmark_case,
                       dtype,
                       "r_cooperative",
                       "aligned_or_alignment_safe",
                       "complete",
                       "packet_adaptive_complete",
                       complete_packet_bytes,
                       complete_plan.block_threads,
                       geometry.reduction_size,
                       1,
                       complete_blocks,
                       input,
                       complete_output,
                       complete_timing,
                       &device_geometry,
                       &complete_occupancy);

    for (uint64_t rows_per_shard : ROW_DEPTHS) {
        const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, rows_per_shard);
        if (shards <= 1) {
            continue;
        }

        if (geometry.inner_size <= 256) {
            Tensor flat_partials(input.getPlacement(),
                                 TensorDescriptor(DataType::FP32,
                                                  {geometry.outer_size, shards, geometry.inner_size}));
            const auto run_flat = [&]() {
                launchFp8NarrowLowPrecisionFlatRCooperativeCalibrationFirstStage(
                    input, flat_partials, geometry, 16, rows_per_shard, shards, stream);
            };
            run_flat();
            stream.synchronize();
            validateSinglePassPartial(flat_partials,
                                      geometry,
                                      shards,
                                      rows_per_shard,
                                      SinglePassShardPolicy::FixedRows,
                                      stream,
                                      "fp8_rcoop_single_pass_flat_rows_packed",
                                      FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming flat_timing = timeRKSinglePass(
                cache_flush, stream, run_flat, FP8_RCOOP_SINGLE_PASS_TIMING_SAMPLES);
            const CubKernelOccupancyInfo flat_occupancy =
                queryFp8NarrowLowPrecisionFlatRCooperativeCalibrationOccupancy(
                    dtype, 16, rows_per_shard, geometry.inner_size);
            const uint64_t flat_blocks = std::min<uint64_t>(
                geometry.outer_size * shards, FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "flat_rows",
                               "staged",
                               "fp8_narrow_flat_p16_adaptive_window_r_cooperative_calibration",
                               16,
                               256,
                               rows_per_shard,
                               shards,
                               flat_blocks,
                               input,
                               flat_partials,
                               flat_timing,
                               &device_geometry,
                               &flat_occupancy);
        }

        struct RotatedP16Layout {
            const char* access;
            const char* implementation;
            bool exact_tile;
        };
        constexpr std::array<RotatedP16Layout, 2> ROTATED_P16_LAYOUTS = {{
            {"rotated_prefix_reserved", "fp8_rotated_prefix_reserved_p16_r_cooperative_calibration", false},
            {"rotated_exact", "fp8_rotated_exact_p16_r_cooperative_calibration", true},
        }};
        for (const RotatedP16Layout& layout : ROTATED_P16_LAYOUTS) {
            Tensor rotated_partials(input.getPlacement(),
                                    TensorDescriptor(DataType::FP32,
                                                     {geometry.outer_size, shards, geometry.inner_size}));
            const auto run_rotated = [&]() {
                launchFp8AwkwardAlignmentRotatedShardedCalibrationFirstStage(
                    input, rotated_partials, geometry, 16, layout.exact_tile, rows_per_shard, shards, stream);
            };
            run_rotated();
            stream.synchronize();
            validateSinglePassPartial(rotated_partials,
                                      geometry,
                                      shards,
                                      rows_per_shard,
                                      SinglePassShardPolicy::FixedRows,
                                      stream,
                                      std::string("fp8_rcoop_single_pass_") + layout.access,
                                      FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming rotated_timing = timeRKSinglePass(
                cache_flush, stream, run_rotated, FP8_RCOOP_SINGLE_PASS_TIMING_SAMPLES);
            const CubKernelOccupancyInfo rotated_occupancy =
                queryFp8AwkwardAlignmentRotatedShardedCalibrationOccupancy(
                    dtype, 16, layout.exact_tile, rows_per_shard);
            const uint64_t rotated_tiles = singlePassRotatedRCoopTiles(geometry, dtype, 16, layout.exact_tile);
            const uint64_t rotated_blocks = std::min<uint64_t>(
                geometry.outer_size * rotated_tiles * shards, FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               layout.access,
                               "staged",
                               layout.implementation,
                               16,
                               512,
                               rows_per_shard,
                               shards,
                               rotated_blocks,
                               input,
                               rotated_partials,
                               rotated_timing,
                               &device_geometry,
                               &rotated_occupancy);
        }
    }
}

void runFp8RCooperativeSinglePassSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    // Preserve the Rotated boundary probes, but densify the FlatRows handoff region. Small odd K establishes whether
    // adaptive row-group depth alone fixes underfilled p16 windows; the 33..249 ladder shows how far multi-component
    // lane ownership remains useful before the two Rotated p16 layouts naturally take over.
    constexpr std::array<uint64_t, 80> K_VALUES = {
        2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 23, 24, 25, 30, 31, 32, 33, 35, 39, 47, 55,
        63, 64, 65, 79, 95, 111, 124, 125, 126, 127, 128, 129, 143, 159, 175, 191, 207, 223, 239, 247,
        248, 249, 250, 255, 256, 257,
        495, 496, 497, 498, 499, 511, 512, 513,
        746, 747, 748,
        993, 994, 995, 996, 997, 1023, 1024, 1025,
        1490, 1491, 1492, 1493, 1494, 1495,
        2047, 2048, 2049,
        4095, 4096, 4097};

    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    printFp8RCooperativeSinglePassHeader(device_geometry);
    uint64_t shape_dtypes = 0;
    for (uint64_t inner : K_VALUES) {
        const Fp8RCooperativeSinglePassShape shape{inner};
        for (DataType dtype : DTYPES) {
            runFp8RCooperativeSinglePassShape(shape, dtype, cache_flush, stream, device_geometry);
            ++shape_dtypes;
        }
    }
    std::cout << "# rk_fp8_rcooperative_single_pass_sweep_complete shape_dtypes=" << shape_dtypes << '\n';
#else
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 RCooperative single-pass sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


constexpr int FP8_RCOOP_FULL_SPACE_TIMING_SAMPLES = 3;
constexpr std::array<uint64_t, 12> FP8_RCOOP_FULL_SPACE_ROW_DEPTHS = {
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

[[nodiscard]] uint64_t fp8RCooperativeOuterNearTarget(uint64_t reduction,
                                                       uint64_t inner,
                                                       uint64_t target_bytes) {
    const unsigned __int128 elements_per_outer =
        static_cast<unsigned __int128>(reduction) * static_cast<unsigned __int128>(inner);
    if (elements_per_outer == 0) {
        throw std::logic_error("FP8 RCooperative full-space sweep requires non-zero R and K.");
    }
    const unsigned __int128 rounded =
        (static_cast<unsigned __int128>(target_bytes) + elements_per_outer / 2) / elements_per_outer;
    return std::max<uint64_t>(1, static_cast<uint64_t>(rounded));
}

[[nodiscard]] const char* fp8RCooperativeTargetLabel(uint64_t bytes) {
    switch (bytes) {
        case 256ULL * 1024ULL: return "target_256kib";
        case 1ULL * 1024ULL * 1024ULL: return "target_1mib";
        case 4ULL * 1024ULL * 1024ULL: return "target_4mib";
        case 16ULL * 1024ULL * 1024ULL: return "target_16mib";
        case 64ULL * 1024ULL * 1024ULL: return "target_64mib";
        case 128ULL * 1024ULL * 1024ULL: return "target_128mib";
        case 256ULL * 1024ULL * 1024ULL: return "target_256mib";
        case 512ULL * 1024ULL * 1024ULL: return "target_512mib";
        default: return "target_custom";
    }
}

void printFp8RCooperativeCoverageHeader(const RKSinglePassDeviceGeometry& device_geometry,
                                            std::string_view mode,
                                            std::string_view objective) {
    std::cout << "# mode=" << mode << " operation=sum comparison=one_physical_pass\n";
    std::cout << "# objective=" << objective << '\n';
    std::cout << "# candidates=exact production KParallel selection; packet-adaptive Complete; adaptive multi-output-warp p4 complete; DEPRECATED one-output-per-warp outer-grouped p4 baseline; "
                 "FlatRows p4/p8/p16 with eight-warp adaptive windows; Rotated PrefixReserved "
                 "p4/4warp,p8/8warp,p16/16warp plus ExactP16. All RCooperative p4/p8 paths are calibration-only and "
                 "production policy is unchanged\n";
    std::cout << "# target problem sizes are approximate because O is integral; duplicate O values for the same R/K target "
                 "ladder are suppressed. Staged candidates require >1 shard, so rows_per_shard<R\n";
    std::cout << "# line_rate_floor_physical_GBps=" << std::fixed << std::setprecision(1)
              << FP8_RCOOP_LINE_RATE_FLOOR_GBPS << " peak_reference_input_GBps="
              << FP8_RCOOP_PEAK_REFERENCE_GBPS << " warmups=" << RK_SINGLE_PASS_WARMUPS
              << " timing_samples=" << FP8_RCOOP_FULL_SPACE_TIMING_SAMPLES << '\n';
    std::cout << "# sm_count=" << device_geometry.sm_count << " warp_size=" << device_geometry.warp_size
              << " max_threads_per_sm=" << device_geometry.max_threads_per_sm << '\n';
    std::cout << "family,case,outer,R,K,dtype,operation,strategy,access,progress,implementation,packet_bytes,block_threads,"
                 "rows_per_shard,shards_per_output,first_stage_blocks,device_sms,device_warp_size,"
                 "device_max_threads_per_sm,device_max_threads_per_block,device_max_blocks_per_sm,"
                 "device_shared_mem_per_sm,device_registers_per_sm,ctas_per_sm,thread_capacity_blocks_per_sm,"
                 "thread_capacity_waves,thread_capacity_occupancy,registers_per_thread,static_shared_bytes,"
                 "dynamic_shared_bytes,occupancy_limit_ctas_per_sm,occupancy_limit_warps_per_sm,launch_warps_per_sm,"
                 "occupancy_waves,input_bytes,output_bytes,median_ms,best_ms,worst_ms,logical_GBps\n";
}

void runFp8RCooperativeFullSpaceCase(const RKSinglePassSweepCase& benchmark_case,
                                     DataType dtype,
                                     Tensor& cache_flush,
                                     Stream& stream,
                                     const RKSinglePassDeviceGeometry& device_geometry,
                                     bool use_rotating_cold_working_set = false,
                                     bool focused_holes_only = false,
                                     bool include_general_rparallel_small_k = false,
                                     bool include_general_rparallel_staged = false) {
    Tensor input(cache_flush.getPlacement(),
                 TensorDescriptor(dtype, {benchmark_case.outer, benchmark_case.reduction, benchmark_case.inner}));
    input.fill(FP8_RCOOP_SINGLE_PASS_FILL_VALUE, stream);
    stream.synchronize();
    const CubReductionGeometry geometry = CubReduction::analyzeValueGeometry(
        CubReductionOp::Sum, input.getDimensions(), {1});
    if (geometry.path != CubReductionPath::TiledFixedSegment) {
        throw std::logic_error("FP8 RCooperative full-space sweep expected ordinary dense RK geometry.");
    }

    uint32_t rotating_slot_count = 0;
    uint64_t rotating_cursor = 0;
    bool rotating_needs_bootstrap_flush = false;
    std::unique_ptr<RKRotatingTensorSlots> rotating_inputs;
    std::unique_ptr<RKRotatingTensorSlots> rotating_complete_outputs;
    if (use_rotating_cold_working_set) {
        rotating_slot_count = rkRotatingColdSlotCount(
            cache_flush.getArraySizeInBytes(), input.getArraySizeInBytes());
        rotating_inputs = std::make_unique<RKRotatingTensorSlots>(
            input.getPlacement(), input.getDescriptor(), rotating_slot_count);
        rotating_inputs->storage().fill(FP8_RCOOP_SINGLE_PASS_FILL_VALUE, stream);
        rotating_complete_outputs = std::make_unique<RKRotatingTensorSlots>(
            input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions), rotating_slot_count);
        stream.synchronize();
        rotating_needs_bootstrap_flush = true;
    }

    const auto time_rotating_candidate = [&](Tensor& output,
                                             RKRotatingTensorSlots& output_slots,
                                             const auto& run) {
        return timeRKSinglePassRotatingWorkingSet(input,
                                                   output,
                                                   *rotating_inputs,
                                                   output_slots,
                                                   rotating_cursor,
                                                   cache_flush,
                                                   rotating_needs_bootstrap_flush,
                                                   stream,
                                                   run,
                                                   FP8_RCOOP_FULL_SPACE_TIMING_SAMPLES);
    };
    const auto time_complete_candidate = [&](Tensor& output, const auto& run) {
        if (!use_rotating_cold_working_set) {
            return timeRKSinglePass(cache_flush, stream, run, FP8_RCOOP_FULL_SPACE_TIMING_SAMPLES);
        }
        return time_rotating_candidate(output, *rotating_complete_outputs, run);
    };

    // The focused hole sweep deliberately stops re-running kernels/geometries whose ownership has already been
    // established. Keep only the physical roofline, the relevant shallow Compact baseline, current RParallel, the
    // K=3 and K=7 phase-stream experiments, and DirectComponent where K=15/31 still has measured wins. Production KParallel,
    // packet-adaptive p16, the deprecated baseline, FlatRows, and Rotated remain covered by the broader sweeps.
    if (focused_holes_only) {
        Tensor roofline_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_roofline = [&]() {
            launchFp8P4ReadWriteRooflineCalibrationReference(input, roofline_output, geometry, stream);
        };
        run_roofline();
        stream.synchronize();
        const RKSinglePassTiming roofline_timing = time_complete_candidate(roofline_output, run_roofline);
        const CubKernelOccupancyInfo roofline_occupancy = queryFp8P4ReadWriteRooflineCalibrationOccupancy();
        const uint64_t output_elements = geometry.outer_size * geometry.inner_size;
        const uint64_t output_packets = output_elements / 4;
        const uint64_t roofline_warp_groups = std::max<uint64_t>(uint64_t{1}, singlePassCeilDiv(output_packets, uint64_t{32}));
        const uint64_t roofline_blocks = std::min<uint64_t>(
            singlePassCeilDiv(roofline_warp_groups, uint64_t{8}), FP8_RCOOP_MAX_GRID_BLOCKS);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "reference",
                           "p4_read_write_roofline",
                           "complete",
                           "fp8_p4_read_write_same_ratio_roofline_reference",
                           4,
                           256,
                           geometry.reduction_size,
                           1,
                           roofline_blocks,
                           input,
                           roofline_output,
                           roofline_timing,
                           &device_geometry,
                           &roofline_occupancy);

        for (const uint32_t compact_packet_bytes : {4u, 8u, 16u}) {
            const uint32_t compact_outputs_per_warp =
                fp8CompactMultiOutputWarpRCooperativeCalibrationOutputsPerWarp(
                    geometry.reduction_size, geometry.inner_size, compact_packet_bytes);
            if (compact_outputs_per_warp <= 1) {
                continue;
            }
            Tensor compact_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_compact = [&]() {
                launchFp8CompactMultiOutputWarpRCooperativeCalibrationComplete(
                    input, compact_output, geometry, compact_packet_bytes, stream);
            };
            run_compact();
            stream.synchronize();
            const std::string compact_label =
                "fp8_rcoop_compact_multi_output_warp_p" + std::to_string(compact_packet_bytes);
            validateSinglePassComplete(compact_output,
                                       geometry,
                                       stream,
                                       compact_label,
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming compact_timing = time_complete_candidate(compact_output, run_compact);
            const CubKernelOccupancyInfo compact_occupancy =
                queryFp8CompactMultiOutputWarpRCooperativeCalibrationOccupancy(
                    dtype, geometry.inner_size, compact_outputs_per_warp, compact_packet_bytes);
            const uint64_t compact_warp_groups = singlePassCeilDiv(
                geometry.outer_size, static_cast<uint64_t>(compact_outputs_per_warp));
            const uint64_t compact_blocks = std::min<uint64_t>(
                singlePassCeilDiv(compact_warp_groups, uint64_t{8}), FP8_RCOOP_MAX_GRID_BLOCKS);
            const std::string compact_implementation =
                "fp8_compact_multi_output_warp_p" + std::to_string(compact_packet_bytes) + "_calibration";
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "multi_output_compact",
                               "complete",
                               compact_implementation,
                               compact_packet_bytes,
                               256,
                               geometry.reduction_size,
                               1,
                               compact_blocks,
                               input,
                               compact_output,
                               compact_timing,
                               &device_geometry,
                               &compact_occupancy);
        }

        Tensor r_parallel_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_r_parallel = [&]() {
            launchFp8RParallelWarpP4RCooperativeCalibrationComplete(
                input, r_parallel_output, geometry, stream);
        };
        run_r_parallel();
        stream.synchronize();
        validateSinglePassComplete(r_parallel_output,
                                   geometry,
                                   stream,
                                   "fp8_rcoop_r_parallel_warp_p4",
                                   FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
        const RKSinglePassTiming r_parallel_timing = time_complete_candidate(r_parallel_output, run_r_parallel);
        const CubKernelOccupancyInfo r_parallel_occupancy =
            queryFp8RParallelWarpP4RCooperativeCalibrationOccupancy(dtype, geometry.inner_size);
        const uint64_t one_output_blocks = std::min<uint64_t>(
            singlePassCeilDiv(geometry.outer_size, uint64_t{8}), FP8_RCOOP_MAX_GRID_BLOCKS);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "r_cooperative",
                           "r_parallel_warp",
                           "complete",
                           "fp8_one_output_warp_r_parallel_p4_calibration",
                           4,
                           256,
                           geometry.reduction_size,
                           1,
                           one_output_blocks,
                           input,
                           r_parallel_output,
                           r_parallel_timing,
                           &device_geometry,
                           &r_parallel_occupancy);

        // Complete the general RParallel packet-width ladder at O1. p4 above remains the established control;
        // p8/p16 reuse the same packed-shared implementation as MultiOutput with OutputsPerWarp=1.
        constexpr std::array<uint32_t, 2> ONE_OUTPUT_RPARALLEL_WIDE_PACKET_BYTES = {8, 16};
        for (uint32_t packet_bytes : ONE_OUTPUT_RPARALLEL_WIDE_PACKET_BYTES) {
            Tensor wide_r_parallel_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_wide_r_parallel = [&]() {
                launchFp8MultiOutputRParallelWarpRCooperativeCalibrationComplete(
                    input, wide_r_parallel_output, geometry, 1, packet_bytes, stream);
            };
            run_wide_r_parallel();
            stream.synchronize();
            const std::string validation_name =
                "fp8_rcoop_r_parallel_warp_p" + std::to_string(packet_bytes);
            validateSinglePassComplete(wide_r_parallel_output,
                                       geometry,
                                       stream,
                                       validation_name,
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming wide_r_parallel_timing =
                time_complete_candidate(wide_r_parallel_output, run_wide_r_parallel);
            const CubKernelOccupancyInfo wide_r_parallel_occupancy =
                queryFp8MultiOutputRParallelWarpRCooperativeCalibrationOccupancy(
                    dtype, geometry.inner_size, 1, packet_bytes);
            const std::string implementation_name =
                "fp8_one_output_warp_r_parallel_p" + std::to_string(packet_bytes) + "_calibration";
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "r_parallel_warp",
                               "complete",
                               implementation_name,
                               packet_bytes,
                               256,
                               geometry.reduction_size,
                               1,
                               one_output_blocks,
                               input,
                               wide_r_parallel_output,
                               wide_r_parallel_timing,
                               &device_geometry,
                               &wide_r_parallel_occupancy);
        }

        // Multi-warp Complete RParallel extends O1 in the other direction: 2/4/8 physical warps cooperate on one
        // output inside a 256-thread CTA, each consuming disjoint R groups through the same p4/p8/p16 packed-shared
        // component/R-worker path. The CTA folds K FP32 warp partials in shared memory and writes the final FP8 output,
        // so this measures how far intra-CTA cooperation can push the Complete boundary before staging is necessary.
        constexpr std::array<uint32_t, 3> MULTI_WARP_RPARALLEL_PACKET_BYTES = {4, 8, 16};
        constexpr std::array<uint32_t, 3> MULTI_WARP_RPARALLEL_WARPS = {2, 4, 8};
        for (uint32_t packet_bytes : MULTI_WARP_RPARALLEL_PACKET_BYTES) {
            for (uint32_t warps_per_output : MULTI_WARP_RPARALLEL_WARPS) {
                // More cooperating warps than R rows can only add idle partials and synchronization; omit those
                // dominated shapes from the focused calibration without changing kernel legality.
                if (warps_per_output > geometry.reduction_size) {
                    continue;
                }
                Tensor multi_warp_r_parallel_output(
                    input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
                const auto run_multi_warp_r_parallel = [&]() {
                    launchFp8MultiWarpRParallelRCooperativeCalibrationComplete(
                        input,
                        multi_warp_r_parallel_output,
                        geometry,
                        warps_per_output,
                        packet_bytes,
                        stream);
                };
                run_multi_warp_r_parallel();
                stream.synchronize();
                const std::string validation_name =
                    "fp8_rcoop_multi_warp_r_parallel_p" + std::to_string(packet_bytes) +
                    "_w" + std::to_string(warps_per_output);
                validateSinglePassComplete(multi_warp_r_parallel_output,
                                           geometry,
                                           stream,
                                           validation_name,
                                           FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
                const RKSinglePassTiming multi_warp_r_parallel_timing =
                    time_complete_candidate(multi_warp_r_parallel_output, run_multi_warp_r_parallel);
                const CubKernelOccupancyInfo multi_warp_r_parallel_occupancy =
                    queryFp8MultiWarpRParallelRCooperativeCalibrationOccupancy(
                        dtype, geometry.inner_size, warps_per_output, packet_bytes);
                const uint32_t outputs_per_block = 8 / warps_per_output;
                const uint64_t blocks = std::min<uint64_t>(
                    singlePassCeilDiv(geometry.outer_size, static_cast<uint64_t>(outputs_per_block)),
                    FP8_RCOOP_MAX_GRID_BLOCKS);
                const std::string access_name =
                    "r_parallel_multi_warp_" + std::to_string(warps_per_output);
                const std::string implementation_name =
                    "fp8_multi_warp_r_parallel_p" + std::to_string(packet_bytes) +
                    "_w" + std::to_string(warps_per_output) + "_calibration";
                printSinglePassRow(benchmark_case,
                                   dtype,
                                   "r_cooperative",
                                   access_name,
                                   "complete",
                                   implementation_name,
                                   packet_bytes,
                                   256,
                                   geometry.reduction_size,
                                   1,
                                   blocks,
                                   input,
                                   multi_warp_r_parallel_output,
                                   multi_warp_r_parallel_timing,
                                   &device_geometry,
                                   &multi_warp_r_parallel_occupancy);
            }
        }

        // Test MultiOutput RParallel as a general packet-width family. p4 is the established control. p8/p16 keep
        // the same component/R-worker ownership and runtime R, but widen each producer lane's global packet and stage
        // the raw FP8 payload through conflict-free 32-bit shared planes before packed decode in the consumers.
        constexpr std::array<uint32_t, 3> MULTI_OUTPUT_RPARALLEL_PACKET_BYTES = {4, 8, 16};
        constexpr std::array<uint32_t, 5> MULTI_OUTPUT_RPARALLEL_OUTPUTS = {2, 4, 8, 16, 32};
        for (uint32_t packet_bytes : MULTI_OUTPUT_RPARALLEL_PACKET_BYTES) {
            for (uint32_t outputs_per_warp : MULTI_OUTPUT_RPARALLEL_OUTPUTS) {
                const uint32_t lanes_per_output = 32 / outputs_per_warp;
                if (geometry.inner_size > lanes_per_output) {
                    continue;
                }
                Tensor multi_output_r_parallel_output(
                    input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
                const auto run_multi_output_r_parallel = [&]() {
                    launchFp8MultiOutputRParallelWarpRCooperativeCalibrationComplete(
                        input,
                        multi_output_r_parallel_output,
                        geometry,
                        outputs_per_warp,
                        packet_bytes,
                        stream);
                };
                run_multi_output_r_parallel();
                stream.synchronize();
                const std::string validation_name =
                    "fp8_rcoop_multi_output_r_parallel_warp_p" + std::to_string(packet_bytes) +
                    "_o" + std::to_string(outputs_per_warp);
                validateSinglePassComplete(multi_output_r_parallel_output,
                                           geometry,
                                           stream,
                                           validation_name,
                                           FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
                const RKSinglePassTiming multi_output_r_parallel_timing =
                    time_complete_candidate(multi_output_r_parallel_output, run_multi_output_r_parallel);
                const CubKernelOccupancyInfo multi_output_r_parallel_occupancy =
                    queryFp8MultiOutputRParallelWarpRCooperativeCalibrationOccupancy(
                        dtype, geometry.inner_size, outputs_per_warp, packet_bytes);
                const uint64_t warp_groups =
                    singlePassCeilDiv(geometry.outer_size, static_cast<uint64_t>(outputs_per_warp));
                const uint64_t blocks = std::min<uint64_t>(
                    singlePassCeilDiv(warp_groups, uint64_t{8}), FP8_RCOOP_MAX_GRID_BLOCKS);
                const std::string access_name =
                    "r_parallel_multi_output_" + std::to_string(outputs_per_warp);
                const std::string implementation_name =
                    "fp8_multi_output_r_parallel_p" + std::to_string(packet_bytes) +
                    "_o" + std::to_string(outputs_per_warp) + "_calibration";
                printSinglePassRow(benchmark_case,
                                   dtype,
                                   "r_cooperative",
                                   access_name,
                                   "complete",
                                   implementation_name,
                                   packet_bytes,
                                   256,
                                   geometry.reduction_size,
                                   1,
                                   blocks,
                                   input,
                                   multi_output_r_parallel_output,
                                   multi_output_r_parallel_timing,
                                   &device_geometry,
                                   &multi_output_r_parallel_occupancy);
            }
        }

        if (geometry.inner_size == 3 &&
            (geometry.reduction_size == 7 || geometry.reduction_size == 8 ||
             geometry.reduction_size == 15 || geometry.reduction_size == 16 ||
             geometry.reduction_size == 31 || geometry.reduction_size == 32 ||
             geometry.reduction_size == 63 || geometry.reduction_size == 64)) {
            Tensor phase_stream_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_phase_stream = [&]() {
                launchFp8K3PhaseStreamRParallelP4RCooperativeCalibrationComplete(
                    input, phase_stream_output, geometry, stream);
            };
            run_phase_stream();
            stream.synchronize();
            validateSinglePassComplete(phase_stream_output,
                                       geometry,
                                       stream,
                                       "fp8_rcoop_k3_phase_stream_r_parallel_p4",
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming phase_stream_timing = time_complete_candidate(phase_stream_output, run_phase_stream);
            const CubKernelOccupancyInfo phase_stream_occupancy =
                queryFp8K3PhaseStreamRParallelP4RCooperativeCalibrationOccupancy(dtype, geometry.reduction_size);
            const uint64_t lanes_per_output = geometry.reduction_size <= 8 ? 8 :
                                              (geometry.reduction_size <= 16 ? 16 : 32);
            const uint64_t outputs_per_warp = 32 / lanes_per_output;
            const uint64_t outputs_per_block = outputs_per_warp * 8;
            const uint64_t phase_stream_blocks = std::min<uint64_t>(
                singlePassCeilDiv(geometry.outer_size, outputs_per_block), FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "r_parallel_k3_phase_stream",
                               "complete",
                               "fp8_k3_phase_stream_r_parallel_p4_calibration",
                               4,
                               256,
                               geometry.reduction_size,
                               1,
                               phase_stream_blocks,
                               input,
                               phase_stream_output,
                               phase_stream_timing,
                               &device_geometry,
                               &phase_stream_occupancy);
        }


        if (geometry.inner_size == 3 &&
            (geometry.reduction_size == 8 || geometry.reduction_size == 16 ||
             geometry.reduction_size == 32 || geometry.reduction_size == 64)) {
            Tensor phase_stream_p8_input_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_phase_stream_p8_input = [&]() {
                launchFp8K3PhaseStreamRParallelP8InputRCooperativeCalibrationComplete(
                    input, phase_stream_p8_input_output, geometry, stream);
            };
            run_phase_stream_p8_input();
            stream.synchronize();
            validateSinglePassComplete(phase_stream_p8_input_output,
                                       geometry,
                                       stream,
                                       "fp8_rcoop_k3_phase_stream_r_parallel_p8_input",
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming phase_stream_p8_input_timing =
                time_complete_candidate(phase_stream_p8_input_output, run_phase_stream_p8_input);
            const CubKernelOccupancyInfo phase_stream_p8_input_occupancy =
                queryFp8K3PhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(dtype, geometry.reduction_size);
            const uint64_t lanes_per_output = geometry.reduction_size <= 8 ? 4 :
                                              (geometry.reduction_size <= 16 ? 8 :
                                               (geometry.reduction_size <= 32 ? 16 : 32));
            const uint64_t outputs_per_warp = 32 / lanes_per_output;
            const uint64_t outputs_per_block = outputs_per_warp * 8;
            const uint64_t phase_stream_p8_input_blocks = std::min<uint64_t>(
                singlePassCeilDiv(geometry.outer_size, outputs_per_block), FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "r_parallel_k3_phase_stream_p8_input",
                               "complete",
                               "fp8_k3_phase_stream_r_parallel_p8_input_calibration",
                               8,
                               256,
                               geometry.reduction_size,
                               1,
                               phase_stream_p8_input_blocks,
                               input,
                               phase_stream_p8_input_output,
                               phase_stream_p8_input_timing,
                               &device_geometry,
                               &phase_stream_p8_input_occupancy);
        }


        if (geometry.inner_size == 3 &&
            (geometry.reduction_size == 7 || geometry.reduction_size == 8 ||
             geometry.reduction_size == 15 || geometry.reduction_size == 16 ||
             geometry.reduction_size == 31 || geometry.reduction_size == 32 ||
             geometry.reduction_size == 63 || geometry.reduction_size == 64)) {
            const auto runtime_r_lanes_per_output = [&](uint64_t packet_bytes) -> uint64_t {
                const uint64_t bytes_per_output = geometry.reduction_size * 3;
                const uint64_t max_packets = (bytes_per_output + 2 * packet_bytes - 2) / packet_bytes;
                if (max_packets <= 4) return 4;
                if (max_packets <= 8) return 8;
                if (max_packets <= 16) return 16;
                return 32;
            };

            Tensor runtime_r_p4_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_runtime_r_p4 = [&]() {
                launchFp8K3RuntimeRPhaseStreamRParallelP4RCooperativeCalibrationComplete(
                    input, runtime_r_p4_output, geometry, stream);
            };
            run_runtime_r_p4();
            stream.synchronize();
            validateSinglePassComplete(runtime_r_p4_output,
                                       geometry,
                                       stream,
                                       "fp8_rcoop_k3_runtime_r_phase_stream_p4",
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming runtime_r_p4_timing = time_complete_candidate(runtime_r_p4_output, run_runtime_r_p4);
            const CubKernelOccupancyInfo runtime_r_p4_occupancy =
                queryFp8K3RuntimeRPhaseStreamRParallelP4RCooperativeCalibrationOccupancy(dtype);
            const uint64_t runtime_r_p4_lanes = runtime_r_lanes_per_output(4);
            const uint64_t runtime_r_p4_outputs_per_block = 8 * (32 / runtime_r_p4_lanes);
            const uint64_t runtime_r_p4_blocks = std::min<uint64_t>(
                singlePassCeilDiv(geometry.outer_size, runtime_r_p4_outputs_per_block), FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "r_parallel_k3_phase_stream_runtime_r_p4",
                               "complete",
                               "fp8_k3_phase_stream_runtime_r_p4_calibration",
                               4,
                               256,
                               geometry.reduction_size,
                               1,
                               runtime_r_p4_blocks,
                               input,
                               runtime_r_p4_output,
                               runtime_r_p4_timing,
                               &device_geometry,
                               &runtime_r_p4_occupancy);

            Tensor runtime_r_p8_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_runtime_r_p8 = [&]() {
                launchFp8K3RuntimeRPhaseStreamRParallelP8InputRCooperativeCalibrationComplete(
                    input, runtime_r_p8_output, geometry, stream);
            };
            run_runtime_r_p8();
            stream.synchronize();
            validateSinglePassComplete(runtime_r_p8_output,
                                       geometry,
                                       stream,
                                       "fp8_rcoop_k3_runtime_r_phase_stream_p8",
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming runtime_r_p8_timing = time_complete_candidate(runtime_r_p8_output, run_runtime_r_p8);
            const CubKernelOccupancyInfo runtime_r_p8_occupancy =
                queryFp8K3RuntimeRPhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(dtype);
            const uint64_t runtime_r_p8_lanes = runtime_r_lanes_per_output(8);
            const uint64_t runtime_r_p8_outputs_per_block = 8 * (32 / runtime_r_p8_lanes);
            const uint64_t runtime_r_p8_blocks = std::min<uint64_t>(
                singlePassCeilDiv(geometry.outer_size, runtime_r_p8_outputs_per_block), FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "r_parallel_k3_phase_stream_runtime_r_p8_input",
                               "complete",
                               "fp8_k3_phase_stream_runtime_r_p8_input_calibration",
                               8,
                               256,
                               geometry.reduction_size,
                               1,
                               runtime_r_p8_blocks,
                               input,
                               runtime_r_p8_output,
                               runtime_r_p8_timing,
                               &device_geometry,
                               &runtime_r_p8_occupancy);
        }


        if (geometry.inner_size == 7 &&
            (geometry.reduction_size == 8 || geometry.reduction_size == 16 ||
             geometry.reduction_size == 32 || geometry.reduction_size == 64)) {
            Tensor k7_phase_stream_p8_input_output(
                input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_k7_phase_stream_p8_input = [&]() {
                launchFp8K7PhaseStreamRParallelP8InputRCooperativeCalibrationComplete(
                    input, k7_phase_stream_p8_input_output, geometry, stream);
            };
            run_k7_phase_stream_p8_input();
            stream.synchronize();
            validateSinglePassComplete(k7_phase_stream_p8_input_output,
                                       geometry,
                                       stream,
                                       "fp8_rcoop_k7_phase_stream_r_parallel_p8_input",
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming k7_phase_stream_p8_input_timing =
                time_complete_candidate(k7_phase_stream_p8_input_output, run_k7_phase_stream_p8_input);
            const CubKernelOccupancyInfo k7_phase_stream_p8_input_occupancy =
                queryFp8K7PhaseStreamRParallelP8InputRCooperativeCalibrationOccupancy(
                    dtype, geometry.reduction_size);
            const uint64_t lanes_per_output = geometry.reduction_size <= 8 ? 8 :
                                              (geometry.reduction_size <= 16 ? 16 : 32);
            const uint64_t outputs_per_warp = 32 / lanes_per_output;
            const uint64_t outputs_per_block = outputs_per_warp * 8;
            const uint64_t k7_phase_stream_p8_input_blocks = std::min<uint64_t>(
                singlePassCeilDiv(geometry.outer_size, outputs_per_block), FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "r_parallel_k7_phase_stream_p8_input",
                               "complete",
                               "fp8_k7_phase_stream_r_parallel_p8_input_calibration",
                               8,
                               256,
                               geometry.reduction_size,
                               1,
                               k7_phase_stream_p8_input_blocks,
                               input,
                               k7_phase_stream_p8_input_output,
                               k7_phase_stream_p8_input_timing,
                               &device_geometry,
                               &k7_phase_stream_p8_input_occupancy);
        }

        if (geometry.inner_size == 7 &&
            (geometry.reduction_size == 16 || geometry.reduction_size == 32 || geometry.reduction_size == 64)) {
            Tensor k7_phase_stream_p16_input_output(
                input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_k7_phase_stream_p16_input = [&]() {
                launchFp8K7PhaseStreamRParallelP16InputRCooperativeCalibrationComplete(
                    input, k7_phase_stream_p16_input_output, geometry, stream);
            };
            run_k7_phase_stream_p16_input();
            stream.synchronize();
            validateSinglePassComplete(k7_phase_stream_p16_input_output,
                                       geometry,
                                       stream,
                                       "fp8_rcoop_k7_phase_stream_r_parallel_p16_input",
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming k7_phase_stream_p16_input_timing =
                time_complete_candidate(k7_phase_stream_p16_input_output, run_k7_phase_stream_p16_input);
            const CubKernelOccupancyInfo k7_phase_stream_p16_input_occupancy =
                queryFp8K7PhaseStreamRParallelP16InputRCooperativeCalibrationOccupancy(
                    dtype, geometry.reduction_size);
            const uint64_t lanes_per_output = geometry.reduction_size <= 16 ? 8 :
                                              (geometry.reduction_size <= 32 ? 16 : 32);
            const uint64_t outputs_per_warp = 32 / lanes_per_output;
            const uint64_t outputs_per_block = outputs_per_warp * 8;
            const uint64_t k7_phase_stream_p16_input_blocks = std::min<uint64_t>(
                singlePassCeilDiv(geometry.outer_size, outputs_per_block), FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "r_parallel_k7_phase_stream_p16_input",
                               "complete",
                               "fp8_k7_phase_stream_r_parallel_p16_input_calibration",
                               16,
                               256,
                               geometry.reduction_size,
                               1,
                               k7_phase_stream_p16_input_blocks,
                               input,
                               k7_phase_stream_p16_input_output,
                               k7_phase_stream_p16_input_timing,
                               &device_geometry,
                               &k7_phase_stream_p16_input_occupancy);
        }

        const bool run_direct_component =
            geometry.inner_size == 5 || geometry.inner_size == 9 || geometry.inner_size == 11 ||
            geometry.inner_size == 13 || geometry.inner_size == 15 || geometry.inner_size == 17 ||
            geometry.inner_size == 19 || geometry.inner_size == 31;
        if (run_direct_component) {
            Tensor direct_component_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_direct_component = [&]() {
                launchFp8DirectComponentWarpP4RCooperativeCalibrationComplete(
                    input, direct_component_output, geometry, stream);
            };
            run_direct_component();
            stream.synchronize();
            validateSinglePassComplete(direct_component_output,
                                       geometry,
                                       stream,
                                       "fp8_rcoop_direct_component_warp_p4",
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming direct_component_timing =
                time_complete_candidate(direct_component_output, run_direct_component);
            const CubKernelOccupancyInfo direct_component_occupancy =
                queryFp8DirectComponentWarpP4RCooperativeCalibrationOccupancy(dtype, geometry.inner_size);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "direct_component_warp",
                               "complete",
                               "fp8_one_output_warp_direct_component_p4_calibration",
                               4,
                               256,
                               geometry.reduction_size,
                               1,
                               one_output_blocks,
                               input,
                               direct_component_output,
                               direct_component_timing,
                               &device_geometry,
                               &direct_component_occupancy);
        }
        return;
    }

    // Let the exact production FP8 KParallel selector compete wherever it has a legal packet/layout. This is not a
    // benchmark recreation of policy: proposeProductionKParallelForCalibration() calls the same selector used by production and
    // returns null when KParallel is inapplicable (notably odd FP8 K).
    DenseReductionOutputSpec output_spec;
    output_spec.produce_value = true;
    output_spec.value_dtype = dtype;
    output_spec.produce_index = false;
    DenseReductionProblem kparallel_problem = makeInitialDenseReductionProblem(
        input.getDimensions(), {1}, DenseReductionAggregateKind::Value, output_spec);
    std::vector<DenseReductionSite> kparallel_sites = enumerateDenseReductionSites(kparallel_problem);
    if (kparallel_sites.size() != 1
        || classifyDenseReducerFamily(kparallel_problem, kparallel_sites.front()) != DenseReducerFamily::RK) {
        throw std::logic_error("FP8 full-space census expected one production KParallel RK site.");
    }
    DenseRKPlanningContext kparallel_context;
    kparallel_context.original_input_dtype = dtype;
    kparallel_context.value_operation = DenseRKValueOperation::Sum;
    kparallel_context.multiprocessors = device_geometry.sm_count;
    // Mirror the device geometry that production passes to the KParallel selector. Exact kernel occupancy answers
    // register/shared-memory residency, but FP8 saturation policy still needs the hardware warp/thread limits to
    // convert the calibrated useful-warps-per-SM target into a device-wide launch-supply target.
    kparallel_context.warp_size = device_geometry.warp_size;
    kparallel_context.max_threads_per_sm = device_geometry.max_threads_per_sm;
    kparallel_context.max_blocks_per_sm = device_geometry.max_blocks_per_sm;
    kparallel_context.occupancy_query = queryDenseRKFamilyExactOccupancy;
    kparallel_context.occupancy_query_context = nullptr;
    kparallel_context.allow_generic_rk_fallback = false;
    std::optional<DenseReductionCandidate> kparallel_candidate = ReducersDenseRK::proposeProductionKParallelForCalibration(
        kparallel_problem, kparallel_sites.front(), kparallel_context);
    if (kparallel_candidate.has_value()) {
        auto kparallel_plan = std::dynamic_pointer_cast<const DenseRKFamilyPhysicalPlan>(
            kparallel_candidate->physical_plan);
        if (!kparallel_plan || kparallel_plan->implementation != DenseRKProductionImplementation::KParallelPass
            || kparallel_plan->strategy != DenseRKStrategy::KParallel) {
            throw std::logic_error("Production KParallel calibration hook returned a non-KParallel plan.");
        }
        Tensor kparallel_output;
        if (kparallel_plan->progress == DenseRKProgress::Complete) {
            kparallel_output = Tensor(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        } else {
            kparallel_output = Tensor(
                input.getPlacement(),
                TensorDescriptor(DataType::FP32,
                                 {geometry.outer_size, kparallel_plan->shards_per_output, geometry.inner_size}));
        }
        const CubReductionStageSemantics kparallel_semantics = makeValueReductionStageSemantics(
            CubReductionOp::Sum, kparallel_plan->role, geometry.reduction_size);
        const KParallelTiledStagePlan physical{
            kparallel_plan->progress == DenseRKProgress::Complete ? TiledRKStageTopology::Complete
                                                                  : TiledRKStageTopology::Staged,
            kparallel_plan->packet_bytes,
            kparallel_plan->block_threads,
            kparallel_plan->shards_per_output};
        const auto run_kparallel = [&]() {
            launchKParallelTiledStage(
                kparallel_semantics, input, kparallel_output, geometry, physical, 1.0f, stream);
        };
        run_kparallel();
        stream.synchronize();
        if (kparallel_plan->progress == DenseRKProgress::Complete) {
            validateSinglePassComplete(
                kparallel_output, geometry, stream, "production_kparallel", FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
        } else {
            validateSinglePassPartial(kparallel_output,
                                      geometry,
                                      kparallel_plan->shards_per_output,
                                      kparallel_plan->rows_per_shard,
                                      SinglePassShardPolicy::Balanced,
                                      stream,
                                      "production_kparallel",
                                      FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
        }
        RKSinglePassTiming kparallel_timing{};
        if (!use_rotating_cold_working_set) {
            kparallel_timing = timeRKSinglePass(
                cache_flush, stream, run_kparallel, FP8_RCOOP_FULL_SPACE_TIMING_SAMPLES);
        } else if (kparallel_plan->progress == DenseRKProgress::Complete) {
            kparallel_timing = time_complete_candidate(kparallel_output, run_kparallel);
        } else {
            RKRotatingTensorSlots rotating_kparallel_outputs(
                kparallel_output.getPlacement(), kparallel_output.getDescriptor(), rotating_slot_count);
            kparallel_timing = time_rotating_candidate(
                kparallel_output, rotating_kparallel_outputs, run_kparallel);
        }
        const CubKernelOccupancyInfo kparallel_occupancy = queryKParallelTiledStageOccupancy(dtype, physical);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "k_parallel",
                           "aligned",
                           kparallel_plan->progress == DenseRKProgress::Complete ? "complete" : "staged",
                           "production_kparallel",
                           kparallel_plan->packet_bytes,
                           kparallel_plan->block_threads,
                           kparallel_plan->rows_per_shard,
                           kparallel_plan->shards_per_output,
                           kparallel_plan->first_stage_blocks,
                           input,
                           kparallel_output,
                           kparallel_timing,
                           &device_geometry,
                           &kparallel_occupancy);
    }

    const CubReductionStageSemantics complete_semantics = makeValueReductionStageSemantics(
        CubReductionOp::Sum, CubReductionStageRole::Complete, geometry.reduction_size);
    const size_t complete_packet_bytes = singlePassPacketAdaptivePacketBytes(geometry, dtype);
    const uint32_t complete_warps = singlePassPacketAdaptiveWarps(geometry, dtype, complete_packet_bytes);
    const uint32_t complete_block_warps = std::max<uint32_t>(
        static_cast<uint32_t>(TILED_REDUCTION_WARPS_PER_BLOCK), complete_warps);
    DenseRKFamilyPhysicalPlan complete_plan;
    complete_plan.packet_bytes = complete_packet_bytes;
    complete_plan.cooperative_warps_per_tile = complete_warps;
    complete_plan.block_threads = complete_block_warps * static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS);
    Tensor complete_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
    const auto run_complete = [&]() {
        launchPacketAdaptiveComplete(complete_semantics, complete_plan, input, complete_output, geometry, stream);
    };
    run_complete();
    stream.synchronize();
    validateSinglePassComplete(complete_output,
                               geometry,
                               stream,
                               "fp8_rcoop_full_space_complete",
                               FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
    const RKSinglePassTiming complete_timing = time_complete_candidate(complete_output, run_complete);
    const CubKernelOccupancyInfo complete_occupancy = querySinglePassPacketAdaptiveOccupancy(
        geometry, dtype, complete_packet_bytes, static_cast<int>(complete_warps));
    const uint64_t complete_tiles = singlePassPacketAdaptiveTiles(geometry, dtype, complete_packet_bytes);
    const uint64_t complete_groups_per_block = complete_block_warps / complete_warps;
    const uint64_t complete_blocks = std::min<uint64_t>(
        singlePassCeilDiv(geometry.outer_size * complete_tiles, complete_groups_per_block),
        FP8_RCOOP_MAX_GRID_BLOCKS);
    printSinglePassRow(benchmark_case,
                       dtype,
                       "r_cooperative",
                       "aligned_or_alignment_safe",
                       "complete",
                       "packet_adaptive_complete",
                       complete_packet_bytes,
                       complete_plan.block_threads,
                       geometry.reduction_size,
                       1,
                       complete_blocks,
                       input,
                       complete_output,
                       complete_timing,
                       &device_geometry,
                       &complete_occupancy);

    // Generalized Complete RParallel is the intended consolidation candidate for medium K. Ownership is now one
    // continuum in two mutually exclusive directions: O1 may grow from W1 to W2/W4/W8 as R deepens, while W1 may
    // grow from O1 to O2/O4/O8 when one warp has enough K/R work to retain multiple outputs. Within each output,
    // narrow K spends spare lanes across R workers/component and wider K assigns multiple components/lane. Packet
    // legality is derived from each output subgroup's p4/p8/p16 staging slice, so awkward K stays in the same family.
    if (geometry.reduction_size <= 1024 && geometry.inner_size <= 256 &&
        (geometry.inner_size > 32 || include_general_rparallel_small_k)) {
        const auto packet_supports_ownership = [&](uint32_t outputs_per_warp, uint32_t packet_bytes) {
            const uint64_t lanes_per_output = uint64_t{32} / outputs_per_warp;
            const uint64_t packet_window = lanes_per_output * packet_bytes;
            const bool row_aligned = (geometry.inner_size % packet_bytes) == 0;
            const uint64_t logical_capacity = row_aligned ? packet_window : packet_window - (packet_bytes - 1);
            return geometry.inner_size <= logical_capacity;
        };
        const auto run_general_r_parallel_candidate =
            [&](uint32_t outputs_per_warp, uint32_t warps_per_output, uint32_t packet_bytes, bool linear_packets) {
                Tensor general_r_parallel_output(
                    input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
                const auto run_general_r_parallel = [&]() {
                    launchFp8GeneralRParallelRCooperativeCalibrationComplete(
                        input,
                        general_r_parallel_output,
                        geometry,
                        outputs_per_warp,
                        warps_per_output,
                        packet_bytes,
                        stream,
                        linear_packets);
                };
                run_general_r_parallel();
                stream.synchronize();
                const std::string ownership_label =
                    "o" + std::to_string(outputs_per_warp) + "_w" + std::to_string(warps_per_output);
                const std::string general_r_parallel_label =
                    "fp8_rcoop_general_r_parallel_" + ownership_label +
                    (linear_packets ? "_linear" : "") + "_p" + std::to_string(packet_bytes);
                validateSinglePassComplete(general_r_parallel_output,
                                           geometry,
                                           stream,
                                           general_r_parallel_label,
                                           FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
                const RKSinglePassTiming general_r_parallel_timing =
                    time_complete_candidate(general_r_parallel_output, run_general_r_parallel);
                const CubKernelOccupancyInfo general_r_parallel_occupancy =
                    queryFp8GeneralRParallelRCooperativeCalibrationOccupancy(
                        dtype,
                        geometry.inner_size,
                        outputs_per_warp,
                        warps_per_output,
                        packet_bytes,
                        linear_packets);
                const uint64_t outputs_per_block =
                    (uint64_t{8} / warps_per_output) * outputs_per_warp;
                const uint64_t general_r_parallel_blocks = std::min<uint64_t>(
                    singlePassCeilDiv(geometry.outer_size, outputs_per_block), FP8_RCOOP_MAX_GRID_BLOCKS);
                const std::string layout_suffix = linear_packets ? "_linear" : "";
                const std::string general_r_parallel_access =
                    (outputs_per_warp == 1
                         ? "r_parallel_general_w" + std::to_string(warps_per_output)
                         : "r_parallel_general_o" + std::to_string(outputs_per_warp) + "_w1") +
                    layout_suffix;
                const std::string general_r_parallel_implementation =
                    "fp8_general_r_parallel_" + ownership_label + layout_suffix + "_p" +
                    std::to_string(packet_bytes) + "_calibration";
                printSinglePassRow(benchmark_case,
                                   dtype,
                                   "r_cooperative",
                                   general_r_parallel_access,
                                   "complete",
                                   general_r_parallel_implementation,
                                   packet_bytes,
                                   256,
                                   geometry.reduction_size,
                                   1,
                                   general_r_parallel_blocks,
                                   input,
                                   general_r_parallel_output,
                                   general_r_parallel_timing,
                                   &device_geometry,
                                   &general_r_parallel_occupancy);
            };

        for (uint32_t warps_per_output : {1u, 2u, 4u, 8u}) {
            if (warps_per_output > geometry.reduction_size) {
                continue;
            }
            for (uint32_t packet_bytes : {4u, 8u, 16u}) {
                if (packet_supports_ownership(1, packet_bytes)) {
                    run_general_r_parallel_candidate(1, warps_per_output, packet_bytes, false);
                }
            }
        }
        for (uint32_t outputs_per_warp : {2u, 4u, 8u}) {
            for (uint32_t packet_bytes : {4u, 8u, 16u}) {
                if (packet_supports_ownership(outputs_per_warp, packet_bytes)) {
                    run_general_r_parallel_candidate(outputs_per_warp, 1, packet_bytes, false);
                }
            }
        }

        // LinearPackets preserves the packet-major byte stream in shared memory. Calibrate it only in the awkward
        // small-K region where FlatRows currently wins; keep O1 so this isolates access layout from output ownership.
        const bool calibrate_linear_packets = include_general_rparallel_staged &&
            (geometry.inner_size == 5 || geometry.inner_size == 7 || geometry.inner_size == 11 ||
             geometry.inner_size == 13 || geometry.inner_size == 15 || geometry.inner_size == 17 ||
             geometry.inner_size == 19 || geometry.inner_size == 31);
        if (calibrate_linear_packets) {
            for (uint32_t warps_per_output : {1u, 2u, 4u, 8u}) {
                if (warps_per_output > geometry.reduction_size) continue;
                for (uint32_t packet_bytes : {8u, 16u}) {
                    if (packet_supports_ownership(1, packet_bytes)) {
                        run_general_r_parallel_candidate(1, warps_per_output, packet_bytes, true);
                    }
                }
            }
        }
    }


    // Staged generalized RParallel extends the same ownership family across CTAs. S2/S4/S8 balance one logical R
    // reduction over independent CTA shards and emit FP32 [outer, shard, K] partials. S already creates independent
    // work, so this calibration intentionally keeps O1 and varies only the useful intra-shard W1/W2/W4/W8 axis.
    if (include_general_rparallel_staged && geometry.inner_size <= 256) {
        const auto packet_supports_staged = [&](uint32_t packet_bytes) {
            const uint64_t packet_window = uint64_t{32} * packet_bytes;
            const bool row_aligned = (geometry.inner_size % packet_bytes) == 0;
            const uint64_t logical_capacity = row_aligned ? packet_window : packet_window - (packet_bytes - 1);
            return geometry.inner_size <= logical_capacity;
        };
        for (uint32_t shards_per_output : {2u, 4u, 8u}) {
            if (shards_per_output > geometry.reduction_size) {
                continue;
            }
            const uint64_t rows_per_shard = singlePassCeilDiv(geometry.reduction_size, shards_per_output);
            Tensor staged_rparallel_partials(
                input.getPlacement(),
                TensorDescriptor(DataType::FP32,
                                 {geometry.outer_size, shards_per_output, geometry.inner_size}));
            std::unique_ptr<RKRotatingTensorSlots> rotating_staged_rparallel_outputs;
            if (use_rotating_cold_working_set) {
                rotating_staged_rparallel_outputs = std::make_unique<RKRotatingTensorSlots>(
                    staged_rparallel_partials.getPlacement(),
                    staged_rparallel_partials.getDescriptor(),
                    rotating_slot_count);
            }
            for (uint32_t warps_per_output : {1u, 2u, 4u, 8u}) {
                if (warps_per_output > rows_per_shard) {
                    continue;
                }
                for (uint32_t packet_bytes : {4u, 8u, 16u}) {
                    if (!packet_supports_staged(packet_bytes)) {
                        continue;
                    }
                    const bool linear_k = geometry.inner_size == 5 || geometry.inner_size == 7 ||
                        geometry.inner_size == 11 || geometry.inner_size == 13 || geometry.inner_size == 15 ||
                        geometry.inner_size == 17 || geometry.inner_size == 19 || geometry.inner_size == 31;
                    for (bool linear_packets : {false, true}) {
                        if (linear_packets && (!linear_k || packet_bytes == 4)) continue;
                    const auto run_staged_rparallel = [&]() {
                        launchFp8GeneralRParallelRCooperativeCalibrationFirstStage(
                            input,
                            staged_rparallel_partials,
                            geometry,
                            warps_per_output,
                            packet_bytes,
                            shards_per_output,
                            stream,
                            linear_packets);
                    };
                    run_staged_rparallel();
                    stream.synchronize();
                    const std::string layout_suffix = linear_packets ? "_linear" : "";
                    const std::string staged_label =
                        "fp8_rcoop_general_r_parallel_o1_w" + std::to_string(warps_per_output) +
                        "_s" + std::to_string(shards_per_output) + layout_suffix + "_p" +
                        std::to_string(packet_bytes);
                    validateSinglePassPartial(staged_rparallel_partials,
                                              geometry,
                                              shards_per_output,
                                              rows_per_shard,
                                              SinglePassShardPolicy::Balanced,
                                              stream,
                                              staged_label,
                                              FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
                    const RKSinglePassTiming staged_timing = use_rotating_cold_working_set
                        ? time_rotating_candidate(
                              staged_rparallel_partials,
                              *rotating_staged_rparallel_outputs,
                              run_staged_rparallel)
                        : timeRKSinglePass(
                              cache_flush,
                              stream,
                              run_staged_rparallel,
                              FP8_RCOOP_FULL_SPACE_TIMING_SAMPLES);
                    const CubKernelOccupancyInfo staged_occupancy =
                        queryFp8GeneralRParallelRCooperativeCalibrationFirstStageOccupancy(
                            dtype, geometry.inner_size, warps_per_output, packet_bytes, linear_packets);
                    const uint64_t outputs_per_block = uint64_t{8} / warps_per_output;
                    const uint64_t blocks = std::min<uint64_t>(
                        singlePassCeilDiv(geometry.outer_size * shards_per_output, outputs_per_block),
                        FP8_RCOOP_MAX_GRID_BLOCKS);
                    const std::string staged_access =
                        "r_parallel_general_w" + std::to_string(warps_per_output) +
                        "_s" + std::to_string(shards_per_output) + layout_suffix;
                    const std::string staged_implementation =
                        "fp8_general_r_parallel_o1_w" + std::to_string(warps_per_output) +
                        "_s" + std::to_string(shards_per_output) + layout_suffix + "_p" +
                        std::to_string(packet_bytes) + "_calibration";
                    printSinglePassRow(benchmark_case,
                                       dtype,
                                       "r_cooperative",
                                       staged_access,
                                       "staged",
                                       staged_implementation,
                                       packet_bytes,
                                       256,
                                       rows_per_shard,
                                       shards_per_output,
                                       blocks,
                                       input,
                                       staged_rparallel_partials,
                                       staged_timing,
                                       &device_geometry,
                                       &staged_occupancy);
                    }
                }
            }
        }
    }

    // DEPRECATED EXPERIMENT ONLY -- NOT A PRODUCTION CANDIDATE. Retain the one-output-per-warp p4 prototype only
    // as a comparison baseline for the adaptive multi-output/R-parallel replacement below.
    if (geometry.reduction_size <= 1024 && geometry.inner_size <= 64) {
        Tensor outer_grouped_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_outer_grouped = [&]() {
            launchDeprecatedExperimentalFp8OuterGroupedP4RCooperativeCalibrationComplete(
                input, outer_grouped_output, geometry, stream);
        };
        run_outer_grouped();
        stream.synchronize();
        validateSinglePassComplete(outer_grouped_output,
                                   geometry,
                                   stream,
                                   "fp8_rcoop_deprecated_experiment_outer_grouped_p4",
                                   FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
        const RKSinglePassTiming outer_grouped_timing =
            time_complete_candidate(outer_grouped_output, run_outer_grouped);
        const CubKernelOccupancyInfo outer_grouped_occupancy =
            queryDeprecatedExperimentalFp8OuterGroupedP4RCooperativeCalibrationOccupancy(dtype, geometry.inner_size);
        const uint64_t outer_grouped_blocks = std::min<uint64_t>(
            singlePassCeilDiv(geometry.outer_size, uint64_t{8}), FP8_RCOOP_MAX_GRID_BLOCKS);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "r_cooperative",
                           "experimental_deprecated_outer_grouped",
                           "complete",
                           "DEPRECATED_EXPERIMENT_ONLY_fp8_outer_grouped_p4_r_cooperative_calibration",
                           4,
                           256,
                           geometry.reduction_size,
                           1,
                           outer_grouped_blocks,
                           input,
                           outer_grouped_output,
                           outer_grouped_timing,
                           &device_geometry,
                           &outer_grouped_occupancy);
    }

    // Raw p4 read+write roofline reference with the same input/output traffic ratio as a complete reduction. It is
    // deliberately excluded from coverage/winner decisions by the analyzer; it answers how much of the remaining gap
    // is physical p4 streaming/writeback versus reduction-kernel overhead.
    if (geometry.reduction_size <= 1024 && geometry.inner_size <= 32) {
        Tensor roofline_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_roofline = [&]() {
            launchFp8P4ReadWriteRooflineCalibrationReference(input, roofline_output, geometry, stream);
        };
        run_roofline();
        stream.synchronize();
        const RKSinglePassTiming roofline_timing =
            time_complete_candidate(roofline_output, run_roofline);
        const CubKernelOccupancyInfo roofline_occupancy = queryFp8P4ReadWriteRooflineCalibrationOccupancy();
        const uint64_t output_elements = geometry.outer_size * geometry.inner_size;
        const uint64_t output_packets = output_elements / 4;
        const uint64_t roofline_warp_groups = std::max<uint64_t>(uint64_t{1}, singlePassCeilDiv(output_packets, uint64_t{32}));
        const uint64_t roofline_blocks = std::min<uint64_t>(
            singlePassCeilDiv(roofline_warp_groups, uint64_t{8}), FP8_RCOOP_MAX_GRID_BLOCKS);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "reference",
                           "p4_read_write_roofline",
                           "complete",
                           "fp8_p4_read_write_same_ratio_roofline_reference",
                           4,
                           256,
                           geometry.reduction_size,
                           1,
                           roofline_blocks,
                           input,
                           roofline_output,
                           roofline_timing,
                           &device_geometry,
                           &roofline_occupancy);
    }

    // Split the shallow/narrow replacement into independent physical mechanisms. This keeps the hot loops simple and
    // lets the census establish the actual handoffs instead of embedding a speculative runtime selector.
    if (geometry.reduction_size <= 1024 && geometry.inner_size <= 32) {
        for (const uint32_t compact_packet_bytes : {4u, 8u, 16u}) {
            const uint32_t compact_outputs_per_warp =
                fp8CompactMultiOutputWarpRCooperativeCalibrationOutputsPerWarp(
                    geometry.reduction_size, geometry.inner_size, compact_packet_bytes);
            if (compact_outputs_per_warp <= 1) {
                continue;
            }
            Tensor compact_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
            const auto run_compact = [&]() {
                launchFp8CompactMultiOutputWarpRCooperativeCalibrationComplete(
                    input, compact_output, geometry, compact_packet_bytes, stream);
            };
            run_compact();
            stream.synchronize();
            const std::string compact_label =
                "fp8_rcoop_compact_multi_output_warp_p" + std::to_string(compact_packet_bytes);
            validateSinglePassComplete(compact_output,
                                       geometry,
                                       stream,
                                       compact_label,
                                       FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming compact_timing =
                time_complete_candidate(compact_output, run_compact);
            const CubKernelOccupancyInfo compact_occupancy =
                queryFp8CompactMultiOutputWarpRCooperativeCalibrationOccupancy(
                    dtype, geometry.inner_size, compact_outputs_per_warp, compact_packet_bytes);
            const uint64_t compact_warp_groups = singlePassCeilDiv(
                geometry.outer_size, static_cast<uint64_t>(compact_outputs_per_warp));
            const uint64_t compact_blocks = std::min<uint64_t>(
                singlePassCeilDiv(compact_warp_groups, uint64_t{8}), FP8_RCOOP_MAX_GRID_BLOCKS);
            const std::string compact_implementation =
                "fp8_compact_multi_output_warp_p" + std::to_string(compact_packet_bytes) + "_calibration";
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "multi_output_compact",
                               "complete",
                               compact_implementation,
                               compact_packet_bytes,
                               256,
                               geometry.reduction_size,
                               1,
                               compact_blocks,
                               input,
                               compact_output,
                               compact_timing,
                               &device_geometry,
                               &compact_occupancy);
        }

        Tensor r_parallel_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_r_parallel = [&]() {
            launchFp8RParallelWarpP4RCooperativeCalibrationComplete(
                input, r_parallel_output, geometry, stream);
        };
        run_r_parallel();
        stream.synchronize();
        validateSinglePassComplete(r_parallel_output,
                                   geometry,
                                   stream,
                                   "fp8_rcoop_r_parallel_warp_p4",
                                   FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
        const RKSinglePassTiming r_parallel_timing =
            time_complete_candidate(r_parallel_output, run_r_parallel);
        const CubKernelOccupancyInfo r_parallel_occupancy =
            queryFp8RParallelWarpP4RCooperativeCalibrationOccupancy(dtype, geometry.inner_size);
        const uint64_t one_output_blocks = std::min<uint64_t>(
            singlePassCeilDiv(geometry.outer_size, uint64_t{8}), FP8_RCOOP_MAX_GRID_BLOCKS);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "r_cooperative",
                           "r_parallel_warp",
                           "complete",
                           "fp8_one_output_warp_r_parallel_p4_calibration",
                           4,
                           256,
                           geometry.reduction_size,
                           1,
                           one_output_blocks,
                           input,
                           r_parallel_output,
                           r_parallel_timing,
                           &device_geometry,
                           &r_parallel_occupancy);

        Tensor direct_component_output(input.getPlacement(), TensorDescriptor(dtype, geometry.output_dimensions));
        const auto run_direct_component = [&]() {
            launchFp8DirectComponentWarpP4RCooperativeCalibrationComplete(
                input, direct_component_output, geometry, stream);
        };
        run_direct_component();
        stream.synchronize();
        validateSinglePassComplete(direct_component_output,
                                   geometry,
                                   stream,
                                   "fp8_rcoop_direct_component_warp_p4",
                                   FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
        const RKSinglePassTiming direct_component_timing =
            time_complete_candidate(direct_component_output, run_direct_component);
        const CubKernelOccupancyInfo direct_component_occupancy =
            queryFp8DirectComponentWarpP4RCooperativeCalibrationOccupancy(dtype, geometry.inner_size);
        printSinglePassRow(benchmark_case,
                           dtype,
                           "r_cooperative",
                           "direct_component_warp",
                           "complete",
                           "fp8_one_output_warp_direct_component_p4_calibration",
                           4,
                           256,
                           geometry.reduction_size,
                           1,
                           one_output_blocks,
                           input,
                           direct_component_output,
                           direct_component_timing,
                           &device_geometry,
                           &direct_component_occupancy);
    }

    struct FlatPacketCandidate {
        size_t packet_bytes;
        uint64_t max_k;
        const char* implementation;
    };
    constexpr std::array<FlatPacketCandidate, 3> FLAT_PACKETS = {{
        {4, 125, "fp8_narrow_flat_p4_adaptive_window_r_cooperative_calibration"},
        {8, 249, "fp8_narrow_flat_p8_adaptive_window_r_cooperative_calibration"},
        {16, 256, "fp8_narrow_flat_p16_adaptive_window_r_cooperative_calibration"},
    }};
    struct RotatedCandidate {
        size_t packet_bytes;
        bool exact_tile;
        const char* access;
        const char* implementation;
    };
    constexpr std::array<RotatedCandidate, 4> ROTATED = {{
        {4, false, "rotated_prefix_reserved", "fp8_rotated_prefix_reserved_p4_r_cooperative_calibration"},
        {8, false, "rotated_prefix_reserved", "fp8_rotated_prefix_reserved_p8_r_cooperative_calibration"},
        {16, false, "rotated_prefix_reserved", "fp8_rotated_prefix_reserved_p16_r_cooperative_calibration"},
        {16, true, "rotated_exact", "fp8_rotated_exact_p16_r_cooperative_calibration"},
    }};

    for (uint64_t rows_per_shard : FP8_RCOOP_FULL_SPACE_ROW_DEPTHS) {
        if (rows_per_shard >= geometry.reduction_size) {
            continue;
        }
        const uint64_t shards = singlePassCeilDiv(geometry.reduction_size, rows_per_shard);
        if (shards <= 1) {
            continue;
        }

        Tensor staged_partials(input.getPlacement(),
                               TensorDescriptor(DataType::FP32,
                                                {geometry.outer_size, shards, geometry.inner_size}));
        std::unique_ptr<RKRotatingTensorSlots> rotating_staged_outputs;
        if (use_rotating_cold_working_set) {
            rotating_staged_outputs = std::make_unique<RKRotatingTensorSlots>(
                staged_partials.getPlacement(), staged_partials.getDescriptor(), rotating_slot_count);
        }

        for (const FlatPacketCandidate& flat : FLAT_PACKETS) {
            if (geometry.inner_size > flat.max_k) {
                continue;
            }
            const auto run_flat = [&]() {
                launchFp8NarrowLowPrecisionFlatRCooperativeCalibrationFirstStage(
                    input, staged_partials, geometry, flat.packet_bytes, rows_per_shard, shards, stream);
            };
            run_flat();
            stream.synchronize();
            validateSinglePassPartial(staged_partials,
                                      geometry,
                                      shards,
                                      rows_per_shard,
                                      SinglePassShardPolicy::FixedRows,
                                      stream,
                                      "fp8_rcoop_full_space_flat_rows_p" + std::to_string(flat.packet_bytes),
                                      FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming timing = use_rotating_cold_working_set
                ? time_rotating_candidate(staged_partials, *rotating_staged_outputs, run_flat)
                : timeRKSinglePass(cache_flush, stream, run_flat, FP8_RCOOP_FULL_SPACE_TIMING_SAMPLES);
            const CubKernelOccupancyInfo occupancy =
                queryFp8NarrowLowPrecisionFlatRCooperativeCalibrationOccupancy(
                    dtype, flat.packet_bytes, rows_per_shard, geometry.inner_size);
            const uint64_t blocks = std::min<uint64_t>(
                geometry.outer_size * shards, FP8_RCOOP_MAX_GRID_BLOCKS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               "flat_rows",
                               "staged",
                               flat.implementation,
                               flat.packet_bytes,
                               256,
                               rows_per_shard,
                               shards,
                               blocks,
                               input,
                               staged_partials,
                               timing,
                               &device_geometry,
                               &occupancy);
        }

        for (const RotatedCandidate& rotated : ROTATED) {
            const auto run_rotated = [&]() {
                launchFp8AwkwardAlignmentRotatedShardedCalibrationFirstStage(
                    input,
                    staged_partials,
                    geometry,
                    rotated.packet_bytes,
                    rotated.exact_tile,
                    rows_per_shard,
                    shards,
                    stream);
            };
            run_rotated();
            stream.synchronize();
            validateSinglePassPartial(staged_partials,
                                      geometry,
                                      shards,
                                      rows_per_shard,
                                      SinglePassShardPolicy::FixedRows,
                                      stream,
                                      std::string("fp8_rcoop_full_space_") + rotated.access + "_p"
                                          + std::to_string(rotated.packet_bytes),
                                      FP8_RCOOP_SINGLE_PASS_FILL_VALUE);
            const RKSinglePassTiming timing = use_rotating_cold_working_set
                ? time_rotating_candidate(staged_partials, *rotating_staged_outputs, run_rotated)
                : timeRKSinglePass(cache_flush, stream, run_rotated, FP8_RCOOP_FULL_SPACE_TIMING_SAMPLES);
            const CubKernelOccupancyInfo occupancy =
                queryFp8AwkwardAlignmentRotatedShardedCalibrationOccupancy(
                    dtype, rotated.packet_bytes, rotated.exact_tile, rows_per_shard);
            const uint64_t tiles = singlePassRotatedRCoopTiles(
                geometry, dtype, rotated.packet_bytes, rotated.exact_tile);
            const uint64_t blocks = std::min<uint64_t>(
                geometry.outer_size * tiles * shards, FP8_RCOOP_MAX_GRID_BLOCKS);
            const uint32_t block_threads =
                static_cast<uint32_t>(rotated.packet_bytes) * static_cast<uint32_t>(TILED_REDUCTION_WARP_THREADS);
            printSinglePassRow(benchmark_case,
                               dtype,
                               "r_cooperative",
                               rotated.access,
                               "staged",
                               rotated.implementation,
                               rotated.packet_bytes,
                               block_threads,
                               rows_per_shard,
                               shards,
                               blocks,
                               input,
                               staged_partials,
                               timing,
                               &device_geometry,
                               &occupancy);
        }
    }
}

void runFp8RCooperativeFullSpaceSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    // Full requested R Cartesian: small/medium/large, with both naturally aligned and awkward depths.
    constexpr std::array<uint64_t, 18> R_VALUES = {
        16, 32, 64, 15, 31, 63,
        256, 512, 1024, 255, 511, 1023,
        4096, 8192, 32768, 4095, 8191, 32767};
    // Actual-work Cartesian. The first four points resolve launch-dominated behavior; 64..512 MiB resolves the
    // saturation crossover seen in the exploratory full-space sweep.
    constexpr std::array<uint64_t, 8> TARGET_BYTES = {
        256ULL * 1024ULL,
        1ULL * 1024ULL * 1024ULL,
        4ULL * 1024ULL * 1024ULL,
        16ULL * 1024ULL * 1024ULL,
        64ULL * 1024ULL * 1024ULL,
        128ULL * 1024ULL * 1024ULL,
        256ULL * 1024ULL * 1024ULL,
        512ULL * 1024ULL * 1024ULL,
    };
    // Full requested K Cartesian: small/medium/large, with nice packet-aligned and awkward/boundary values.
    constexpr std::array<uint64_t, 22> K_VALUES = {
        8, 16, 32, 7, 15, 31,
        64, 128, 256, 63, 127, 129, 255,
        512, 1024, 4096, 497, 511, 993, 1023, 1493, 4095};

    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    printFp8RCooperativeCoverageHeader(
        device_geometry,
        "rk_fp8_rcooperative_full_space_sweep",
        "map modern FP8 RK coverage across the full R x K x actual-input-byte Cartesian; production KParallel "
        "competes wherever its real selector has a legal plan, while calibration-only RCooperative candidates expose "
        "genuine holes and transition regions");
    uint64_t exact_shape_dtypes = 0;
    for (uint64_t reduction : R_VALUES) {
        for (uint64_t inner : K_VALUES) {
            std::array<uint64_t, TARGET_BYTES.size()> seen_outer{};
            size_t seen_count = 0;
            for (uint64_t target_bytes : TARGET_BYTES) {
                const uint64_t outer = fp8RCooperativeOuterNearTarget(reduction, inner, target_bytes);
                bool duplicate = false;
                for (size_t index = 0; index < seen_count; ++index) {
                    duplicate = duplicate || seen_outer[index] == outer;
                }
                if (duplicate) {
                    continue;
                }
                seen_outer[seen_count++] = outer;
                RKSinglePassSweepCase benchmark_case{
                    fp8RCooperativeTargetLabel(target_bytes),
                    "o" + std::to_string(outer) + "_r" + std::to_string(reduction) + "_k" + std::to_string(inner),
                    outer,
                    reduction,
                    inner};
                for (DataType dtype : DTYPES) {
                    runFp8RCooperativeFullSpaceCase(
                        benchmark_case, dtype, cache_flush, stream, device_geometry);
                    ++exact_shape_dtypes;
                }
            }
        }
    }
    std::cout << "# rk_fp8_rcooperative_full_space_sweep_complete exact_shape_dtypes="
              << exact_shape_dtypes << '\n';
#else
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 RCooperative full-space sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

void runFp8RCooperativeSmallRSmallKSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    // Focus the shallow/narrow hole aggressively, including genuinely tiny reductions. Powers of two are the nice
    // controls; their immediately-lower awkward neighbors expose alignment / utilization effects.
    constexpr std::array<uint64_t, 11> R_VALUES = {2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64};
    // Nice/aligned K={4,8,16,32} is already covered by Compact/RParallel and no longer belongs in this
    // focused benchmark. Sweep every odd K through 19 to map where phase-stream-style coverage is actually needed,
    // and retain K=31 as the larger odd-K control.
    constexpr std::array<uint64_t, 10> K_VALUES = {3, 5, 7, 9, 11, 13, 15, 17, 19, 31};
    // Keep only substantial work sizes here. The purpose is to distinguish a physical geometry hole from launch
    // underfill, not to repeat the full small-work Cartesian.
    constexpr std::array<uint64_t, 4> TARGET_BYTES = {
        64ULL * 1024ULL * 1024ULL,
        128ULL * 1024ULL * 1024ULL,
        256ULL * 1024ULL * 1024ULL,
        512ULL * 1024ULL * 1024ULL,
    };

    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    printFp8RCooperativeCoverageHeader(
        device_geometry,
        "rk_fp8_rcooperative_small_r_small_k_sweep",
        "focused unresolved shallow-R/narrow-K FP8 holes only; skips already-established KParallel/generic-p16/FlatRows/Rotated "
        "coverage and compares the p4 roofline, Compact where relevant, one-output RParallel p4/p8/p16, the subgrouped "
        "multi-output RParallel p4/p8/p16 packet shapes, multi-warp Complete RParallel W2/W4/W8 p4/p8/p16, the "
        "established K=3/K=7 phase-stream candidates, and DirectComponent across the odd-K census");
    std::cout << "# focused_holes_only=1 odd_k_census=3,5,7,9,11,13,15,17,19,31 "
                 "candidates=roofline,compact_when_applicable,r_parallel_p4_p8_p16,r_parallel_multi_output_2_4_8_16_32_p4_p8_p16,"
                 "r_parallel_multi_warp_2_4_8_p4_p8_p16,k3_phase_stream,k7_phase_stream_p8_p16,direct_odd_k\n";
    uint64_t exact_shape_dtypes = 0;
    for (uint64_t reduction : R_VALUES) {
        for (uint64_t inner : K_VALUES) {
            // K=2 deep R is now covered by paired-row RParallel and no longer belongs in this sweep. Compact already
            // owns K=3 at R<=4, so keep its established phase-stream census starting at R=7. The newly-added odd K
            // values run across the complete shallow-R ladder so the benchmark can reveal both aligned-R and awkward-R
            // holes without assuming phase-stream is required. K=31 remains the larger shallow-R control.
            const bool newly_swept_odd_k =
                inner == 5 || inner == 9 || inner == 11 || inner == 13 || inner == 17 || inner == 19;
            const bool unresolved_cell =
                (inner == 3 && reduction >= 7) ||
                (inner == 7) ||
                newly_swept_odd_k ||
                (inner == 15) ||
                (inner == 31 && reduction <= 16);
            if (!unresolved_cell) {
                continue;
            }
            std::array<uint64_t, TARGET_BYTES.size()> seen_outer{};
            size_t seen_count = 0;
            for (uint64_t target_bytes : TARGET_BYTES) {
                const uint64_t outer = fp8RCooperativeOuterNearTarget(reduction, inner, target_bytes);
                bool duplicate = false;
                for (size_t index = 0; index < seen_count; ++index) {
                    duplicate = duplicate || seen_outer[index] == outer;
                }
                if (duplicate) {
                    continue;
                }
                seen_outer[seen_count++] = outer;
                RKSinglePassSweepCase benchmark_case{
                    fp8RCooperativeTargetLabel(target_bytes),
                    "o" + std::to_string(outer) + "_r" + std::to_string(reduction) + "_k" + std::to_string(inner),
                    outer,
                    reduction,
                    inner};
                for (DataType dtype : DTYPES) {
                    runFp8RCooperativeFullSpaceCase(
                        benchmark_case, dtype, cache_flush, stream, device_geometry, true, true);
                    ++exact_shape_dtypes;
                }
            }
        }
    }
    std::cout << "# rk_fp8_rcooperative_small_r_small_k_sweep_complete exact_shape_dtypes="
              << exact_shape_dtypes << '\n';
#else
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 RCooperative small-R/small-K sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void runFp8RCooperativeSmallRMediumKSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    // Reuse the established small-R nice/awkward ladder so this report lines up directly with the preceding
    // shallow/narrow census. The R=2/3/4 points are intentionally retained: medium K provides ample retained-width
    // parallelism, so they reveal whether production KParallel or a one-pass RCooperative layout already covers the
    // formerly difficult ultra-shallow reductions without relying on the K<=32 specialized kernels.
    constexpr std::array<uint64_t, 11> R_VALUES = {2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64};
    // Keep this exactly aligned with the analyzer's medium-K regime: packet-friendly controls at 64/128/256 and the
    // adjacent awkward/boundary widths that historically expose alignment and packetization holes.
    constexpr std::array<uint64_t, 7> K_VALUES = {63, 64, 127, 128, 129, 255, 256};
    constexpr std::array<uint64_t, 4> TARGET_BYTES = {
        64ULL * 1024ULL * 1024ULL,
        128ULL * 1024ULL * 1024ULL,
        256ULL * 1024ULL * 1024ULL,
        512ULL * 1024ULL * 1024ULL,
    };

    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    printFp8RCooperativeCoverageHeader(
        device_geometry,
        "rk_fp8_rcooperative_small_r_medium_k_sweep",
        "focused small-R x medium-K FP8 coverage census; production KParallel competes wherever its real selector is legal "
        "against generalized Complete RParallel O1/O2/O4/O8 x W1/W2/W4/W8 (mutually exclusive axes) p4/p8/p16, packet-adaptive Complete, and the existing "
        "FlatRows/Rotated layouts that the generalized family is intended to replace where possible");
    std::cout << "# focused_small_r_medium_k=1 "
                 "r_census=2,3,4,7,8,15,16,31,32,63,64 "
                 "k_census=63,64,127,128,129,255,256 "
                 "targets_mib=64,128,256,512 "
                 "candidates=production_kparallel,general_r_parallel_o1_o2_o4_o8_w1_w2_w4_w8_p4_p8_p16,packet_adaptive_complete,"
                 "flat_rows_p4_p8_p16,rotated_prefix_p4_p8_p16,rotated_exact_p16\n";

    uint64_t exact_shape_dtypes = 0;
    for (uint64_t reduction : R_VALUES) {
        for (uint64_t inner : K_VALUES) {
            std::array<uint64_t, TARGET_BYTES.size()> seen_outer{};
            size_t seen_count = 0;
            for (uint64_t target_bytes : TARGET_BYTES) {
                const uint64_t outer = fp8RCooperativeOuterNearTarget(reduction, inner, target_bytes);
                bool duplicate = false;
                for (size_t index = 0; index < seen_count; ++index) {
                    duplicate = duplicate || seen_outer[index] == outer;
                }
                if (duplicate) {
                    continue;
                }
                seen_outer[seen_count++] = outer;
                RKSinglePassSweepCase benchmark_case{
                    fp8RCooperativeTargetLabel(target_bytes),
                    "o" + std::to_string(outer) + "_r" + std::to_string(reduction) + "_k" + std::to_string(inner),
                    outer,
                    reduction,
                    inner};
                for (DataType dtype : DTYPES) {
                    // Use the broad family case rather than focused_holes_only so generalized medium-K RParallel
                    // competes directly with production KParallel and the existing Complete/FlatRows/Rotated baselines.
                    runFp8RCooperativeFullSpaceCase(
                        benchmark_case, dtype, cache_flush, stream, device_geometry, true, false);
                    ++exact_shape_dtypes;
                }
            }
        }
    }
    std::cout << "# rk_fp8_rcooperative_small_r_medium_k_sweep_complete exact_shape_dtypes="
              << exact_shape_dtypes << '\n';
#else
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 RCooperative small-R/medium-K sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}


void runFp8RCooperativeMediumRSmallKSweepImpl(Tensor& cache_flush, Stream& stream) {
#if THOR_CUB_ENABLE_FP8_TYPES
    constexpr std::array<DataType, 2> DTYPES = {DataType::FP8_E4M3, DataType::FP8_E5M2};
    // Use the analyzer's complete medium-R domain. Adjacent nice/awkward pairs make the R ownership transition
    // visible without inventing a benchmark-only boundary: 255/256, 511/512, and 1023/1024.
    constexpr std::array<uint64_t, 6> R_VALUES = {255, 256, 511, 512, 1023, 1024};
    // Use the complete analyzer-defined small-K domain rather than a hand-picked subset. This includes the power-of-two
    // packet-friendly controls plus every odd width already used by the shallow/narrow calibration work.
    constexpr std::array<uint64_t, 15> K_VALUES = {
        2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 31, 32};
    constexpr std::array<uint64_t, 4> TARGET_BYTES = {
        64ULL * 1024ULL * 1024ULL,
        128ULL * 1024ULL * 1024ULL,
        256ULL * 1024ULL * 1024ULL,
        512ULL * 1024ULL * 1024ULL,
    };

    const RKSinglePassDeviceGeometry device_geometry = querySinglePassDeviceGeometry();
    printFp8RCooperativeCoverageHeader(
        device_geometry,
        "rk_fp8_rcooperative_medium_r_small_k_sweep",
        "focused medium-R x small-K FP8 coverage census; generalized RParallel enables the same "
        "O1/O2/O4/O8 x W1/W2/W4/W8 Complete ownership plus balanced staged S2/S4/S8 with O1 x W1/W2/W4/W8, "
        "with LinearPackets calibrated at K=5/7/11/13/15/17/19/31 against PackedPlanes and FlatRows");
    std::cout << "# focused_medium_r_small_k=1 "
                 "r_census=255,256,511,512,1023,1024 "
                 "k_census=2,3,4,5,7,8,9,11,13,15,16,17,19,31,32 "
                 "targets_mib=64,128,256,512 "
                 "candidates=production_kparallel,general_r_parallel_complete_o1_o2_o4_o8_w1_w2_w4_w8_p4_p8_p16,"
                 "general_r_parallel_staged_s2_s4_s8_o1_w1_w2_w4_w8_p4_p8_p16,"
                 "general_r_parallel_linear_k5_k7_k11_k13_k15_k17_k19_k31_o1_w1_w2_w4_w8_s1_s2_s4_s8_p8_p16,"
                 "packet_adaptive_complete,compact_when_applicable,legacy_r_parallel_p4,direct_component_p4,"
                 "flat_rows_p4_p8_p16,rotated_prefix_p4_p8_p16,rotated_exact_p16\n";

    uint64_t exact_shape_dtypes = 0;
    for (uint64_t reduction : R_VALUES) {
        for (uint64_t inner : K_VALUES) {
            std::array<uint64_t, TARGET_BYTES.size()> seen_outer{};
            size_t seen_count = 0;
            for (uint64_t target_bytes : TARGET_BYTES) {
                const uint64_t outer = fp8RCooperativeOuterNearTarget(reduction, inner, target_bytes);
                bool duplicate = false;
                for (size_t index = 0; index < seen_count; ++index) {
                    duplicate = duplicate || seen_outer[index] == outer;
                }
                if (duplicate) {
                    continue;
                }
                seen_outer[seen_count++] = outer;
                RKSinglePassSweepCase benchmark_case{
                    fp8RCooperativeTargetLabel(target_bytes),
                    "o" + std::to_string(outer) + "_r" + std::to_string(reduction) + "_k" + std::to_string(inner),
                    outer,
                    reduction,
                    inner};
                for (DataType dtype : DTYPES) {
                    // Broad-family mode retains current production/staged baselines. The final two true flags are
                    // scoped to this census: expose generalized RParallel at K<=32 and its S2/S4/S8 staged topology.
                    runFp8RCooperativeFullSpaceCase(
                        benchmark_case, dtype, cache_flush, stream, device_geometry, true, false, true, true);
                    ++exact_shape_dtypes;
                }
            }
        }
    }
    std::cout << "# rk_fp8_rcooperative_medium_r_small_k_sweep_complete exact_shape_dtypes="
              << exact_shape_dtypes << '\n';
#else
    static_cast<void>(cache_flush);
    static_cast<void>(stream);
    throw std::logic_error("FP8 RCooperative medium-R/small-K sweep requires THOR_CUB_ENABLE_FP8_TYPES.");
#endif
}

const ReductionCandidateRegistrar<DenseRKFamilyCandidate> dense_rk_family_candidate_registrar;

}  // namespace

void runModernRKSinglePassSweep(Tensor& cache_flush, Stream& stream) {
    runModernRKSinglePassSweepImpl(cache_flush, stream);
}

void runModernRKSinglePassFocusSweep(Tensor& cache_flush, Stream& stream) {
    runModernRKSinglePassFocusSweepImpl(cache_flush, stream);
}

void runModernRKGapCensus(Tensor& cache_flush, Stream& stream) {
    runModernRKGapCensusImpl(cache_flush, stream);
}

void runRKNarrowLowPrecisionCalibration(Tensor& cache_flush, Stream& stream) {
    runRKNarrowLowPrecisionCalibrationImpl(cache_flush, stream);
}

void runRKLeanCompleteCalibration(Tensor& cache_flush, Stream& stream) {
    runRKLeanCompleteCalibrationImpl(cache_flush, stream);
}

void runRKKParallelStageCrossoverCalibration(Tensor& cache_flush, Stream& stream) {
    runRKKParallelStageCrossoverCalibrationImpl(cache_flush, stream);
}

void runRKKParallelStagedGeometryCalibration(Tensor& cache_flush, Stream& stream) {
    runRKKParallelStagedGeometryCalibrationImpl(cache_flush, stream);
}

void runFp8KParallelPacketABCensus(Tensor& cache_flush, Stream& stream) {
    runFp8KParallelPacketABCensusImpl(cache_flush, stream);
}

void runFp8KParallelSmallPacketSweep(Tensor& cache_flush, Stream& stream) {
    runFp8KParallelSmallPacketSweepImpl(cache_flush, stream);
}

void runFp8KParallelStraddledSweep(Tensor& cache_flush, Stream& stream) {
    runFp8KParallelStraddledSweepImpl(cache_flush, stream);
}

void runFp8KParallelCompleteLayoutSweep(Tensor& cache_flush, Stream& stream) {
    runFp8KParallelCompleteLayoutSweepImpl(cache_flush, stream);
}

void runFp8KParallelStagedLayoutSweep(Tensor& cache_flush, Stream& stream) {
    runFp8KParallelStagedLayoutSweepImpl(cache_flush, stream);
}

void runFp8RCooperativeSinglePassSweep(Tensor& cache_flush, Stream& stream) {
    runFp8RCooperativeSinglePassSweepImpl(cache_flush, stream);
}

void runFp8RCooperativeFullSpaceSweep(Tensor& cache_flush, Stream& stream) {
    runFp8RCooperativeFullSpaceSweepImpl(cache_flush, stream);
}

void runFp8RCooperativeSmallRSmallKSweep(Tensor& cache_flush, Stream& stream) {
    runFp8RCooperativeSmallRSmallKSweepImpl(cache_flush, stream);
}

void runFp8RCooperativeSmallRMediumKSweep(Tensor& cache_flush, Stream& stream) {
    runFp8RCooperativeSmallRMediumKSweepImpl(cache_flush, stream);
}

void runFp8RCooperativeMediumRSmallKSweep(Tensor& cache_flush, Stream& stream) {
    runFp8RCooperativeMediumRSmallKSweepImpl(cache_flush, stream);
}

void runRKKParallelEndToEndCrossoverCalibration(Tensor& cache_flush, Stream& stream) {
    runRKKParallelEndToEndCrossoverCalibrationImpl(cache_flush, stream);
}

void runRKRealisticEndToEndProgressCalibration(Tensor& cache_flush, Stream& stream) {
    runRKRealisticEndToEndProgressCalibrationImpl(cache_flush, stream);
}

void runRKFamilyBoundaryCalibration(Tensor& cache_flush, Stream& stream) {
    runRKFamilyBoundaryCalibrationImpl(cache_flush, stream);
}

}  // namespace ThorImplementation::CubReductionBenchmarking
