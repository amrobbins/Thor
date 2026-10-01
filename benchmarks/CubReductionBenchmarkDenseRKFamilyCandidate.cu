#include "benchmarks/CubReductionBenchmarkCandidate.h"
#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionOperation.cuh"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"

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
    if ((packet_bytes != 4 && packet_bytes != 8 && packet_bytes != 16)
        || packet_bytes < sizeof(InputT) || packet_bytes % sizeof(InputT) != 0) {
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
    const uint64_t items_per_lane = dtype == DataType::FP32 ? 4ULL : 8ULL;
    return singlePassCeilDiv(geometry.inner_size, 32ULL * items_per_lane);
}

[[nodiscard]] bool singlePassAlignedRCoopSupports(const CubReductionGeometry& geometry, DataType dtype) {
    const uint64_t items_per_lane = dtype == DataType::FP32 ? 4ULL : 8ULL;
    return geometry.inner_size % items_per_lane == 0;
}

[[nodiscard]] uint64_t singlePassRotatedRCoopTiles(const CubReductionGeometry& geometry, DataType dtype) {
    const uint64_t items_per_lane = dtype == DataType::FP32 ? 4ULL : 8ULL;
    const uint64_t logical_tile = 32ULL * items_per_lane - (items_per_lane - 1ULL);
    return singlePassCeilDiv(geometry.inner_size, logical_tile);
}

[[nodiscard]] size_t singlePassPacketAdaptivePacketBytes(const CubReductionGeometry& geometry, DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return chooseAdaptiveCooperativeTiledPacketBytes<__half>(geometry);
        case DataType::BF16:
            return chooseAdaptiveCooperativeTiledPacketBytes<__nv_bfloat16>(geometry);
        case DataType::FP32:
            return chooseAdaptiveCooperativeTiledPacketBytes<float>(geometry);
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
            return querySinglePassPacketAdaptiveOccupancyForWarps<InputT, PacketBytes, 32, NaturallyAlignedRows>();
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

[[nodiscard]] RKSinglePassTiming timeRKSinglePass(Tensor& cache_flush, Stream& stream, const auto& run) {
    for (int i = 0; i < RK_SINGLE_PASS_WARMUPS; ++i) {
        run();
    }
    stream.synchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkSinglePassCuda(cudaEventCreate(&start), "cudaEventCreate(rk_single_pass_start)");
    checkSinglePassCuda(cudaEventCreate(&stop), "cudaEventCreate(rk_single_pass_stop)");
    std::array<double, RK_SINGLE_PASS_SAMPLES> samples{};
    for (int sample = 0; sample < RK_SINGLE_PASS_SAMPLES; ++sample) {
        checkSinglePassCuda(cudaMemsetAsync(cache_flush.getMemPtr<void>(),
                                            0x41 + sample,
                                            cache_flush.getArraySizeInBytes(),
                                            stream.getStream()),
                            "cudaMemsetAsync(rk_single_pass_cache_flush)");
        stream.synchronize();
        checkSinglePassCuda(cudaEventRecord(start, stream.getStream()), "cudaEventRecord(rk_single_pass_start)");
        run();
        checkSinglePassCuda(cudaEventRecord(stop, stream.getStream()), "cudaEventRecord(rk_single_pass_stop)");
        checkSinglePassCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(rk_single_pass_stop)");
        float elapsed_ms = 0.0f;
        checkSinglePassCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime(rk_single_pass)");
        samples[static_cast<size_t>(sample)] = elapsed_ms;
    }
    checkSinglePassCuda(cudaEventDestroy(start), "cudaEventDestroy(rk_single_pass_start)");
    checkSinglePassCuda(cudaEventDestroy(stop), "cudaEventDestroy(rk_single_pass_stop)");
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
                                std::string_view implementation) {
    const double expected = static_cast<double>(geometry.reduction_size) * RK_SINGLE_PASS_FILL_VALUE;
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
                               std::string_view implementation) {
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
                             static_cast<double>(first_rows) * RK_SINGLE_PASS_FILL_VALUE,
                             stream,
                             implementation);
    validateSinglePassSample(output,
                             output.getTotalNumElements() - 1,
                             static_cast<double>(last_rows) * RK_SINGLE_PASS_FILL_VALUE,
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
