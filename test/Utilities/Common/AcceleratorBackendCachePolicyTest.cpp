#include "Utilities/Common/AcceleratorBackendCachePolicy.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

struct ExampleSelectionRecipe : AcceleratorBackendSelectionRecipeTag {
    int algorithm = 0;
    uint64_t workspace_bytes = 0;
};

struct ExampleLocalExecutionState : AcceleratorBackendLocalExecutionStateTag {
    ExampleLocalExecutionState() = default;
    ExampleLocalExecutionState(ExampleLocalExecutionState&&) noexcept = default;
    ExampleLocalExecutionState& operator=(ExampleLocalExecutionState&&) noexcept = default;
};

optional<filesystem::path> findThorSourceRootFrom(filesystem::path current) {
    while (!current.empty()) {
        if (filesystem::exists(current / "CMakeLists.txt") && filesystem::exists(current / "Utilities" / "TensorOperations")) {
            return current;
        }
        const filesystem::path parent = current.parent_path();
        if (parent == current)
            break;
        current = parent;
    }
    return nullopt;
}

filesystem::path findThorSourceRoot() {
    if (const optional<filesystem::path> fromSource =
            findThorSourceRootFrom(filesystem::absolute(filesystem::path(__FILE__)).parent_path());
        fromSource.has_value()) {
        return fromSource.value();
    }
    if (const optional<filesystem::path> fromWorkingDirectory = findThorSourceRootFrom(filesystem::current_path());
        fromWorkingDirectory.has_value()) {
        return fromWorkingDirectory.value();
    }
    throw runtime_error("Could not locate Thor source root from source path or working directory.");
}

string readTextFile(const filesystem::path& path) {
    ifstream input(path);
    if (!input)
        throw runtime_error("Could not read " + path.string());
    ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

bool isAuditedSourceFile(const filesystem::path& path) {
    const string extension = path.extension().string();
    return extension == ".h" || extension == ".hpp" || extension == ".cpp" || extension == ".cc" || extension == ".cu" ||
           extension == ".cuh";
}

set<string> filesContaining(const filesystem::path& root, string_view needle) {
    set<string> matches;
    for (const string_view sourceRoot : {string_view("Utilities"), string_view("DeepLearning")}) {
        for (const auto& entry : filesystem::recursive_directory_iterator(root / sourceRoot)) {
            if (!entry.is_regular_file() || !isAuditedSourceFile(entry.path()))
                continue;
            if (readTextFile(entry.path()).find(needle) != string::npos) {
                matches.insert(filesystem::relative(entry.path(), root).generic_string());
            }
        }
    }
    return matches;
}

set<string> filesMatching(const filesystem::path& root, const regex& pattern) {
    set<string> matches;
    for (const string_view sourceRoot : {string_view("Utilities"), string_view("DeepLearning")}) {
        for (const auto& entry : filesystem::recursive_directory_iterator(root / sourceRoot)) {
            if (!entry.is_regular_file() || !isAuditedSourceFile(entry.path()))
                continue;
            if (regex_search(readTextFile(entry.path()), pattern)) {
                matches.insert(filesystem::relative(entry.path(), root).generic_string());
            }
        }
    }
    return matches;
}

void expectNoProductionSourceMatches(const filesystem::path& root,
                                     const regex& pattern,
                                     string_view stateDescription) {
    const set<string> matches = filesMatching(root, pattern);
    EXPECT_TRUE(matches.empty()) << "Forbidden accelerator backend execution state found for " << stateDescription
                                 << ": " << ([&] {
                                        ostringstream out;
                                        bool first = true;
                                        for (const string& match : matches) {
                                            if (!first)
                                                out << ", ";
                                            out << match;
                                            first = false;
                                        }
                                        return out.str();
                                    })();
}

void expectOnlyListedCacheSites(const filesystem::path& root,
                                string_view needle,
                                const set<string>& expectedFiles,
                                string_view stateDescription) {
    const set<string> actualFiles = filesContaining(root, needle);
    EXPECT_EQ(actualFiles, expectedFiles)
        << "Accelerator backend cache policy sites changed for " << stateDescription
        << ". Global selection-cache inventories are explicit; backend execution state must never become globally cached.";
}

}  // namespace

static_assert(isAcceleratorBackendSelectionRecipeV<ExampleSelectionRecipe>);
static_assert(AcceleratorBackendSelectionRecipe<ExampleSelectionRecipe>);
static_assert(!isAcceleratorBackendLocalExecutionStateV<ExampleSelectionRecipe>);
static_assert(is_copy_constructible_v<ExampleSelectionRecipe>);
static_assert(is_move_constructible_v<ExampleSelectionRecipe>);

static_assert(isAcceleratorBackendLocalExecutionStateV<ExampleLocalExecutionState>);
static_assert(AcceleratorBackendLocalExecutionState<ExampleLocalExecutionState>);
static_assert(!isAcceleratorBackendSelectionRecipeV<ExampleLocalExecutionState>);
static_assert(!is_copy_constructible_v<ExampleLocalExecutionState>);
static_assert(!is_copy_assignable_v<ExampleLocalExecutionState>);
static_assert(is_move_constructible_v<ExampleLocalExecutionState>);
static_assert(is_move_assignable_v<ExampleLocalExecutionState>);

TEST(AcceleratorBackendCachePolicy, GlobalBackendExecutionStateCachesAreForbidden) {
    const filesystem::path root = findThorSourceRoot();

    // C12 deletes the transitional cached-executable model entirely. Split the
    // spelling so this regression test does not itself keep the retired symbol
    // alive in repository-wide source audits.
    const string retiredCudnnCacheType = string("CudnnCached") + "ExecutionPlan<";
    EXPECT_TRUE(filesContaining(root, retiredCudnnCacheType).empty());

    // Direct cache-container audits catch descriptor/graph/executable values even
    // when a future implementation gives the cache a new variable name. Local
    // operation-owned maps are allowed; process-style cache containers are not.
    const regex cudnnGraphOrDescriptorCache(
        R"((?:LruCacheThreadSafe|[A-Za-z_][A-Za-z0-9_]*Cache)\s*<[^;{}]{0,2048}(?:fe::graph::Graph|cudnn(?:Tensor|Filter|Convolution|Backend)Descriptor_t|Cudnn[A-Za-z0-9_]*ExecutablePlan|BuiltGraph|BuiltSoftmax|BuiltConvolution)[^;{}]{0,2048}>)");
    expectNoProductionSourceMatches(root, cudnnGraphOrDescriptorCache, "globally cached cuDNN graph/descriptor/executable state");

    const regex cublasKernelCache(
        R"((?:LruCacheThreadSafe|[A-Za-z_][A-Za-z0-9_]*Cache)\s*<[^;{}]{0,2048}\bCublasKernel\b[^;{}]{0,2048}>)");
    expectNoProductionSourceMatches(root, cublasKernelCache, "globally cached descriptor-bearing CublasKernel state");

    const regex staticCudnnExecutionState(
        R"(\bstatic\s+(?:(?:std::)?(?:shared_ptr|unique_ptr)\s*<\s*)?(?:fe::graph::Graph|cudnn(?:Tensor|Filter|Convolution|Backend)Descriptor_t)\s*>?\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:=|;))");
    expectNoProductionSourceMatches(root, staticCudnnExecutionState, "static cuDNN graph/descriptor objects");

    const regex staticCublasKernelContainer(
        R"(\bstatic\s+(?:std::)?(?:unordered_map|map)\s*<[^;{}]{0,2048}\bCublasKernel\b[^;{}]{0,2048}>\s+[A-Za-z_][A-Za-z0-9_]*)");
    expectNoProductionSourceMatches(root, staticCublasKernelContainer, "static CublasKernel associative containers");

    // Backend handles are execution-domain state too. Thor Streams own their
    // cuDNN/cuBLAS/cuBLASLt handles; raw handles must not be parked in global
    // or process-style containers as an alternate ownership mechanism.
    const regex rawBackendHandleContainer(
        R"((?:std::)?(?:unordered_map|map|vector|deque|list)\s*<[^;{}]{0,2048}\b(?:cudnnHandle_t|cublasHandle_t|cublasLtHandle_t)\b[^;{}]{0,2048}>\s+[A-Za-z_][A-Za-z0-9_]*)");
    expectNoProductionSourceMatches(root, rawBackendHandleContainer, "raw backend-handle containers");

    const regex staticRawBackendHandle(
        R"(\bstatic\s+(?:(?:std::)?optional\s*<\s*)?(?:cudnnHandle_t|cublasHandle_t|cublasLtHandle_t)\s*>?\s+[A-Za-z_][A-Za-z0-9_]*)");
    expectNoProductionSourceMatches(root, staticRawBackendHandle, "static raw backend handles");

    // Also reject the historical named cache sites so accidental resurrection is
    // diagnosed clearly even if their declaration shape changes.
    EXPECT_TRUE(filesContaining(root, "builtMatmulCache").empty());
    EXPECT_TRUE(filesContaining(root, "builtSoftmaxCache").empty());
    EXPECT_TRUE(filesContaining(root, "LruCacheThreadSafe<ConvolutionKernelRequirement, cudnnConvolution").empty());
}


TEST(AcceleratorBackendCachePolicy, BackendHandlesRemainStreamOwned) {
    const filesystem::path root = findThorSourceRoot();
    const string helperHeader = readTextFile(root / "Utilities/Common/CudnnHelper.h");
    const string streamSource = readTextFile(root / "Utilities/Common/Stream.cpp");

    EXPECT_EQ(helperHeader.find("getCudnnHandle"), string::npos);
    EXPECT_EQ(helperHeader.find("cudnnHandle_t"), string::npos);
    EXPECT_FALSE(filesystem::exists(root / "Utilities/Common/CudnnHelper.cpp"));

    EXPECT_NE(streamSource.find("optional<cudnnHandle_t> cudnnHandle"), string::npos);
    EXPECT_NE(streamSource.find("optional<cublasHandle_t> cublasHandle"), string::npos);
    EXPECT_NE(streamSource.find("optional<cublasLtHandle_t> cublasLtHandle"), string::npos);
    EXPECT_NE(streamSource.find("cudnnSetStream(handle, state->cudaStream)"), string::npos);
    EXPECT_NE(streamSource.find("cublasSetStream(handle, state->cudaStream)"), string::npos);
}

TEST(AcceleratorBackendCachePolicy, MigratedLayerNormGlobalStateIsSelectionOnly) {
    const filesystem::path root = findThorSourceRoot();
    const string source = readTextFile(root / "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.cpp");

    EXPECT_NE(source.find("CudnnFrontendPlanSelectionCache<string> selections"), string::npos);
    EXPECT_EQ(source.find("unordered_map<string, BuiltGraph>"), string::npos);
    EXPECT_EQ(source.find("static shared_ptr<fe::graph::Graph>"), string::npos);
}

TEST(AcceleratorBackendCachePolicy, MigratedInstanceNormGlobalStateIsSelectionOnly) {
    const filesystem::path root = findThorSourceRoot();
    const string source = readTextFile(root / "Utilities/TensorOperations/DeepLearning/CudnnInstanceNorm.cpp");
    const string layerHeader = readTextFile(root / "DeepLearning/Implementation/Layers/NeuralNetwork/InstanceNorm.h");
    const string layerSource = readTextFile(root / "DeepLearning/Implementation/Layers/NeuralNetwork/InstanceNorm.cpp");

    EXPECT_NE(source.find("CudnnFrontendPlanSelectionCache<string> selections"), string::npos);
    EXPECT_EQ(source.find("unordered_map<string, BuiltGraph>"), string::npos);
    EXPECT_EQ(source.find("static shared_ptr<fe::graph::Graph>"), string::npos);

    EXPECT_NE(layerHeader.find("vector<std::optional<CudnnInstanceNormExecutablePlan>> forwardPlans"), string::npos);
    EXPECT_NE(layerHeader.find("vector<std::optional<CudnnInstanceNormExecutablePlan>> backwardPlans"), string::npos);
    EXPECT_NE(layerSource.find("prepareForward(forwardDescriptor, streams[i])"), string::npos);
    EXPECT_NE(layerSource.find("prepareBackward(backwardDescriptor, streams[i])"), string::npos);
    EXPECT_EQ(layerSource.find("forwardWorkspaceSizeInBytes"), string::npos);
    EXPECT_EQ(layerSource.find("backwardWorkspaceSizeInBytes"), string::npos);
}

TEST(AcceleratorBackendCachePolicy, MigratedAdaptiveLayerNormGlobalStateIsSelectionOnly) {
    const filesystem::path root = findThorSourceRoot();
    const string source = readTextFile(root / "Utilities/TensorOperations/DeepLearning/CudnnAdaptiveLayerNorm.cpp");
    const string layerHeader = readTextFile(root / "DeepLearning/Implementation/Layers/NeuralNetwork/AdaptiveLayerNorm.h");
    const string layerSource = readTextFile(root / "DeepLearning/Implementation/Layers/NeuralNetwork/AdaptiveLayerNorm.cpp");

    EXPECT_NE(source.find("CudnnFrontendPlanSelectionCache<string> selections"), string::npos);
    EXPECT_EQ(source.find("unordered_map<string, BuiltGraph>"), string::npos);
    EXPECT_EQ(source.find("static shared_ptr<fe::graph::Graph>"), string::npos);

    EXPECT_NE(layerHeader.find("std::optional<CudnnAdaptiveLayerNormExecutablePlan> forwardPlan"), string::npos);
    EXPECT_NE(layerHeader.find("std::optional<CudnnAdaptiveLayerNormExecutablePlan> backwardPlan"), string::npos);
    EXPECT_NE(layerSource.find("prepareForward(forwardDescriptor, computeStream())"), string::npos);
    EXPECT_NE(layerSource.find("prepareBackward(backwardDescriptor, computeStream())"), string::npos);
    EXPECT_EQ(layerSource.find("forwardWorkspaceSizeInBytes"), string::npos);
    EXPECT_EQ(layerSource.find("backwardWorkspaceSizeInBytes"), string::npos);
}

TEST(AcceleratorBackendCachePolicy, MigratedRmsNormGlobalStateIsSelectionOnly) {
    const filesystem::path root = findThorSourceRoot();
    const string source = readTextFile(root / "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.cpp");
    const string layerHeader = readTextFile(root / "DeepLearning/Implementation/Layers/NeuralNetwork/RMSNorm.h");
    const string layerSource = readTextFile(root / "DeepLearning/Implementation/Layers/NeuralNetwork/RMSNorm.cpp");
    const string stampedHeader = readTextFile(root / "Utilities/Expression/StampedEquation.h");
    const string stampedSource = readTextFile(root / "Utilities/Expression/StampedEquation.cpp");

    EXPECT_NE(source.find("CudnnFrontendPlanSelectionCache<string> selections"), string::npos);
    EXPECT_EQ(source.find("unordered_map<string, BuiltGraph>"), string::npos);
    EXPECT_EQ(source.find("static shared_ptr<fe::graph::Graph>"), string::npos);

    EXPECT_NE(layerHeader.find("vector<std::optional<CudnnRmsNormExecutablePlan>> forwardExecutablePlans"), string::npos);
    EXPECT_NE(layerHeader.find("vector<std::optional<CudnnRmsNormExecutablePlan>> backwardExecutablePlans"), string::npos);
    EXPECT_NE(layerSource.find("prepareForward(forwardDescriptor, streams[i])"), string::npos);
    EXPECT_NE(layerSource.find("prepareBackward(backwardDescriptor, streams[i])"), string::npos);

    EXPECT_NE(stampedHeader.find("forward_executable_plans"), string::npos);
    EXPECT_NE(stampedHeader.find("backward_executable_plans"), string::npos);
    EXPECT_NE(stampedHeader.find("fallback_forward_executable_plans"), string::npos);
    EXPECT_NE(stampedSource.find("prepareForwardExecutableFamily"), string::npos);
    EXPECT_NE(stampedSource.find("prepareBackwardExecutableFamilies"), string::npos);

    // Runtime must select only from already-prepared operation-local families.
    const size_t forwardRun = stampedSource.find("void StampedRmsNorm::runOn(Stream& run_stream) const");
    const size_t forwardEnd = stampedSource.find("void StampedRmsNorm::retainForwardStateForBackward()", forwardRun);
    const size_t backwardRun = stampedSource.find("void StampedRmsNormBackward::runOn(Stream& run_stream) const");
    const size_t backwardEnd = stampedSource.find("StampedEmbeddingLookup::", backwardRun);
    ASSERT_NE(forwardRun, string::npos);
    ASSERT_NE(forwardEnd, string::npos);
    ASSERT_NE(backwardRun, string::npos);
    ASSERT_NE(backwardEnd, string::npos);
    const string forwardBody = stampedSource.substr(forwardRun, forwardEnd - forwardRun);
    const string backwardBody = stampedSource.substr(backwardRun, backwardEnd - backwardRun);
    EXPECT_EQ(forwardBody.find("prepareForward("), string::npos);
    EXPECT_EQ(forwardBody.find("forwardWorkspaceSizeInBytes("), string::npos);
    EXPECT_EQ(backwardBody.find("prepareForward("), string::npos);
    EXPECT_EQ(backwardBody.find("prepareBackward("), string::npos);
    EXPECT_EQ(backwardBody.find("backwardWorkspaceSizeInBytes("), string::npos);
}

TEST(AcceleratorBackendCachePolicy, MigratedAttentionGlobalStateIsSelectionOnly) {
    const filesystem::path root = findThorSourceRoot();
    const string source = readTextFile(root / "Utilities/TensorOperations/GpuAttention/CudnnAttention.cpp");
    const string header = readTextFile(root / "Utilities/TensorOperations/GpuAttention/CudnnAttention.h");
    const string stampedHeader = readTextFile(root / "Utilities/Expression/StampedEquation.h");
    const string stampedSource = readTextFile(root / "Utilities/Expression/StampedEquation.cpp");

    EXPECT_NE(source.find("CudnnFrontendPlanSelectionCache<string> selections"), string::npos);
    EXPECT_EQ(source.find("unordered_map<string, BuiltGraph>"), string::npos);
    EXPECT_EQ(source.find("static shared_ptr<fe::graph::Graph>"), string::npos);
    EXPECT_NE(header.find("class CudnnAttentionExecutablePlan final"), string::npos);

    // Attention selection is intentionally more expensive than the normalization
    // wrappers: Mode B ranks the candidates, Thor builds at most the first 16 successfully buildable
    // plans, and cuDNN's own autotuner chooses the measured winner.  The
    // winner is cached only as immutable serialized replay bytes because autotune
    // reorders execution plans independently of the heuristic config vector.
    EXPECT_NE(source.find("kAutotuneCandidateLimit = 16"), string::npos);
    EXPECT_NE(source.find("create_execution_plans({fe::HeurMode_t::B})"), string::npos);
    EXPECT_EQ(source.find("create_execution_plans({fe::HeurMode_t::A})"), string::npos);
    EXPECT_NE(source.find("builtCandidates < kAutotuneCandidateLimit"), string::npos);
    EXPECT_NE(source.find("graph->autotune(stream.getCudnnHandle(), buffers.pack, workspacePtr)"), string::npos);
    EXPECT_NE(source.find("cudnnFrontendSelectedSerializedPlanSelection(*graph, operation)"), string::npos);

    EXPECT_NE(stampedHeader.find("std::optional<CudnnAttentionExecutablePlan> forward_plan"), string::npos);
    EXPECT_NE(stampedHeader.find("std::optional<CudnnAttentionExecutablePlan> backward_plan"), string::npos);
    EXPECT_NE(stampedHeader.find("std::optional<CudnnAttentionExecutablePlan> fallback_forward_plan"), string::npos);
    EXPECT_NE(stampedSource.find("prepareForward(descriptor, workspaceArgs, stream)"), string::npos);
    EXPECT_NE(stampedSource.find("prepareBackward(descriptor, backwardWorkspaceArgs, stream)"), string::npos);

    // Runtime may bind tensors and prepare ragged metadata, but it must never
    // consult selection state or construct/replay a Frontend executable.
    const size_t forwardRun = stampedSource.find("void StampedAttention::runOn(Stream& run_stream) const");
    const size_t forwardEnd = stampedSource.find("bool StampedAttention::canProvideForwardStateFor", forwardRun);
    const size_t backwardRun = stampedSource.find("void StampedAttentionBackward::runOn(Stream& run_stream) const");
    const size_t backwardEnd = stampedSource.find("StampedAttentionBackward::StampedAttentionBackward", backwardRun);
    ASSERT_NE(forwardRun, string::npos);
    ASSERT_NE(forwardEnd, string::npos);
    ASSERT_NE(backwardRun, string::npos);
    ASSERT_NE(backwardEnd, string::npos);
    const string forwardBody = stampedSource.substr(forwardRun, forwardEnd - forwardRun);
    const string backwardBody = stampedSource.substr(backwardRun, backwardEnd - backwardRun);
    EXPECT_EQ(forwardBody.find("prepareForward("), string::npos);
    EXPECT_EQ(forwardBody.find("forwardWorkspaceSizeInBytes("), string::npos);
    EXPECT_EQ(backwardBody.find("prepareForward("), string::npos);
    EXPECT_EQ(backwardBody.find("prepareBackward("), string::npos);
    EXPECT_EQ(backwardBody.find("backwardWorkspaceSizeInBytes("), string::npos);
}

TEST(AcceleratorBackendCachePolicy, C11ConvolutionGlobalStateIsValidatedSelectionOnly) {
    const filesystem::path root = findThorSourceRoot();
    const string header = readTextFile(root / "Utilities/Expression/StampedEquation.h");
    const string source = readTextFile(root / "Utilities/Expression/StampedEquation.cpp");

    // The only process-global convolution value is the immutable C2 selection
    // recipe.  Placement-time scratch graphs are local variables and completed
    // BuiltConvolution objects retain only their own move-only executable.
    EXPECT_NE(source.find("static CudnnFrontendPlanSelectionCache<std::string> selections"), string::npos);
    EXPECT_EQ(header.find("frontend_autotune_graph"), string::npos);
    EXPECT_EQ(source.find("static shared_ptr<fe::graph::Graph>"), string::npos);

    // Publication happens only through the selector lambda, whose autotune path
    // returns after exact replay plus the independent convolution oracle passed.
    EXPECT_NE(source.find("frontendConvolutionSelectionCache().getOrSelect(selection_cache_key"), string::npos);
    EXPECT_NE(source.find("return selected_plan->selection();"), string::npos);
    EXPECT_NE(source.find("validateFrontendConvolutionCandidate(candidate_plan"), string::npos);
    EXPECT_NE(source.find("appendFrontendConvolutionCacheVector(out, \"stride\", strides)"), string::npos);
    EXPECT_NE(source.find("appendFrontendConvolutionCacheVector(out, \"dilation\", dilations)"), string::npos);
    EXPECT_NE(source.find("appendFrontendConvolutionCacheVector(out, \"pre\", pre_padding)"), string::npos);
    EXPECT_NE(source.find("appendFrontendConvolutionCacheVector(out, \"post\", post_padding)"), string::npos);

    // Cache hits and misses both terminate in a fresh stamp-local replay; the
    // executable object itself can never be a cache value.
    EXPECT_NE(source.find("replayCudnnFrontendExecutablePlan(graph_factory, selection"), string::npos);
    EXPECT_NE(header.find("struct BuiltConvolution final : AcceleratorBackendLocalExecutionStateTag"), string::npos);
    EXPECT_NE(header.find("std::optional<CudnnFrontendExecutablePlan> frontend_plan"), string::npos);
    EXPECT_EQ(source.find("CudnnFrontendPlanSelectionCache<std::string, CudnnFrontendExecutablePlan"), string::npos);
}



TEST(AcceleratorBackendCachePolicy, RmsNormRhtAmaxCacheNamesResolvedCudaKernelMetadataPrecisely) {
    const filesystem::path root = findThorSourceRoot();
    const string header = readTextFile(root / "Utilities/TensorOperations/DeepLearning/CudnnRmsNormRhtAbsMax.h");
    const string source = readTextFile(root / "Utilities/TensorOperations/DeepLearning/CudnnRmsNormRhtAbsMax.cu");

    EXPECT_NE(header.find("cachedResolvedKernelCount() const"), string::npos);
    EXPECT_EQ(header.find("cachedGraphCount"), string::npos);
    EXPECT_NE(source.find("struct ResolvedRhtAmaxKernel final : AcceleratorBackendSelectionRecipeTag"), string::npos);
    EXPECT_NE(source.find("class ResolvedRhtAmaxKernelCache"), string::npos);
    EXPECT_NE(source.find("unordered_map<string, ResolvedRhtAmaxKernel> resolved_kernels"), string::npos);
    EXPECT_EQ(source.find("RhtAmaxPlanCache"), string::npos);
}

TEST(AcceleratorBackendCachePolicy, KnownGlobalSelectionCachesRemainSelectionOnly) {
    const filesystem::path root = findThorSourceRoot();

    expectOnlyListedCacheSites(root,
                                 "CudnnFrontendPlanSelectionCache<",
                                 {"Utilities/Expression/StampedEquation.cpp",
                                  "Utilities/TensorOperations/DeepLearning/CudnnAdaptiveLayerNorm.cpp",
                                  "Utilities/TensorOperations/DeepLearning/CudnnInstanceNorm.cpp",
                                  "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.cpp",
                                  "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.cpp",
                                  "Utilities/TensorOperations/GpuAttention/CudnnAttention.cpp"},
                                 "cuDNN Frontend immutable plan-selection caches");
    expectOnlyListedCacheSites(root,
                                 "LruCacheThreadSafe<CublasKernelRequirement, CublasKernelSelection>",
                                 {"Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"},
                                 "measured ordinary cuBLASLt kernel-selection cache");
    expectOnlyListedCacheSites(root,
                                 "LruCacheThreadSafe<CublasKernelRequirement, cublasLtMatmulAlgo_t>",
                                 {"Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"},
                                 "known-good cuBLASLt heuristic algorithm-selection cache");
    expectOnlyListedCacheSites(root,
                                 "LruCacheThreadSafe<std::string, CublasMatrixMultiply::LtMatmulAlgorithmSelection>",
                                 {"Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.cpp"},
                                 "measured cuBLASLt epilogue algorithm-selection cache");
}

TEST(AcceleratorBackendCachePolicy, C8CublasKernelExecutionStateIsMoveOnlyAndSelectionOnlyGlobally) {
    const filesystem::path root = findThorSourceRoot();
    const string kernelHeader =
        readTextFile(root / "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h");
    const string optionsHeader =
        readTextFile(root / "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelOptions.h");
    const string matrixHeader =
        readTextFile(root / "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h");
    const string matrixSource =
        readTextFile(root / "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.cpp");

    EXPECT_NE(kernelHeader.find("class CublasKernel final : public AcceleratorBackendLocalExecutionStateTag"), string::npos);
    EXPECT_NE(kernelHeader.find("CublasKernel(const CublasKernel &other) = delete"), string::npos);
    EXPECT_NE(kernelHeader.find("std::unique_ptr<State> state"), string::npos);
    EXPECT_EQ(kernelHeader.find("std::shared_ptr<State> state"), string::npos);

    const size_t selectionBegin = optionsHeader.find("struct CublasKernelSelection : AcceleratorBackendSelectionRecipeTag");
    ASSERT_NE(selectionBegin, string::npos);
    const size_t selectionEnd = optionsHeader.find("};", selectionBegin);
    ASSERT_NE(selectionEnd, string::npos);
    const string selectionBody = optionsHeader.substr(selectionBegin, selectionEnd - selectionBegin);
    EXPECT_EQ(selectionBody.find("RunStats"), string::npos);
    EXPECT_NE(matrixHeader.find("LruCacheThreadSafe<CublasKernelRequirement, CublasKernelSelection> optimalKernelSelections"),
              string::npos);
    EXPECT_EQ(matrixHeader.find("LruCacheThreadSafe<CublasKernelRequirement, CublasKernel>"), string::npos);
    EXPECT_NE(matrixHeader.find("CublasKernel materializeSelectedGemmKernel("), string::npos);
    EXPECT_EQ(matrixHeader.find("getCachedGemmKernel"), string::npos);

    EXPECT_NE(matrixSource.find("const CublasKernelSelection bestSelection = bestKernel.getSelectionRecipe()"), string::npos);
    EXPECT_NE(matrixSource.find("return materializeCublasKernel(cublasKernelRequirement, optimalSelection.value())"), string::npos);
}


TEST(AcceleratorBackendCachePolicy, C9ExpressionAndEpilogueMatmulStateIsOperationLocal) {
    const filesystem::path root = findThorSourceRoot();
    const string matrixHeader =
        readTextFile(root / "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h");
    const string matrixSource =
        readTextFile(root / "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.cpp");
    const string stampedHeader = readTextFile(root / "Utilities/Expression/StampedEquation.h");
    const string stampedSource = readTextFile(root / "Utilities/Expression/StampedEquation.cpp");

    EXPECT_EQ(stampedSource.find("builtMatmulCache"), string::npos);
    EXPECT_NE(stampedHeader.find("struct BuiltMatmul : AcceleratorBackendLocalExecutionStateTag"), string::npos);
    EXPECT_NE(stampedHeader.find("const std::unique_ptr<BuiltMatmul> built_matmul"), string::npos);
    EXPECT_EQ(stampedHeader.find("std::shared_ptr<BuiltMatmul>"), string::npos);
    EXPECT_NE(stampedHeader.find("std::unique_ptr<CublasMatrixMultiply::LtMatmulPlan> epilogue_plan"), string::npos);

    EXPECT_NE(matrixHeader.find("struct LtMatmulAlgorithmSelection : AcceleratorBackendSelectionRecipeTag"), string::npos);
    EXPECT_NE(matrixHeader.find("struct LtMatmulPlan : AcceleratorBackendLocalExecutionStateTag"), string::npos);
    EXPECT_NE(matrixHeader.find("LtMatmulPlan(const LtMatmulPlan &) = delete"), string::npos);
    EXPECT_NE(matrixSource.find("LruCacheThreadSafe<std::string, CublasMatrixMultiply::LtMatmulAlgorithmSelection> selections"),
              string::npos);
    EXPECT_EQ(matrixSource.find("LruCacheThreadSafe<std::string, CublasMatrixMultiply::LtMatmulPlan>"), string::npos);
    EXPECT_EQ(matrixSource.find("LruCacheThreadSafe<std::string, std::shared_ptr<CublasMatrixMultiply::LtMatmulPlan>>"), string::npos);

    // Expensive epilogue contests are selection-time only. A stamped runtime stage
    // may execute its retained plan, but must not select/build one.
    const size_t runBegin = stampedSource.find("void StampedMatmul::runOn(Stream& run_stream,");
    const size_t runEnd = stampedSource.find("void StampedMatmul::runOnConditionalGraphCapture", runBegin);
    ASSERT_NE(runBegin, string::npos);
    ASSERT_NE(runEnd, string::npos);
    const string runBody = stampedSource.substr(runBegin, runEnd - runBegin);
    EXPECT_EQ(runBody.find("selectGemmWithEpilogueAlgorithm"), string::npos);
    EXPECT_EQ(runBody.find("selectGemmWithBackwardEpilogueAlgorithm"), string::npos);
    EXPECT_EQ(runBody.find("buildGemmWithEpiloguePlan"), string::npos);
    EXPECT_EQ(runBody.find("buildGemmWithBackwardEpiloguePlan"), string::npos);

    const size_t conditionalBegin = stampedSource.find("void StampedMatmul::runOnConditionalGraphCapture", runEnd);
    const size_t conditionalEnd = stampedSource.find("StampedScanMinMaxBackward::StampedScanMinMaxBackward", conditionalBegin);
    ASSERT_NE(conditionalBegin, string::npos);
    ASSERT_NE(conditionalEnd, string::npos);
    const string conditionalBody = stampedSource.substr(conditionalBegin, conditionalEnd - conditionalBegin);
    EXPECT_EQ(conditionalBody.find("selectGemmWithEpilogueAlgorithm"), string::npos);
    EXPECT_EQ(conditionalBody.find("selectGemmWithBackwardEpilogueAlgorithm"), string::npos);
    EXPECT_EQ(conditionalBody.find("buildGemmWithEpiloguePlan"), string::npos);
    EXPECT_EQ(conditionalBody.find("buildGemmWithBackwardEpiloguePlan"), string::npos);
    EXPECT_EQ(conditionalBody.find("materializeSelectedGemmKernel"), string::npos);
}

TEST(AcceleratorBackendCachePolicy, C13ConstructionCountersArePlacementOnly) {
    const filesystem::path root = findThorSourceRoot();
    const string cudnnPlanSource = readTextFile(root / "Utilities/Common/CudnnFrontendPlan.cpp");
    const string cublasKernelHeader =
        readTextFile(root / "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernel.h");
    const string cublasSource =
        readTextFile(root / "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.cpp");

    const size_t cudnnReplayBegin = cudnnPlanSource.find("CudnnFrontendExecutablePlan replayCudnnFrontendExecutablePlan(");
    const size_t cudnnExecuteBegin = cudnnPlanSource.find("void CudnnFrontendExecutablePlan::execute(");
    ASSERT_NE(cudnnReplayBegin, string::npos);
    ASSERT_NE(cudnnExecuteBegin, string::npos);
    EXPECT_NE(cudnnPlanSource.find("executable_preparation_count.fetch_add", cudnnReplayBegin), string::npos);
    const size_t cudnnExecuteEnd = cudnnPlanSource.find("CudnnFrontendPlanSelection cudnnFrontendPlanSelectionAtIndex", cudnnExecuteBegin);
    ASSERT_NE(cudnnExecuteEnd, string::npos);
    EXPECT_EQ(cudnnPlanSource.substr(cudnnExecuteBegin, cudnnExecuteEnd - cudnnExecuteBegin)
                  .find("executable_preparation_count"),
              string::npos);

    const size_t kernelConstructBegin = cublasKernelHeader.find("void construct(CublasKernelRequirement");
    ASSERT_NE(kernelConstructBegin, string::npos);
    EXPECT_NE(cublasKernelHeader.find("materializationCount.fetch_add", kernelConstructBegin), string::npos);
    const size_t launchBegin = cublasKernelHeader.find("cublasStatus_t launchUncheckedPrevalidated");
    if (launchBegin != string::npos) {
        EXPECT_LT(launchBegin, kernelConstructBegin);
        EXPECT_EQ(cublasKernelHeader.substr(launchBegin, kernelConstructBegin - launchBegin)
                      .find("materializationCount.fetch_add"),
                  string::npos);
    }

    const size_t epilogueRunBegin = cublasSource.find("void CublasMatrixMultiply::LtMatmulPlan::runGemmWithEpilogue(");
    const size_t epilogueBuildBegin = cublasSource.find("CublasMatrixMultiply::buildGemmWithEpiloguePlan(");
    ASSERT_NE(epilogueRunBegin, string::npos);
    ASSERT_NE(epilogueBuildBegin, string::npos);
    EXPECT_EQ(cublasSource.substr(epilogueRunBegin, epilogueBuildBegin - epilogueRunBegin)
                  .find("lt_matmul_plan_build_count.fetch_add"),
              string::npos);
    EXPECT_NE(cublasSource.find("lt_matmul_plan_build_count.fetch_add", epilogueBuildBegin), string::npos);
}

TEST(AcceleratorBackendCachePolicy, ClassicCudnnSoftmaxDescriptorsAreStampLocal) {
    const filesystem::path root = findThorSourceRoot();
    const string header = readTextFile(root / "Utilities/Expression/StampedEquation.h");
    const string source = readTextFile(root / "Utilities/Expression/StampedEquation.cpp");
    const string stamping = readTextFile(root / "Utilities/Expression/FusedEquation.cpp");

    EXPECT_NE(header.find("struct BuiltSoftmax final : AcceleratorBackendLocalExecutionStateTag"), string::npos);
    EXPECT_NE(header.find("const std::unique_ptr<BuiltSoftmax> built_softmax"), string::npos);
    EXPECT_NE(header.find("static std::unique_ptr<BuiltSoftmax> buildSoftmax"), string::npos);
    EXPECT_NE(stamping.find("std::unique_ptr<BuiltSoftmax> built = StampedEquation::buildSoftmax"), string::npos);
    EXPECT_EQ(source.find("builtSoftmaxCache"), string::npos);
    EXPECT_EQ(source.find("shared_ptr<BuiltSoftmax>"), string::npos);
}

TEST(AcceleratorBackendCachePolicy, LegacyGpuConvolutionProductionFacilityIsRemoved) {
    const filesystem::path root = findThorSourceRoot();

    EXPECT_FALSE(filesystem::exists(root / "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"));
    EXPECT_FALSE(filesystem::exists(root / "Utilities/TensorOperations/GpuConvolution/GpuConvolution.cpp"));
    EXPECT_FALSE(filesystem::exists(root / "Utilities/TensorOperations/GpuConvolution/ConvolutionKernelRequirement.h"));
    EXPECT_FALSE(filesystem::exists(root / "Utilities/TensorOperations/GpuConvolution/GpuConvolutionKernels.cu"));
    EXPECT_TRUE(filesContaining(root, "GpuConvolution::instance").empty());
    EXPECT_TRUE(filesContaining(root, "ConvolutionKernelRequirement").empty());
}
