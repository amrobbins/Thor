#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to read source file: " + path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

bool isSourceFile(const std::filesystem::path& path) {
    const std::string extension = path.extension().string();
    return extension == ".h" || extension == ".hpp" || extension == ".cuh" || extension == ".cpp" ||
           extension == ".cc" || extension == ".cxx" || extension == ".cu";
}

bool isBuildArtifactDirectory(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    return name == "build" || name == "_build" || name == "_deps" || name == "dist" ||
           name == "wheelhouse" || name == "__pycache__" || name == ".venv" || name == "venv" ||
           name == "_skbuild" || name.rfind("cmake-build-", 0) == 0;
}

template <typename Fn>
void forEachActiveSourceFile(const std::vector<std::filesystem::path>& sourceRoots, Fn&& fn) {
    for (const std::filesystem::path& sourceRoot : sourceRoots) {
        std::filesystem::recursive_directory_iterator entry(sourceRoot);
        const std::filesystem::recursive_directory_iterator end;
        while (entry != end) {
            if (entry->is_directory() && isBuildArtifactDirectory(entry->path())) {
                // In-tree CMake and wheel builds can contain installed copies of headers from an earlier build.
                // Those files are generated artifacts, not active Thor source, and must not influence this guard.
                entry.disable_recursion_pending();
            } else if (entry->is_regular_file() && isSourceFile(entry->path())) {
                fn(entry->path());
            }
            ++entry;
        }
    }
}

}  // namespace

TEST(ExpressionReductionArchitecture, SourceGuardsIgnoreGeneratedBuildTrees) {
    EXPECT_TRUE(isBuildArtifactDirectory("build"));
    EXPECT_TRUE(isBuildArtifactDirectory("cmake-build-debug"));
    EXPECT_TRUE(isBuildArtifactDirectory("_skbuild"));
    EXPECT_FALSE(isBuildArtifactDirectory("src"));
    EXPECT_FALSE(isBuildArtifactDirectory("Utilities"));
}

TEST(ExpressionReductionArchitecture, ActiveSourcesDoNotUseCudnnReductionApis) {
    const std::vector<std::filesystem::path> source_roots = {
        std::filesystem::path(SOURCE_DIR) / "Utilities",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning",
        std::filesystem::path(SOURCE_DIR) / "bindings",
    };
    for (const std::filesystem::path& source_root : source_roots) {
        ASSERT_TRUE(std::filesystem::exists(source_root));
    }

    const std::vector<std::string> forbidden_tokens = {
        "cudnnReduceTensor(",
        "cudnnReduceTensorDescriptor_t",
        "cudnnCreateReduceTensorDescriptor",
        "cudnnSetReduceTensorDescriptor",
        "cudnnGetReductionWorkspaceSize",
        "cudnnGetReductionIndicesSize",
        "CUDNN_REDUCE_TENSOR_",
    };

    std::vector<std::string> violations;
    forEachActiveSourceFile(source_roots, [&](const std::filesystem::path& path) {
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) {
                violations.push_back(std::filesystem::relative(path, SOURCE_DIR).generic_string() + ": " + token);
            }
        }
    });

    EXPECT_TRUE(violations.empty()) << "Active Thor reductions must not use cuDNN reduction APIs.\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}

TEST(ExpressionReductionArchitecture, GeneralReductionsUseThorsReducer) {
    const std::vector<std::filesystem::path> source_roots = {
        std::filesystem::path(SOURCE_DIR) / "Utilities",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning",
        std::filesystem::path(SOURCE_DIR) / "bindings",
    };
    for (const std::filesystem::path& source_root : source_roots) {
        ASSERT_TRUE(std::filesystem::exists(source_root));
    }

    // ReduceByKey is intentionally excluded: it performs keyed grouping, not tensor-axis reduction.
    const std::vector<std::string> forbidden_tokens = {
        "cub::DeviceReduce::Sum(",
        "cub::DeviceReduce::Min(",
        "cub::DeviceReduce::Max(",
        "cub::DeviceReduce::Reduce(",
        "cub::DeviceReduce::TransformReduce(",
        "cub::DeviceReduce::ArgMin(",
        "cub::DeviceReduce::ArgMax(",
        "cub::DeviceSegmentedReduce::Sum(",
        "cub::DeviceSegmentedReduce::Min(",
        "cub::DeviceSegmentedReduce::Max(",
        "cub::DeviceSegmentedReduce::Reduce(",
        "cub::DeviceSegmentedReduce::ArgMin(",
        "cub::DeviceSegmentedReduce::ArgMax(",
        "launchVectorSegmentedReduction",
        "launchVectorSegmentedReduceMinMaxBackward",
    };

    std::vector<std::string> violations;
    forEachActiveSourceFile(source_roots, [&](const std::filesystem::path& path) {
        const std::filesystem::path relative = std::filesystem::relative(path, SOURCE_DIR);
        const std::string relative_string = relative.generic_string();
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) {
                violations.push_back(relative_string + ": " + token);
            }
        }
    });

    EXPECT_TRUE(violations.empty()) << "General value and arg reductions must use Thor's reducer "
                                       "(Utilities/TensorOperations/Reduction).\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}

// The Cub/ directory is gone: CUB is called only where a kernel uses it directly, and the argument checks it held are
// in Utilities/Common/TensorRequirements.h.
TEST(ExpressionReductionArchitecture, CubWrapperDirectoryStaysDeleted) {
    const std::filesystem::path cub_root = std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Cub";
    EXPECT_TRUE(!std::filesystem::exists(cub_root) || std::filesystem::is_empty(cub_root)) << cub_root;
}

TEST(ExpressionReductionArchitecture, ArgReductionDoesNotRetainSupersededKernelFamilies) {
    const std::filesystem::path cub_root = std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Cub";
    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubArgReductionMin.cu"));
    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubArgReductionMax.cu"));
    // The header went with its last users, the offset-segmented ARG kernels.
    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubArgReductionOperation.cuh"));
}

// The offset-segmented kernels (one owner per segment: CUB DeviceSegmentedReduce over offset pointers for rank-1
// values, the vector narrow / wide kernels for [N, D...]) are deleted and must stay deleted. Every segmented stamp runs
// the merge-path ragged kernels (Ragged*.cu, RaggedStampAutotuner.cpp); a site no ragged family runs throws.
TEST(ExpressionReductionArchitecture, SegmentedReductionDoesNotRetainLegacyOffsetSegmentedKernels) {
    const std::filesystem::path cub_root = std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Cub";
    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubReductionOffsetSegmented.cu"));
    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubArgReductionOffsetSegmented.cu"));

    const std::vector<std::filesystem::path> source_roots = {
        std::filesystem::path(SOURCE_DIR) / "Utilities",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning",
        std::filesystem::path(SOURCE_DIR) / "bindings",
    };
    const std::vector<std::string> forbidden_tokens = {
        "CubReductionOffsetSegmented",
        "CubArgReductionOffsetSegmented",
        "CubArgReductionOperation.cuh",
        "queryOffsetSegmentedReductionBytes",
        "launchOffsetSegmentedReduction",
        "queryOffsetSegmentedArgReductionBytes",
        "launchOffsetSegmentedArgReduction",
        "offsetSegmentedReductionBackendName",
        "offsetSegmentedArgReductionBackendName",
        "narrowVectorOffsetSegmentedReductionKernel",
        "vectorOffsetSegmentedReductionKernel",
        "narrowVectorOffsetSegmentedArgReductionKernel",
        "vectorOffsetSegmentedArgReductionKernel",
        "OffsetSegmentedArgCandidateInput",
        "device_segmented_reduce.cuh",
    };
    // The dense fixed-size segmented CUB reduction (ContiguousFixedSegment) is deleted too, so no file is exempt.
    std::vector<std::string> violations;
    forEachActiveSourceFile(source_roots, [&](const std::filesystem::path& path) {
        const std::string relative = std::filesystem::relative(path, SOURCE_DIR).generic_string();
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) violations.push_back(relative + ": " + token);
        }
    });
    EXPECT_TRUE(violations.empty()) << "The legacy offset-segmented reduction kernels must stay deleted.\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}

// No value or arg reduction calls a CUB device reduction. Dense inputs run the dense planner, compact-permutation views
// run it on the dense span of their bytes, rank-1 views run StridedR, and the reducer refuses the other views (a caller
// runs them in two parts). No CUB device-reduce call is left anywhere (the last, FlatScatterAddKernel's ReduceByKey,
// left with the scan min/max backward's old path in 1508).
TEST(ExpressionReductionArchitecture, NoReductionCallsACubDeviceReduction) {
    const std::vector<std::filesystem::path> source_roots = {
        std::filesystem::path(SOURCE_DIR) / "Utilities",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning",
        std::filesystem::path(SOURCE_DIR) / "bindings",
    };
    const std::vector<std::string> forbidden_tokens = {
        "cub::DeviceReduce::",
        "cub::DeviceSegmentedReduce::",
        "cub/device/device_reduce.cuh",
        "cub/device/device_segmented_reduce.cuh",
        "makeAffineStridedFp32Iterator",
        "makeContiguousFp32Iterator",
        "makeRuntimeFp32OutputIterator",
    };

    std::vector<std::string> violations;
    forEachActiveSourceFile(source_roots, [&](const std::filesystem::path& path) {
        const std::string relative = std::filesystem::relative(path, SOURCE_DIR).generic_string();
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) violations.push_back(relative + ": " + token);
        }
    });
    EXPECT_TRUE(violations.empty()) << "Reductions must not call CUB device reductions.\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}

// The Thor tiled view family (direct and pitched component-tiled, the 32x33 payload transpose, the wide
// shared-transpose kernels, their async full-row pipeline and the per-operation executors over them) is deleted. L2
// keeps one finalizer type (AdditiveFinalizeFp32's square_root flag).
TEST(ExpressionReductionArchitecture, ValueReductionDoesNotRetainTiledViewFamily) {
    const std::filesystem::path cub_root =
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Cub";
    const std::filesystem::path reduction_root =
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Reduction";
    const std::string operation_header = readTextFile(reduction_root / "Kernels/ReductionOperation.cuh");
    EXPECT_EQ(operation_header.find("SquareRootFinalizeFp32"), std::string::npos)
        << "L2 must not regain an independent square-root finalizer template type.";
    EXPECT_NE(operation_header.find("bool square_root"), std::string::npos);

    for (const char* deleted : {"CubReductionSum.cu",
                                "CubReductionSumSquares.cu",
                                "CubReductionMean.cu",
                                "CubReductionL2Norm.cu",
                                "CubReductionL1Norm.cu",
                                "CubReductionMin.cu",
                                "CubReductionMax.cu",
                                "CubReductionProduct.cu"}) {
        EXPECT_FALSE(std::filesystem::exists(cub_root / deleted)) << deleted;
    }

    const std::vector<std::filesystem::path> files = {
        reduction_root / "Reducer.h",
        reduction_root / "ArgReducer.h",
        reduction_root / "SegmentedReducer.h",
        reduction_root / "ReductionTypes.h",
        reduction_root / "Layout/ReductionGeometry.h",
        reduction_root / "Layout/ViewSplit.h",
        reduction_root / "Planner/DenseExecutablePlan.h",
        reduction_root / "Reducer.cpp",
        reduction_root / "ArgReducer.cpp",
        reduction_root / "SegmentedReducer.cpp",
        reduction_root / "ReducerBenchmarkHooks.cpp",
        reduction_root / "FrontEnd.h",
        reduction_root / "FrontEnd.cpp",
        reduction_root / "Layout/ReductionGeometry.cpp",
        reduction_root / "Layout/ViewSplit.cpp",
        reduction_root / "Planner/DensePhysicalPassExecution.cpp",
        reduction_root / "Kernels/KernelLaunchers.h",
        reduction_root / "Kernels/ReductionOperation.cuh",
    };
    const std::vector<std::string> forbidden_tokens = {
        "CubReductionTiledPolicy",
        "TILED_REDUCTION_BLOCK_THREADS",
        "directTiledFixedSegmentReductionKernel",
        "pitchedTiledFixedSegmentReductionKernel",
        "payloadTranspose32x33ReductionKernel",
        "wideComponentTiledSharedTransposeReductionKernel",
        "launchTiledFixedSegmentReduction",
        "enqueueAsyncFullRowReductionStage",
        "queryOperationReductionBytes",
        "launchOperationReduction",
        "querySumReductionBytes",
        "launchSumReduction",
        "stampPhysicalStage",
        "queryPhysicalStageWorkspace",
    };
    for (const std::filesystem::path& path : files) {
        ASSERT_TRUE(std::filesystem::exists(path)) << path;
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            EXPECT_EQ(contents.find(token), std::string::npos) << token << " in " << path;
        }
    }
}

// The packet-adaptive cooperative tiled kernel was unreachable (dense geometry runs the dense planner, and its gate
// refused every view candidate), so it and its policy constants are deleted.
TEST(ExpressionReductionArchitecture, ValueReductionDoesNotRetainAdaptiveCooperativeTiledKernel) {
    const std::filesystem::path reduction_root =
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Reduction";
    const std::vector<std::filesystem::path> files = {
        reduction_root / "Reducer.h",
        reduction_root / "ArgReducer.h",
        reduction_root / "SegmentedReducer.h",
        reduction_root / "ReductionTypes.h",
        reduction_root / "Layout/ReductionGeometry.h",
        reduction_root / "Layout/ViewSplit.h",
        reduction_root / "Planner/DenseExecutablePlan.h",
        reduction_root / "Kernels/ReductionOperation.cuh",
    };
    const std::vector<std::string> forbidden_tokens = {
        "adaptiveCooperativeTiled",
        "AdaptiveCooperativeTiled",
        "PACKET_COOPERATIVE_",
        "ADAPTIVE_COOPERATIVE_TILED",
    };
    for (const std::filesystem::path& path : files) {
        ASSERT_TRUE(std::filesystem::exists(path)) << path;
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            EXPECT_EQ(contents.find(token), std::string::npos) << token << " in " << path;
        }
    }
}

TEST(ExpressionReductionArchitecture, ValueReductionDoesNotRetainLegacyLogicalIndexFallback) {
    const std::filesystem::path cub_root =
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Cub";
    const std::filesystem::path reduction_root =
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Reduction";

    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubReductionIndexing.cuh"))
        << "DELETE requires the legacy logical-index mapper header to stay removed.";

    const std::vector<std::filesystem::path> files = {
        reduction_root / "Reducer.h",
        reduction_root / "ArgReducer.h",
        reduction_root / "SegmentedReducer.h",
        reduction_root / "ReductionTypes.h",
        reduction_root / "Layout/ReductionGeometry.h",
        reduction_root / "Layout/ViewSplit.h",
        reduction_root / "Planner/DenseExecutablePlan.h",
        reduction_root / "Reducer.cpp",
        reduction_root / "ArgReducer.cpp",
        reduction_root / "SegmentedReducer.cpp",
        reduction_root / "ReducerBenchmarkHooks.cpp",
        reduction_root / "FrontEnd.h",
        reduction_root / "FrontEnd.cpp",
        reduction_root / "Layout/ReductionGeometry.cpp",
        reduction_root / "Layout/ViewSplit.cpp",
        reduction_root / "Planner/DensePhysicalPassExecution.cpp",
        reduction_root / "Kernels/ReductionOperation.cuh",
    };
    const std::vector<std::string> forbidden_tokens = {
        "StridedFixedSegment",
        "CubReductionIndexing",
        "CubReductionDeviceIndexing",
        "LogicalAxesToFp32",
        "makeStridedFp32Iterator",
        "mapLogicalReductionIndex",
        "stampDeviceIndexingMetadata",
        "stampLegacyStridedForBenchmark",
        "strided_value_indexing_fits_uint32",
    };

    for (const std::filesystem::path& path : files) {
        ASSERT_TRUE(std::filesystem::exists(path)) << path;
        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            EXPECT_EQ(contents.find(token), std::string::npos)
                << "DELETE requires legacy value-reduction machinery to stay removed: " << token
                << " in " << path;
        }
    }
}
