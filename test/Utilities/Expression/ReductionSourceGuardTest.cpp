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

TEST(ExpressionReductionArchitecture, GeneralReductionsAreCentralizedUnderCubReduction) {
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
        if (relative_string.rfind("Utilities/TensorOperations/Cub/CubReduction", 0) == 0
            || relative_string.rfind("Utilities/TensorOperations/Cub/CubArgReduction", 0) == 0) {
            return;
        }

        const std::string contents = readTextFile(path);
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) {
                violations.push_back(relative_string + ": " + token);
            }
        }
    });

    EXPECT_TRUE(violations.empty()) << "General value and arg reductions must use the central CUB reduction utility.\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}

TEST(ExpressionReductionArchitecture, ArgReductionDoesNotRetainSupersededKernelFamilies) {
    const std::filesystem::path cub_root = std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Cub";
    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubArgReductionMin.cu"));
    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubArgReductionMax.cu"));
    // 1330 (ragged phase 9): the header's last users, the offset-segmented ARG kernels, are deleted, so the header is
    // too (1261 had cut it to their FP32 candidate rules and stores).
    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubArgReductionOperation.cuh"));
}

// 1330 (ragged phase 9): the legacy offset-segmented kernels (one owner per segment: CUB DeviceSegmentedReduce over
// offset pointers for rank-1 values, the vector narrow / wide kernels for [N, D...]) are deleted. Every segmented
// stamp runs the merge-path ragged kernels (CubRagged*.cu, RaggedStampAutotuner.cpp); a site no ragged family runs
// throws. They must stay deleted.
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
    // 1373: the dense fixed-size segmented CUB reduction (ContiguousFixedSegment) is deleted too, so no file is exempt.
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

// 1373 (plans/Thor_View_Reduction_Plan_2026-11-08.md, phase 4): no value or arg reduction calls a CUB device
// reduction. Dense inputs run the retained planner, compact-permutation views run it on the dense span of their bytes,
// rank-1 views run StridedR, and the other views run the Thor tiled kernels. The one CUB device-reduce call left is
// FlatScatterAddKernel's ReduceByKey, a keyed grouping (it includes <cub/cub.cuh>, not the device-reduce headers).
TEST(ExpressionReductionArchitecture, NoReductionCallsACubDeviceReduction) {
    const std::vector<std::filesystem::path> source_roots = {
        std::filesystem::path(SOURCE_DIR) / "Utilities",
        std::filesystem::path(SOURCE_DIR) / "DeepLearning",
        std::filesystem::path(SOURCE_DIR) / "bindings",
    };
    const std::string keyed_grouping_owner = "Utilities/Expression/FlatScatterAddKernel.cu";
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
    size_t keyed_grouping_calls = 0;
    forEachActiveSourceFile(source_roots, [&](const std::filesystem::path& path) {
        const std::string relative = std::filesystem::relative(path, SOURCE_DIR).generic_string();
        std::string contents = readTextFile(path);
        if (relative == keyed_grouping_owner) {
            const std::string keyed = "cub::DeviceReduce::ReduceByKey(";
            for (size_t at = contents.find(keyed); at != std::string::npos; at = contents.find(keyed, at)) {
                contents.erase(at, keyed.size());
                ++keyed_grouping_calls;
            }
        }
        for (const std::string& token : forbidden_tokens) {
            if (contents.find(token) != std::string::npos) violations.push_back(relative + ": " + token);
        }
    });
    EXPECT_GT(keyed_grouping_calls, 0U) << "The guard's one exemption no longer matches FlatScatterAddKernel.cu.";
    EXPECT_TRUE(violations.empty()) << "Reductions must not call CUB device reductions.\n"
                                    << [&]() {
                                           std::ostringstream message;
                                           for (const std::string& violation : violations) {
                                               message << violation << '\n';
                                           }
                                           return message.str();
                                       }();
}

// 1393: the Thor tiled view family (direct and pitched component-tiled, the 32x33 payload transpose, the wide
// shared-transpose kernels, their async full-row pipeline and the per-operation executors over them) is deleted. No
// stamp reached it after 1390. L2 keeps one finalizer type (AdditiveFinalizeFp32's square_root flag).
TEST(ExpressionReductionArchitecture, ValueReductionDoesNotRetainTiledViewFamily) {
    const std::filesystem::path cub_root =
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Cub";
    const std::string operation_header = readTextFile(cub_root / "CubReductionOperation.cuh");
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
        cub_root / "CubReduction.h",
        cub_root / "CubReduction.cpp",
        cub_root / "CubReductionInternal.h",
        cub_root / "CubReductionOperation.cuh",
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

// 1385: the packet-adaptive cooperative tiled kernel was unreachable (dense geometry runs the dense planner, and its
// gate refused every view candidate), so it and its policy constants are deleted.
TEST(ExpressionReductionArchitecture, ValueReductionDoesNotRetainAdaptiveCooperativeTiledKernel) {
    const std::filesystem::path cub_root =
        std::filesystem::path(SOURCE_DIR) / "Utilities/TensorOperations/Cub";
    const std::vector<std::filesystem::path> files = {
        cub_root / "CubReduction.h",
        cub_root / "CubReductionOperation.cuh",
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

    EXPECT_FALSE(std::filesystem::exists(cub_root / "CubReductionIndexing.cuh"))
        << "DELETE requires the legacy logical-index mapper header to stay removed.";

    const std::vector<std::filesystem::path> files = {
        cub_root / "CubReduction.h",
        cub_root / "CubReduction.cpp",
        cub_root / "CubReductionOperation.cuh",
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
