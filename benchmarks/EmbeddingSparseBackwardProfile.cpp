#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/CudaDriver/CudaGraph.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

struct RunningStats {
    uint64_t n = 0;
    double sum = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = 0.0;

    void add(double v) {
        ++n;
        sum += v;
        min = std::min(min, v);
        max = std::max(max, v);
    }

    double mean() const { return n == 0 ? 0.0 : sum / static_cast<double>(n); }
};

struct StageStats {
    RunningStats materialize;
    RunningStats sort;
    RunningStats clearCounts;
    RunningStats rle;
    RunningStats finalize;
    RunningStats scan;
    RunningStats reduce;
    RunningStats sparseGradientTotal;
    RunningStats sparseSgdUpdate;
    RunningStats totalBackwardAndUpdate;
    RunningStats graphSparseGradientTotal;
    RunningStats activeRows;
    RunningStats singletonRows;
    RunningStats duplicateRows;
    RunningStats maxRunCount;
};

std::string dtypeName(DataType dtype);
std::vector<std::string> parseStringList(const std::string& raw);
std::vector<uint64_t> parseU64List(const std::string& raw);
std::vector<DataType> parseDTypeList(const std::string& raw);

std::string envString(const char* name, std::string defaultValue) {
    const char* raw = std::getenv(name);
    return raw == nullptr ? std::move(defaultValue) : std::string(raw);
}

bool envBool(const char* name, bool defaultValue) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return defaultValue;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value.empty() || value == "0" || value == "false" || value == "no" || value == "off");
}

int envInt(const char* name, int defaultValue) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return defaultValue;
    }
    return std::stoi(raw);
}

uint64_t envU64(const char* name, uint64_t defaultValue) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return defaultValue;
    }
    return static_cast<uint64_t>(std::stoull(raw));
}

struct Options {
    int gpu = envInt("THOR_EMBEDDING_BACKWARD_PROFILE_GPU", 0);
    int warmupIters = envInt("THOR_EMBEDDING_BACKWARD_PROFILE_WARMUP_ITERS", 6);
    int measureIters = envInt("THOR_EMBEDDING_BACKWARD_PROFILE_MEASURE_ITERS", 24);
    uint64_t minPoolBytes = envU64("THOR_EMBEDDING_BACKWARD_PROFILE_MIN_POOL_BYTES", 512ULL * 1024ULL * 1024ULL);
    int minPoolSlots = envInt("THOR_EMBEDDING_BACKWARD_PROFILE_MIN_POOL_SLOTS", 8);
    int maxPoolSlots = envInt("THOR_EMBEDDING_BACKWARD_PROFILE_MAX_POOL_SLOTS", 512);
    int explicitPoolSlots = envInt("THOR_EMBEDDING_BACKWARD_PROFILE_POOL_SLOTS", 0);
    bool profileGraph = envBool("THOR_EMBEDDING_BACKWARD_PROFILE_GRAPH", true);
    bool profileUpdate = envBool("THOR_EMBEDDING_BACKWARD_PROFILE_UPDATE", true);
    bool useFusedUpdate = envBool("THOR_EMBEDDING_BACKWARD_PROFILE_FUSED_UPDATE", true);
    bool initializeWeights = envBool("THOR_EMBEDDING_BACKWARD_PROFILE_INITIALIZE_WEIGHTS", false);
    uint32_t batchSize = static_cast<uint32_t>(envInt("THOR_EMBEDDING_BACKWARD_PROFILE_BATCH_SIZE", 1));
    std::string caseFilter = envString("THOR_EMBEDDING_BACKWARD_PROFILE_CASE_FILTER", "");
    std::vector<uint64_t> vocabs = parseU64List(envString("THOR_EMBEDDING_BACKWARD_PROFILE_VOCABS", "32768,131072,1048576"));
    std::vector<uint64_t> dims = parseU64List(envString("THOR_EMBEDDING_BACKWARD_PROFILE_DIMS", "16,32,64,128,256"));
    std::vector<uint64_t> tokens = parseU64List(envString("THOR_EMBEDDING_BACKWARD_PROFILE_TOKENS", "4096,16384,65536"));
    std::vector<DataType> upstreamDTypes = parseDTypeList(envString("THOR_EMBEDDING_BACKWARD_PROFILE_UPSTREAM_DTYPES", "fp16,bf16,fp32"));
    std::vector<std::string> duplicateModes = parseStringList(envString("THOR_EMBEDDING_BACKWARD_PROFILE_DUPLICATE_MODES", "unique,moderate,high"));
    std::vector<std::string> optimizers = parseStringList(envString("THOR_EMBEDDING_BACKWARD_PROFILE_OPTIMIZERS", "sgd"));
};
struct CaseConfig {
    uint64_t vocabularySize = 0;
    uint64_t embeddingDim = 0;
    uint64_t numTokens = 0;
    DataType upstreamDType = DataType::FP32;
    std::string duplicateMode;
    std::string optimizer;

    std::string name() const {
        std::ostringstream ss;
        ss << "V" << vocabularySize << "_D" << embeddingDim << "_N" << numTokens << "_" << dtypeName(upstreamDType) << "_" << duplicateMode
           << "_" << optimizer;
        return ss.str();
    }
};

struct PoolSlot {
    Tensor indices;
    Tensor upstream;
    Tensor weights;
    std::unique_ptr<Sgd> optimizer;
    SparseRowGradient gradient;
    std::shared_ptr<PreparedEmbeddingSparseGradient> prepared;
    bool fusedSparseRowUpdate = false;
    CapturedEmbeddingSparseGradient captured;
    std::optional<CudaGraphExecutable> graph;
};

std::string dtypeName(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        case DataType::FP32:
            return "fp32";
        case DataType::UINT16:
            return "uint16";
        case DataType::UINT32:
            return "uint32";
        case DataType::UINT64:
            return "uint64";
        default:
            return TensorDescriptor::getElementTypeName(dtype);
    }
}

bool hasFixedSparseOptimizerFusionReducer(uint64_t embeddingDim) {
    return embeddingDim == 16ULL || embeddingDim == 32ULL || embeddingDim == 64ULL || embeddingDim == 128ULL || embeddingDim == 256ULL;
}


std::vector<std::string> parseStringList(const std::string& raw) {
    std::vector<std::string> out;
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }), item.end());
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<uint64_t> parseU64List(const std::string& raw) {
    std::vector<uint64_t> out;
    for (const std::string& item : parseStringList(raw)) {
        out.push_back(static_cast<uint64_t>(std::stoull(item)));
    }
    return out;
}

std::vector<DataType> parseDTypeList(const std::string& raw) {
    std::vector<DataType> out;
    for (std::string item : parseStringList(raw)) {
        std::transform(item.begin(), item.end(), item.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (item == "fp16") {
            out.push_back(DataType::FP16);
        } else if (item == "bf16") {
            out.push_back(DataType::BF16);
        } else if (item == "fp32") {
            out.push_back(DataType::FP32);
        } else {
            throw std::invalid_argument("Unsupported upstream dtype: " + item);
        }
    }
    return out;
}

uint64_t elementBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP16:
        case DataType::BF16:
        case DataType::UINT16:
            return 2;
        case DataType::FP32:
        case DataType::UINT32:
            return 4;
        case DataType::UINT64:
            return 8;
        default:
            throw std::invalid_argument("Unsupported dtype byte-width request.");
    }
}

uint64_t checkedMul(uint64_t a, uint64_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::overflow_error(std::string(label) + " overflow");
    }
    return a * b;
}

uint64_t estimateTouchedBytesPerSlot(const CaseConfig& cfg) {
    const uint64_t capacity = std::min(cfg.numTokens, cfg.vocabularySize);
    const DataType rowDType = SparseRowGradient::chooseRowDataType(cfg.vocabularySize);
    const uint64_t upstreamBytes = checkedMul(checkedMul(cfg.numTokens, cfg.embeddingDim, "upstream elements"), elementBytes(cfg.upstreamDType), "upstream bytes");
    const uint64_t gradientValueBytes = checkedMul(checkedMul(capacity, cfg.embeddingDim, "gradient elements"), sizeof(float), "gradient bytes");
    const uint64_t rowTraffic = checkedMul(cfg.numTokens, elementBytes(rowDType) * 2 + sizeof(uint32_t) * 2, "row staging bytes");
    const uint64_t weightsReadWrite = checkedMul(checkedMul(capacity, cfg.embeddingDim, "weight elements"), elementBytes(cfg.upstreamDType) * 2, "weight bytes");
    return upstreamBytes + gradientValueBytes + rowTraffic + weightsReadWrite;
}

int choosePoolSlots(const Options& opts, const CaseConfig& cfg, uint64_t l2Bytes) {
    if (opts.explicitPoolSlots > 0) {
        return std::max(1, opts.explicitPoolSlots);
    }
    const uint64_t perSlotBytes = std::max<uint64_t>(estimateTouchedBytesPerSlot(cfg), 1);
    const uint64_t targetPoolBytes = std::max(opts.minPoolBytes, 4 * l2Bytes);
    uint64_t slots = (targetPoolBytes + perSlotBytes - 1) / perSlotBytes;
    slots = std::max<uint64_t>(slots, static_cast<uint64_t>(opts.minPoolSlots));
    slots = std::min<uint64_t>(slots, static_cast<uint64_t>(opts.maxPoolSlots));
    return static_cast<int>(std::max<uint64_t>(slots, 1));
}

void fillCpuIndices(Tensor& tensor, const CaseConfig& cfg, int slot) {
    uint32_t* ptr = tensor.getMemPtr<uint32_t>();
    const uint64_t validRows = std::max<uint64_t>(cfg.vocabularySize - 1, 1);
    uint64_t uniqueRows = std::min(cfg.numTokens, validRows);
    if (cfg.duplicateMode == "low") {
        uniqueRows = std::max<uint64_t>(1, std::min(validRows, (cfg.numTokens * 3 + 3) / 4));
    } else if (cfg.duplicateMode == "moderate") {
        uniqueRows = std::max<uint64_t>(1, std::min(validRows, std::max<uint64_t>(1, cfg.numTokens / 4)));
    } else if (cfg.duplicateMode == "high") {
        uniqueRows = std::max<uint64_t>(1, std::min(validRows, std::max<uint64_t>(1, cfg.numTokens / 32)));
    } else if (cfg.duplicateMode == "zipf") {
        uniqueRows = std::max<uint64_t>(1, std::min(validRows, std::max<uint64_t>(1, cfg.numTokens / 8)));
    } else if (cfg.duplicateMode != "unique") {
        throw std::invalid_argument("Unsupported duplicate mode: " + cfg.duplicateMode);
    }

    const uint64_t stride = std::max<uint64_t>(uniqueRows, 1);
    const uint64_t base = 1 + ((static_cast<uint64_t>(slot) * stride * 1315423911ULL) % validRows);
    for (uint64_t t = 0; t < cfg.numTokens; ++t) {
        uint64_t within;
        if (cfg.duplicateMode == "zipf") {
            const uint64_t bucket = (t * 1103515245ULL + 12345ULL + static_cast<uint64_t>(slot) * 17ULL) % uniqueRows;
            within = bucket == 0 ? 0 : (bucket / (1ULL + (bucket % 17ULL)));
            within %= uniqueRows;
        } else {
            within = t % uniqueRows;
        }
        uint64_t row = 1 + ((base + within - 1) % validRows);
        ptr[t] = static_cast<uint32_t>(row);
    }
}

void fillCpuUpstream(Tensor& tensor, const CaseConfig& cfg, int slot) {
    const uint64_t n = checkedMul(cfg.numTokens, cfg.embeddingDim, "upstream element count");
    switch (cfg.upstreamDType) {
        case DataType::FP32: {
            float* ptr = tensor.getMemPtr<float>();
            for (uint64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<float>(((i * 17ULL + static_cast<uint64_t>(slot) * 8191ULL) % 1024ULL) - 512.0) * (1.0f / 1024.0f);
            }
            break;
        }
        case DataType::FP16: {
            auto* ptr = static_cast<__half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < n; ++i) {
                float v = static_cast<float>(((i * 17ULL + static_cast<uint64_t>(slot) * 8191ULL) % 1024ULL) - 512.0) * (1.0f / 1024.0f);
                ptr[i] = __float2half(v);
            }
            break;
        }
        case DataType::BF16: {
            auto* ptr = static_cast<__nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < n; ++i) {
                float v = static_cast<float>(((i * 17ULL + static_cast<uint64_t>(slot) * 8191ULL) % 1024ULL) - 512.0) * (1.0f / 1024.0f);
                ptr[i] = __float2bfloat16(v);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported upstream dtype.");
    }
}

void fillCpuWeights(Tensor& tensor, const CaseConfig& cfg, int slot) {
    const uint64_t n = checkedMul(cfg.vocabularySize, cfg.embeddingDim, "weight element count");
    switch (cfg.upstreamDType) {
        case DataType::FP32: {
            float* ptr = tensor.getMemPtr<float>();
            for (uint64_t i = 0; i < n; ++i) {
                ptr[i] = static_cast<float>(((i * 3ULL + static_cast<uint64_t>(slot) * 19ULL) % 256ULL) - 128.0) * (1.0f / 256.0f);
            }
            break;
        }
        case DataType::FP16: {
            auto* ptr = static_cast<__half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < n; ++i) {
                float v = static_cast<float>(((i * 3ULL + static_cast<uint64_t>(slot) * 19ULL) % 256ULL) - 128.0) * (1.0f / 256.0f);
                ptr[i] = __float2half(v);
            }
            break;
        }
        case DataType::BF16: {
            auto* ptr = static_cast<__nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < n; ++i) {
                float v = static_cast<float>(((i * 3ULL + static_cast<uint64_t>(slot) * 19ULL) % 256ULL) - 128.0) * (1.0f / 256.0f);
                ptr[i] = __float2bfloat16(v);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported weight dtype.");
    }
}

template <typename FillFn>
Tensor makeGpuTensorFromCpu(TensorPlacement cpuPlacement, TensorPlacement gpuPlacement, TensorDescriptor desc, Stream stream, FillFn fillFn) {
    Tensor cpu(cpuPlacement, desc);
    fillFn(cpu);
    Tensor gpu(gpuPlacement, desc);
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

float timeCallableMs(Stream stream, const std::function<void()>& fn) {
    Event start(stream.getGpuNum(), /*enableTiming=*/true);
    Event end(stream.getGpuNum(), /*enableTiming=*/true);
    start.record(stream);
    fn();
    end.record(stream);
    return end.synchronizeAndReportElapsedTimeInMilliseconds(start);
}

std::pair<float, EmbeddingSparseGradientProfileResult> profileSparseGradient(PoolSlot& slot, Stream stream, uint32_t batchSize) {
    EmbeddingSparseGradientProfileResult result;
    if (slot.fusedSparseRowUpdate) {
        result = profilePreparedEmbeddingSparseGradientWithSparseRowUpdate(*slot.prepared,
                                                                          slot.indices,
                                                                          slot.upstream,
                                                                          slot.gradient,
                                                                          slot.optimizer->sparseRowUpdateRuntimeScalars(batchSize),
                                                                          stream);
    } else {
        result = profilePreparedEmbeddingSparseGradient(*slot.prepared, slot.indices, slot.upstream, slot.gradient, stream);
    }
    return {result.totalMs, result};
}

float profileGraphSparseGradient(PoolSlot& slot, Stream stream) {
    if (!slot.graph.has_value()) {
        return 0.0f;
    }
    return timeCallableMs(stream, [&] { slot.graph->launch(stream); });
}

float profileSparseUpdate(PoolSlot& slot, Stream stream, uint32_t batchSize) {
    if (slot.optimizer == nullptr) {
        return 0.0f;
    }
    return timeCallableMs(stream, [&] { slot.optimizer->updateSparseRows(batchSize); });
}

std::vector<PoolSlot> buildPool(const Options& opts, const CaseConfig& cfg, int poolSlots, Stream stream) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, opts.gpu);
    std::vector<PoolSlot> pool;
    pool.reserve(static_cast<size_t>(poolSlots));

    Tensor sharedWeights(gpuPlacement, TensorDescriptor(cfg.upstreamDType, {cfg.vocabularySize, cfg.embeddingDim}));
    if (opts.initializeWeights) {
        Tensor cpuWeights(cpuPlacement, TensorDescriptor(cfg.upstreamDType, {cfg.vocabularySize, cfg.embeddingDim}));
        fillCpuWeights(cpuWeights, cfg, 0);
        sharedWeights.copyFromAsync(cpuWeights, stream);
        stream.synchronize();
    } else {
        sharedWeights.memsetAsync(stream, 0);
        stream.synchronize();
    }

    const uint64_t capacity = std::min(cfg.numTokens, cfg.vocabularySize);
    for (int i = 0; i < poolSlots; ++i) {
        PoolSlot slot;
        slot.indices = makeGpuTensorFromCpu(cpuPlacement,
                                            gpuPlacement,
                                            TensorDescriptor(DataType::UINT32, {cfg.numTokens}),
                                            stream,
                                            [&](Tensor& cpu) { fillCpuIndices(cpu, cfg, i); });
        slot.upstream = makeGpuTensorFromCpu(cpuPlacement,
                                             gpuPlacement,
                                             TensorDescriptor(cfg.upstreamDType, {cfg.numTokens, cfg.embeddingDim}),
                                             stream,
                                             [&](Tensor& cpu) { fillCpuUpstream(cpu, cfg, i); });

        slot.weights = sharedWeights;

        float momentum = 0.0f;
        bool nesterov = false;
        if (cfg.optimizer == "momentum") {
            momentum = 0.9f;
        } else if (cfg.optimizer == "nesterov") {
            momentum = 0.9f;
            nesterov = true;
        } else if (cfg.optimizer != "sgd") {
            throw std::invalid_argument("Unsupported optimizer mode: " + cfg.optimizer);
        }
        slot.optimizer = std::make_unique<Sgd>(static_cast<uint64_t>(i + 1), 0.01f, 0.0f, momentum, nesterov);
        slot.gradient = slot.optimizer->compileSparseRows(slot.weights, capacity, stream);
        stream.synchronize();

        slot.fusedSparseRowUpdate = opts.useFusedUpdate && slot.optimizer->supportsSparseRowUpdateFusion() &&
                                    hasFixedSparseOptimizerFusionReducer(cfg.embeddingDim);
        if (slot.fusedSparseRowUpdate) {
            SparseRowOptimizerExpression updateExpression = slot.optimizer->toSparseRowUpdateExpression(slot.weights, slot.gradient);
            slot.prepared = prepareEmbeddingSparseGradientWithSparseRowUpdate(slot.indices,
                                                                             slot.upstream,
                                                                             slot.gradient,
                                                                             updateExpression.outputs,
                                                                             updateExpression.inputs,
                                                                             updateExpression.indexedOutputs,
                                                                             std::nullopt);
        } else {
            slot.prepared = prepareEmbeddingSparseGradient(slot.indices, slot.upstream, slot.gradient, std::nullopt);
        }

        if (opts.profileGraph) {
            slot.captured = CapturedEmbeddingSparseGradient(opts.gpu);
            CudaGraphCaptureBuilder builder(stream);
            if (slot.fusedSparseRowUpdate) {
                capturePreparedEmbeddingSparseGradientWithSparseRowUpdate(builder,
                                                                         *slot.prepared,
                                                                         slot.indices,
                                                                         slot.upstream,
                                                                         slot.gradient,
                                                                         slot.optimizer->sparseRowUpdateRuntimeScalars(opts.batchSize),
                                                                         slot.captured);
            } else {
                capturePreparedEmbeddingSparseGradient(builder, *slot.prepared, slot.indices, slot.upstream, slot.gradient, slot.captured);
            }
            slot.graph.emplace(builder.endCaptureAndInstantiate(stream));
            slot.captured.uploadTargetNodes(stream);
        }
        pool.push_back(std::move(slot));
    }
    stream.synchronize();
    return pool;
}

void printCsvHeader() {
    std::cout << "case,vocab,dim,tokens,upstream_dtype,row_dtype,dup_mode,optimizer,update_path,pool_slots,pool_bytes,l2_bytes,"
                 "materialize_ms,sort_ms,clear_counts_ms,rle_ms,finalize_ms,scan_ms,reduce_ms,"
                 "sparse_gradient_ms,sparse_update_ms,total_backward_update_ms,graph_sparse_gradient_ms,"
                 "active_rows,singleton_rows,duplicate_rows,max_run_count,"
                 "sort_temp_bytes,rle_temp_bytes,scan_temp_bytes\n";
}

void printCsvRow(const CaseConfig& cfg,
                 const EmbeddingSparseGradientProfileResult& meta,
                 const StageStats& stats,
                 const std::string& updatePath,
                 int poolSlots,
                 uint64_t poolBytes,
                 uint64_t l2Bytes) {
    std::cout << cfg.name() << ',' << cfg.vocabularySize << ',' << cfg.embeddingDim << ',' << cfg.numTokens << ',' << dtypeName(cfg.upstreamDType) << ','
              << dtypeName(meta.rowDataType) << ',' << cfg.duplicateMode << ',' << cfg.optimizer << ',' << updatePath << ',' << poolSlots << ',' << poolBytes << ',' << l2Bytes << ','
              << stats.materialize.mean() << ',' << stats.sort.mean() << ',' << stats.clearCounts.mean() << ',' << stats.rle.mean() << ','
              << stats.finalize.mean() << ',' << stats.scan.mean() << ',' << stats.reduce.mean() << ',' << stats.sparseGradientTotal.mean() << ','
              << stats.sparseSgdUpdate.mean() << ',' << stats.totalBackwardAndUpdate.mean() << ',' << stats.graphSparseGradientTotal.mean() << ','
              << stats.activeRows.mean() << ',' << stats.singletonRows.mean() << ',' << stats.duplicateRows.mean() << ','
              << stats.maxRunCount.mean() << ',' << meta.sortTempBytes << ',' << meta.rleTempBytes << ',' << meta.scanTempBytes << '\n';
}

bool keepCase(const Options& opts, const CaseConfig& cfg) {
    if (opts.caseFilter.empty()) {
        return true;
    }
    return cfg.name().find(opts.caseFilter) != std::string::npos;
}

void profileCase(const Options& opts, const CaseConfig& cfg, const cudaDeviceProp& prop) {
    Stream stream(opts.gpu);
    const uint64_t l2Bytes = static_cast<uint64_t>(prop.l2CacheSize);
    const int poolSlots = choosePoolSlots(opts, cfg, l2Bytes);
    const uint64_t poolBytes = checkedMul(static_cast<uint64_t>(poolSlots), estimateTouchedBytesPerSlot(cfg), "pool bytes");

    std::cerr << "profiling " << cfg.name() << " pool_slots=" << poolSlots << " estimated_pool_bytes=" << poolBytes << " l2_bytes=" << l2Bytes << '\n';
    const uint64_t targetPoolBytes = std::max(opts.minPoolBytes, 4 * l2Bytes);
    if (poolBytes < targetPoolBytes) {
        std::cerr << "  warning: estimated rotating pool bytes are below the requested anti-L2 target; "
                  << "increase THOR_EMBEDDING_BACKWARD_PROFILE_MAX_POOL_SLOTS or THOR_EMBEDDING_BACKWARD_PROFILE_POOL_SLOTS "
                  << "if this case is being used for final bandwidth claims. target_pool_bytes=" << targetPoolBytes << '\n';
    }
    std::vector<PoolSlot> pool = buildPool(opts, cfg, poolSlots, stream);

    for (int i = 0; i < opts.warmupIters; ++i) {
        PoolSlot& slot = pool[static_cast<size_t>(i % poolSlots)];
        if (slot.fusedSparseRowUpdate) {
            launchPreparedEmbeddingSparseGradientWithSparseRowUpdate(*slot.prepared,
                                                                     slot.indices,
                                                                     slot.upstream,
                                                                     slot.gradient,
                                                                     slot.optimizer->sparseRowUpdateRuntimeScalars(opts.batchSize),
                                                                     stream);
        } else {
            launchPreparedEmbeddingSparseGradient(*slot.prepared, slot.indices, slot.upstream, slot.gradient, stream);
            if (opts.profileUpdate) {
                slot.optimizer->updateSparseRows(opts.batchSize);
            }
        }
        if (opts.profileGraph && slot.graph.has_value()) {
            slot.graph->launch(stream);
        }
    }
    stream.synchronize();

    StageStats stats;
    EmbeddingSparseGradientProfileResult meta;
    for (int i = 0; i < opts.measureIters; ++i) {
        PoolSlot& slot = pool[static_cast<size_t>(i % poolSlots)];
        const auto [totalMs, profile] = profileSparseGradient(slot, stream, opts.batchSize);
        meta = profile;
        stats.materialize.add(profile.materializeSortPairsMs);
        stats.sort.add(profile.cubSortMs);
        stats.clearCounts.add(profile.clearRunCountsMs);
        stats.rle.add(profile.cubRleMs);
        stats.finalize.add(profile.finalizeRowsMs);
        stats.scan.add(profile.cubScanOffsetsMs);
        stats.reduce.add(profile.reduceValuesMs);
        stats.sparseGradientTotal.add(totalMs);
        stats.activeRows.add(static_cast<double>(profile.activeRows));
        stats.singletonRows.add(static_cast<double>(profile.singletonRows));
        stats.duplicateRows.add(static_cast<double>(profile.duplicateRows));
        stats.maxRunCount.add(static_cast<double>(profile.maxRunCount));

        float updateMs = 0.0f;
        if (slot.fusedSparseRowUpdate) {
            // The production fused path performs the optimizer update inside the timed reducer stage.
            stats.sparseSgdUpdate.add(0.0);
        } else if (opts.profileUpdate) {
            updateMs = profileSparseUpdate(slot, stream, opts.batchSize);
            stats.sparseSgdUpdate.add(updateMs);
        }
        stats.totalBackwardAndUpdate.add(totalMs + updateMs);

        if (opts.profileGraph && slot.graph.has_value()) {
            stats.graphSparseGradientTotal.add(profileGraphSparseGradient(slot, stream));
        }
    }

    const bool fusedUpdatePath = !pool.empty() && pool.front().fusedSparseRowUpdate;
    printCsvRow(cfg, meta, stats, fusedUpdatePath ? "fused" : "materialized", poolSlots, poolBytes, l2Bytes);
}

std::vector<CaseConfig> makeCases(const Options& opts) {
    std::vector<CaseConfig> cases;
    for (uint64_t vocab : opts.vocabs) {
        if (vocab == 0 || vocab > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument("The profiling suite currently generates uint32 token indices and requires 1 <= vocab <= UINT32_MAX.");
        }
        for (uint64_t dim : opts.dims) {
            for (uint64_t tokens : opts.tokens) {
                for (DataType dtype : opts.upstreamDTypes) {
                    for (const std::string& dup : opts.duplicateModes) {
                        for (const std::string& opt : opts.optimizers) {
                            CaseConfig cfg{vocab, dim, tokens, dtype, dup, opt};
                            if (keepCase(opts, cfg)) {
                                cases.push_back(std::move(cfg));
                            }
                        }
                    }
                }
            }
        }
    }
    return cases;
}

void printUsage() {
    std::cerr << "Thor embedding sparse backward profiling suite\n"
              << "\nEnvironment controls:\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_GPU=0\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_WARMUP_ITERS=6\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_MEASURE_ITERS=24\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_MIN_POOL_BYTES=536870912\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_MIN_POOL_SLOTS=8\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_MAX_POOL_SLOTS=512\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_POOL_SLOTS=0\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_GRAPH=1\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_UPDATE=1\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_INITIALIZE_WEIGHTS=0\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_CASE_FILTER=\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_VOCABS=32768,131072,1048576\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_DIMS=16,32,64,128,256\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_TOKENS=4096,16384,65536\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_UPSTREAM_DTYPES=fp16,bf16,fp32\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_DUPLICATE_MODES=unique,moderate,high\n"
              << "  THOR_EMBEDDING_BACKWARD_PROFILE_OPTIMIZERS=sgd\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) {
            std::string_view arg(argv[i]);
            if (arg == "--help" || arg == "-h") {
                printUsage();
                return 0;
            }
        }

        Options opts;
        ScopedGpu scopedGpu(opts.gpu);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.gpu));

        std::cerr << "GPU " << opts.gpu << ": " << prop.name << " l2_bytes=" << prop.l2CacheSize << '\n';
        std::vector<CaseConfig> cases = makeCases(opts);
        if (cases.empty()) {
            std::cerr << "No cases selected. Check THOR_EMBEDDING_BACKWARD_PROFILE_CASE_FILTER.\n";
            return 2;
        }

        printCsvHeader();
        for (const CaseConfig& cfg : cases) {
            profileCase(opts, cfg, prop);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "embedding sparse backward profile failed: " << e.what() << '\n';
        return 1;
    }
}
