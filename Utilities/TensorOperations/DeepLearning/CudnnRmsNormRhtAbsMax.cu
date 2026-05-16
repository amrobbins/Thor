#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNormRhtAbsMax.h"

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Expression/CudaHelpers.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

using namespace ThorImplementation;
using namespace std;

namespace {

[[noreturn]] void throwInvalidRhtAmax(const string& message) {
    throw invalid_argument("Invalid cuDNN RMSNorm+RHT+Amax descriptor: " + message);
}

string dtypeName(TensorDescriptor::DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

uint64_t checkedMul(uint64_t a, uint64_t b, string_view what) {
    if (a != 0 && b > numeric_limits<uint64_t>::max() / a) {
        throwInvalidRhtAmax(string(what) + " element count overflows uint64_t");
    }
    return a * b;
}

void requireInitialized(const Tensor& tensor, string_view name) {
    if (!tensor.isInitialized()) {
        throw invalid_argument("cuDNN RMSNorm+RHT+Amax tensor '" + string(name) + "' is not initialized.");
    }
}

void requireSameGpu(const Tensor& tensor, int gpuNum, string_view name) {
    requireInitialized(tensor, name);
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw invalid_argument("cuDNN RMSNorm+RHT+Amax tensor '" + string(name) + "' must be a GPU tensor.");
    }
    if (tensor.getPlacement().getDeviceNum() != gpuNum) {
        throw invalid_argument("cuDNN RMSNorm+RHT+Amax tensor '" + string(name) + "' is on GPU " +
                               to_string(tensor.getPlacement().getDeviceNum()) + ", expected GPU " + to_string(gpuNum) + ".");
    }
}

void requireDtype(const Tensor& tensor, TensorDescriptor::DataType expected, string_view name) {
    if (tensor.getDataType() != expected) {
        throw invalid_argument("cuDNN RMSNorm+RHT+Amax tensor '" + string(name) + "' dtype mismatch. Expected " + dtypeName(expected) +
                               ", got " + dtypeName(tensor.getDataType()) + ".");
    }
}

void requireNumElements(const Tensor& tensor, uint64_t expected, string_view name) {
    const uint64_t actual = tensor.getTotalNumElements();
    if (actual != expected) {
        throw invalid_argument("cuDNN RMSNorm+RHT+Amax tensor '" + string(name) + "' element-count mismatch. Expected " +
                               to_string(expected) + ", got " + to_string(actual) + ".");
    }
}

void requireTensor(
    const Tensor& tensor, TensorDescriptor::DataType expectedDtype, uint64_t expectedElements, int gpuNum, string_view name) {
    requireSameGpu(tensor, gpuNum, name);
    requireDtype(tensor, expectedDtype, name);
    requireNumElements(tensor, expectedElements, name);
}

struct ResolvedRhtAmaxKernel {
    uint32_t num_threads = 0;
    uint32_t rows_per_cta = 0;
};

class RhtAmaxPlanCache {
   public:
    ResolvedRhtAmaxKernel getOrBuild(const CudnnRmsNormRhtAbsMaxDescriptor& descriptor, int gpuNum) {
        const string key = descriptor.cacheKey("forward", gpuNum);
        unique_lock<mutex> lock(mtx);
        auto iter = plans.find(key);
        if (iter != plans.end()) {
            return iter->second;
        }
        descriptor.validate();
        ResolvedRhtAmaxKernel plan{descriptor.resolvedNumThreads(), descriptor.resolvedRowsPerCta()};
        plans.emplace(key, plan);
        return plan;
    }

    void clear() {
        unique_lock<mutex> lock(mtx);
        plans.clear();
    }

    size_t size() const {
        unique_lock<mutex> lock(mtx);
        return plans.size();
    }

   private:
    mutable mutex mtx;
    unordered_map<string, ResolvedRhtAmaxKernel> plans;
};

RhtAmaxPlanCache& cache() {
    static RhtAmaxPlanCache instance;
    return instance;
}

__device__ __forceinline__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warpReduceMax(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <typename ReduceFn>
__device__ float blockReduce(float value, float identity, ReduceFn reduce_fn, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = reduce_fn(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    value = threadIdx.x < num_warps ? shared[lane] : identity;
    if (warp == 0) {
        value = reduce_fn(value);
    }
    __syncthreads();
    return value;
}

struct SumReduce {
    __device__ float operator()(float value) const { return warpReduceSum(value); }
};

struct MaxReduce {
    __device__ float operator()(float value) const { return warpReduceMax(value); }
};

__global__ void rmsNormRhtAmaxBf16Kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ scale,
                                         __nv_bfloat16* __restrict__ y,
                                         float* __restrict__ amax,
                                         uint64_t outer_size,
                                         uint64_t hidden,
                                         float epsilon,
                                         uint32_t rows_per_cta) {
    extern __shared__ float shared[];

    const uint64_t cta = blockIdx.x;
    const uint64_t row_begin = cta * static_cast<uint64_t>(rows_per_cta);
    float thread_amax = 0.0f;

    for (uint32_t r = 0; r < rows_per_cta; ++r) {
        const uint64_t row = row_begin + r;
        if (row >= outer_size) {
            continue;
        }

        float sum_sq = 0.0f;
        for (uint64_t n = threadIdx.x; n < hidden; n += blockDim.x) {
            const float xv = __bfloat162float(x[row * hidden + n]);
            sum_sq += xv * xv;
        }
        sum_sq = blockReduce(sum_sq, 0.0f, SumReduce{}, shared);
        const float inv_rms = rsqrtf(sum_sq / static_cast<float>(hidden) + epsilon);

        for (uint64_t n = threadIdx.x; n < hidden; n += blockDim.x) {
            const uint64_t block_base = (n / 16) * 16;
            const uint32_t col = static_cast<uint32_t>(n & 15u);
            float acc = 0.0f;
#pragma unroll
            for (uint32_t j = 0; j < 16; ++j) {
                const uint64_t idx = row * hidden + block_base + j;
                const float v = __bfloat162float(x[idx]) * inv_rms * __bfloat162float(scale[block_base + j]);
                const int sign = (__popc(j & col) & 1) ? -1 : 1;
                acc += sign > 0 ? v : -v;
            }
            const float out = acc * 0.25f;
            y[row * hidden + n] = __float2bfloat16(out);
            thread_amax = fmaxf(thread_amax, fabsf(out));
        }
    }

    thread_amax = blockReduce(thread_amax, 0.0f, MaxReduce{}, shared);
    if (threadIdx.x == 0) {
        amax[cta] = thread_amax;
    }
}

}  // namespace

uint32_t CudnnRmsNormRhtAbsMaxDescriptor::resolvedNumThreads() const {
    if (numThreads != 0) {
        return numThreads;
    }

    struct TunedEntry {
        uint64_t n;
        uint32_t threads;
    };
    constexpr TunedEntry tuned[] = {
        {2048, 128},
        {4096, 256},
        {7168, 128},
        {8192, 512},
        {16384, 1024},
        {32768, 512},
    };
    for (const auto& entry : tuned) {
        if (normalizedFeatureCount == entry.n) {
            return entry.threads;
        }
    }

    constexpr uint32_t candidates[] = {1024, 512, 256, 128, 64, 32};
    for (uint32_t candidate : candidates) {
        if (normalizedFeatureCount % candidate != 0) {
            continue;
        }
        const uint64_t ept = normalizedFeatureCount / candidate;
        if (ept >= 8 && ept % 8 == 0) {
            return candidate;
        }
    }
    throwInvalidRhtAmax("could not resolve numThreads satisfying N / numThreads >= 8 and divisible by 8");
}

uint32_t CudnnRmsNormRhtAbsMaxDescriptor::resolvedRowsPerCta() const {
    if (rowsPerCta != 0) {
        return rowsPerCta;
    }
    for (uint32_t candidate : {2u, 4u, 8u}) {
        if (outerSize % candidate == 0) {
            return candidate;
        }
    }
    throwInvalidRhtAmax("could not resolve rowsPerCta; outerSize must be divisible by one of {2, 4, 8}");
}

uint64_t CudnnRmsNormRhtAbsMaxDescriptor::absMaxElementCount() const {
    const uint32_t rows = resolvedRowsPerCta();
    if (outerSize % rows != 0) {
        throwInvalidRhtAmax("outerSize must be divisible by rowsPerCta");
    }
    return outerSize / rows;
}

void CudnnRmsNormRhtAbsMaxDescriptor::validate() const {
    if (outerSize == 0) {
        throwInvalidRhtAmax("outerSize must be non-zero");
    }
    if (normalizedFeatureCount == 0) {
        throwInvalidRhtAmax("normalizedFeatureCount must be non-zero");
    }
    (void)checkedMul(outerSize, normalizedFeatureCount, "IO");
    if (inputDataType != TensorDescriptor::DataType::BF16 || outputDataType != TensorDescriptor::DataType::BF16 ||
        parameterDataType != TensorDescriptor::DataType::BF16) {
        throwInvalidRhtAmax("x, scale, and output tensors must all be bf16; got input " + dtypeName(inputDataType) + ", output " +
                            dtypeName(outputDataType) + ", scale " + dtypeName(parameterDataType));
    }
    if (absMaxDataType != TensorDescriptor::DataType::FP32) {
        throwInvalidRhtAmax("amax tensor must be fp32; got " + dtypeName(absMaxDataType));
    }
    if (!(epsilon > 0.0f)) {
        throwInvalidRhtAmax("epsilon must be > 0");
    }
    if (normalizedFeatureCount % 16 != 0) {
        throwInvalidRhtAmax("normalizedFeatureCount must be divisible by the fixed 16-wide Hadamard block size");
    }
    const uint32_t threads = resolvedNumThreads();
    if (threads == 0 || threads > 1024 || (threads & (threads - 1)) != 0) {
        throwInvalidRhtAmax("numThreads must be a power of two in [1, 1024]");
    }
    if (normalizedFeatureCount % threads != 0) {
        throwInvalidRhtAmax("normalizedFeatureCount must be divisible by numThreads");
    }
    const uint64_t ept = normalizedFeatureCount / threads;
    if (ept < 8 || ept % 8 != 0) {
        throwInvalidRhtAmax("EPT = normalizedFeatureCount / numThreads must be at least 8 and divisible by 8");
    }
    const uint32_t rows = resolvedRowsPerCta();
    if (rows != 2 && rows != 4 && rows != 8) {
        throwInvalidRhtAmax("rowsPerCta must be one of {2, 4, 8}");
    }
    if (outerSize % rows != 0) {
        throwInvalidRhtAmax("outerSize must be divisible by rowsPerCta");
    }
}

string CudnnRmsNormRhtAbsMaxDescriptor::cacheKey(string_view passName, int gpuNum) const {
    ostringstream out;
    out << "rmsnorm_rht_amax:" << passName << ":gpu=" << gpuNum << ":outer=" << outerSize << ":hidden=" << normalizedFeatureCount
        << ":eps=" << epsilon << ":threads=" << resolvedNumThreads() << ":rows=" << resolvedRowsPerCta();
    return out.str();
}

CudnnRmsNormRhtAbsMax& CudnnRmsNormRhtAbsMax::instance() {
    static CudnnRmsNormRhtAbsMax singleton;
    return singleton;
}

void CudnnRmsNormRhtAbsMax::forward(const CudnnRmsNormRhtAbsMaxDescriptor& descriptor,
                                    const CudnnRmsNormRhtAbsMaxForwardArgs& args,
                                    Stream stream) {
    descriptor.validate();
    const int gpuNum = stream.getGpuNum();
    requireTensor(args.x, descriptor.inputDataType, checkedMul(descriptor.outerSize, descriptor.normalizedFeatureCount, "x"), gpuNum, "x");
    requireTensor(args.scale, descriptor.parameterDataType, descriptor.normalizedFeatureCount, gpuNum, "scale");
    requireTensor(args.y, descriptor.outputDataType, checkedMul(descriptor.outerSize, descriptor.normalizedFeatureCount, "y"), gpuNum, "y");
    requireTensor(args.absMax, descriptor.absMaxDataType, descriptor.absMaxElementCount(), gpuNum, "amax");

    ScopedGpu scopedGpu(gpuNum);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpuNum));
    if (prop.major < 10) {
        throw runtime_error("RMSNorm+RHT+Amax follows the cuDNN Frontend SM100+ support surface and requires compute capability >= 10.0.");
    }

    const ResolvedRhtAmaxKernel plan = cache().getOrBuild(descriptor, gpuNum);
    const dim3 grid(static_cast<unsigned int>(descriptor.absMaxElementCount()));
    const dim3 block(plan.num_threads);
    const size_t shared_bytes = ((plan.num_threads + 31) / 32) * sizeof(float);
    rmsNormRhtAmaxBf16Kernel<<<grid, block, shared_bytes, stream>>>(args.x.getMemPtr<__nv_bfloat16>(),
                                                                    args.scale.getMemPtr<__nv_bfloat16>(),
                                                                    (__nv_bfloat16*)args.y.getMemPtr<__nv_bfloat16>(),
                                                                    (float*)args.absMax.getMemPtr<float>(),
                                                                    descriptor.outerSize,
                                                                    descriptor.normalizedFeatureCount,
                                                                    descriptor.epsilon,
                                                                    plan.rows_per_cta);
    CUDA_CHECK(cudaPeekAtLastError());
}

void CudnnRmsNormRhtAbsMax::warmForward(const CudnnRmsNormRhtAbsMaxDescriptor& descriptor, int gpuNum) {
    (void)cache().getOrBuild(descriptor, gpuNum);
}

void CudnnRmsNormRhtAbsMax::clearCache() { cache().clear(); }

size_t CudnnRmsNormRhtAbsMax::cachedGraphCount() const { return cache().size(); }

bool CudnnRmsNormRhtAbsMax::frontendAvailable() { return true; }
