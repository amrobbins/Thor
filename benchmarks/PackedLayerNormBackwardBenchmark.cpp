// Packed LayerNorm backward (launchPackedLayerNormBackwardActivePrefix: dX, dScale, dBias over the active rows) against
// cuDNN's LayerNorm backward on the same dense [rows, hidden] tensors (the kernel Thor's non-packed LayerNorm backward
// runs), and against a device copy of one [rows, hidden] tensor.
//
// Reported per variant: the median us per launch and TB/s of the bytes the operation has to move at least: read x and
// dY, write dX (3 x rows x hidden x element size), read mean and inv_std (8 bytes a row), read scale and write dScale
// and dBias (12 bytes a column). The copy's TB/s counts its read and its write.
//
// A sample is `sets` launches back to back between two events, each on its own x / dY / dX / statistics / gradients
// (so a set's inputs are not left in L2 by the launch before when the sets together are larger than L2); the reported
// time is the sample over the sets, the median of the samples. Every variant is checked once per case against a CPU
// reference in double (dX: every element; dScale, dBias: every column), see checkResult.
//
// Options: --hidden a,b (default 256,512,768,1024,2048,4096,8192,16384), --rows a,b (default 2048,8192,32768),
// --dtypes bf16,fp16,fp32, --variants thor,cudnn,copy, --samples n, --warmup n, --pool-bytes n (the sets together are
// at least this, default 1 GiB), --max-sets n, --max-elements n (cases with more rows x hidden are skipped, default
// 2^29), --no-verify, --csv path.

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/Expression/PackedLayerNormBackwardKernel.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

struct Options {
    std::vector<uint64_t> hidden{256, 512, 768, 1024, 2048, 4096, 8192, 16384};
    std::vector<uint64_t> rows{2048, 8192, 32768};
    std::vector<std::string> dtypes{"bf16", "fp16", "fp32"};
    std::vector<std::string> variants{"thor", "cudnn", "copy"};
    int samples = 20;
    int warmup = 3;
    uint64_t poolBytes = 1ULL << 30;
    int maxSets = 64;
    uint64_t maxElements = 1ULL << 29;
    bool verify = true;
    std::string csv;
};

std::vector<std::string> split(const std::string& raw) {
    std::vector<std::string> out;
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(arg + " takes a value");
            return argv[++i];
        };
        auto numbers = [&]() {
            std::vector<uint64_t> out;
            for (const std::string& s : split(value())) out.push_back(std::stoull(s));
            return out;
        };
        if (arg == "--hidden") {
            o.hidden = numbers();
        } else if (arg == "--rows") {
            o.rows = numbers();
        } else if (arg == "--dtypes") {
            o.dtypes = split(value());
            for (const std::string& d : o.dtypes) {
                if (d != "bf16" && d != "fp16" && d != "fp32") throw std::invalid_argument("--dtypes takes bf16, fp16, fp32");
            }
        } else if (arg == "--variants") {
            o.variants = split(value());
            for (const std::string& v : o.variants) {
                if (v != "thor" && v != "cudnn" && v != "copy") throw std::invalid_argument("--variants takes thor, cudnn, copy");
            }
        } else if (arg == "--samples") {
            o.samples = std::stoi(value());
        } else if (arg == "--warmup") {
            o.warmup = std::stoi(value());
        } else if (arg == "--pool-bytes") {
            o.poolBytes = std::stoull(value());
        } else if (arg == "--max-sets") {
            o.maxSets = std::stoi(value());
        } else if (arg == "--max-elements") {
            o.maxElements = std::stoull(value());
        } else if (arg == "--no-verify") {
            o.verify = false;
        } else if (arg == "--csv") {
            o.csv = value();
        } else {
            throw std::invalid_argument(
                "usage: thor_packed_layernorm_backward_benchmark [--hidden a,b] [--rows a,b] [--dtypes bf16,fp16,fp32] "
                "[--variants thor,cudnn,copy] [--samples n] [--warmup n] [--pool-bytes n] [--max-sets n] "
                "[--max-elements n] [--no-verify] [--csv path]");
        }
    }
    if (o.samples < 1 || o.warmup < 0 || o.maxSets < 1) throw std::invalid_argument("bad --samples, --warmup or --max-sets");
    return o;
}

// Element types: host storage, rounding of a float to the type, and the type's unit roundoff (for the dX check).
template <typename T>
struct Element;
template <>
struct Element<float> {
    static constexpr DataType dtype = DataType::FP32;
    static constexpr const char* name = "fp32";
    static constexpr double unitRoundoff = 0x1p-24;
    static float fromFloat(float v) { return v; }
    static float toFloat(float v) { return v; }
};
template <>
struct Element<__half> {
    static constexpr DataType dtype = DataType::FP16;
    static constexpr const char* name = "fp16";
    static constexpr double unitRoundoff = 0x1p-11;
    static __half fromFloat(float v) { return __float2half_rn(v); }
    static float toFloat(__half v) { return __half2float(v); }
};
template <>
struct Element<__nv_bfloat16> {
    static constexpr DataType dtype = DataType::BF16;
    static constexpr const char* name = "bf16";
    static constexpr double unitRoundoff = 0x1p-8;
    static __nv_bfloat16 fromFloat(float v) { return __float2bfloat16_rn(v); }
    static float toFloat(__nv_bfloat16 v) { return __bfloat162float(v); }
};

// Deterministic, parallel-friendly pseudo-random values: a hash of the index, two uniforms summed (roughly normal).
uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
float noise(uint64_t seed, uint64_t i) {
    const uint64_t h = splitmix(seed * 0x2545F4914F6CDD1DULL + i);
    const float a = static_cast<float>(h >> 40) * 0x1p-24f;
    const float b = static_cast<float>((h >> 16) & 0xFFFFFF) * 0x1p-24f;
    return (a + b - 1.0f) * 1.7320508f;  // about unit variance
}

// The case's inputs on the host, already rounded to T, and the reference gradients in double.
template <typename T>
struct HostCase {
    uint64_t rows = 0, hidden = 0;
    std::vector<T> x, dy;
    std::vector<float> gamma, mean, invStd;
    // Reference: dX per element; per row the scale of the dX terms (for the check's bound); dScale, dBias and the sums
    // of their terms' magnitudes.
    std::vector<double> dx, rowTermScale;
    std::vector<double> dgamma, dbeta, dgammaAbs, dbetaAbs;
};

template <typename T>
HostCase<T> makeHostCase(uint64_t rows, uint64_t hidden, bool reference) {
    HostCase<T> c;
    c.rows = rows;
    c.hidden = hidden;
    const uint64_t n = rows * hidden;
    c.x.resize(n);
    c.dy.resize(n);
    c.gamma.resize(hidden);
    c.mean.resize(rows);
    c.invStd.resize(rows);
    for (uint64_t j = 0; j < hidden; ++j) c.gamma[j] = 1.0f + 0.1f * noise(3, j);
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
        const float offset = 0.5f * noise(5, r);
        const float spread = 0.5f + 0.25f * (noise(7, r) + 1.8f);
        double sum = 0.0;
        for (uint64_t j = 0; j < hidden; ++j) {
            const uint64_t i = r * hidden + j;
            c.x[i] = Element<T>::fromFloat(offset + spread * noise(11, i));
            c.dy[i] = Element<T>::fromFloat(0.1f * noise(13, i));
            sum += Element<T>::toFloat(c.x[i]);
        }
        const double mu = sum / static_cast<double>(hidden);
        double var = 0.0;
        for (uint64_t j = 0; j < hidden; ++j) {
            const double d = Element<T>::toFloat(c.x[r * hidden + j]) - mu;
            var += d * d;
        }
        var /= static_cast<double>(hidden);
        c.mean[r] = static_cast<float>(mu);
        c.invStd[r] = static_cast<float>(1.0 / std::sqrt(var + 1.0e-5));
    }
    if (!reference) return c;

    // Reference with the statistics as stored (FP32), as the kernels use them.
    c.dx.resize(n);
    c.rowTermScale.resize(rows);
    const int threads = omp_get_max_threads();
    std::vector<std::vector<double>> partG(threads, std::vector<double>(hidden, 0.0)),
        partB(threads, std::vector<double>(hidden, 0.0)), partGA(threads, std::vector<double>(hidden, 0.0)),
        partBA(threads, std::vector<double>(hidden, 0.0));
#pragma omp parallel
    {
        const int t = omp_get_thread_num();
        std::vector<double>& g = partG[t];
        std::vector<double>& b = partB[t];
        std::vector<double>& ga = partGA[t];
        std::vector<double>& ba = partBA[t];
#pragma omp for schedule(static)
        for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
            const double mu = c.mean[r], inv = c.invStd[r];
            double sumQ = 0.0, sumQZ = 0.0, absQ = 0.0, absQZ = 0.0;
            for (uint64_t j = 0; j < hidden; ++j) {
                const uint64_t i = r * hidden + j;
                const double dyv = Element<T>::toFloat(c.dy[i]);
                const double z = (Element<T>::toFloat(c.x[i]) - mu) * inv;
                const double q = dyv * c.gamma[j];
                sumQ += q;
                sumQZ += q * z;
                absQ += std::fabs(q);
                absQZ += std::fabs(q * z);
                g[j] += dyv * z;
                b[j] += dyv;
                ga[j] += std::fabs(dyv * z);
                ba[j] += std::fabs(dyv);
            }
            const double h = static_cast<double>(hidden);
            for (uint64_t j = 0; j < hidden; ++j) {
                const uint64_t i = r * hidden + j;
                const double z = (Element<T>::toFloat(c.x[i]) - mu) * inv;
                const double q = Element<T>::toFloat(c.dy[i]) * c.gamma[j];
                c.dx[i] = inv * (q - sumQ / h - z * sumQZ / h);
            }
            // The FP32 sums' rounding shows in dX in proportion to inv x (mean |q| + max |z| x mean |q z|); max |z| is
            // at most sqrt(hidden).
            c.rowTermScale[r] = inv * (absQ / h + std::sqrt(h) * absQZ / h);
        }
    }
    c.dgamma.assign(hidden, 0.0);
    c.dbeta.assign(hidden, 0.0);
    c.dgammaAbs.assign(hidden, 0.0);
    c.dbetaAbs.assign(hidden, 0.0);
    for (int t = 0; t < threads; ++t) {
        for (uint64_t j = 0; j < hidden; ++j) {
            c.dgamma[j] += partG[t][j];
            c.dbeta[j] += partB[t][j];
            c.dgammaAbs[j] += partGA[t][j];
            c.dbetaAbs[j] += partBA[t][j];
        }
    }
    return c;
}

TensorPlacement gpuPlacement() { return TensorPlacement(TensorPlacement::MemDevices::GPU, 0); }

Tensor uploadBytes(const void* data, DataType dtype, const std::vector<uint64_t>& dims, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(dtype, dims));
    std::memcpy(host.getMemPtr(), data, host.getArraySizeInBytes());
    Tensor device(gpuPlacement(), host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

template <typename U>
std::vector<U> download(const Tensor& device, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), device.getDescriptor());
    host.copyFromAsync(device, stream);
    stream.synchronize();
    std::vector<U> out(host.getArraySizeInBytes() / sizeof(U));
    std::memcpy(out.data(), host.getMemPtr(), out.size() * sizeof(U));
    return out;
}

struct Set {
    Tensor x, dy, dx, mean, invStd, dgamma, dbeta;
};

struct Result {
    std::string variant;
    double median = 0, p10 = 0, p90 = 0, tbps = 0;
    // Largest error over its bound (dX, dScale, dBias); a check passes at <= 1.
    double dxErr = 0, dgammaErr = 0, dbetaErr = 0;
    std::string check = "-";
};

// dX: |got - ref| <= 2 u(T) |ref| + 1e-5 x the row's term scale (FP32 sums: ~hidden x 2^-24 relative at worst for
// hidden <= 16384, which 1e-5 covers; the output's rounding is u(T)). dScale / dBias: within 1e-5 of the sum of the
// column's term magnitudes (FP32 sums over the rows), plus 1e-30.
template <typename T>
void checkResult(const HostCase<T>& c, const Set& set, Stream& stream, Result& r) {
    const std::vector<T> dx = download<T>(set.dx, stream);
    const std::vector<float> dg = download<float>(set.dgamma, stream);
    const std::vector<float> db = download<float>(set.dbeta, stream);
    double dxErr = 0.0;
#pragma omp parallel for schedule(static) reduction(max : dxErr)
    for (int64_t r0 = 0; r0 < static_cast<int64_t>(c.rows); ++r0) {
        for (uint64_t j = 0; j < c.hidden; ++j) {
            const uint64_t i = r0 * c.hidden + j;
            const double got = Element<T>::toFloat(dx[i]);
            const double bound = 2.0 * Element<T>::unitRoundoff * std::fabs(c.dx[i]) + 1.0e-5 * c.rowTermScale[r0] + 1e-30;
            const double e = std::isfinite(got) ? std::fabs(got - c.dx[i]) / bound : 1e30;
            dxErr = std::max(dxErr, e);
        }
    }
    double gErr = 0.0, bErr = 0.0;
    for (uint64_t j = 0; j < c.hidden; ++j) {
        const double ge = std::isfinite(dg[j]) ? std::fabs(dg[j] - c.dgamma[j]) / (1.0e-5 * c.dgammaAbs[j] + 1e-30) : 1e30;
        const double be = std::isfinite(db[j]) ? std::fabs(db[j] - c.dbeta[j]) / (1.0e-5 * c.dbetaAbs[j] + 1e-30) : 1e30;
        gErr = std::max(gErr, ge);
        bErr = std::max(bErr, be);
    }
    r.dxErr = dxErr;
    r.dgammaErr = gErr;
    r.dbetaErr = bErr;
    r.check = (dxErr <= 1.0 && gErr <= 1.0 && bErr <= 1.0) ? "ok" : "FAIL";
}

template <typename T>
void runCase(const Options& o, uint64_t rows, uint64_t hidden, Stream& stream, std::ostream* csv) {
    const DataType dtype = Element<T>::dtype;
    const uint64_t n = rows * hidden;
    const uint64_t tensorBytes = n * sizeof(T);
    const uint64_t setBytes = 3 * tensorBytes + rows * 8 + hidden * 8;
    const int sets = static_cast<int>(std::clamp<uint64_t>(o.poolBytes / setBytes + 1, 1, static_cast<uint64_t>(o.maxSets)));
    const double movedBytes = 3.0 * static_cast<double>(tensorBytes) + 8.0 * rows + 12.0 * hidden;
    cudaStream_t s = stream.getStream();

    const HostCase<T> host = makeHostCase<T>(rows, hidden, o.verify);
    const std::vector<uint64_t> dims{rows, hidden};
    Tensor gamma = uploadBytes(host.gamma.data(), DataType::FP32, {hidden}, stream);
    std::vector<Set> pool;
    pool.reserve(sets);
    {
        Set first;
        first.x = uploadBytes(host.x.data(), dtype, dims, stream);
        first.dy = uploadBytes(host.dy.data(), dtype, dims, stream);
        first.mean = uploadBytes(host.mean.data(), DataType::FP32, {rows}, stream);
        first.invStd = uploadBytes(host.invStd.data(), DataType::FP32, {rows}, stream);
        pool.push_back(first);
    }
    for (int i = 1; i < sets; ++i) {
        Set set;
        set.x = pool[0].x.clone();
        set.dy = pool[0].dy.clone();
        set.mean = pool[0].mean.clone();
        set.invStd = pool[0].invStd.clone();
        set.x.copyFromAsync(pool[0].x, stream);
        set.dy.copyFromAsync(pool[0].dy, stream);
        set.mean.copyFromAsync(pool[0].mean, stream);
        set.invStd.copyFromAsync(pool[0].invStd, stream);
        pool.push_back(set);
    }
    for (Set& set : pool) {
        set.dx = Tensor(gpuPlacement(), TensorDescriptor(dtype, dims));
        set.dgamma = Tensor(gpuPlacement(), TensorDescriptor(DataType::FP32, {hidden}));
        set.dbeta = Tensor(gpuPlacement(), TensorDescriptor(DataType::FP32, {hidden}));
    }
    stream.synchronize();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Poison the outputs, run once on set 0 and check, then time.
    auto measure = [&](Result& r, bool computes, const auto& launch) {
        if (o.verify && computes) {
            CUDA_CHECK(cudaMemsetAsync(pool[0].dx.getMemPtr(), 0xFF, pool[0].dx.getArraySizeInBytes(), s));
            CUDA_CHECK(cudaMemsetAsync(pool[0].dgamma.getMemPtr(), 0xFF, hidden * sizeof(float), s));
            CUDA_CHECK(cudaMemsetAsync(pool[0].dbeta.getMemPtr(), 0xFF, hidden * sizeof(float), s));
            launch(pool[0]);
            CUDA_CHECK(cudaStreamSynchronize(s));
            checkResult(host, pool[0], stream, r);
        }
        std::vector<double> times;
        for (int sample = 0; sample < o.warmup + o.samples; ++sample) {
            CUDA_CHECK(cudaEventRecord(start, s));
            for (Set& set : pool) launch(set);
            CUDA_CHECK(cudaEventRecord(stop, s));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (sample >= o.warmup) times.push_back(static_cast<double>(ms) * 1000.0 / sets);
        }
        std::sort(times.begin(), times.end());
        r.median = times[times.size() / 2];
        r.p10 = times[times.size() / 10];
        r.p90 = times[times.size() * 9 / 10];
    };

    const auto wanted = [&](const char* v) { return std::find(o.variants.begin(), o.variants.end(), v) != o.variants.end(); };
    std::vector<Result> results;
    if (wanted("thor")) {
        Result r;
        r.variant = "thor";
        measure(r, true, [&](Set& set) {
            launchPackedLayerNormBackwardActivePrefix(
                set.x, gamma, set.dy, set.mean, set.invStd, set.dx, set.dgamma, set.dbeta, rows, s);
        });
        r.tbps = movedBytes / (r.median * 1e6);
        results.push_back(r);
    }
    if (wanted("cudnn")) {
        CudnnLayerNormDescriptor d;
        d.outerSize = rows;
        d.normalizedFeatureCount = hidden;
        d.inputDataType = dtype;
        d.outputDataType = dtype;
        d.parameterDataType = DataType::FP32;
        d.computeDataType = DataType::FP32;
        d.epsilon = 1.0e-5f;
        d.training = true;
        d.debugName = "packed_layernorm_backward_benchmark";
        Result r;
        r.variant = "cudnn";
        try {
            CudnnLayerNormExecutablePlan plan = CudnnLayerNorm::instance().prepareBackward(d, stream);
            std::optional<Tensor> workspace;
            if (plan.workspaceBytes() > 0) {
                workspace = Tensor(gpuPlacement(), TensorDescriptor(DataType::UINT8, {plan.workspaceBytes()}), 256);
            }
            measure(r, true, [&](Set& set) {
                CudnnLayerNormBackwardArgs args;
                args.dy = set.dy;
                args.x = set.x;
                args.scale = gamma;
                args.mean = set.mean;
                args.invVariance = set.invStd;
                args.dx = set.dx;
                args.dscale = set.dgamma;
                args.dbias = set.dbeta;
                CudnnLayerNorm::instance().backward(plan, args, workspace, stream);
            });
            r.tbps = movedBytes / (r.median * 1e6);
        } catch (const std::exception& e) {
            std::cerr << "cudnn unavailable for rows " << rows << " hidden " << hidden << " " << Element<T>::name << ": "
                      << e.what() << '\n';
            r.check = "unavailable";
        }
        results.push_back(r);
    }
    if (wanted("copy")) {
        Result r;
        r.variant = "copy";
        measure(r, false, [&](Set& set) {
            CUDA_CHECK(cudaMemcpyAsync(set.dx.getMemPtr(), set.x.getMemPtr(), tensorBytes, cudaMemcpyDeviceToDevice, s));
        });
        r.tbps = 2.0 * static_cast<double>(tensorBytes) / (r.median * 1e6);
        results.push_back(r);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::printf("%s rows %-7llu hidden %-6llu sets %-2d:", Element<T>::name, static_cast<unsigned long long>(rows),
                static_cast<unsigned long long>(hidden), sets);
    for (const Result& r : results) {
        std::printf("  %s %.2f us %.3f TB/s [%s]", r.variant.c_str(), r.median, r.tbps, r.check.c_str());
    }
    std::printf("\n");
    std::fflush(stdout);
    if (csv != nullptr) {
        for (const Result& r : results) {
            *csv << r.variant << ',' << Element<T>::name << ',' << rows << ',' << hidden << ',' << sets << ',' << r.median
                 << ',' << r.p10 << ',' << r.p90 << ',' << r.tbps << ',' << r.dxErr << ',' << r.dgammaErr << ','
                 << r.dbetaErr << ',' << r.check << '\n';
        }
        csv->flush();
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        ScopedGpu scopedGpu(0);
        Stream stream(0);
        std::ofstream csvFile;
        if (!o.csv.empty()) {
            csvFile.open(o.csv);
            csvFile << "variant,dtype,rows,hidden,sets,median_us,p10_us,p90_us,tbps,dx_err,dscale_err,dbias_err,check\n";
        }
        std::ostream* csv = o.csv.empty() ? nullptr : &csvFile;
        for (const std::string& dtype : o.dtypes) {
            for (const uint64_t hidden : o.hidden) {
                for (const uint64_t rows : o.rows) {
                    if (rows == 0 || hidden == 0) throw std::invalid_argument("--rows and --hidden take positive values");
                    if (rows * hidden > o.maxElements) continue;
                    if (dtype == "bf16") {
                        runCase<__nv_bfloat16>(o, rows, hidden, stream, csv);
                    } else if (dtype == "fp16") {
                        runCase<__half>(o, rows, hidden, stream, csv);
                    } else {
                        runCase<float>(o, rows, hidden, stream, csv);
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "packed layernorm backward benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
