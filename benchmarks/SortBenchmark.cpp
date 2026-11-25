// Thor's radix sort of (key, UINT32 value) pairs against cub::DeviceRadixSort::SortPairs, in us per sort.
//
// A sample is `sets` sorts back to back between two events, each on its own input arrays (restored from a master copy
// before the sample, outside the timed interval: a sort overwrites its input); the reported time is the sample over
// the sets, the median of the samples. Inputs are in L2 or not as their size has it, as a sort's are behind the kernel
// that wrote its keys. Every variant is checked once per case against std::stable_sort (CUB's radix sort is stable).
//
// Variants: cub (host item count = the active items, bits [0, key_bits)); thor (the plan's own tile), thor_t512 /
// _t4096 (tiles of that many pairs; --thor-tiles), all with the item count read on the device and grids sized
// for the capacity; thor_one (the one-block sort of RadixSortOneBlock.h, in place, for capacities that fit a block);
// copy (cudaMemcpyAsync of the active keys and values: one pass's traffic). --variants runs only some of cub, thor,
// thor_one and copy. --keys random (the default: uniform over the key bits) and / or consecutive (base + i: the order
// an embedding of positions gives a sort, and the one in which a block's stores to shared memory can all meet one bank).

#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Sort/RadixSort.h"
#include "Utilities/TensorOperations/Sort/RadixSortOneBlock.h"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

struct Options {
    std::vector<uint64_t> items{4096, 16384, 65536, 262144, 1048576, 4194304};
    std::vector<uint32_t> keyBits{12, 17, 21, 32};
    std::vector<double> activeFractions{1.0, 0.5};
    std::vector<uint32_t> thorTiles{0, 512, 4096};
    std::vector<std::string> variants{"cub", "thor", "thor_one", "copy"};  // thor: every --thor-tiles entry
    std::vector<std::string> keyOrders{"random"};
    int samples = 30;
    int warmup = 3;
    uint64_t poolBytes = 256ULL << 20;
    int maxSets = 32;
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
        if (arg == "--items") {
            o.items.clear();
            for (const std::string& s : split(value())) o.items.push_back(std::stoull(s));
        } else if (arg == "--key-bits") {
            o.keyBits.clear();
            for (const std::string& s : split(value())) o.keyBits.push_back(static_cast<uint32_t>(std::stoul(s)));
        } else if (arg == "--active-fractions") {
            o.activeFractions.clear();
            for (const std::string& s : split(value())) o.activeFractions.push_back(std::stod(s));
        } else if (arg == "--thor-tiles") {
            o.thorTiles.clear();
            for (const std::string& s : split(value())) o.thorTiles.push_back(static_cast<uint32_t>(std::stoul(s)));
        } else if (arg == "--variants") {
            o.variants = split(value());
            for (const std::string& v : o.variants) {
                if (v != "cub" && v != "thor" && v != "thor_one" && v != "copy") {
                    throw std::invalid_argument("--variants takes cub, thor, thor_one, copy");
                }
            }
        } else if (arg == "--keys") {
            o.keyOrders = split(value());
            for (const std::string& k : o.keyOrders) {
                if (k != "random" && k != "consecutive") throw std::invalid_argument("--keys takes random, consecutive");
            }
        } else if (arg == "--samples") {
            o.samples = std::stoi(value());
        } else if (arg == "--warmup") {
            o.warmup = std::stoi(value());
        } else if (arg == "--max-sets") {
            o.maxSets = std::stoi(value());
        } else if (arg == "--no-verify") {
            o.verify = false;
        } else if (arg == "--csv") {
            o.csv = value();
        } else {
            throw std::invalid_argument(
                "usage: thor_sort_benchmark [--items a,b] [--key-bits a,b] [--active-fractions a,b] [--thor-tiles "
                "0,512,4096] [--variants cub,thor,thor_one,copy] [--keys random,consecutive] "
                "[--samples n] [--warmup n] [--max-sets n] [--no-verify] [--csv path]");
        }
    }
    return o;
}

struct DeviceArray {
    void* pointer = nullptr;
    explicit DeviceArray(uint64_t bytes) { CUDA_CHECK(cudaMalloc(&pointer, std::max<uint64_t>(bytes, 16))); }
    ~DeviceArray() { cudaFree(pointer); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

struct Set {
    DeviceArray keys, keysAlternate, values, valuesAlternate;
    Set(uint64_t keyBytes, uint64_t valueBytes)
        : keys(keyBytes), keysAlternate(keyBytes), values(valueBytes), valuesAlternate(valueBytes) {}
};

struct Result {
    std::string variant;
    uint32_t passes = 0;
    uint32_t tilePairs = 0;
    uint32_t tiles = 0;
    double median = 0, p10 = 0, p90 = 0;
    std::string check;
};

template <typename KeyT>
void runCase(const Options& o,
             uint64_t capacity,
             uint32_t keyBits,
             double fraction,
             const std::string& keyOrder,
             Stream stream,
             std::ostream* csv) {
    const uint64_t active =
        std::min<uint64_t>(capacity, static_cast<uint64_t>(std::floor(fraction * static_cast<double>(capacity))));
    if (active == 0) return;
    const DataType keyType = sizeof(KeyT) == 2 ? DataType::UINT16 : DataType::UINT32;
    const uint64_t keyBytes = capacity * sizeof(KeyT);
    const uint64_t valueBytes = capacity * sizeof(uint32_t);
    const int sets = static_cast<int>(
        std::clamp<uint64_t>(o.poolBytes / (2 * (keyBytes + valueBytes)), 1, static_cast<uint64_t>(o.maxSets)));
    cudaStream_t s = stream.getStream();

    // Host input, and its stable order.
    std::vector<KeyT> hostKeys(capacity);
    std::vector<uint32_t> hostValues(capacity);
    std::mt19937_64 rng(capacity * 131 + keyBits);
    const uint64_t mask = (uint64_t{1} << keyBits) - 1;
    const uint64_t firstKey = std::mt19937_64(capacity * 977 + keyBits)() & mask;  // rng's keys stay as they were
    for (uint64_t i = 0; i < capacity; ++i) {
        hostKeys[i] = static_cast<KeyT>((keyOrder == "consecutive" ? firstKey + i : rng()) & mask);
        hostValues[i] = static_cast<uint32_t>(i);
    }
    std::vector<uint32_t> order(active);
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return hostKeys[a] < hostKeys[b]; });

    DeviceArray masterKeys(keyBytes), masterValues(valueBytes), activeCount(sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(masterKeys.pointer, hostKeys.data(), keyBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(masterValues.pointer, hostValues.data(), valueBytes, cudaMemcpyHostToDevice));
    const uint32_t active32 = static_cast<uint32_t>(active);
    CUDA_CHECK(cudaMemcpy(activeCount.pointer, &active32, sizeof(active32), cudaMemcpyHostToDevice));
    std::vector<std::unique_ptr<Set>> pool;
    for (int i = 0; i < sets; ++i) pool.push_back(std::make_unique<Set>(keyBytes, valueBytes));

    size_t cubTempBytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                               cubTempBytes,
                                               static_cast<const KeyT*>(pool[0]->keys.pointer),
                                               static_cast<KeyT*>(pool[0]->keysAlternate.pointer),
                                               static_cast<const uint32_t*>(pool[0]->values.pointer),
                                               static_cast<uint32_t*>(pool[0]->valuesAlternate.pointer),
                                               static_cast<int>(active),
                                               0,
                                               static_cast<int>(keyBits),
                                               s));
    DeviceArray cubTemp(cubTempBytes);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto restore = [&]() {
        for (auto& set : pool) {
            CUDA_CHECK(cudaMemcpyAsync(set->keys.pointer, masterKeys.pointer, keyBytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(set->values.pointer, masterValues.pointer, valueBytes, cudaMemcpyDeviceToDevice, s));
        }
    };

    // launch(set) enqueues one sort (or copy) of the set; sortedInAlternate says where its result is.
    auto measure = [&](Result& result, bool sortedInAlternate, bool sorts, const auto& launch) {
        if (o.verify && sorts) {
            restore();
            launch(*pool[0]);
            std::vector<KeyT> gotKeys(active);
            std::vector<uint32_t> gotValues(active);
            CUDA_CHECK(cudaStreamSynchronize(s));
            CUDA_CHECK(cudaMemcpy(gotKeys.data(),
                                  sortedInAlternate ? pool[0]->keysAlternate.pointer : pool[0]->keys.pointer,
                                  active * sizeof(KeyT),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gotValues.data(),
                                  sortedInAlternate ? pool[0]->valuesAlternate.pointer : pool[0]->values.pointer,
                                  active * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < active; ++i) {
                if (gotKeys[i] != hostKeys[order[i]] || gotValues[i] != hostValues[order[i]]) {
                    throw std::runtime_error(result.variant + " FAILED its check at pair " + std::to_string(i) +
                                             " (items " + std::to_string(capacity) + ", active " +
                                             std::to_string(active) + ", key bits " + std::to_string(keyBits) + ")");
                }
            }
            result.check = "ok";
        } else {
            result.check = sorts ? "off" : "-";
        }
        std::vector<double> times;
        for (int sample = 0; sample < o.warmup + o.samples; ++sample) {
            restore();
            CUDA_CHECK(cudaEventRecord(start, s));
            for (auto& set : pool) launch(*set);
            CUDA_CHECK(cudaEventRecord(stop, s));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (sample >= o.warmup) times.push_back(static_cast<double>(ms) * 1000.0 / sets);
        }
        std::sort(times.begin(), times.end());
        result.median = times[times.size() / 2];
        result.p10 = times[times.size() / 10];
        result.p90 = times[times.size() * 9 / 10];
    };

    const auto wanted = [&](const char* variant) {
        return std::find(o.variants.begin(), o.variants.end(), variant) != o.variants.end();
    };
    std::vector<Result> results;
    if (wanted("cub")) {
        Result r;
        r.variant = "cub";
        measure(r, true, true, [&](Set& set) {
            size_t bytes = cubTempBytes;
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(cubTemp.pointer,
                                                       bytes,
                                                       static_cast<const KeyT*>(set.keys.pointer),
                                                       static_cast<KeyT*>(set.keysAlternate.pointer),
                                                       static_cast<const uint32_t*>(set.values.pointer),
                                                       static_cast<uint32_t*>(set.valuesAlternate.pointer),
                                                       static_cast<int>(active),
                                                       0,
                                                       static_cast<int>(keyBits),
                                                       s));
        });
        results.push_back(r);
    }
    for (const uint32_t tilePairs : o.thorTiles) {
        if (!wanted("thor")) break;
        const RadixSortPairsPlan plan = planRadixSortPairs(keyType, capacity, keyBits, tilePairs);
        DeviceArray scratch(plan.scratchWords * sizeof(uint32_t));
        Result r;
        r.variant = tilePairs == 0 ? "thor" : "thor_t" + std::to_string(tilePairs);
        r.passes = plan.passes;
        r.tilePairs = plan.tilePairs;
        r.tiles = plan.tiles;
        measure(r, plan.sortedInAlternate, true, [&](Set& set) {
            launchRadixSortPairs(plan,
                                 set.keys.pointer,
                                 set.keysAlternate.pointer,
                                 static_cast<uint32_t*>(set.values.pointer),
                                 static_cast<uint32_t*>(set.valuesAlternate.pointer),
                                 static_cast<uint32_t*>(scratch.pointer),
                                 activeCount.pointer,
                                 DataType::UINT32,
                                 stream);
        });
        results.push_back(r);
    }
    // The one-block sort, where the capacity fits a block: in place, no scratch.
    if (wanted("thor_one") && capacity <= radixSortOneBlockMaxPairs(keyType)) {
        const RadixSortOneBlockPlan plan = planRadixSortPairsOneBlock(keyType, capacity, keyBits);
        Result r;
        r.variant = "thor_one";
        r.passes = plan.passes;
        r.tilePairs = plan.warps * RadixSortOneBlockPlan::kWarpPairs;
        r.tiles = 1;
        measure(r, false, true, [&](Set& set) {
            launchRadixSortPairsOneBlock(plan,
                                         set.keys.pointer,
                                         static_cast<uint32_t*>(set.values.pointer),
                                         activeCount.pointer,
                                         DataType::UINT32,
                                         stream);
        });
        results.push_back(r);
    }
    if (wanted("copy")) {
        Result r;
        r.variant = "copy";
        measure(r, true, false, [&](Set& set) {
            CUDA_CHECK(cudaMemcpyAsync(
                set.keysAlternate.pointer, set.keys.pointer, active * sizeof(KeyT), cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(set.valuesAlternate.pointer,
                                       set.values.pointer,
                                       active * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice,
                                       s));
        });
        results.push_back(r);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::printf("items %-8llu active %-8llu key bits %-2u (%zu-byte keys, %s), %d sets:",
                static_cast<unsigned long long>(capacity),
                static_cast<unsigned long long>(active),
                keyBits,
                sizeof(KeyT),
                keyOrder.c_str(),
                sets);
    for (const Result& r : results) std::printf("  %s %.2f", r.variant.c_str(), r.median);
    std::printf(" us\n");
    std::fflush(stdout);
    if (csv != nullptr) {
        for (const Result& r : results) {
            *csv << r.variant << ',' << capacity << ',' << active << ',' << keyBits << ',' << sizeof(KeyT) << ','
                 << r.passes << ',' << r.tilePairs << ',' << r.tiles << ',' << sets << ',' << r.median << ',' << r.p10
                 << ',' << r.p90 << ',' << r.check << ',' << keyOrder << '\n';
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
            csvFile << "variant,items,active_items,key_bits,key_bytes,passes,tile_pairs,tiles,sets,median_us,p10_"
                       "us,p90_us,"
                       "check,keys\n";
        }
        std::ostream* csv = o.csv.empty() ? nullptr : &csvFile;
        for (const uint64_t items : o.items) {
            if (items == 0 || items >= (uint64_t{1} << 30)) throw std::invalid_argument("--items takes 1 to 2^30 - 1");
            for (const uint32_t bits : o.keyBits) {
                if (bits == 0 || bits > 32) throw std::invalid_argument("--key-bits takes 1 to 32");
                for (const double fraction : o.activeFractions) {
                    for (const std::string& keyOrder : o.keyOrders) {
                        if (bits <= 16) {
                            runCase<uint16_t>(o, items, bits, fraction, keyOrder, stream, csv);
                        } else {
                            runCase<uint32_t>(o, items, bits, fraction, keyOrder, stream, csv);
                        }
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sort benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
