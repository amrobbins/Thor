#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

// Host-side submission scheduling only. The CUDA benchmark owns the device
// buffers, streams and timestamp events, and keeps each Work's captures alive
// until collect() returns after the batch-wide stream synchronization.
namespace ThorImplementation::CubReductionBenchmarking {

class Fp8OracleBatchQueue {
public:
    struct Timing {
        double warmup_ms = 0;
        double submit_ms = 0;
        double wait_ms = 0;
        double collect_ms = 0;
    };
    struct Work {
        std::function<void()> warmup;
        std::function<void(int)> submit_pass;
        std::function<void()> collect;
    };

    Fp8OracleBatchQueue(std::size_t max_cells, int passes,
                        std::function<void()> synchronize)
        : max_cells_(max_cells), passes_(passes), synchronize_(std::move(synchronize)) {
        if (max_cells_ == 0 || passes_ <= 0 || !synchronize_)
            throw std::invalid_argument("Invalid FP8 oracle batch scheduler configuration");
        pending_.reserve(max_cells_);
    }

    void add(Work work) {
        if (!work.warmup || !work.submit_pass || !work.collect)
            throw std::invalid_argument("Incomplete FP8 oracle batch work item");
        pending_.push_back(std::move(work));
        if (pending_.size() == max_cells_) flush();
    }

    void flush() {
        if (pending_.empty()) return;
        const auto clock_now = [] { return std::chrono::steady_clock::now(); };
        const auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        if (!warmed_) {
            const auto start = clock_now();
            pending_.front().warmup();
            synchronize_(); // once, outside the timed measurement rounds
            timing_.warmup_ms += ms(start, clock_now());
            warmed_ = true;
        }
        auto start = clock_now();
        for (int pass = 0; pass < passes_; ++pass)
            for (const Work& work : pending_) work.submit_pass(pass);
        timing_.submit_ms += ms(start, clock_now());
        start = clock_now();
        synchronize_(); // only measured-launch sync within this batch
        timing_.wait_ms += ms(start, clock_now());
        start = clock_now();
        for (const Work& work : pending_) work.collect();
        timing_.collect_ms += ms(start, clock_now());
        total_cells_ += pending_.size();
        pending_.clear(); // release all captured CUDA buffers after completion
        ++batches_;
    }

    [[nodiscard]] std::size_t pending() const { return pending_.size(); }
    [[nodiscard]] std::size_t batches() const { return batches_; }
    [[nodiscard]] std::size_t completedCells() const { return total_cells_; }
    [[nodiscard]] Timing timing() const { return timing_; }

private:
    std::size_t max_cells_;
    int passes_;
    std::function<void()> synchronize_;
    std::vector<Work> pending_;
    bool warmed_ = false;
    std::size_t batches_ = 0;
    std::size_t total_cells_ = 0;
    Timing timing_{};
};

} // namespace ThorImplementation::CubReductionBenchmarking
