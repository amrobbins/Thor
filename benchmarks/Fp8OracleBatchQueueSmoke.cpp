#include "benchmarks/Fp8OracleBatchQueue.h"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

using ThorImplementation::CubReductionBenchmarking::Fp8OracleBatchQueue;

int main() {
    struct Action { int type, cell, pass; };
    std::vector<Action> log;
    Fp8OracleBatchQueue queue(128, 5, [&] { log.push_back({2, -1, -1}); });
    for (int id = 0; id < 257; ++id) {
        queue.add({
            [&, id] { log.push_back({0, id, -1}); },
            [&, id](int pass) { log.push_back({1, id, pass}); },
            [&, id] { log.push_back({3, id, -1}); },
        });
    }
    assert(queue.batches() == 2 && queue.pending() == 1);
    queue.flush();
    assert(queue.batches() == 3 && queue.completedCells() == 257);
    assert(queue.pending() == 0);
    const auto previous = log.size();
    queue.flush();
    assert(log.size() == previous);
    std::size_t cursor = 0;
    auto get = [&]() -> Action { assert(cursor < log.size()); return log[cursor++]; };
    assert(get().type == 0); // initial warmup, then sync
    assert(get().type == 2);
    for (int chunk = 0; chunk < 3; ++chunk) {
        const int first = chunk * 128;
        const int count = chunk == 2 ? 1 : 128;
        for (int pass = 0; pass < 5; ++pass)
            for (int id = first; id < first + count; ++id) {
                const auto action = get();
                assert(action.type == 1 && action.cell == id && action.pass == pass);
            }
        assert(get().type == 2); // exactly one measured sync per batch
        for (int id = first; id < first + count; ++id) {
            const auto action = get();
            assert(action.type == 3 && action.cell == id);
        }
    }
    assert(cursor == log.size());
    bool caught = false;
    try { Fp8OracleBatchQueue invalid(0, 5, []{}); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
}
