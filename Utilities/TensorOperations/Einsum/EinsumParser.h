#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ThorImplementation {

// Einsum labels are normalized to stable integer ids so execution backends do
// not need to reason about the textual equation.  Regular labels occupy ids
// [0, 52); expanded ellipsis axes start at kEllipsisLabelBase.
struct EinsumSubscript {
    std::vector<int32_t> labels;
    std::optional<size_t> ellipsis_position;
};

struct EinsumEquation {
    std::vector<EinsumSubscript> inputs;
    EinsumSubscript output;
    bool explicit_output = false;
};

struct ResolvedEinsumOperand {
    // One label id per physical input axis, after expanding any ellipsis.
    std::vector<int32_t> axis_labels;
    uint32_t ellipsis_rank = 0;
};

struct ResolvedEinsumEquation {
    std::vector<ResolvedEinsumOperand> inputs;
    std::vector<int32_t> output_labels;
    std::vector<int32_t> reduction_labels;

    // Indexed by normalized label id.  Entries for unused regular labels are
    // zero.  Every label referenced by inputs/output has a non-zero dimension.
    std::vector<uint64_t> label_dimensions;
    std::vector<uint64_t> output_dimensions;

    uint32_t ellipsis_rank = 0;
    bool explicit_output = false;
};

class EinsumParser {
   public:
    static constexpr int32_t kRegularLabelCount = 52;
    static constexpr int32_t kEllipsisLabelBase = kRegularLabelCount;

    // Parse textual einsum syntax.  Whitespace is ignored.  When no explicit
    // output is present, NumPy-compatible implicit output is synthesized:
    // ellipsis axes first, followed by labels that occur exactly once across
    // all inputs in alphabetical order.
    static EinsumEquation parse(const std::string& equation);

    // Resolve a parsed equation against concrete operand dimensions.  This
    // expands ellipses, validates repeated-label diagonals and broadcasting,
    // and infers the output dimensions and reduction labels.
    static ResolvedEinsumEquation resolve(const EinsumEquation& equation,
                                          const std::vector<std::vector<uint64_t>>& input_dimensions);

    static ResolvedEinsumEquation parseAndResolve(const std::string& equation,
                                                  const std::vector<std::vector<uint64_t>>& input_dimensions);

    static int32_t labelId(char label);
    static char labelCharacter(int32_t label_id);
    static bool isEllipsisLabel(int32_t label_id);
};

}  // namespace ThorImplementation
