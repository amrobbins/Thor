#include "Utilities/TensorOperations/Einsum/EinsumParser.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ThorImplementation {
namespace {

std::string removeWhitespace(const std::string& equation) {
    std::string normalized;
    normalized.reserve(equation.size());
    for (unsigned char c : equation) {
        if (!std::isspace(c)) {
            normalized.push_back(static_cast<char>(c));
        }
    }
    return normalized;
}

EinsumSubscript parseSubscript(std::string_view text, const std::string& context) {
    EinsumSubscript subscript;
    subscript.labels.reserve(text.size());

    for (size_t i = 0; i < text.size();) {
        const char c = text[i];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            subscript.labels.push_back(EinsumParser::labelId(c));
            ++i;
            continue;
        }

        if (c == '.') {
            if (i + 2 >= text.size() || text[i + 1] != '.' || text[i + 2] != '.') {
                throw std::invalid_argument(std::string("Einsum ") + context + " contains '.' outside an ellipsis ('...').");
            }
            if (subscript.ellipsis_position.has_value()) {
                throw std::invalid_argument(std::string("Einsum ") + context + " contains more than one ellipsis.");
            }
            subscript.ellipsis_position = subscript.labels.size();
            i += 3;
            continue;
        }

        throw std::invalid_argument(std::string("Einsum ") + context + " contains invalid character '" + c + "'.");
    }

    return subscript;
}

std::vector<std::string_view> splitInputs(std::string_view lhs) {
    std::vector<std::string_view> terms;
    size_t begin = 0;
    while (true) {
        const size_t comma = lhs.find(',', begin);
        if (comma == std::string_view::npos) {
            terms.push_back(lhs.substr(begin));
            break;
        }
        terms.push_back(lhs.substr(begin, comma - begin));
        begin = comma + 1;
    }
    return terms;
}

uint64_t mergeBroadcastDimension(uint64_t current, uint64_t incoming, int32_t label_id) {
    if (current == 0 || current == incoming) {
        return incoming;
    }
    if (current == 1) {
        return incoming;
    }
    if (incoming == 1) {
        return current;
    }

    std::string label;
    if (EinsumParser::isEllipsisLabel(label_id)) {
        label = "ellipsis axis " + std::to_string(label_id - EinsumParser::kEllipsisLabelBase);
    } else {
        label = std::string("label '") + EinsumParser::labelCharacter(label_id) + "'";
    }
    throw std::invalid_argument("Einsum dimension mismatch for " + label + ": " + std::to_string(current) + " vs " +
                                std::to_string(incoming) + ".");
}

std::vector<int32_t> expandSubscript(const EinsumSubscript& subscript, uint32_t local_ellipsis_rank, uint32_t max_ellipsis_rank) {
    std::vector<int32_t> expanded;
    expanded.reserve(subscript.labels.size() + local_ellipsis_rank);

    if (!subscript.ellipsis_position.has_value()) {
        expanded = subscript.labels;
        return expanded;
    }

    const size_t ellipsis_position = *subscript.ellipsis_position;
    expanded.insert(expanded.end(), subscript.labels.begin(), subscript.labels.begin() + ellipsis_position);

    const uint32_t first_ellipsis_axis = max_ellipsis_rank - local_ellipsis_rank;
    for (uint32_t axis = first_ellipsis_axis; axis < max_ellipsis_rank; ++axis) {
        expanded.push_back(EinsumParser::kEllipsisLabelBase + static_cast<int32_t>(axis));
    }

    expanded.insert(expanded.end(), subscript.labels.begin() + ellipsis_position, subscript.labels.end());
    return expanded;
}

std::vector<int32_t> expandOutputSubscript(const EinsumSubscript& subscript, uint32_t ellipsis_rank) {
    if (!subscript.ellipsis_position.has_value()) {
        return subscript.labels;
    }

    std::vector<int32_t> expanded;
    expanded.reserve(subscript.labels.size() + ellipsis_rank);
    const size_t ellipsis_position = *subscript.ellipsis_position;
    expanded.insert(expanded.end(), subscript.labels.begin(), subscript.labels.begin() + ellipsis_position);
    for (uint32_t axis = 0; axis < ellipsis_rank; ++axis) {
        expanded.push_back(EinsumParser::kEllipsisLabelBase + static_cast<int32_t>(axis));
    }
    expanded.insert(expanded.end(), subscript.labels.begin() + ellipsis_position, subscript.labels.end());
    return expanded;
}

}  // namespace

int32_t EinsumParser::labelId(char label) {
    if (label >= 'A' && label <= 'Z') {
        return static_cast<int32_t>(label - 'A');
    }
    if (label >= 'a' && label <= 'z') {
        return 26 + static_cast<int32_t>(label - 'a');
    }
    throw std::invalid_argument(std::string("Invalid einsum label '") + label + "'. Labels must be ASCII letters.");
}

char EinsumParser::labelCharacter(int32_t label_id) {
    if (label_id >= 0 && label_id < 26) {
        return static_cast<char>('A' + label_id);
    }
    if (label_id >= 26 && label_id < kRegularLabelCount) {
        return static_cast<char>('a' + (label_id - 26));
    }
    throw std::invalid_argument("Einsum label id does not identify a regular ASCII label.");
}

bool EinsumParser::isEllipsisLabel(int32_t label_id) {
    return label_id >= kEllipsisLabelBase;
}

EinsumEquation EinsumParser::parse(const std::string& equation) {
    const std::string normalized = removeWhitespace(equation);

    const size_t arrow = normalized.find("->");
    const bool explicit_output = arrow != std::string::npos;
    if (explicit_output && normalized.find("->", arrow + 2) != std::string::npos) {
        throw std::invalid_argument("Einsum equation contains more than one '->'.");
    }

    const std::string_view normalized_view(normalized);
    const std::string_view lhs = explicit_output ? normalized_view.substr(0, arrow) : normalized_view;
    const std::string_view rhs = explicit_output ? normalized_view.substr(arrow + 2) : std::string_view{};

    EinsumEquation parsed;
    parsed.explicit_output = explicit_output;

    const std::vector<std::string_view> input_terms = splitInputs(lhs);
    parsed.inputs.reserve(input_terms.size());

    std::array<uint32_t, kRegularLabelCount> occurrence_counts{};
    bool any_input_ellipsis = false;
    for (size_t operand = 0; operand < input_terms.size(); ++operand) {
        EinsumSubscript subscript = parseSubscript(input_terms[operand], "input operand " + std::to_string(operand));
        any_input_ellipsis = any_input_ellipsis || subscript.ellipsis_position.has_value();
        for (int32_t label : subscript.labels) {
            ++occurrence_counts.at(static_cast<size_t>(label));
        }
        parsed.inputs.push_back(std::move(subscript));
    }

    if (explicit_output) {
        parsed.output = parseSubscript(rhs, "output");

        std::array<bool, kRegularLabelCount> output_seen{};
        for (int32_t label : parsed.output.labels) {
            const size_t index = static_cast<size_t>(label);
            if (output_seen.at(index)) {
                throw std::invalid_argument(std::string("Einsum output repeats label '") + labelCharacter(label) + "'.");
            }
            output_seen[index] = true;
            if (occurrence_counts.at(index) == 0) {
                throw std::invalid_argument(std::string("Einsum output label '") + labelCharacter(label) +
                                            "' does not appear in any input operand.");
            }
        }
    } else {
        if (any_input_ellipsis) {
            parsed.output.ellipsis_position = 0;
        }
        for (int32_t label = 0; label < kRegularLabelCount; ++label) {
            if (occurrence_counts.at(static_cast<size_t>(label)) == 1) {
                parsed.output.labels.push_back(label);
            }
        }
    }

    return parsed;
}

ResolvedEinsumEquation EinsumParser::resolve(const EinsumEquation& equation,
                                             const std::vector<std::vector<uint64_t>>& input_dimensions) {
    if (equation.inputs.size() != input_dimensions.size()) {
        throw std::invalid_argument("Einsum operand count does not match the number of supplied input shapes.");
    }
    if (equation.inputs.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("Einsum operand count exceeds the supported uint32_t range.");
    }

    std::vector<uint32_t> local_ellipsis_ranks(equation.inputs.size(), 0);
    uint32_t max_ellipsis_rank = 0;

    for (size_t operand = 0; operand < equation.inputs.size(); ++operand) {
        const EinsumSubscript& subscript = equation.inputs[operand];
        const size_t rank = input_dimensions[operand].size();
        const size_t named_axis_count = subscript.labels.size();

        for (uint64_t dimension : input_dimensions[operand]) {
            if (dimension == 0) {
                throw std::invalid_argument("Einsum does not support zero-sized Thor tensor dimensions.");
            }
        }

        if (!subscript.ellipsis_position.has_value()) {
            if (rank != named_axis_count) {
                throw std::invalid_argument("Einsum input operand " + std::to_string(operand) + " has rank " +
                                            std::to_string(rank) + " but its subscript names " +
                                            std::to_string(named_axis_count) + " axes and contains no ellipsis.");
            }
            continue;
        }

        if (rank < named_axis_count) {
            throw std::invalid_argument("Einsum input operand " + std::to_string(operand) + " has rank " +
                                        std::to_string(rank) + " but its subscript names " +
                                        std::to_string(named_axis_count) + " non-ellipsis axes.");
        }
        const size_t ellipsis_rank = rank - named_axis_count;
        constexpr size_t max_synthetic_ellipsis_rank =
            static_cast<size_t>(std::numeric_limits<int32_t>::max() - kEllipsisLabelBase);
        if (ellipsis_rank > max_synthetic_ellipsis_rank) {
            throw std::invalid_argument("Einsum ellipsis rank exceeds the normalized label-id range.");
        }
        local_ellipsis_ranks[operand] = static_cast<uint32_t>(ellipsis_rank);
        max_ellipsis_rank = std::max(max_ellipsis_rank, local_ellipsis_ranks[operand]);
    }

    // NumPy requires non-empty input ellipses to survive into the explicit
    // output.  Thor's CPU implementation can therefore delegate to np.einsum
    // without a semantic mismatch.
    if (equation.explicit_output && max_ellipsis_rank > 0 && !equation.output.ellipsis_position.has_value()) {
        throw std::invalid_argument("Einsum explicit output must contain '...' when input ellipsis expands to one or more axes.");
    }

    ResolvedEinsumEquation resolved;
    resolved.explicit_output = equation.explicit_output;
    resolved.ellipsis_rank = max_ellipsis_rank;
    resolved.label_dimensions.resize(kEllipsisLabelBase + max_ellipsis_rank, 0);
    resolved.inputs.reserve(equation.inputs.size());

    std::vector<bool> label_present(resolved.label_dimensions.size(), false);

    for (size_t operand = 0; operand < equation.inputs.size(); ++operand) {
        ResolvedEinsumOperand resolved_operand;
        resolved_operand.ellipsis_rank = local_ellipsis_ranks[operand];
        resolved_operand.axis_labels = expandSubscript(equation.inputs[operand], local_ellipsis_ranks[operand], max_ellipsis_rank);

        if (resolved_operand.axis_labels.size() != input_dimensions[operand].size()) {
            throw std::logic_error("Internal einsum parser error: expanded input labels do not match input rank.");
        }

        std::array<uint64_t, kRegularLabelCount> local_regular_dimensions{};
        for (size_t axis = 0; axis < resolved_operand.axis_labels.size(); ++axis) {
            const int32_t label = resolved_operand.axis_labels[axis];
            const uint64_t dimension = input_dimensions[operand][axis];

            if (!isEllipsisLabel(label)) {
                uint64_t& prior_local_dimension = local_regular_dimensions.at(static_cast<size_t>(label));
                if (prior_local_dimension != 0 && prior_local_dimension != dimension) {
                    throw std::invalid_argument(std::string("Einsum repeated label '") + labelCharacter(label) +
                                                "' in input operand " + std::to_string(operand) +
                                                " must refer to axes with identical dimensions, got " +
                                                std::to_string(prior_local_dimension) + " and " + std::to_string(dimension) + ".");
                }
                prior_local_dimension = dimension;
            }

            const size_t label_index = static_cast<size_t>(label);
            resolved.label_dimensions.at(label_index) =
                mergeBroadcastDimension(resolved.label_dimensions.at(label_index), dimension, label);
            label_present.at(label_index) = true;
        }

        resolved.inputs.push_back(std::move(resolved_operand));
    }

    resolved.output_labels = expandOutputSubscript(equation.output, max_ellipsis_rank);
    resolved.output_dimensions.reserve(resolved.output_labels.size());

    std::vector<bool> output_label_present(resolved.label_dimensions.size(), false);
    for (int32_t label : resolved.output_labels) {
        const size_t label_index = static_cast<size_t>(label);
        if (label_index >= label_present.size() || !label_present[label_index]) {
            // A zero-rank output ellipsis expands to no labels, so every actual
            // expanded output label must have originated in an input.
            throw std::logic_error("Internal einsum parser error: resolved output label is absent from all inputs.");
        }
        resolved.output_dimensions.push_back(resolved.label_dimensions[label_index]);
        output_label_present[label_index] = true;
    }

    for (size_t label = 0; label < label_present.size(); ++label) {
        if (label_present[label] && !output_label_present[label]) {
            resolved.reduction_labels.push_back(static_cast<int32_t>(label));
        }
    }

    return resolved;
}

ResolvedEinsumEquation EinsumParser::parseAndResolve(const std::string& equation,
                                                     const std::vector<std::vector<uint64_t>>& input_dimensions) {
    return resolve(parse(equation), input_dimensions);
}

}  // namespace ThorImplementation
