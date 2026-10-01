#pragma once

#include <cstdint>

namespace ThorImplementation {

/**
 * Semantic I/O role of one physical reduction pass.
 *
 * This role says only whether a pass consumes original input or an intermediate aggregate and whether it produces
 * another aggregate or the requested final output. It does not select reduction topology or constrain pass ordering.
 */
enum class CubReductionPassRole : uint8_t {
    Complete = 0,
    First = 1,
    Intermediate = 2,
    Final = 3,
};

}  // namespace ThorImplementation
