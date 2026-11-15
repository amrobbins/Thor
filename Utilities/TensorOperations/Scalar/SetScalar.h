#pragma once

#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

// Writes *first_d = first and *second_d = second with one launch.
void launchSetInt64Pair(int64_t* first_d, int64_t first, int64_t* second_d, int64_t second, Stream stream);

}  // namespace ThorImplementation
