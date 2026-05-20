#pragma once

#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

void launchSetInt64Pair(int64_t* dest_d, int64_t first, int64_t second, Stream stream);

}  // namespace ThorImplementation
