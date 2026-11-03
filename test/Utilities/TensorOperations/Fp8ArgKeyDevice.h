#pragma once

// Device evaluation of the FP8 ARG key helpers for Fp8ArgKeyTest (kernels in Fp8ArgKeyDevice.cu).

#include "Utilities/TensorOperations/Cub/Fp8ArgKey.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation::Fp8ArgKeyDevice {

// keys[i], negative_zero[i] = fp8ArgKeys4(words[i], fold) evaluated on GPU 0.
void keys(const std::vector<uint32_t>& words,
          CubReductionInternal::Fp8ArgFold fold,
          std::vector<uint32_t>& keys,
          std::vector<uint32_t>& negative_zero);

// Column ARG of a row-major [rows, cols] FP8 byte matrix on GPU 0: one thread per column splits its rows into `lanes`
// consecutive lane chunks, reduces each with lane candidates, widens with the chunk's row base and merges.
void columns(const std::vector<uint8_t>& input,
             uint32_t rows,
             uint32_t cols,
             uint32_t lanes,
             CubReductionInternal::Fp8ArgFold fold,
             std::vector<uint32_t>& index,
             std::vector<uint32_t>& value_byte);

}  // namespace ThorImplementation::Fp8ArgKeyDevice
