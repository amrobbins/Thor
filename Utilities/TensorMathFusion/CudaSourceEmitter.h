#pragma once

#include <cstdint>
#include <string>

#include "Utilities/TensorMathFusion/Expression.h"
#include "Utilities/TensorMathFusion/StampedEquation.h"

namespace ThorImplementation {

class CudaSourceEmitter {
   public:
    static std::string emit(const PhysicalExpression& expr, const std::string& kernel_name);

   private:
    static std::string ref(uint32_t idx);
};

}  // namespace ThorImplementation
