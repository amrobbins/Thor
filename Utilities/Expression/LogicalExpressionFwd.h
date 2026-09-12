#pragma once

#include <memory>

namespace ThorImplementation {

class LogicalExpressionNode;
class LogicalCudaKernelApplication;

using LogicalExpression = std::shared_ptr<const LogicalExpressionNode>;
using LogicalCudaKernelApplicationPtr = std::shared_ptr<const LogicalCudaKernelApplication>;

}  // namespace ThorImplementation
