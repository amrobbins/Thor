#include "Utilities/TensorOperations/DeepLearning/ExpressionDenseSoftmax.h"

#include "Utilities/Expression/Expression.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

[[noreturn]] void throwInvalidSoftmax(const string& message) {
    throw invalid_argument("Invalid Expression dense Softmax descriptor: " + message);
}

bool isSupportedInputDtype(DataType dtype) {
    return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

bool isSupportedOutputDtype(DataType dtype) {
    switch (dtype) {
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
        default:
            return false;
    }
}

string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

uint64_t checkedElementCount(const ExpressionDenseSoftmaxDescriptor& descriptor) {
    if (descriptor.outerSize == 0) throwInvalidSoftmax("outerSize must be non-zero");
    if (descriptor.channelCount == 0) throwInvalidSoftmax("channelCount must be non-zero");
    if (descriptor.outerSize > numeric_limits<uint64_t>::max() / descriptor.channelCount) {
        throwInvalidSoftmax("element count overflows uint64_t");
    }
    return descriptor.outerSize * descriptor.channelCount;
}

void requireTensor(const Tensor& tensor,
                   DataType dtype,
                   const ExpressionDenseSoftmaxDescriptor& descriptor,
                   int gpuNum,
                   string_view name) {
    if (!tensor.isInitialized()) {
        throw invalid_argument("Expression dense Softmax tensor '" + string(name) + "' is not initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw invalid_argument("Expression dense Softmax tensor '" + string(name) + "' must be a GPU tensor.");
    }
    if (tensor.getPlacement().getDeviceNum() != gpuNum) {
        throw invalid_argument("Expression dense Softmax tensor '" + string(name) + "' is on GPU " +
                               to_string(tensor.getPlacement().getDeviceNum()) + ", expected GPU " + to_string(gpuNum) + ".");
    }
    if (tensor.getDataType() != dtype) {
        throw invalid_argument("Expression dense Softmax tensor '" + string(name) + "' has dtype " +
                               dtypeName(tensor.getDataType()) + ", expected " + dtypeName(dtype) + ".");
    }
    const vector<uint64_t> expectedDims{descriptor.outerSize, descriptor.channelCount};
    if (tensor.getDimensions() != expectedDims) {
        throw invalid_argument("Expression dense Softmax tensor '" + string(name) +
                               "' must have dimensions [outerSize, channelCount].");
    }
    if (tensor.getTotalNumElements() != checkedElementCount(descriptor)) {
        throw invalid_argument("Expression dense Softmax tensor '" + string(name) + "' has an unexpected element count.");
    }
}

Expression fp32Binary(const Expression& expression) {
    return expression.withDTypes(DataType::FP32, DataType::FP32);
}

Expression fp32Unary(const Expression& expression) {
    return expression.withDTypes(DataType::FP32, DataType::FP32);
}

shared_ptr<FusedEquation> compileSingleOutput(const Expression& output, int gpuNum) {
    return make_shared<FusedEquation>(FusedEquation::compile(Expression::outputs({{"output", output}}).physicalOutputs(), gpuNum));
}

vector<shared_ptr<FusedEquation>> buildForwardEquations(const ExpressionDenseSoftmaxDescriptor& descriptor, int gpuNum) {
    const DataType outputDtype = descriptor.resolvedOutputDataType();
    vector<shared_ptr<FusedEquation>> equations;
    equations.reserve(4);

    // Stage 0: one final-axis max reduction into [outer, 1] FP32 storage.
    {
        const Expression x = Expression::input("x", DataType::FP32, descriptor.inputDataType);
        const Expression rowMax =
            x.reduce_max(/*reduction_axes=*/{1}, /*squeeze_axes=*/{}, DataType::FP32).withOutputDType(DataType::FP32);
        equations.push_back(compileSingleOutput(rowMax, gpuNum));
    }

    // Stage 1: exp(x - rowMax), materialized as FP32.  Keeping this as an
    // explicit boundary avoids immutable Expression composition cloning the max
    // reduction when the numerator and denominator are later recombined.
    {
        const Expression x = Expression::input("x", DataType::FP32, descriptor.inputDataType);
        const Expression rowMax = Expression::input("row_max", DataType::FP32, DataType::FP32);
        const Expression expCentered = fp32Unary(fp32Binary(x - rowMax).exp());
        equations.push_back(compileSingleOutput(expCentered, gpuNum));
    }

    // Stage 2: one final-axis sum reduction into [outer, 1] FP32 storage.
    {
        const Expression expCentered = Expression::input("exp_centered", DataType::FP32, DataType::FP32);
        const Expression rowSum =
            expCentered.reduce_sum(/*reduction_axes=*/{1}, /*squeeze_axes=*/{}, DataType::FP32).withOutputDType(DataType::FP32);
        equations.push_back(compileSingleOutput(rowSum, gpuNum));
    }

    // Stage 3: final materialization in the caller-selected storage dtype.
    if (descriptor.kind == ExpressionDenseSoftmaxKind::Softmax) {
        const Expression expCentered = Expression::input("exp_centered", DataType::FP32, DataType::FP32);
        const Expression rowSum = Expression::input("row_sum", DataType::FP32, DataType::FP32);
        Expression y = fp32Binary(expCentered / rowSum);
        y = y.withDTypes(DataType::FP32, outputDtype);
        equations.push_back(compileSingleOutput(y, gpuNum));
    } else {
        const Expression x = Expression::input("x", DataType::FP32, descriptor.inputDataType);
        const Expression rowMax = Expression::input("row_max", DataType::FP32, DataType::FP32);
        const Expression rowSum = Expression::input("row_sum", DataType::FP32, DataType::FP32);
        const Expression centered = fp32Binary(x - rowMax);
        Expression y = fp32Binary(centered - fp32Unary(rowSum.ln()));
        y = y.withDTypes(DataType::FP32, outputDtype);
        equations.push_back(compileSingleOutput(y, gpuNum));
    }

    return equations;
}

vector<shared_ptr<FusedEquation>> buildBackwardEquations(const ExpressionDenseSoftmaxDescriptor& descriptor, int gpuNum) {
    const DataType outputDtype = descriptor.resolvedOutputDataType();
    vector<shared_ptr<FusedEquation>> equations;

    if (descriptor.kind == ExpressionDenseSoftmaxKind::Softmax) {
        equations.reserve(3);

        // Stage 0: y * dy in FP32.
        {
            const Expression y = Expression::input("y", DataType::FP32, outputDtype);
            const Expression dy = Expression::input("dy", DataType::FP32, outputDtype);
            equations.push_back(compileSingleOutput(fp32Binary(y * dy), gpuNum));
        }

        // Stage 1: reduce_sum(y * dy) in FP32.
        {
            const Expression yDy = Expression::input("y_dy", DataType::FP32, DataType::FP32);
            const Expression rowDot =
                yDy.reduce_sum(/*reduction_axes=*/{1}, /*squeeze_axes=*/{}, DataType::FP32).withOutputDType(DataType::FP32);
            equations.push_back(compileSingleOutput(rowDot, gpuNum));
        }

        // Stage 2: dx = y * (dy - rowDot), narrowed only at the final output.
        {
            const Expression y = Expression::input("y", DataType::FP32, outputDtype);
            const Expression dy = Expression::input("dy", DataType::FP32, outputDtype);
            const Expression rowDot = Expression::input("row_dot", DataType::FP32, DataType::FP32);
            Expression dx = fp32Binary(y * fp32Binary(dy - rowDot));
            dx = dx.withDTypes(DataType::FP32, descriptor.inputDataType);
            equations.push_back(compileSingleOutput(dx, gpuNum));
        }
    } else {
        equations.reserve(2);

        // Stage 0: reduce_sum(dy) in FP32.
        {
            const Expression dy = Expression::input("dy", DataType::FP32, outputDtype);
            const Expression rowDySum =
                dy.reduce_sum(/*reduction_axes=*/{1}, /*squeeze_axes=*/{}, DataType::FP32).withOutputDType(DataType::FP32);
            equations.push_back(compileSingleOutput(rowDySum, gpuNum));
        }

        // Stage 1: dx = dy - exp(y) * sum(dy), narrowed only at the final output.
        {
            const Expression y = Expression::input("y", DataType::FP32, outputDtype);
            const Expression dy = Expression::input("dy", DataType::FP32, outputDtype);
            const Expression rowDySum = Expression::input("row_dy_sum", DataType::FP32, DataType::FP32);
            const Expression expY = fp32Unary(y.exp());
            Expression dx = fp32Binary(dy - fp32Binary(expY * rowDySum));
            dx = dx.withDTypes(DataType::FP32, descriptor.inputDataType);
            equations.push_back(compileSingleOutput(dx, gpuNum));
        }
    }

    return equations;
}

Tensor makeScratchTensor(int gpuNum, DataType dtype, const vector<uint64_t>& dims) {
    return Tensor(TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum), TensorDescriptor(dtype, dims));
}

void runEquation(const FusedEquation& equation,
                 const unordered_map<string, Tensor>& inputs,
                 Tensor& output,
                 Stream& stream) {
    StampedExecutionPlan stamped = equation.stampSingleOutput(inputs, stream, {}, output);
    stamped.run();
}

}  // namespace

const char* ThorImplementation::toString(ExpressionDenseSoftmaxKind kind) {
    switch (kind) {
        case ExpressionDenseSoftmaxKind::Softmax:
            return "softmax";
        case ExpressionDenseSoftmaxKind::LogSoftmax:
            return "log_softmax";
    }
    return "unknown";
}

void ExpressionDenseSoftmaxDescriptor::validateForward() const {
    (void)checkedElementCount(*this);
    if (!isSupportedInputDtype(inputDataType)) {
        if (inputDataType == DataType::FP8_E4M3 || inputDataType == DataType::FP8_E5M2) {
            throwInvalidSoftmax("FP8 input storage is not supported; cast logits to FP16, BF16, or FP32 before Softmax");
        }
        throwInvalidSoftmax("input dtype " + dtypeName(inputDataType) + " is not a supported Softmax input dtype");
    }
    if (!isSupportedOutputDtype(resolvedOutputDataType())) {
        throwInvalidSoftmax("output dtype " + dtypeName(resolvedOutputDataType()) + " is not a supported floating storage dtype");
    }
    if (computeDataType != DataType::FP32) {
        throwInvalidSoftmax("compute dtype must be FP32, got " + dtypeName(computeDataType));
    }
}

void ExpressionDenseSoftmaxDescriptor::validateBackward() const { validateForward(); }

ExpressionDenseSoftmax& ExpressionDenseSoftmax::instance() {
    static ExpressionDenseSoftmax singleton;
    return singleton;
}

ExpressionDenseSoftmaxPlan ExpressionDenseSoftmax::prepareForward(const ExpressionDenseSoftmaxDescriptor& descriptor,
                                                                   int gpuNum) {
    descriptor.validateForward();
    if (gpuNum < 0) throw invalid_argument("Expression dense Softmax GPU number must be non-negative.");
    return ExpressionDenseSoftmaxPlan(
        descriptor, ExpressionDenseSoftmaxPlan::Pass::Forward, gpuNum, buildForwardEquations(descriptor, gpuNum));
}

ExpressionDenseSoftmaxPlan ExpressionDenseSoftmax::prepareBackward(const ExpressionDenseSoftmaxDescriptor& descriptor,
                                                                    int gpuNum) {
    descriptor.validateBackward();
    if (gpuNum < 0) throw invalid_argument("Expression dense Softmax GPU number must be non-negative.");
    return ExpressionDenseSoftmaxPlan(
        descriptor, ExpressionDenseSoftmaxPlan::Pass::Backward, gpuNum, buildBackwardEquations(descriptor, gpuNum));
}

void ExpressionDenseSoftmax::forward(const ExpressionDenseSoftmaxPlan& plan,
                                     const ExpressionDenseSoftmaxForwardArgs& args,
                                     Stream& stream) const {
    if (!plan.isForward()) throw invalid_argument("Expression dense Softmax forward received a backward plan.");
    if (stream.getGpuNum() != plan.gpuNum()) {
        throw invalid_argument("Expression dense Softmax forward stream GPU does not match prepared plan GPU.");
    }
    const ExpressionDenseSoftmaxDescriptor& descriptor = plan.descriptor();
    requireTensor(args.x, descriptor.inputDataType, descriptor, plan.gpuNum(), "x");
    requireTensor(args.y, descriptor.resolvedOutputDataType(), descriptor, plan.gpuNum(), "y");
    if (plan.equationCount() != 4) {
        throw runtime_error("Expression dense Softmax forward plan has an unexpected equation count.");
    }

    const vector<uint64_t> fullDims{descriptor.outerSize, descriptor.channelCount};
    const vector<uint64_t> rowDims{descriptor.outerSize, 1};
    Tensor rowMax = makeScratchTensor(plan.gpuNum(), DataType::FP32, rowDims);
    Tensor expCentered = makeScratchTensor(plan.gpuNum(), DataType::FP32, fullDims);
    Tensor rowSum = makeScratchTensor(plan.gpuNum(), DataType::FP32, rowDims);

    runEquation(plan.equation(0), {{"x", args.x}}, rowMax, stream);
    runEquation(plan.equation(1), {{"x", args.x}, {"row_max", rowMax}}, expCentered, stream);
    runEquation(plan.equation(2), {{"exp_centered", expCentered}}, rowSum, stream);

    Tensor y = args.y;
    if (descriptor.kind == ExpressionDenseSoftmaxKind::Softmax) {
        runEquation(plan.equation(3), {{"exp_centered", expCentered}, {"row_sum", rowSum}}, y, stream);
    } else {
        runEquation(plan.equation(3), {{"x", args.x}, {"row_max", rowMax}, {"row_sum", rowSum}}, y, stream);
    }
}

void ExpressionDenseSoftmax::backward(const ExpressionDenseSoftmaxPlan& plan,
                                      const ExpressionDenseSoftmaxBackwardArgs& args,
                                      Stream& stream) const {
    if (!plan.isBackward()) throw invalid_argument("Expression dense Softmax backward received a forward plan.");
    if (stream.getGpuNum() != plan.gpuNum()) {
        throw invalid_argument("Expression dense Softmax backward stream GPU does not match prepared plan GPU.");
    }
    const ExpressionDenseSoftmaxDescriptor& descriptor = plan.descriptor();
    const DataType outputDtype = descriptor.resolvedOutputDataType();
    requireTensor(args.y, outputDtype, descriptor, plan.gpuNum(), "y");
    requireTensor(args.dy, outputDtype, descriptor, plan.gpuNum(), "dy");
    requireTensor(args.dx, descriptor.inputDataType, descriptor, plan.gpuNum(), "dx");

    const vector<uint64_t> fullDims{descriptor.outerSize, descriptor.channelCount};
    const vector<uint64_t> rowDims{descriptor.outerSize, 1};
    Tensor dx = args.dx;

    if (descriptor.kind == ExpressionDenseSoftmaxKind::Softmax) {
        if (plan.equationCount() != 3) {
            throw runtime_error("Expression dense Softmax backward plan has an unexpected equation count.");
        }
        Tensor yDy = makeScratchTensor(plan.gpuNum(), DataType::FP32, fullDims);
        Tensor rowDot = makeScratchTensor(plan.gpuNum(), DataType::FP32, rowDims);
        runEquation(plan.equation(0), {{"y", args.y}, {"dy", args.dy}}, yDy, stream);
        runEquation(plan.equation(1), {{"y_dy", yDy}}, rowDot, stream);
        runEquation(plan.equation(2), {{"y", args.y}, {"dy", args.dy}, {"row_dot", rowDot}}, dx, stream);
    } else {
        if (plan.equationCount() != 2) {
            throw runtime_error("Expression dense LogSoftmax backward plan has an unexpected equation count.");
        }
        Tensor rowDySum = makeScratchTensor(plan.gpuNum(), DataType::FP32, rowDims);
        runEquation(plan.equation(0), {{"dy", args.dy}}, rowDySum, stream);
        runEquation(plan.equation(1), {{"y", args.y}, {"dy", args.dy}, {"row_dy_sum", rowDySum}}, dx, stream);
    }
}
