#include "Utilities/TensorMathFusion/StampedEquation.h"
#include "Utilities/TensorMathFusion/CudaHelpers.h"
#include "Utilities/TensorMathFusion/EquationRunner.h"
#include "Utilities/TensorMathFusion/FusedEquation.h"

#include <stdexcept>
#include <vector>

namespace ThorImplementation {

void StampedEquation::run() {
    if (deviceBroadcastInfo.isPresent())
        EquationRunner::run(compiledEquation, inputs, output, stream, deviceBroadcastInfo);
    else
        EquationRunner::run(compiledEquation, inputs, output, stream);
}

StampedReduction::StampedReduction(
    std::shared_ptr<BuiltReduction> built, const Tensor& input, const Tensor& output, const Stream& stream, Optional<Tensor> workspace)
    : built_reduction(built), input(input), output(output), workspace(workspace), stream(stream) {
    if (built_reduction->workspace_bytes != 0) {
        assert(workspace.isPresent());
        assert(workspace.get().getArraySizeInBytes() >= built_reduction->workspace_bytes);
    }
    assert(input.getDataType() == built_reduction->key.inout_dtype);
    assert(output.getDataType() == built_reduction->key.inout_dtype);
}

void StampedReduction::run() {
    // FIXME: Indices when support arg max
    CUDNN_CHECK(cudnnReduceTensor(stream.getCudnnHandle(),
                                  built_reduction->reduce_desc,
                                  nullptr,
                                  0,
                                  workspace.get().getMemPtr(),
                                  built_reduction->workspace_bytes,
                                  alpha,
                                  built_reduction->a_desc,
                                  input.getMemPtr(),
                                  beta,
                                  built_reduction->c_desc,
                                  output.getMemPtr()));
}

}  // namespace ThorImplementation
