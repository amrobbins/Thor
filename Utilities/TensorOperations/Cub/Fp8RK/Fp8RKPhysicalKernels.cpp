#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
void launchPhysicalFirstStage(const PhysicalKernel& p, const CubReductionStageSemantics& semantics,
                              const Tensor& input, Tensor& output,
                              const CubReductionGeometry& geometry, float output_scale,
                              Stream& stream) {
    switch (p.family) {
        case PhysicalFamily::OneWindow:
            launchOneWindow(p, semantics, input, output, geometry, output_scale, stream); return;
        case PhysicalFamily::DeepRSmallK:
            launchDeepRSmallK(p, semantics, input, output, geometry, stream); return;
        case PhysicalFamily::AlignedK:
            launchAlignedK(p, semantics, input, output, geometry, output_scale, stream); return;
        case PhysicalFamily::RuntimeK:
            launchRuntimeK(p, semantics, input, output, geometry, output_scale, stream); return;
    }
    throw std::logic_error("Unknown physical FP8 RK family");
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8RK
