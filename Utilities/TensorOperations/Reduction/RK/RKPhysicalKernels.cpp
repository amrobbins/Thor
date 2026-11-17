#include "Utilities/TensorOperations/Reduction/RK/RKPhysicalKernels.h"

#include <stdexcept>

namespace ThorImplementation::ReductionInternal::RK {
void launchPhysicalFirstStage(const PhysicalKernel& p, const ReductionStageSemantics& semantics,
                              const StridedSpan& input, const StridedSpan& output,
                              const ReductionGeometry& geometry, float output_scale,
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

}  // namespace ThorImplementation::ReductionInternal::RK
