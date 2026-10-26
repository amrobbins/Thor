#pragma once
// Conservative FP8 inventory pruning using both broad GPU sweeps,
// joint ablation, and isolated sm_120 SASS comparison.
// Codes: dtype 0=E4M3,1=E5M2; combine 0=SUM,1=PRODUCT,2=MIN,3=MAX;
// transform 0=identity,1=ABS,2=square. Packet/CTA/staged are compiled axes.
// Keep this predicate shared by benchmark candidate enumeration and CUDA
// launch/occupancy instantiation so unsupported specializations stay unreachable.
namespace ThorImplementation::CubReductionInternal {
constexpr bool fp8AlignedTypedPruned(unsigned dtype, unsigned combine,
                                     unsigned transform, unsigned packet,
                                     unsigned cta, bool staged) {
    // Cross-validated against BOTH the 17,784-cell cold-ring census and the
    // earlier 5,472-cell broad typed sweep. Ten of the original 14 proposed
    // deletions actually win in the older sweep (20 winners >=1.3 TB/s).
    // Only four consistently non-winning typed specializations are retired.
    if (staged || packet != 8 || cta != 64 || transform == 0) return false;
    if (dtype == 0)
        return combine == 2 && transform == 2; // E4M3 MIN(square)
    if (dtype == 1)
        return (combine == 3 && (transform == 1 || transform == 2)) ||
               (combine == 2 && transform == 2); // E5M2 MAX(abs/square), MIN(square)
    return false;
}
} // namespace ThorImplementation::CubReductionInternal
