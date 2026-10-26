// CPU-only parity driver. Fields are supplied by the test fixture generator:
// outer R K dtype combine transform family local_rank p1 p2 p3 p4 p5
#include "benchmarks/Fp8GeneralRParallelSelector.h"
#include <iomanip>
#include <iostream>
using namespace ThorImplementation::CubReductionBenchmarking;
namespace G = Fp8GeneralRParallelPolicy;
int main() {
    uint64_t outer,r,k,a,b,c,d,e;
    unsigned fmt,comb,transform,family,rank;
    std::cout << std::setprecision(10);
    while (std::cin >> outer >> r >> k >> fmt >> comb >> transform >> family >> rank >> a >> b >> c >> d >> e) {
        G::Selection sel{};
        sel.mode = static_cast<G::Mode>(family);
        switch (sel.mode) {
            case G::Mode::OneWindow:
                sel.configuration = G::OneWindowConfiguration{{static_cast<uint32_t>(a),static_cast<uint32_t>(b),static_cast<uint32_t>(c)}};
                break;
            case G::Mode::DeepRSmallK:
                sel.configuration = G::DeepRSmallKConfiguration{{static_cast<uint32_t>(a)}};
                break;
            case G::Mode::AlignedK:
                sel.configuration = G::AlignedKConfiguration{{static_cast<uint32_t>(a),static_cast<uint32_t>(b),c}};
                break;
            case G::Mode::RuntimeK:
                sel.configuration = G::RuntimeKConfiguration{{static_cast<Fp8RuntimeKPolicy::Layout>(a),static_cast<uint32_t>(b),static_cast<uint32_t>(c),static_cast<uint32_t>(d),e,Fp8RuntimeKPolicy::ceilDiv(r,e)}};
                break;
        }
        G::CandidatePlan plan{sel,1,0,1};
        G::RankerSemantics semantics{static_cast<G::Fp8Format>(fmt),static_cast<G::Combine>(comb),static_cast<G::InputTransform>(transform)};
        const auto features = G::Detail::rankerFeatures(plan,outer,r,k,semantics,rank);
        for (float f : features) std::cout << f << ' ';
        std::cout << Fp8GeometryRankerModel::margin(features) << '\n';
    }
}
