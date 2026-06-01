#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Muon.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_muon(nb::module_& optimizers) {
    auto muon = nb::class_<Muon, Optimizer>(optimizers, "Muon");
    muon.attr("__module__") = "thor.optimizers";

    muon.def_static(
        "__new__",
        [](nb::handle cls,
           float alpha,
           float beta,
           float epsilon,
           float weight_decay,
           bool nesterov,
           uint32_t num_iterations,
           float coefficient_a,
           float coefficient_b,
           float coefficient_c,
           bool transpose_tall_matrices,
           shared_ptr<Optimizer> fallback_optimizer,
           shared_ptr<Network> network) -> shared_ptr<Muon> {
            (void)cls;
            if (alpha <= 0.0f)
                throw nb::value_error(("Muon builder: alpha must be > 0. alpha: " + to_string(alpha)).c_str());
            if (beta < 0.0f || beta >= 1.0f)
                throw nb::value_error(("Muon builder: assertion 0 <= beta < 1 failed. beta: " + to_string(beta)).c_str());
            if (epsilon <= 0.0f)
                throw nb::value_error(("Muon builder: epsilon must be > 0. epsilon: " + to_string(epsilon)).c_str());
            if (weight_decay < 0.0f)
                throw nb::value_error(("Muon builder: weight_decay must be >= 0. weight_decay: " + to_string(weight_decay)).c_str());
            if (num_iterations == 0)
                throw nb::value_error("Muon builder: num_iterations must be > 0.");

            Muon::Builder builder;
            builder.alpha(alpha)
                .beta(beta)
                .epsilon(epsilon)
                .weightDecay(weight_decay)
                .nesterov(nesterov)
                .numIterations(num_iterations)
                .coefficientA(coefficient_a)
                .coefficientB(coefficient_b)
                .coefficientC(coefficient_c)
                .transposeTallMatrices(transpose_tall_matrices);

            if (fallback_optimizer != nullptr)
                builder.fallbackOptimizer(fallback_optimizer);
            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<Muon> muon = dynamic_pointer_cast<Muon>(base);
            if (!muon)
                throw nb::type_error("Muon builder did not return a Muon instance");
            return muon;
        },
        "cls"_a,
        "alpha"_a = 0.02f,
        "beta"_a = 0.95f,
        "epsilon"_a = 1.0e-8f,
        "weight_decay"_a = 0.0f,
        "nesterov"_a = true,
        "num_iterations"_a = 5,
        "coefficient_a"_a = 3.4445f,
        "coefficient_b"_a = -4.775f,
        "coefficient_c"_a = 2.0315f,
        "transpose_tall_matrices"_a = true,
        "fallback_optimizer"_a.none() = nb::none(),
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct a Muon optimizer.)nbdoc");

    muon.def(
        "__init__",
        [](Muon* self,
           float alpha,
           float beta,
           float epsilon,
           float weight_decay,
           bool nesterov,
           uint32_t num_iterations,
           float coefficient_a,
           float coefficient_b,
           float coefficient_c,
           bool transpose_tall_matrices,
           shared_ptr<Optimizer> fallback_optimizer,
           shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)beta;
            (void)epsilon;
            (void)weight_decay;
            (void)nesterov;
            (void)num_iterations;
            (void)coefficient_a;
            (void)coefficient_b;
            (void)coefficient_c;
            (void)transpose_tall_matrices;
            (void)fallback_optimizer;
            (void)network;
        },
        "alpha"_a = 0.02f,
        "beta"_a = 0.95f,
        "epsilon"_a = 1.0e-8f,
        "weight_decay"_a = 0.0f,
        "nesterov"_a = true,
        "num_iterations"_a = 5,
        "coefficient_a"_a = 3.4445f,
        "coefficient_b"_a = -4.775f,
        "coefficient_c"_a = 2.0315f,
        "transpose_tall_matrices"_a = true,
        "fallback_optimizer"_a.none() = nb::none(),
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct a Muon optimizer.)nbdoc");

    muon.attr("__doc__") = R"doc(
Muon optimizer.

Muon applies momentum to dense rank-2 matrix parameters, orthogonalizes the
resulting update with Newton-Schulz iterations, and applies a decoupled weight
decay term. Non-matrix parameters and sparse-row updates are routed to a fallback
optimizer. The builder default fallback is AdamW.

Parameters
----------
alpha : float, default 0.02
    Matrix-path learning rate.
beta : float, default 0.95
    Momentum coefficient.
epsilon : float, default 1e-8
    Newton-Schulz normalization epsilon.
weight_decay : float, default 0.0
    Decoupled matrix-path weight decay.
nesterov : bool, default True
    Whether to use a Nesterov-style momentum source before orthogonalization.
num_iterations : int, default 5
    Number of Newton-Schulz iterations.
fallback_optimizer : thor.optimizers.Optimizer, default None
    Optimizer used for non-matrix parameters and sparse-row updates. When omitted,
    AdamW is used.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.
)doc";
}
