#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/NAdam.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_nadam(nb::module_& optimizers) {
    auto nadam = nb::class_<NAdam, Optimizer>(optimizers, "NAdam");
    nadam.attr("__module__") = "thor.optimizers";

    nadam.def_static(
        "__new__",
        [](nb::handle cls, float alpha, float beta1, float beta2, float epsilon, shared_ptr<Network> network) -> shared_ptr<NAdam> {
            if (alpha <= 0.0f) {
                string error_message = "NAdam builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (beta1 < 0.0f || beta1 >= 1.0f) {
                string error_message = "NAdam builder: assertion 0 <= beta1 < 1 failed. beta1: " + to_string(beta1);
                throw nb::value_error(error_message.c_str());
            }
            if (beta2 < 0.0f || beta2 >= 1.0f) {
                string error_message = "NAdam builder: assertion 0 <= beta2 < 1 failed. beta2: " + to_string(beta2);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "NAdam builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }

            NAdam::Builder builder;
            builder.alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<NAdam> nadam = dynamic_pointer_cast<NAdam>(base);
            if (!nadam)
                throw nb::type_error("NAdam builder did not return a NAdam instance");
            return nadam;
        },
        "cls"_a,
        "alpha"_a = 0.002f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct a NAdam optimizer.)nbdoc");

    nadam.def(
        "__init__",
        [](NAdam* self, float alpha, float beta1, float beta2, float epsilon, shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)beta1;
            (void)beta2;
            (void)epsilon;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.002f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct a NAdam optimizer.)nbdoc");

    nadam.attr("__doc__") = R"doc(
NAdam optimizer.

NAdam combines Adam's adaptive moments with a Nesterov-style first-moment lookahead.
It maintains first-moment ``m`` and second-moment ``v`` buffers for each parameter.

Parameters
----------
alpha : float, default 0.002
    Base learning rate.
beta1 : float, default 0.9
    Exponential decay rate for the first-moment estimate. Must be in ``[0, 1)``.
beta2 : float, default 0.999
    Exponential decay rate for the second-moment estimate. Must be in ``[0, 1)``.
epsilon : float, default 1e-7
    Small constant added to the denominator for numerical stability.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
The dense update is:

- ``m <- beta1 * m + (1 - beta1) * g``
- ``v <- beta2 * v + (1 - beta2) * g * g``
- ``w <- w - (m_scale * m + gradient_scale * g) / (sqrt(v) + epsilon)``

where ``m_scale`` and ``gradient_scale`` include the learning rate and NAdam bias-correction terms,
and ``g`` is Thor's batch/loss-scale normalized gradient.

For sparse-row embedding updates, the same expression is applied to the touched rows.

Examples
--------
Basic usage::

    from thor.optimizers import NAdam

    opt = NAdam(alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-7)

See Also
--------
Adam : Adaptive Moment Estimation optimizer.
Adamax : Infinity-norm variant of Adam.
AdamW : Adam with decoupled weight decay.
RMSprop : RMSprop optimizer.
)doc";
}
