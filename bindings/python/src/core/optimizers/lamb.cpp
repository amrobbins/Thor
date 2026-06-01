#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Lamb.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_lamb(nb::module_& optimizers) {
    auto lamb = nb::class_<Lamb, Optimizer>(optimizers, "Lamb");
    lamb.attr("__module__") = "thor.optimizers";

    lamb.def_static(
        "__new__",
        [](nb::handle cls,
           float alpha,
           float beta1,
           float beta2,
           float epsilon,
           float weight_decay,
           float trust_ratio_epsilon,
           shared_ptr<Network> network) -> shared_ptr<Lamb> {
            if (alpha <= 0.0f) {
                string error_message = "Lamb builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (beta1 < 0.0f || beta1 >= 1.0f) {
                string error_message = "Lamb builder: assertion 0 <= beta1 < 1 failed. beta1: " + to_string(beta1);
                throw nb::value_error(error_message.c_str());
            }
            if (beta2 < 0.0f || beta2 >= 1.0f) {
                string error_message = "Lamb builder: assertion 0 <= beta2 < 1 failed. beta2: " + to_string(beta2);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "Lamb builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }
            if (weight_decay < 0.0f) {
                string error_message = "Lamb builder: weight_decay must be >= 0. weight_decay: " + to_string(weight_decay);
                throw nb::value_error(error_message.c_str());
            }
            if (trust_ratio_epsilon <= 0.0f) {
                string error_message = "Lamb builder: trust_ratio_epsilon must be > 0. trust_ratio_epsilon: " + to_string(trust_ratio_epsilon);
                throw nb::value_error(error_message.c_str());
            }

            Lamb::Builder builder;
            builder.alpha(alpha)
                .beta1(beta1)
                .beta2(beta2)
                .epsilon(epsilon)
                .weightDecay(weight_decay)
                .trustRatioEpsilon(trust_ratio_epsilon);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<Lamb> lamb = dynamic_pointer_cast<Lamb>(base);
            if (!lamb)
                throw nb::type_error("Lamb builder did not return a Lamb instance");
            return lamb;
        },
        "cls"_a,
        "alpha"_a = 0.001f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-6f,
        "weight_decay"_a = 0.01f,
        "trust_ratio_epsilon"_a = 1e-6f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct a LAMB optimizer.)nbdoc");

    lamb.def(
        "__init__",
        [](Lamb* self,
           float alpha,
           float beta1,
           float beta2,
           float epsilon,
           float weight_decay,
           float trust_ratio_epsilon,
           shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)beta1;
            (void)beta2;
            (void)epsilon;
            (void)weight_decay;
            (void)trust_ratio_epsilon;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.001f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-6f,
        "weight_decay"_a = 0.01f,
        "trust_ratio_epsilon"_a = 1e-6f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct a LAMB optimizer.)nbdoc");

    lamb.attr("__doc__") = R"doc(
LAMB optimizer.

LAMB combines Adam-style first/second moment adaptation with a layer-wise trust
ratio. It is commonly used for large-batch transformer training.

Parameters
----------
alpha : float, default 0.001
    Base learning rate.
beta1 : float, default 0.9
    Exponential decay rate for the first-moment estimate.
beta2 : float, default 0.999
    Exponential decay rate for the second-moment estimate.
epsilon : float, default 1e-6
    Small constant added to the Adam denominator for numerical stability.
weight_decay : float, default 0.01
    Weight decay coefficient included in the layer-wise update vector.
trust_ratio_epsilon : float, default 1e-6
    Small constant added to the trust-ratio denominator.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
The dense update is:

- ``m <- beta1 * m + (1 - beta1) * g``
- ``v <- beta2 * v + (1 - beta2) * (g * g)``
- ``u <- m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w``
- ``trust_ratio <- ||w||_2 / (||u||_2 + trust_ratio_epsilon)``
- ``w <- w - alpha * trust_ratio * u``

where ``g`` is Thor's batch/loss-scale normalized gradient. For 1-D tensors,
Thor uses a trust ratio of 1.0, matching the common practice of excluding bias
and normalization parameters from LAMB's layer-wise scaling.

Sparse-row embedding updates are intentionally not supported because true LAMB
needs layer-wide norms, while sparse-row optimizer fusion only sees touched rows.

Examples
--------
Basic usage::

    from thor.optimizers import Lamb

    opt = Lamb(alpha=1e-3, weight_decay=0.01)

See Also
--------
AdamW : Adam with decoupled weight decay.
Muon : Momentum optimizer with Newton-Schulz orthogonalized matrix updates.
)doc";
}
