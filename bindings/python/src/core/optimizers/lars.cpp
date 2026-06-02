#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Lars.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_lars(nb::module_& optimizers) {
    auto lars = nb::class_<Lars, Optimizer>(optimizers, "Lars");
    lars.attr("__module__") = "thor.optimizers";

    lars.def_static(
        "__new__",
        [](nb::handle cls,
           float alpha,
           float momentum,
           float weight_decay,
           float trust_coefficient,
           float epsilon,
           bool nesterov_momentum,
           shared_ptr<Network> network) -> shared_ptr<Lars> {
            if (alpha <= 0.0f) {
                string error_message = "Lars builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (momentum < 0.0f) {
                string error_message = "Lars builder: momentum must be >= 0. momentum: " + to_string(momentum);
                throw nb::value_error(error_message.c_str());
            }
            if (weight_decay < 0.0f) {
                string error_message = "Lars builder: weight_decay must be >= 0. weight_decay: " + to_string(weight_decay);
                throw nb::value_error(error_message.c_str());
            }
            if (trust_coefficient <= 0.0f) {
                string error_message = "Lars builder: trust_coefficient must be > 0. trust_coefficient: " + to_string(trust_coefficient);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "Lars builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }

            Lars::Builder builder;
            builder.alpha(alpha)
                .momentum(momentum)
                .weightDecay(weight_decay)
                .trustCoefficient(trust_coefficient)
                .epsilon(epsilon)
                .useNesterovMomentum(nesterov_momentum);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<Lars> lars = dynamic_pointer_cast<Lars>(base);
            if (!lars)
                throw nb::type_error("Lars builder did not return a Lars instance");
            return lars;
        },
        "cls"_a,
        "alpha"_a = 0.01f,
        "momentum"_a = 0.9f,
        "weight_decay"_a = 0.0f,
        "trust_coefficient"_a = 0.001f,
        "epsilon"_a = 1e-8f,
        "nesterov_momentum"_a = false,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct a LARS optimizer.)nbdoc");

    lars.def(
        "__init__",
        [](Lars* self,
           float alpha,
           float momentum,
           float weight_decay,
           float trust_coefficient,
           float epsilon,
           bool nesterov_momentum,
           shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)momentum;
            (void)weight_decay;
            (void)trust_coefficient;
            (void)epsilon;
            (void)nesterov_momentum;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.01f,
        "momentum"_a = 0.9f,
        "weight_decay"_a = 0.0f,
        "trust_coefficient"_a = 0.001f,
        "epsilon"_a = 1e-8f,
        "nesterov_momentum"_a = false,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct a LARS optimizer.)nbdoc");

    lars.attr("__doc__") = R"doc(
LARS optimizer.

LARS applies SGD with momentum and a layer-wise adaptive trust ratio. It is often
used for large-batch convolutional training.

Parameters
----------
alpha : float, default 0.01
    Base learning rate.
momentum : float, default 0.9
    Momentum coefficient.
weight_decay : float, default 0.0
    Coupled weight decay coefficient included in the LARS update vector.
trust_coefficient : float, default 0.001
    Coefficient used to scale the layer-wise trust ratio.
epsilon : float, default 1e-8
    Small constant added to the trust-ratio denominator.
nesterov_momentum : bool, default False
    Whether to use Nesterov momentum.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
The dense update is:

- ``u <- g + weight_decay * w``
- ``trust_ratio <- trust_coefficient * ||w||_2 / (||g||_2 + weight_decay * ||w||_2 + epsilon)``
- ``v <- momentum * v + alpha * trust_ratio * u``
- ``w <- w - v``

where ``g`` is Thor's batch/loss-scale normalized gradient. For 1-D tensors,
Thor uses a trust ratio of 1.0, matching the common practice of excluding bias
and normalization parameters from layer-wise scaling.

Sparse-row embedding updates are intentionally not supported because true LARS
needs layer-wide norms, while sparse-row optimizer fusion only sees touched rows.

Examples
--------
Basic usage::

    from thor.optimizers import Lars

    opt = Lars(alpha=0.1, momentum=0.9, weight_decay=1e-4)

See Also
--------
Sgd : Stochastic gradient descent with optional momentum.
Lamb : Adam-style optimizer with a layer-wise trust ratio.
)doc";
}
