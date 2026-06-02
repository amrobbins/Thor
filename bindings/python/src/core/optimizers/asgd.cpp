#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/ASGD.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_asgd(nb::module_& optimizers) {
    auto asgd = nb::class_<ASGD, Optimizer>(optimizers, "ASGD");
    asgd.attr("__module__") = "thor.optimizers";

    asgd.def_static(
        "__new__",
        [](nb::handle cls,
           float alpha,
           float lambd,
           float power,
           float t0,
           float weight_decay,
           shared_ptr<Network> network) -> shared_ptr<ASGD> {
            if (alpha <= 0.0f) {
                string error_message = "ASGD builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (lambd < 0.0f) {
                string error_message = "ASGD builder: lambd must be >= 0. lambd: " + to_string(lambd);
                throw nb::value_error(error_message.c_str());
            }
            if (power < 0.0f) {
                string error_message = "ASGD builder: power must be >= 0. power: " + to_string(power);
                throw nb::value_error(error_message.c_str());
            }
            if (t0 < 1.0f) {
                string error_message = "ASGD builder: t0 must be >= 1. t0: " + to_string(t0);
                throw nb::value_error(error_message.c_str());
            }
            if (weight_decay < 0.0f) {
                string error_message = "ASGD builder: weight_decay must be >= 0. weight_decay: " + to_string(weight_decay);
                throw nb::value_error(error_message.c_str());
            }

            ASGD::Builder builder;
            builder.alpha(alpha).lambd(lambd).power(power).t0(t0).weightDecay(weight_decay);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<ASGD> asgd = dynamic_pointer_cast<ASGD>(base);
            if (!asgd)
                throw nb::type_error("ASGD builder did not return an ASGD instance");
            return asgd;
        },
        "cls"_a,
        "alpha"_a = 0.01f,
        "lambd"_a = 1e-4f,
        "power"_a = 0.75f,
        "t0"_a = 1e6f,
        "weight_decay"_a = 0.0f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an ASGD optimizer.)nbdoc");

    asgd.def(
        "__init__",
        [](ASGD* self, float alpha, float lambd, float power, float t0, float weight_decay, shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)lambd;
            (void)power;
            (void)t0;
            (void)weight_decay;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.01f,
        "lambd"_a = 1e-4f,
        "power"_a = 0.75f,
        "t0"_a = 1e6f,
        "weight_decay"_a = 0.0f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an ASGD optimizer.)nbdoc");

    asgd.attr("__doc__") = R"doc(
ASGD optimizer (Averaged Stochastic Gradient Descent).

ASGD applies a decayed SGD update and maintains a separate running average of
the dense parameter tensor. The averaged tensor is optimizer state and is saved
when optimizer state saving is enabled.

Parameters
----------
alpha : float, default 0.01
    Base learning rate.
lambd : float, default 1e-4
    ASGD decay coefficient used in the decayed step size and multiplicative
    weight shrinkage.
power : float, default 0.75
    Exponent for the decayed step size schedule.
t0 : float, default 1e6
    First update step at which the averaged parameter tensor starts tracking
    the running average.
weight_decay : float, default 0.0
    Additional coupled weight decay added to the gradient update vector.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
Thor's dense ASGD update is:

- ``eta <- alpha / (1 + lambd * alpha * t) ** power``
- ``w <- (1 - lambd * eta) * w - eta * (g + weight_decay * w)``
- before ``t0``: ``averaged_weights`` is unchanged
- from ``t0`` onward: ``averaged_weights`` tracks the running average of ``w``

where ``g`` is Thor's batch/loss-scale normalized gradient and ``t`` is the
optimizer update count. Sparse-row updates are intentionally not supported
because the averaged parameter tensor is full-weight state and untouched rows
would otherwise become stale.

Examples
--------
Basic usage::

    from thor.optimizers import ASGD

    opt = ASGD(alpha=0.01, lambd=1e-4, power=0.75, t0=1000)

See Also
--------
Sgd : Stochastic gradient descent with optional momentum.
)doc";
}
