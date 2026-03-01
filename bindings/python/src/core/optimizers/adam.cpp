#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Adam.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_adam(nb::module_ &optimizers) {
    auto adam = nb::class_<Adam, Optimizer>(optimizers, "Adam");
    adam.attr("__module__") = "thor.optimizers";

    adam.def_static(
        "__new__",
        [](nb::handle cls, float alpha, float beta1, float beta2, float epsilon, shared_ptr<Network> network) -> std::shared_ptr<Adam> {
            if (alpha <= 0.0f) {
                string error_message = "Adam builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            // beta1, beta2 are exponential decay rates; require [0, 1)
            if (beta1 < 0.0f || beta1 >= 1.0f) {
                string error_message = "Adam builder: assertion 0 <= beta1 < 1 failed. beta1: " + to_string(beta1);
                throw nb::value_error(error_message.c_str());
            }
            if (beta2 < 0.0f || beta2 >= 1.0f) {
                string error_message = "Adam builder: assertion 0 <= beta2 < 1 failed. beta2: " + to_string(beta2);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "Adam builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }

            Adam::Builder builder;
            builder.alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon);

            if (network != nullptr)
                builder.network(*network);

            std::shared_ptr<Optimizer> base = builder.build();  // Builder returns shared_ptr<Optimizer>
            std::shared_ptr<Adam> adam = std::dynamic_pointer_cast<Adam>(base);
            if (!adam)
                throw nb::type_error("Adam builder did not return an Adam instance");
            return adam;  // nanobind converts shared_ptr<Adam> to an Adam Python object
        },
        "cls"_a,
        "alpha"_a = 0.001f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an ADAM optimizer.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    adam.def(
        "__init__",
        [](Adam *self, float alpha, float beta1, float beta2, float epsilon, shared_ptr<Network> network) -> void {
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.001f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an ADAM optimizer.)nbdoc");

    adam.attr("__doc__") = R"doc(
ADAM optimizer (Adaptive Moment Estimation).

Adam is an adaptive learning-rate optimizer that combines ideas from momentum and
RMSProp by maintaining exponentially decayed moving averages of the gradient
(first moment) and of the squared gradient (second moment). These moment estimates
are bias-corrected and used to scale the parameter update for each weight (and bias)
individually.

Parameters
----------
alpha : float, default 0.001
    Base learning rate (step size).
beta1 : float, default 0.9
    Exponential decay rate for the first-moment estimate (mean of gradients).
    Typical values are close to 0.9.
beta2 : float, default 0.999
    Exponential decay rate for the second-moment estimate (mean of squared gradients).
    Typical values are close to 0.999.
epsilon : float, default fp32: 1e-7, fp16: 1e-4
    Small constant added to the denominator for numerical stability.
network : thor.Network, default None
    When network is passed in, then this optimizer will be set as the default optimizer in
    the network and attached to all layers that do not have a layer specific optimizer
    already attached, at network stamping time. You would not pass network here when you
    want this optimizer to be specific to one or more layers, but not applied to the others
    by default.

Notes
-----
Adam maintains, for each parameter, a first-moment buffer ``m`` and a second-moment
buffer ``v``:

- ``m <- beta1 * m + (1 - beta1) * g``
- ``v <- beta2 * v + (1 - beta2) * (g * g)``

It then uses bias-corrected moments:

- ``m_hat = m / (1 - beta1^t)``
- ``v_hat = v / (1 - beta2^t)``

and applies the update:

- ``w <- w - alpha * m_hat / (sqrt(v_hat) + epsilon)``

where ``t`` is the update step count.

Bias parameters (if present) are updated in the same way as weights, each with their
own moment buffers.

Examples
--------
Basic usage::

    from thor.optimizers import Adam

    opt = Adam(network)

Custom hyperparameters::

    opt = Adam(network, alpha=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-7)

See Also
--------
Sgd : Stochastic Gradient Descent optimizer (optionally with momentum / Nesterov).
RmsProp : RMSProp optimizer.
)doc";
}
