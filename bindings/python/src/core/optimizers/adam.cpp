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
        [](nb::handle cls, Network &network, float alpha, float beta1, float beta2, float epsilon) -> std::shared_ptr<Adam> {
            Adam::Builder builder;
            builder.network(network).alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon);

            std::shared_ptr<Optimizer> base = builder.build();  // Builder returns shared_ptr<Optimizer>
            std::shared_ptr<Adam> adam = std::dynamic_pointer_cast<Adam>(base);
            if (!adam)
                throw nb::type_error("Adam builder did not return an Adam instance");
            return adam;  // nanobind converts shared_ptr<Adam> to an Adam Python object
        },
        "cls"_a,
        "network"_a,
        "alpha"_a = 0.001f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        // nb::sig("def __new__(cls, network: thor.Network, alpha: float = 0.01, beta1: float = 0.0, beta2: float = 0.0, epsilon: float =
        // 1e-7"
        //         "= False) -> thor.optimizers.Adam"),
        R"nbdoc(Construct an ADAM optimizer.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    adam.def(
        "__init__",
        [](Adam *self, Network &network, float alpha, float beta1, float beta2, float epsilon) -> void {
            // no-op: constructed in __new__
        },
        "network"_a,
        "alpha"_a = 0.001f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        // nb::sig("def __init__(self, network: thor.Network, alpha: float = 0.01, beta1: float = 0.0, beta2: float = 0.0, epsilon: float =
        // "
        //         "1e-7) -> None"),
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
network : thor.Network
    The network whose trainable parameters will be updated.
alpha : float, optional
    Base learning rate (step size). Default is 0.001.
beta1 : float, optional
    Exponential decay rate for the first-moment estimate (mean of gradients).
    Typical values are close to 0.9. Default is 0.9.
beta2 : float, optional
    Exponential decay rate for the second-moment estimate (mean of squared gradients).
    Typical values are close to 0.999. Default is 0.999.
epsilon : float, optional
    Small constant added to the denominator for numerical stability. Default is 1e-7.

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
