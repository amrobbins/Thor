#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Adamax.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_adamax(nb::module_& optimizers) {
    auto adamax = nb::class_<Adamax, Optimizer>(optimizers, "Adamax");
    adamax.attr("__module__") = "thor.optimizers";

    adamax.def_static(
        "__new__",
        [](nb::handle cls, float alpha, float beta1, float beta2, float epsilon, shared_ptr<Network> network) -> shared_ptr<Adamax> {
            if (alpha <= 0.0f) {
                string error_message = "Adamax builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (beta1 < 0.0f || beta1 >= 1.0f) {
                string error_message = "Adamax builder: assertion 0 <= beta1 < 1 failed. beta1: " + to_string(beta1);
                throw nb::value_error(error_message.c_str());
            }
            if (beta2 < 0.0f || beta2 >= 1.0f) {
                string error_message = "Adamax builder: assertion 0 <= beta2 < 1 failed. beta2: " + to_string(beta2);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "Adamax builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }

            Adamax::Builder builder;
            builder.alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<Adamax> adamax = dynamic_pointer_cast<Adamax>(base);
            if (!adamax)
                throw nb::type_error("Adamax builder did not return an Adamax instance");
            return adamax;
        },
        "cls"_a,
        "alpha"_a = 0.002f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an Adamax optimizer.)nbdoc");

    adamax.def(
        "__init__",
        [](Adamax* self, float alpha, float beta1, float beta2, float epsilon, shared_ptr<Network> network) -> void {
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
        R"nbdoc(Construct an Adamax optimizer.)nbdoc");

    adamax.attr("__doc__") = R"doc(
Adamax optimizer.

Adamax is the infinity-norm variant of Adam. It maintains a first-moment buffer
``m`` and an exponentially decayed infinity-norm buffer ``u`` for each parameter.

Parameters
----------
alpha : float, default 0.002
    Base learning rate.
beta1 : float, default 0.9
    Exponential decay rate for the first-moment estimate. Must be in ``[0, 1)``.
beta2 : float, default 0.999
    Exponential decay rate for the infinity-norm estimate. Must be in ``[0, 1)``.
epsilon : float, default 1e-7
    Small constant added to the denominator for numerical stability.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
The dense update is:

- ``m <- beta1 * m + (1 - beta1) * g``
- ``u <- max(beta2 * u, abs(g))``
- ``w <- w - (alpha / (1 - beta1**t)) * m / (u + epsilon)``

where ``g`` is Thor's batch/loss-scale normalized gradient.

For sparse-row embedding updates, the same expression is applied to the touched rows.

Examples
--------
Basic usage::

    from thor.optimizers import Adamax

    opt = Adamax(alpha=0.002, beta1=0.9, beta2=0.999, epsilon=1e-7)

See Also
--------
Adam : Adaptive Moment Estimation optimizer.
AdamW : Adam with decoupled weight decay.
RMSprop : RMSprop optimizer.
Sgd : Stochastic Gradient Descent optimizer.
)doc";
}
