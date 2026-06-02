#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Adadelta.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_adadelta(nb::module_& optimizers) {
    auto adadelta = nb::class_<Adadelta, Optimizer>(optimizers, "Adadelta");
    adadelta.attr("__module__") = "thor.optimizers";

    adadelta.def_static(
        "__new__",
        [](nb::handle cls, float alpha, float rho, float epsilon, shared_ptr<Network> network) -> shared_ptr<Adadelta> {
            if (alpha <= 0.0f) {
                string error_message = "Adadelta builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (rho < 0.0f || rho >= 1.0f) {
                string error_message = "Adadelta builder: assertion 0 <= rho < 1 failed. rho: " + to_string(rho);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "Adadelta builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }

            Adadelta::Builder builder;
            builder.alpha(alpha).rho(rho).epsilon(epsilon);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<Adadelta> adadelta = dynamic_pointer_cast<Adadelta>(base);
            if (!adadelta)
                throw nb::type_error("Adadelta builder did not return an Adadelta instance");
            return adadelta;
        },
        "cls"_a,
        "alpha"_a = 1.0f,
        "rho"_a = 0.95f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an Adadelta optimizer.)nbdoc");

    adadelta.def(
        "__init__",
        [](Adadelta* self, float alpha, float rho, float epsilon, shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)rho;
            (void)epsilon;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 1.0f,
        "rho"_a = 0.95f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an Adadelta optimizer.)nbdoc");

    adadelta.attr("__doc__") = R"doc(
Adadelta optimizer.

Adadelta adapts each parameter's step size using exponentially decayed moving
averages of squared gradients and squared updates. Unlike Adagrad, the update
history keeps the effective learning rate from shrinking monotonically forever.

Parameters
----------
alpha : float, default 1.0
    Global learning-rate multiplier applied to the Adadelta update.
rho : float, default 0.95
    Exponential decay rate for the running averages. Must be in ``[0, 1)``.
epsilon : float, default 1e-7
    Small constant added inside the root-mean-square terms for numerical stability.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
The dense update is:

- ``gradient_square_average <- rho * gradient_square_average + (1 - rho) * g * g``
- ``update <- sqrt(update_square_average + epsilon) / sqrt(gradient_square_average + epsilon) * g``
- ``update_square_average <- rho * update_square_average + (1 - rho) * update * update``
- ``w <- w - alpha * update``

where ``g`` is Thor's batch/loss-scale normalized gradient.

For sparse-row embedding updates, the same expression is applied to the touched rows.

Examples
--------
Basic usage::

    from thor.optimizers import Adadelta

    opt = Adadelta(alpha=1.0, rho=0.95, epsilon=1e-7)

See Also
--------
Adagrad : Accumulated-gradient adaptive optimizer.
RMSprop : Moving-average adaptive optimizer.
Adam : Adaptive Moment Estimation optimizer.
)doc";
}
