#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Optimizers/RMSprop.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_rmsprop(nb::module_& optimizers) {
    auto rmsprop = nb::class_<RMSprop, Optimizer>(optimizers, "RMSprop");
    rmsprop.attr("__module__") = "thor.optimizers";

    rmsprop.def_static(
        "__new__",
        [](nb::handle cls, float alpha, float rho, float epsilon, shared_ptr<Network> network) -> shared_ptr<RMSprop> {
            if (alpha <= 0.0f) {
                string error_message = "RMSprop builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (rho < 0.0f || rho >= 1.0f) {
                string error_message = "RMSprop builder: assertion 0 <= rho < 1 failed. rho: " + to_string(rho);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "RMSprop builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }

            RMSprop::Builder builder;
            builder.alpha(alpha).rho(rho).epsilon(epsilon);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<RMSprop> rmsprop = dynamic_pointer_cast<RMSprop>(base);
            if (!rmsprop)
                throw nb::type_error("RMSprop builder did not return an RMSprop instance");
            return rmsprop;
        },
        "cls"_a,
        "alpha"_a = 0.001f,
        "rho"_a = 0.9f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an RMSprop optimizer.)nbdoc");

    rmsprop.def(
        "__init__",
        [](RMSprop* self, float alpha, float rho, float epsilon, shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)rho;
            (void)epsilon;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.001f,
        "rho"_a = 0.9f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an RMSprop optimizer.)nbdoc");

    rmsprop.attr("__doc__") = R"doc(
RMSprop optimizer.

RMSprop adapts each parameter's step size using an exponentially decayed moving
average of squared gradients. Compared with Adagrad, the exponential decay keeps
the effective learning rate from shrinking monotonically forever.

Parameters
----------
alpha : float, default 0.001
    Base learning rate.
rho : float, default 0.9
    Exponential decay rate for the running average of squared gradients. Must be
    in ``[0, 1)``.
epsilon : float, default 1e-7
    Small constant added to the denominator for numerical stability.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
The dense update is:

- ``square_average <- rho * square_average + (1 - rho) * g * g``
- ``w <- w - alpha * g / (sqrt(square_average) + epsilon)``

where ``g`` is Thor's batch/loss-scale normalized gradient.

For sparse-row embedding updates, the same expression is applied to the touched rows.

Examples
--------
Basic usage::

    from thor.optimizers import RMSprop

    opt = RMSprop(alpha=0.001, rho=0.9, epsilon=1e-7)

See Also
--------
Adagrad : Accumulated-gradient adaptive optimizer.
Adam : Adaptive Moment Estimation optimizer.
AdamW : Adam with decoupled weight decay.
Sgd : Stochastic Gradient Descent optimizer.
)doc";
}
