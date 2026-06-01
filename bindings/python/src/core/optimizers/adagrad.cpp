#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Adagrad.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_adagrad(nb::module_& optimizers) {
    auto adagrad = nb::class_<Adagrad, Optimizer>(optimizers, "Adagrad");
    adagrad.attr("__module__") = "thor.optimizers";

    adagrad.def_static(
        "__new__",
        [](nb::handle cls, float alpha, float epsilon, shared_ptr<Network> network) -> shared_ptr<Adagrad> {
            if (alpha <= 0.0f) {
                string error_message = "Adagrad builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "Adagrad builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }

            Adagrad::Builder builder;
            builder.alpha(alpha).epsilon(epsilon);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<Adagrad> adagrad = dynamic_pointer_cast<Adagrad>(base);
            if (!adagrad)
                throw nb::type_error("Adagrad builder did not return an Adagrad instance");
            return adagrad;
        },
        "cls"_a,
        "alpha"_a = 0.01f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an Adagrad optimizer.)nbdoc");

    adagrad.def(
        "__init__",
        [](Adagrad* self, float alpha, float epsilon, shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)epsilon;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.01f,
        "epsilon"_a = 1e-7f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an Adagrad optimizer.)nbdoc");

    adagrad.attr("__doc__") = R"doc(
Adagrad optimizer.

Adagrad adapts the learning rate for each parameter using a running sum of
squared gradients. It is often useful for sparse features because frequently
updated parameters receive smaller effective steps over time.

Parameters
----------
alpha : float, default 0.01
    Base learning rate.
epsilon : float, default 1e-7
    Small constant added to the denominator for numerical stability.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
The dense update is:

- ``accumulator <- accumulator + g * g``
- ``w <- w - alpha * g / (sqrt(accumulator) + epsilon)``

where ``g`` is Thor's batch/loss-scale normalized gradient.

For sparse-row embedding updates, the same expression is applied to the touched rows.

Examples
--------
Basic usage::

    from thor.optimizers import Adagrad

    opt = Adagrad(alpha=0.01, epsilon=1e-7)

See Also
--------
Adam : Adaptive Moment Estimation optimizer.
AdamW : Adam with decoupled weight decay.
Sgd : Stochastic Gradient Descent optimizer.
)doc";
}
