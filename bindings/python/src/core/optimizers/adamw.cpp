#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/AdamW.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_adamw(nb::module_ &optimizers) {
    auto adamw = nb::class_<AdamW, Optimizer>(optimizers, "AdamW");
    adamw.attr("__module__") = "thor.optimizers";

    adamw.def_static(
        "__new__",
        [](nb::handle cls,
           float alpha,
           float beta1,
           float beta2,
           float epsilon,
           float weight_decay,
           shared_ptr<Network> network) -> std::shared_ptr<AdamW> {
            if (alpha <= 0.0f) {
                string error_message = "AdamW builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (beta1 < 0.0f || beta1 >= 1.0f) {
                string error_message = "AdamW builder: assertion 0 <= beta1 < 1 failed. beta1: " + to_string(beta1);
                throw nb::value_error(error_message.c_str());
            }
            if (beta2 < 0.0f || beta2 >= 1.0f) {
                string error_message = "AdamW builder: assertion 0 <= beta2 < 1 failed. beta2: " + to_string(beta2);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "AdamW builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }
            if (weight_decay < 0.0f) {
                string error_message = "AdamW builder: weight_decay must be >= 0. weight_decay: " + to_string(weight_decay);
                throw nb::value_error(error_message.c_str());
            }

            AdamW::Builder builder;
            builder.alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).weightDecay(weight_decay);

            if (network != nullptr)
                builder.network(*network);

            std::shared_ptr<Optimizer> base = builder.build();
            std::shared_ptr<AdamW> adamw = std::dynamic_pointer_cast<AdamW>(base);
            if (!adamw)
                throw nb::type_error("AdamW builder did not return an AdamW instance");
            return adamw;
        },
        "cls"_a,
        "alpha"_a = 0.001f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        "weight_decay"_a = 0.01f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an AdamW optimizer.)nbdoc");

    adamw.def(
        "__init__",
        [](AdamW *self,
           float alpha,
           float beta1,
           float beta2,
           float epsilon,
           float weight_decay,
           shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)beta1;
            (void)beta2;
            (void)epsilon;
            (void)weight_decay;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.001f,
        "beta1"_a = 0.9f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-7f,
        "weight_decay"_a = 0.01f,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an AdamW optimizer.)nbdoc");

    adamw.attr("__doc__") = R"doc(
AdamW optimizer.

AdamW is Adam with decoupled weight decay. It maintains first- and second-moment
buffers like Adam, but applies weight decay directly to the parameter rather than
adding an L2 penalty into the gradient.

Parameters
----------
alpha : float, default 0.001
    Base learning rate.
beta1 : float, default 0.9
    Exponential decay rate for the first-moment estimate.
beta2 : float, default 0.999
    Exponential decay rate for the second-moment estimate.
epsilon : float, default 1e-7
    Small constant added to the denominator for numerical stability.
weight_decay : float, default 0.01
    Decoupled weight decay coefficient. Set to 0.0 for Adam-equivalent behavior.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.

Notes
-----
The dense update is:

- ``m <- beta1 * m + (1 - beta1) * g``
- ``v <- beta2 * v + (1 - beta2) * (g * g)``
- ``w <- w - alpha * weight_decay * w - alpha_t * m / (sqrt(v) + epsilon)``

where ``alpha_t`` is the Adam bias-corrected learning rate.

For sparse-row embedding updates, the same expression is applied to the touched rows.

Examples
--------
Basic usage::

    from thor.optimizers import AdamW

    opt = AdamW(alpha=1e-3, weight_decay=0.01)

See Also
--------
Adam : Adaptive Moment Estimation optimizer without decoupled weight decay.
Sgd : Stochastic Gradient Descent optimizer.
)doc";
}
