#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Optimizers/Adafactor.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_adafactor(nb::module_& optimizers) {
    auto adafactor = nb::class_<Adafactor, Optimizer>(optimizers, "Adafactor");
    adafactor.attr("__module__") = "thor.optimizers";

    adafactor.def_static(
        "__new__",
        [](nb::handle cls,
           float alpha,
           float beta2,
           float epsilon,
           float weight_decay,
           bool factor_second_moment,
           shared_ptr<Network> network) -> shared_ptr<Adafactor> {
            (void)cls;
            if (alpha <= 0.0f) {
                string error_message = "Adafactor builder: alpha must be > 0. alpha: " + to_string(alpha);
                throw nb::value_error(error_message.c_str());
            }
            if (beta2 < 0.0f || beta2 >= 1.0f) {
                string error_message = "Adafactor builder: assertion 0 <= beta2 < 1 failed. beta2: " + to_string(beta2);
                throw nb::value_error(error_message.c_str());
            }
            if (epsilon <= 0.0f) {
                string error_message = "Adafactor builder: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(error_message.c_str());
            }
            if (weight_decay < 0.0f) {
                string error_message = "Adafactor builder: weight_decay must be >= 0. weight_decay: " + to_string(weight_decay);
                throw nb::value_error(error_message.c_str());
            }

            Adafactor::Builder builder;
            builder.alpha(alpha)
                .beta2(beta2)
                .epsilon(epsilon)
                .weightDecay(weight_decay)
                .factorSecondMoment(factor_second_moment);

            if (network != nullptr)
                builder.network(*network);

            shared_ptr<Optimizer> base = builder.build();
            shared_ptr<Adafactor> adafactor = dynamic_pointer_cast<Adafactor>(base);
            if (!adafactor)
                throw nb::type_error("Adafactor builder did not return an Adafactor instance");
            return adafactor;
        },
        "cls"_a,
        "alpha"_a = 0.001f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-30f,
        "weight_decay"_a = 0.0f,
        "factor_second_moment"_a = true,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an Adafactor optimizer.)nbdoc");

    adafactor.def(
        "__init__",
        [](Adafactor* self,
           float alpha,
           float beta2,
           float epsilon,
           float weight_decay,
           bool factor_second_moment,
           shared_ptr<Network> network) -> void {
            (void)self;
            (void)alpha;
            (void)beta2;
            (void)epsilon;
            (void)weight_decay;
            (void)factor_second_moment;
            (void)network;
            // no-op: constructed in __new__
        },
        "alpha"_a = 0.001f,
        "beta2"_a = 0.999f,
        "epsilon"_a = 1e-30f,
        "weight_decay"_a = 0.0f,
        "factor_second_moment"_a = true,
        "network"_a.none() = nb::none(),
        R"nbdoc(Construct an Adafactor optimizer.)nbdoc");

    adafactor.attr("__doc__") = R"doc(
Adafactor optimizer.

Adafactor uses an exponential moving average of squared gradients to normalize
updates. For rank-2 and higher dense tensors, Thor uses Adafactor's memory-saving
factored second-moment estimate over the final two dimensions. Rank-1 tensors and
sparse-row embedding updates use the unfactored second-moment fallback.

Parameters
----------
alpha : float, default 0.001
    Learning rate.
beta2 : float, default 0.999
    Exponential decay rate for the second-moment estimate. Must be in ``[0, 1)``.
epsilon : float, default 1e-30
    Small constant added for numerical stability.
weight_decay : float, default 0.0
    Decoupled weight-decay coefficient.
factor_second_moment : bool, default True
    Use factored second-moment state for rank-2 and higher dense tensors.
network : thor.Network, default None
    When network is passed in, this optimizer is set as the network default optimizer.
)doc";
}
