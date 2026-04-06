// #include <nanobind/nanobind.h>
// #include <nanobind/stl/shared_ptr.h>
//
// #include "DeepLearning/Api/Optimizers/Optimizer.h"
// #include "DeepLearning/Api/Optimizers/Sgd.h"
//
// namespace nb = nanobind;
// using namespace nb::literals;
// using namespace std;
// using namespace Thor;
//
// void bind_sgd(nb::module_ &optimizers) {
//     auto sgd = nb::class_<Sgd, Optimizer>(optimizers, "Sgd");
//     sgd.attr("__module__") = "thor.optimizers";
//
//     sgd.def_static(
//         "__new__",
//         [](nb::handle cls, float initial_learning_rate, float decay, float momentum, bool nesterov_momentum, shared_ptr<Network> network)
//             -> std::shared_ptr<Sgd> {
//             if (initial_learning_rate <= 0.0f) {
//                 string error_message =
//                     "Sgd builder: initial_learning_rate must be > 0. initial_learning_rate: " + to_string(initial_learning_rate);
//                 throw nb::value_error(error_message.c_str());
//             }
//             if (decay < 0.0f || decay > 1.0f) {
//                 string error_message = "Sgd builder: assertion 0 <= decay <= 1 failed. decay: " + to_string(decay);
//                 throw nb::value_error(error_message.c_str());
//             }
//             if (momentum < 0.0f) {
//                 string error_message = "Sgd builder: momentum must be >= 0. momentum: " + to_string(momentum);
//                 throw nb::value_error(error_message.c_str());
//             }
//
//             Sgd::Builder builder;
//             builder.initialLearningRate(initial_learning_rate).decay(decay).momentum(momentum).useNesterovMomentum(nesterov_momentum);
//
//             if (network != nullptr)
//                 builder.network(*network);
//
//             std::shared_ptr<Optimizer> base = builder.build();  // Builder returns shared_ptr<Optimizer>
//             std::shared_ptr<Sgd> sgd = std::dynamic_pointer_cast<Sgd>(base);
//             if (!sgd)
//                 throw nb::type_error("Sgd builder did not return an Sgd instance");
//             return sgd;  // nanobind converts shared_ptr<Sgd> to an Sgd Python object
//         },
//         "cls"_a,
//         "initial_learning_rate"_a = 0.01f,
//         "decay"_a = 0.0f,
//         "momentum"_a = 0.0f,
//         "nesterov_momentum"_a = false,
//         "network"_a.none() = nb::none(),
//         R"nbdoc(Construct an SGD optimizer.)nbdoc");
//
//     // No-op __init__ (construction happens in __new__)
//     sgd.def(
//         "__init__",
//         [](Sgd *self, float initial_learning_rate, float decay, float momentum, bool nesterov_momentum, shared_ptr<Network> network)
//             -> void {
//             // no-op: constructed in __new__
//         },
//         "initial_learning_rate"_a = 0.01f,
//         "decay"_a = 0.0f,
//         "momentum"_a = 0.0f,
//         "nesterov_momentum"_a = false,
//         "network"_a.none() = nb::none(),
//         R"nbdoc(Construct an SGD optimizer.)nbdoc");
//
//     sgd.attr("__doc__") = R"doc(
// Stochastic Gradient Descent (SGD) optimizer.
//
// This optimizer updates a layer's trainable parameters (weights and, if present, biases)
// using classic SGD with optional momentum, optional Nesterov momentum, and optional learning
// rate decay.
//
// Parameters
// ----------
// initial_learning_rate : float, default 0.01
//     Base learning rate used for the update step.
// decay : float, default 0.0
//     Per-epoch learning rate decay factor. When decay is non-zero, the effective learning rate
//     is reduced each epoch, e.g. ``lr <- lr * (1 - decay)`` each epoch.
// momentum : float, default 0.0
//     Momentum coefficient in ``[0, 1]``. When non-zero, the optimizer maintains a
//     velocity buffer for each parameter and performs momentum updates.
// nesterov_momentum : bool, default False
//     If True, use Nesterov momentum (lookahead / projected parameters) for training-time
//     forward passes.
// network : thor.Network, default None
//     When network is passed in, then this optimizer will be set as the default optimizer in
//     the network and attached to all layers that do not have a layer specific optimizer
//     already attached, at network stamping time. You would not pass network here when you
//     want this optimizer to be specific to one or more layers, but not applied to the others
//     by default.
//
//
// Notes
// -----
// **Momentum.**
// With momentum enabled, SGD maintains a velocity buffer ``u`` per parameter:
//
// - ``u <- momentum * update - learning_rate * gradient``
//
// and applies the parameter update using the velocity.
//
// **Nesterov momentum.**
// When Nesterov is enabled, training-time forward passes use a *projected* (lookahead)
// parameter:
//
// - ``p = w + mu * u``
//
// where ``w`` is the current parameter and ``u`` is its velocity buffer. Backprop computes
// gradients at ``p``. Inference-time forward passes use the real (non-projected) parameters.
//
// **Bias parameters.**
// If the underlying layer has biases, SGD maintains separate velocity/gradient buffers
// for biases as well.
//
// Examples
// --------
// Basic usage::
//
//     from thor.optimizers import Sgd
//
//     opt = Sgd(initial_learning_rate=0.1)
//
// With momentum::
//
//     opt = Sgd(initial_learning_rate=0.1, momentum=0.9)
//
// With Nesterov momentum and decay::
//
//     opt = Sgd(initial_learning_rate=0.1, decay=0.1, momentum=0.9, nesterov_momentum=True)
//
// See Also
// --------
// Adam : Adaptive Moment Estimation optimizer.
// RmsProp : RMSProp optimizer.
// )doc";
// }
