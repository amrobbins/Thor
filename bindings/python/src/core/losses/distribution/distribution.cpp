#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_poisson_nll_loss(nb::module_ &distribution);
void bind_gaussian_nll_loss(nb::module_ &distribution);
void bind_negative_binomial_nll_loss(nb::module_ &distribution);
void bind_laplace_nll_loss(nb::module_ &distribution);
void bind_student_t_nll_loss(nb::module_ &distribution);
void bind_gamma_tweedie_losses(nb::module_ &distribution);

void bind_distribution_losses(nb::module_ &distribution) {
    distribution.doc() = "Thor distributional likelihood and deviance losses";

    bind_poisson_nll_loss(distribution);
    bind_gaussian_nll_loss(distribution);
    bind_negative_binomial_nll_loss(distribution);
    bind_laplace_nll_loss(distribution);
    bind_student_t_nll_loss(distribution);
    bind_gamma_tweedie_losses(distribution);
}
