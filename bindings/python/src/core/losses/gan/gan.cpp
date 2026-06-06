#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_hinge_gan_discriminator_loss(nb::module_ &gan);
void bind_hinge_gan_generator_loss(nb::module_ &gan);
void bind_wasserstein_gan_losses(nb::module_ &gan);
void bind_lsgan_losses(nb::module_ &gan);

void bind_gan_losses(nb::module_ &gan) {
    gan.doc() = "Thor GAN losses";

    bind_hinge_gan_discriminator_loss(gan);
    bind_hinge_gan_generator_loss(gan);
    bind_wasserstein_gan_losses(gan);
    bind_lsgan_losses(gan);
}
