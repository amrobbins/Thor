"""Generative adversarial network losses."""

import thor
import thor.losses


class HingeGANDiscriminatorLoss(thor.losses.Loss):
    """
    Discriminator-side hinge GAN loss.

    The raw elementwise loss is:

        relu(1 - real_scores) + relu(1 + fake_scores)

    Gradients are produced for both real_scores and fake_scores. Use StopGradient on
    the generated samples feeding the discriminator when the discriminator step should
    not backpropagate into the generator.
    """

    def __init__(self, network: thor.Network, real_scores: thor.Tensor, fake_scores: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """
        Construct the discriminator-side hinge GAN loss over real and fake discriminator scores.
        """

    def get_real_scores(self) -> thor.Tensor: ...

    def get_fake_scores(self) -> thor.Tensor: ...

class HingeGANGeneratorLoss(thor.losses.Loss):
    """
    Generator-side hinge GAN loss.

    The raw elementwise loss is:

        -fake_scores

    Gradients are produced for fake_scores so that the generator step can backpropagate
    through the discriminator graph to the generator, while step-scoped update sets
    control which parameters are updated.
    """

    def __init__(self, network: thor.Network, fake_scores: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """
        Construct the generator-side hinge GAN loss over fake discriminator scores.
        """

    def get_fake_scores(self) -> thor.Tensor: ...

class LSGANDiscriminatorLoss(thor.losses.Loss):
    """
    Discriminator-side least-squares GAN loss.

    The raw elementwise loss is:

        0.5 * ((real_scores - real_target)^2 + (fake_scores - fake_target)^2)

    The default targets are real_target=1.0 and fake_target=0.0. Gradients are
    produced for both real_scores and fake_scores. Use StopGradient on generated
    samples feeding the discriminator when the discriminator step should not
    backpropagate into the generator.
    """

    def __init__(self, network: thor.Network, real_scores: thor.Tensor, fake_scores: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, real_target: float = 1.0, fake_target: float = 0.0, *, loss_weight: float | None = None) -> None:
        """
        Construct the discriminator-side least-squares GAN loss over real and fake discriminator scores.
        """

    def get_real_scores(self) -> thor.Tensor: ...

    def get_fake_scores(self) -> thor.Tensor: ...

    @property
    def real_target(self) -> float: ...

    @property
    def fake_target(self) -> float: ...

class LSGANGeneratorLoss(thor.losses.Loss):
    """
    Generator-side least-squares GAN loss.

    The raw elementwise loss is:

        0.5 * (fake_scores - target)^2

    The default target is 1.0.
    """

    def __init__(self, network: thor.Network, fake_scores: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, target: float = 1.0, *, loss_weight: float | None = None) -> None:
        """
        Construct the generator-side least-squares GAN loss over fake discriminator scores.
        """

    def get_fake_scores(self) -> thor.Tensor: ...

    @property
    def target(self) -> float: ...

class WassersteinGANCriticGradientPenaltyLoss(thor.losses.Loss):
    """
    Critic-side Wasserstein GAN loss with gradient penalty.

    The raw per-example scalar loss is:

        fake_scores - real_scores + gradient_penalty_weight * (||sample_gradients||_2 - target_gradient_norm)^2

    ``sample_gradients`` is expected to be the gradient of the critic output with
    respect to interpolated input samples. This class deliberately consumes that
    gradient tensor; trainer/autodiff scaffolding must produce it as a differentiable
    first-class tensor for end-to-end WGAN-GP training.
    """

    def __init__(self, network: thor.Network, real_scores: thor.Tensor, fake_scores: thor.Tensor, sample_gradients: thor.Tensor, gradient_penalty_weight: float = 10.0, target_gradient_norm: float = 1.0, eps: float = 9.999999960041972e-13, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """
        Construct the WGAN-GP critic loss over scalar scores and a materialized per-sample input-gradient tensor.
        """

    def get_real_scores(self) -> thor.Tensor: ...

    def get_fake_scores(self) -> thor.Tensor: ...

    def get_sample_gradients(self) -> thor.Tensor: ...

    @property
    def gradient_penalty_weight(self) -> float: ...

    @property
    def target_gradient_norm(self) -> float: ...

    @property
    def eps(self) -> float: ...

class WassersteinGANCriticLoss(thor.losses.Loss):
    """
    Critic-side Wasserstein GAN loss.

    The raw elementwise loss is:

        fake_scores - real_scores

    Minimizing this critic loss maximizes the Wasserstein distance estimate. For
    original WGAN weight clipping, use a post-update parameter constraint hook; this
    loss intentionally only models the differentiable score objective.
    """

    def __init__(self, network: thor.Network, real_scores: thor.Tensor, fake_scores: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """
        Construct the critic-side Wasserstein GAN loss over real and fake critic scores.
        """

    def get_real_scores(self) -> thor.Tensor: ...

    def get_fake_scores(self) -> thor.Tensor: ...

class WassersteinGANGeneratorLoss(thor.losses.Loss):
    """
    Generator-side Wasserstein GAN loss.

    The raw elementwise loss is:

        -fake_scores
    """

    def __init__(self, network: thor.Network, fake_scores: thor.Tensor, loss_data_type: thor.DataType | None = None, reported_loss_shape: thor.losses.LossShape | None = thor.losses.LossShape.batch, *, loss_weight: float | None = None) -> None:
        """
        Construct the generator-side Wasserstein GAN loss over fake critic scores.
        """

    def get_fake_scores(self) -> thor.Tensor: ...

__all__: list = ['HingeGANDiscriminatorLoss', 'HingeGANGeneratorLoss', 'WassersteinGANCriticLoss', 'WassersteinGANGeneratorLoss', 'WassersteinGANCriticGradientPenaltyLoss', 'LSGANDiscriminatorLoss', 'LSGANGeneratorLoss']
