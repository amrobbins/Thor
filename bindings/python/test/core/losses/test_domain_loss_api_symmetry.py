import pytest
import thor


_RAW = thor.losses.LossShape.raw
_FP32 = thor.DataType.fp32


def _tensor(dims, dtype=_FP32):
    return thor.Tensor(list(dims), dtype)


def _domain_loss_factories():
    return [
        pytest.param(
            "classification.binary_focal",
            lambda n, loss_weight: thor.losses.classification.BinaryFocalLoss(
                n, _tensor([1]), _tensor([1]), 1.5, 0.35, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="classification.binary_focal",
        ),
        pytest.param(
            "classification.categorical_focal",
            lambda n, loss_weight: thor.losses.classification.CategoricalFocalLoss(
                n, _tensor([3]), _tensor([3]), 1.5, 0.8, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="classification.categorical_focal",
        ),
        *[
            pytest.param(
                f"detection.{name}",
                lambda n, loss_weight, cls=cls: cls(
                    n, _tensor([4]), _tensor([4]), "xyxy", 1.0e-6, _FP32, _RAW, loss_weight=loss_weight
                ),
                id=f"detection.{name}",
            )
            for name, cls in [
                ("iou", thor.losses.detection.IoULoss),
                ("giou", thor.losses.detection.GIoULoss),
                ("diou", thor.losses.detection.DIoULoss),
                ("ciou", thor.losses.detection.CIoULoss),
            ]
        ],
        pytest.param(
            "distribution.poisson_nll",
            lambda n, loss_weight: thor.losses.distribution.PoissonNLLLoss(
                n, _tensor([3]), _tensor([3]), False, True, 1.0e-5, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="distribution.poisson_nll",
        ),
        pytest.param(
            "distribution.gaussian_nll",
            lambda n, loss_weight: thor.losses.distribution.GaussianNLLLoss(
                n, _tensor([3]), _tensor([3]), _tensor([3]), True, 1.0e-5, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="distribution.gaussian_nll",
        ),
        pytest.param(
            "distribution.gamma_nll",
            lambda n, loss_weight: thor.losses.distribution.GammaNLLLoss(
                n, _tensor([3]), _tensor([3]), 1.0e-5, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="distribution.gamma_nll",
        ),
        pytest.param(
            "distribution.tweedie",
            lambda n, loss_weight: thor.losses.distribution.TweedieLoss(
                n, _tensor([3]), _tensor([3]), 1.35, 1.0e-5, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="distribution.tweedie",
        ),
        pytest.param(
            "gan.hinge_discriminator",
            lambda n, loss_weight: thor.losses.gan.HingeGANDiscriminatorLoss(
                n, _tensor([3]), _tensor([3]), _FP32, _RAW, loss_weight=loss_weight
            ),
            id="gan.hinge_discriminator",
        ),
        pytest.param(
            "gan.hinge_generator",
            lambda n, loss_weight: thor.losses.gan.HingeGANGeneratorLoss(
                n, _tensor([3]), _FP32, _RAW, loss_weight=loss_weight
            ),
            id="gan.hinge_generator",
        ),
        pytest.param(
            "gan.wasserstein_critic",
            lambda n, loss_weight: thor.losses.gan.WassersteinGANCriticLoss(
                n, _tensor([3]), _tensor([3]), _FP32, _RAW, loss_weight=loss_weight
            ),
            id="gan.wasserstein_critic",
        ),
        pytest.param(
            "gan.wasserstein_generator",
            lambda n, loss_weight: thor.losses.gan.WassersteinGANGeneratorLoss(
                n, _tensor([3]), _FP32, _RAW, loss_weight=loss_weight
            ),
            id="gan.wasserstein_generator",
        ),
        pytest.param(
            "gan.wasserstein_gradient_penalty",
            lambda n, loss_weight: thor.losses.gan.WassersteinGANCriticGradientPenaltyLoss(
                n, _tensor([1]), _tensor([1]), _tensor([2, 2]), 4.0, 1.25, 1.0e-8, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="gan.wasserstein_gradient_penalty",
        ),
        pytest.param(
            "gan.lsgan_discriminator",
            lambda n, loss_weight: thor.losses.gan.LSGANDiscriminatorLoss(
                n, _tensor([3]), _tensor([3]), _FP32, _RAW, 0.9, -0.1, loss_weight=loss_weight
            ),
            id="gan.lsgan_discriminator",
        ),
        pytest.param(
            "gan.lsgan_generator",
            lambda n, loss_weight: thor.losses.gan.LSGANGeneratorLoss(
                n, _tensor([3]), _FP32, _RAW, 0.9, loss_weight=loss_weight
            ),
            id="gan.lsgan_generator",
        ),
        pytest.param(
            "metric_learning.contrastive",
            lambda n, loss_weight: thor.losses.metric_learning.ContrastiveLoss(
                n, _tensor([3]), _tensor([3]), 1.25, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="metric_learning.contrastive",
        ),
        pytest.param(
            "metric_learning.info_nce",
            lambda n, loss_weight: thor.losses.metric_learning.InfoNCELoss(
                n, _tensor([3]), _tensor([3]), 0.7, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="metric_learning.info_nce",
        ),
        pytest.param(
            "metric_learning.triplet",
            lambda n, loss_weight: thor.losses.metric_learning.TripletLoss(
                n, _tensor([3]), _tensor([3]), _tensor([3]), 0.75, 1.0e-5, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="metric_learning.triplet",
        ),
        pytest.param(
            "metric_learning.cosine_embedding",
            lambda n, loss_weight: thor.losses.metric_learning.CosineEmbeddingLoss(
                n, _tensor([3]), _tensor([3]), _tensor([1]), 0.2, 1.0e-6, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="metric_learning.cosine_embedding",
        ),
        pytest.param(
            "ranking.margin_ranking",
            lambda n, loss_weight: thor.losses.ranking.MarginRankingLoss(
                n, _tensor([3]), _tensor([3]), _tensor([3]), 0.15, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="ranking.margin_ranking",
        ),
        pytest.param(
            "ranking.list_net",
            lambda n, loss_weight: thor.losses.ranking.ListNetLoss(
                n, _tensor([4]), _tensor([4]), 0.8, 1.3, _FP32, _RAW, _tensor([4], thor.DataType.uint8), loss_weight=loss_weight
            ),
            id="ranking.list_net",
        ),
        pytest.param(
            "ranking.listwise_softmax_cross_entropy",
            lambda n, loss_weight: thor.losses.ranking.ListwiseSoftmaxCrossEntropyLoss(
                n, _tensor([4]), _tensor([4]), 0.8, _FP32, _RAW, _tensor([4], thor.DataType.uint8), loss_weight=loss_weight
            ),
            id="ranking.listwise_softmax_cross_entropy",
        ),
        pytest.param(
            "segmentation.dice",
            lambda n, loss_weight: thor.losses.segmentation.DiceLoss(
                n, _tensor([2, 3]), _tensor([2, 3]), 0.7, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="segmentation.dice",
        ),
        pytest.param(
            "segmentation.tversky",
            lambda n, loss_weight: thor.losses.segmentation.TverskyLoss(
                n, _tensor([2, 3]), _tensor([2, 3]), 0.4, 0.6, 0.7, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="segmentation.tversky",
        ),
        pytest.param(
            "segmentation.focal_tversky",
            lambda n, loss_weight: thor.losses.segmentation.FocalTverskyLoss(
                n, _tensor([2, 3]), _tensor([2, 3]), 0.4, 0.6, 1.25, 0.7, _FP32, _RAW, loss_weight=loss_weight
            ),
            id="segmentation.focal_tversky",
        ),
    ]


@pytest.mark.parametrize("name, factory", _domain_loss_factories())
@pytest.mark.parametrize("loss_weight, expected", [(None, None), (1.0, None), (2.25, 2.25)])
def test_all_domain_loss_python_constructors_expose_normalized_loss_weight(name, factory, loss_weight, expected):
    network = thor.Network(f"test_net_domain_loss_api_symmetry_{name}_{loss_weight}".replace(".", "_"))

    loss = factory(network, loss_weight)

    if expected is None:
        assert loss.loss_weight is None
    else:
        assert loss.loss_weight == pytest.approx(expected)
