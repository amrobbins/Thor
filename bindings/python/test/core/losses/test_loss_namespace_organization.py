import importlib

import thor


_CORE_FLAT_LOSSES = [
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "SparseCategoricalCrossEntropy",
    "MAE",
    "MAPE",
    "MSE",
    "SmoothL1Loss",
    "HuberLoss",
    "SoftTargetCrossEntropy",
    "KLDivLoss",
]

_DOMAIN_LOSSES = {
    "classification": ["BinaryFocalLoss", "CategoricalFocalLoss"],
    "detection": ["IoULoss", "GIoULoss", "DIoULoss", "CIoULoss"],
    "distribution": ["PoissonNLLLoss", "GaussianNLLLoss", "GammaNLLLoss", "TweedieLoss"],
    "gan": [
        "HingeGANDiscriminatorLoss",
        "HingeGANGeneratorLoss",
        "WassersteinGANCriticLoss",
        "WassersteinGANGeneratorLoss",
        "WassersteinGANCriticGradientPenaltyLoss",
        "LSGANDiscriminatorLoss",
        "LSGANGeneratorLoss",
    ],
    "metric_learning": ["ContrastiveLoss", "InfoNCELoss", "TripletLoss", "CosineEmbeddingLoss"],
    "ranking": ["MarginRankingLoss", "ListNetLoss", "ListwiseSoftmaxCrossEntropyLoss"],
    "segmentation": ["DiceLoss", "TverskyLoss", "FocalTverskyLoss"],
}


def test_core_losses_remain_available_from_flat_losses_namespace():
    for name in _CORE_FLAT_LOSSES:
        assert hasattr(thor.losses, name), name


def test_domain_loss_submodules_are_available_from_native_namespace_after_importing_thor():
    for domain, names in _DOMAIN_LOSSES.items():
        module = getattr(thor.losses, domain)
        assert module is not None
        for name in names:
            cls = getattr(module, name)
            assert cls.__module__ == f"thor.losses.{domain}"


def test_domain_loss_submodules_are_importable_as_python_modules():
    for domain, names in _DOMAIN_LOSSES.items():
        imported = importlib.import_module(f"thor.losses.{domain}")
        native = getattr(thor.losses, domain)
        for name in names:
            assert getattr(imported, name) is getattr(native, name)
            assert name in imported.__all__


def test_domain_specific_losses_are_not_in_flat_losses_namespace():
    assert thor.losses.LossShape.batch is not None
    for names in _DOMAIN_LOSSES.values():
        for name in names:
            assert not hasattr(thor.losses, name), name
