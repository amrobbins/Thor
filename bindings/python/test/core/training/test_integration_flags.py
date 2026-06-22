from integration_flags import ALL_INTEGRATION_TESTS_ENV, integration_flag_enabled, integration_skip_reason


def test_integration_flag_enabled_is_false_when_specific_and_global_flags_are_unset(monkeypatch):
    monkeypatch.delenv(ALL_INTEGRATION_TESTS_ENV, raising=False)
    monkeypatch.delenv("THOR_RUN_TRAINING_INTEGRATION", raising=False)

    assert not integration_flag_enabled("THOR_RUN_TRAINING_INTEGRATION")


def test_integration_flag_enabled_honors_specific_flag(monkeypatch):
    monkeypatch.delenv(ALL_INTEGRATION_TESTS_ENV, raising=False)
    monkeypatch.setenv("THOR_RUN_TRAINING_INTEGRATION", "1")

    assert integration_flag_enabled("THOR_RUN_TRAINING_INTEGRATION")


def test_integration_flag_enabled_honors_all_integration_tests(monkeypatch):
    monkeypatch.setenv(ALL_INTEGRATION_TESTS_ENV, "1")
    monkeypatch.delenv("THOR_RUN_TRAINING_BYTE_LM_INTEGRATION", raising=False)

    assert integration_flag_enabled("THOR_RUN_TRAINING_BYTE_LM_INTEGRATION")


def test_integration_flag_enabled_keeps_existing_exact_one_semantics(monkeypatch):
    monkeypatch.setenv(ALL_INTEGRATION_TESTS_ENV, "true")
    monkeypatch.setenv("THOR_RUN_TRAINING_INTEGRATION", "yes")

    assert not integration_flag_enabled("THOR_RUN_TRAINING_INTEGRATION")


def test_integration_skip_reason_mentions_global_and_specific_flags():
    reason = integration_skip_reason(
        "THOR_RUN_TRAINING_IMAGENET100_INTEGRATION",
        "THOR_RUN_TRAINING_IMAGENET100_CV5_ALEXNET_INTEGRATION",
        description="heavyweight ImageNet-100 tests",
    )

    assert "THOR_ALL_INTEGRATION_TESTS=1" in reason
    assert "THOR_RUN_TRAINING_IMAGENET100_INTEGRATION=1" in reason
    assert "THOR_RUN_TRAINING_IMAGENET100_CV5_ALEXNET_INTEGRATION=1" in reason
