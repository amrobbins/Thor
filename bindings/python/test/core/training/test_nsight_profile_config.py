from pathlib import Path

import pytest

import thor


def test_nsight_profile_contains_multiple_named_phase_relative_captures(tmp_path: Path):
    glm_output = tmp_path / "glm.nsys-rep"
    transformer_output = tmp_path / "transformer.nsys-rep"
    profile = thor.training.NsightProfile(
        captures=[
            thor.training.NsightProfileCapture(
                phase="poisson_glm_pretrain",
                start_epoch=100,
                epoch_count=2,
                output=glm_output,
            ),
            thor.training.NsightProfileCapture(
                phase="transformer_residual",
                start_epoch=3,
                epoch_count=2,
                output=transformer_output,
            ),
        ]
    )

    assert len(profile.captures) == 2
    assert profile.captures[0].phase == "poisson_glm_pretrain"
    assert profile.captures[0].start_epoch == 100
    assert profile.captures[0].epoch_count == 2
    assert profile.captures[0].output == str(glm_output)
    assert profile.captures[1].phase == "transformer_residual"
    assert profile.captures[1].start_epoch == 3
    assert profile.captures[1].epoch_count == 2
    assert profile.captures[1].output == str(transformer_output)


def test_nsight_profile_rejects_invalid_capture_specification(tmp_path: Path):
    output = tmp_path / "steady-state.nsys-rep"

    with pytest.raises(ValueError, match="phase"):
        thor.training.NsightProfileCapture(phase="", start_epoch=1, epoch_count=2, output=output)
    with pytest.raises(ValueError, match="start_epoch"):
        thor.training.NsightProfileCapture(phase="phase", start_epoch=0, epoch_count=2, output=output)
    with pytest.raises(ValueError, match="epoch_count"):
        thor.training.NsightProfileCapture(phase="phase", start_epoch=1, epoch_count=0, output=output)
    with pytest.raises(ValueError, match=".nsys-rep"):
        thor.training.NsightProfileCapture(
            phase="phase", start_epoch=1, epoch_count=2, output=tmp_path / "profile.txt"
        )
    with pytest.raises(ValueError, match="captures"):
        thor.training.NsightProfile(captures=[])
