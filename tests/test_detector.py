"""Fast tests that don't need the 167 MB model. The last test runs only if the model is present."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import detector  # noqa: E402

EXAMPLES = sorted((ROOT / "examples").glob("*.jpg"))


def test_preprocess_matches_training_input():
    x = detector.preprocess(Image.new("RGB", (500, 300), (255, 0, 0)))
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert x.max() == pytest.approx(255.0)  # pixels stay in [0, 255]; no /255


def test_preprocess_converts_grey_and_alpha_images():
    assert detector.preprocess(Image.new("L", (64, 64))).shape == (1, 224, 224, 3)
    assert detector.preprocess(Image.new("RGBA", (64, 64))).shape == (1, 224, 224, 3)


@pytest.mark.parametrize("p_real, label", [(0.95, "REAL"), (0.30, "REAL"), (0.29, "FAKE"), (0.01, "FAKE")])
def test_decision_uses_threshold(p_real, label):
    assert detector.decide(p_real, threshold=0.30).label == label


def test_confidence_is_between_half_and_one():
    for p in np.linspace(0, 1, 21):
        assert 0.5 <= detector.decide(float(p), threshold=0.30).confidence <= 1.0


def test_overlay_keeps_size():
    heat = np.random.rand(224, 224)
    assert detector.overlay(Image.new("RGB", (300, 300)), heat).size == (224, 224)


def test_examples_are_labelled_by_name():
    assert EXAMPLES and all(p.name.startswith(("real_", "fake_")) for p in EXAMPLES)


@pytest.mark.skipif(not detector.MODEL_PATH.exists(), reason="model file not downloaded")
def test_model_and_gradcam_on_examples():
    fake = Image.open(ROOT / "examples" / "fake_1.jpg")
    pred = detector.predict(fake)
    assert pred.label == "FAKE"
    heat = detector.gradcam(fake, pred.label)
    assert heat.shape == (224, 224) and 0 <= heat.min() and heat.max() <= 1
