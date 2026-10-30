"""
detector.py — Loads the trained VGG16 model, classifies a face and explains the decision with Grad-CAM.

The model (training/train.py) takes a 224×224 RGB image with pixels in [0, 255]
(VGG16 preprocessing is inside the model) and outputs P(real): labels were fake=0, real=1.
"""

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "vgg16_deepfake.h5"))
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/FAROUK-debug/deepfake-detector/releases/download/v1.0/vgg16_deepfake.h5",
)
MODEL_SHA256 = "445dc25c136d3eaa3c3e7e8bef390e601b9ab581cc7873f8bf9c25e24de09db0"

IMG_SIZE = 224
# Chosen on the validation set (scripts/evaluate.py), never on the test set.
# Below 0.5 because training up-weighted the fake class, which makes the model
# too quick to call a real face fake.
THRESHOLD = float(os.getenv("THRESHOLD", "0.27"))
LAST_CONV = "block5_conv3"


@dataclass
class Prediction:
    label: str          # "REAL" or "FAKE"
    p_real: float       # raw model output
    confidence: float   # probability of the predicted label, rescaled around the threshold


def ensure_model() -> Path:
    """Downloads the model from the GitHub release if it's not on disk, and checks its hash."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        print(f"Downloading model (167 MB) from {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if sha != MODEL_SHA256:
        raise RuntimeError(f"Model file {MODEL_PATH} has an unexpected SHA-256 ({sha}). Delete it and retry.")
    return MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(ensure_model(), compile=False)


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image → (1, 224, 224, 3) float32 in [0, 255], the exact input used in training."""
    import tensorflow as tf
    arr = np.asarray(img.convert("RGB"), dtype="float32")
    # tf.image.resize (bilinear), like training: PIL's resize gives slightly different pixels.
    return tf.image.resize(arr, [IMG_SIZE, IMG_SIZE]).numpy()[None]


def decide(p_real: float, threshold: float = THRESHOLD) -> Prediction:
    if p_real >= threshold:
        label, conf = "REAL", 0.5 + 0.5 * (p_real - threshold) / (1 - threshold)
    else:
        label, conf = "FAKE", 0.5 + 0.5 * (threshold - p_real) / threshold
    return Prediction(label, p_real, float(min(conf, 1.0)))


def predict(img: Image.Image) -> Prediction:
    return decide(float(load_model().predict(preprocess(img), verbose=0)[0][0]))


@lru_cache(maxsize=1)
def _gradcam_parts():
    """Splits the saved model into: VGG16 (conv features + output) and the classifier head."""
    import tensorflow as tf
    model = load_model()
    vgg = model.get_layer("vgg16")
    features = tf.keras.Model(vgg.inputs, [vgg.get_layer(LAST_CONV).output, vgg.output])
    head = model.layers[model.layers.index(vgg) + 1:]   # GAP → BN → Dense → BN → Dense → BN → output
    return features, head


def gradcam(img: Image.Image, label: str) -> np.ndarray:
    """
    Heatmap (224×224, values 0–1) of the regions that pushed the model towards `label`.
    Grad-CAM: weight each channel of the last conv layer by the gradient of the score, then sum.
    """
    import tensorflow as tf
    features, head = _gradcam_parts()
    x = tf.keras.applications.vgg16.preprocess_input(tf.constant(preprocess(img)))
    with tf.GradientTape() as tape:
        conv, out = features(x, training=False)
        tape.watch(conv)
        for layer in head[:-1]:
            out = layer(out, training=False)
        # Use the logit (before the sigmoid): on very confident images the sigmoid is flat
        # and its gradients vanish, which would give an empty heatmap.
        final = head[-1]
        logit = tf.matmul(out, final.kernel)[:, 0] + final.bias[0]
        score = logit if label == "REAL" else -logit
    grads = tape.gradient(score, conv)[0]                     # (14, 14, 512)
    weights = tf.reduce_mean(grads, axis=(0, 1))              # one weight per channel
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)) / 255.0


def overlay(img: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Draws the heatmap on top of the face (blue = low, red = high)."""
    import matplotlib
    colored = matplotlib.colormaps["jet"](heat)[..., :3]
    base = np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype="float32") / 255.0
    mixed = (1 - alpha) * base + alpha * colored
    return Image.fromarray(np.uint8(mixed * 255))
