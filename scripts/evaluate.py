"""
evaluate.py — Picks the decision threshold on the validation set, then measures the model on the test set.

The threshold is chosen on validation images only; the test images are used once,
for the final numbers. Writes docs/results.md, docs/confusion_matrix.png and docs/roc_curve.png.

Run: python scripts/evaluate.py --data "path/to/real-vs-fake" [--valid 1000] [--test 2000]
Dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import detector  # noqa: E402

DOCS = ROOT / "docs"


def sample(split_dir: Path, n: int, seed: int):
    """Balanced random sample: n/2 fake (label 0) and n/2 real (label 1)."""
    rng = random.Random(seed)
    paths, labels = [], []
    for label, cls in ((0, "fake"), (1, "real")):
        files = sorted(os.listdir(split_dir / cls))
        for f in rng.sample(files, n // 2):
            paths.append(str(split_dir / cls / f))
            labels.append(label)
    return paths, np.array(labels)


def p_real(paths):
    def load(p):
        img = tf.image.decode_jpeg(tf.io.read_file(p), channels=3)
        return tf.image.resize(tf.cast(img, tf.float32), [detector.IMG_SIZE, detector.IMG_SIZE])
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    return detector.load_model().predict(ds, verbose=0).ravel()


def metrics(y, p, t):
    # Positive class for precision/recall = FAKE (that's what we want to catch).
    pred_fake = (p < t).astype(int)
    is_fake = 1 - y
    return {
        "accuracy": accuracy_score(is_fake, pred_fake),
        "precision (fake)": precision_score(is_fake, pred_fake),
        "recall (fake)": recall_score(is_fake, pred_fake),
        "f1 (fake)": f1_score(is_fake, pred_fake),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder that contains train/ valid/ test/")
    ap.add_argument("--valid", type=int, default=1000)
    ap.add_argument("--test", type=int, default=2000)
    args = ap.parse_args()
    data = Path(args.data)

    # 1. Threshold on validation
    vp, vy = sample(data / "valid", args.valid, seed=1)
    v = p_real(vp)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    accs = [accuracy_score(vy, (v >= t).astype(int)) for t in grid]
    best_t = float(grid[int(np.argmax(accs))])
    print(f"validation: best threshold {best_t} (accuracy {max(accs):.3f}; at 0.5: {accuracy_score(vy, (v >= 0.5).astype(int)):.3f})")

    # 2. Test, used once
    tp, ty = sample(data / "test", args.test, seed=2)
    started = time.perf_counter()
    t = p_real(tp)
    ms_per_image = (time.perf_counter() - started) * 1000 / len(tp)
    auc = roc_auc_score(ty, t)
    m50, mbest = metrics(ty, t, 0.5), metrics(ty, t, best_t)

    # 3. Plots
    DOCS.mkdir(exist_ok=True)
    cm = confusion_matrix(1 - ty, (t < best_t).astype(int), labels=[0, 1])  # rows: true REAL, FAKE
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center", fontsize=14,
                color="white" if val > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["REAL", "FAKE"]); ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Test set, threshold {best_t}")
    fig.tight_layout(); fig.savefig(DOCS / "confusion_matrix.png", dpi=120); plt.close(fig)

    fpr, tpr, _ = roc_curve(1 - ty, 1 - t)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"VGG16 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax.set_xlabel("False positive rate (real called fake)"); ax.set_ylabel("True positive rate (fakes caught)")
    ax.legend(loc="lower right"); ax.set_title("ROC curve, test set")
    fig.tight_layout(); fig.savefig(DOCS / "roc_curve.png", dpi=120); plt.close(fig)

    # 4. Report
    rows = "\n".join(f"| {k} | {m50[k]:.1%} | {mbest[k]:.1%} |" for k in m50)
    report = f"""# Results

Validation images used to pick the threshold: {len(vp)} (balanced). Test images: {len(tp)} (balanced, different from validation).

**AUC: {auc:.3f}**, inference on CPU: {ms_per_image:.0f} ms per image (batches of 32).

| Metric | Threshold 0.5 | Threshold {best_t} (from validation) |
|---|---|---|
{rows}

Confusion matrix at threshold {best_t} (rows = true class):

| | predicted REAL | predicted FAKE |
|---|---|---|
| **true REAL** | {cm[0, 0]} | {cm[0, 1]} |
| **true FAKE** | {cm[1, 0]} | {cm[1, 1]} |
"""
    (DOCS / "results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
