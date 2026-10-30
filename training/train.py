"""
train.py — How models/vgg16_deepfake.h5 was trained.

VGG16 pretrained on ImageNet, block4 + block5 fine-tuned, new classifier head (no dropout, L2 + BatchNorm).
Data: a balanced random subset of the Kaggle "140k Real and Fake Faces" dataset
(10k real + 10k fake for training, 2k + 2k for validation).
Labels: fake = 0, real = 1, so the model outputs P(real).

Run: python training/train.py --data "path/to/real-vs-fake"     (a GPU is strongly recommended)
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
SEED = 42
TRAIN_PER_CLASS = 10_000
VALID_PER_CLASS = 2_000


def subset(split_dir: Path, per_class: int):
    samples = []
    for label, cls in ((0, "fake"), (1, "real")):
        files = [f for f in os.listdir(split_dir / cls) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        samples += [(str(split_dir / cls / f), float(label)) for f in random.sample(files, min(per_class, len(files)))]
    random.shuffle(samples)
    return samples


def make_dataset(samples, augment=False):
    def load(path, label):
        img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return tf.cast(img, tf.float32), label  # stays in [0, 255]: VGG16 preprocessing is inside the model

    def aug(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1 * 255)
        return tf.clip_by_value(img, 0.0, 255.0), label

    paths, labels = zip(*samples)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels))).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    base = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    for layer in base.layers:  # fine-tune only the two last blocks
        if "block4" in layer.name or "block5" in layer.name:
            layer.trainable = True

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="VGG16_Deepfake")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder that contains train/ and valid/")
    ap.add_argument("--out", default="models/vgg16_deepfake.h5")
    args = ap.parse_args()
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    data = Path(args.data)
    train = subset(data / "train", TRAIN_PER_CLASS)
    valid = subset(data / "valid", VALID_PER_CLASS)

    # Fake up-weighted to raise recall on fakes. Side effect: the model calls real faces fake
    # more often, which is why the app uses a threshold tuned on validation instead of 0.5.
    class_weight = {0: 1.2, 1: 0.8}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = build_model()
    model.fit(
        make_dataset(train, augment=True),
        validation_data=make_dataset(valid),
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[
            ModelCheckpoint(args.out, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-8, verbose=1),
            CSVLogger("training_logs.csv"),
        ],
    )


if __name__ == "__main__":
    main()
