# src/datasets/div2k.py
import os
from glob import glob
import tensorflow as tf

def _load_png(path: str) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return tf.image.convert_image_dtype(img, tf.float32)

def _make_pairs(hr_dir: str, lr_dir: str):
    hr_paths = sorted(glob(os.path.join(hr_dir, "*.png")))
    lr_paths = sorted(glob(os.path.join(lr_dir, "*.png")))
    assert len(hr_paths) == len(lr_paths), "HR/LR count mismatch"
    return lr_paths, hr_paths

def get_div2k_dataset(
    root: str,
    scale: int = 4,
    hr_crop: int = 256,
    batch_size: int = 16,
    training: bool = True,
) -> tf.data.Dataset:
    split = "train" if training else "valid"
    base = os.path.join(root, "div2k")
    hr_dir = os.path.join(base, f"DIV2K_{split}_HR")
    lr_dir = os.path.join(base, f"DIV2K_{split}_LR_bicubic_X{scale}")
    lr_paths, hr_paths = _make_pairs(hr_dir, lr_dir)

    ds = tf.data.Dataset.from_tensor_slices((lr_paths, hr_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(lr_paths))

    def _process(lr_path, hr_path):
        hr = _load_png(hr_path)
        if training:
            hr_patch = tf.image.random_crop(hr, [hr_crop, hr_crop, 3])
            lr_patch = tf.image.resize(
                hr_patch,
                [hr_crop//scale, hr_crop//scale],
                method="bicubic"
            )
            return lr_patch, hr_patch
        else:
            lr = _load_png(lr_path)
            return lr, hr

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
