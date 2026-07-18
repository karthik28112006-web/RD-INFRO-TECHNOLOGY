# utils.py
import numpy as np
import cv2
import tensorflow as tf


def load_trained_model(model_path: str = "mnist_model.h5") -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def preprocess_canvas_image(image_data: np.ndarray) -> np.ndarray:
    if image_data is None:
        return None

    if np.all(image_data[:, :, :3] == 0) or image_data.sum() == 0:
        return None

    img = cv2.cvtColor(image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    if not np.any(img):
        return None

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    if img.mean() > 127:
        img = cv2.bitwise_not(img)

    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


