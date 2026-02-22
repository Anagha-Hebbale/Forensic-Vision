import os
import random
from typing import Dict, List, Optional, Tuple
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (128, 128)
METADATA_PATH = "models/model_metadata.json"


def load_detector_model(model_path: str = "ai_detector_model.h5") -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def load_model_metadata(metadata_path: str = METADATA_PATH) -> Dict[str, object]:
    if not os.path.exists(metadata_path):
        return {
            "class_names": ["FAKE", "REAL"],
            "threshold": 0.5,
            "img_size": [IMG_SIZE[0], IMG_SIZE[1]],
        }
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_valid_image(test_folder: str = "dataset/test") -> str:
    classes = os.listdir(test_folder)

    while True:
        chosen_class = random.choice(classes)
        class_path = os.path.join(test_folder, chosen_class)
        img_name = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, img_name)

        try:
            with Image.open(img_path) as img:
                img.verify()
            return img_path
        except Exception:
            print("Skipping corrupted:", img_path)


def preprocess_pil_image(image: Image.Image, img_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    rgb = image.convert("RGB").resize(img_size)
    arr = np.array(rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

        # Handle nested models (e.g., MobileNetV2 used as a base inside Sequential).
        if hasattr(layer, "layers") and getattr(layer, "layers", None):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer

    raise ValueError(
        "No convolutional layer found in model. Ensure the loaded model contains Conv2D layers."
    )


def find_last_feature_layer(model: tf.keras.Model) -> Optional[tf.keras.layers.Layer]:
    # Prefer a top-level layer with 4D output so graph wiring stays connected.
    for layer in reversed(model.layers):
        out_shape = getattr(layer, "output_shape", None)
        if isinstance(out_shape, tuple) and len(out_shape) == 4:
            return layer
    return None


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    input_img: np.ndarray,
    last_conv_name: Optional[str] = None,
    output_size: Tuple[int, int] = IMG_SIZE,
) -> np.ndarray:
    # Ensure symbolic input/output tensors exist for loaded Sequential models.
    _ = model(input_img, training=False)

    try:
        if last_conv_name is not None:
            feature_layer = model.get_layer(last_conv_name)
        else:
            feature_layer = find_last_feature_layer(model)

        if feature_layer is None:
            raise ValueError("No top-level 4D feature layer found.")

        grad_model = tf.keras.models.Model(
            [model.inputs[0]],
            [feature_layer.output, model.outputs[0]],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_img)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
    except Exception:
        # Fallback: input-gradient saliency heatmap.
        with tf.GradientTape() as tape:
            inp = tf.convert_to_tensor(input_img)
            tape.watch(inp)
            preds = model(inp, training=False)
            loss = preds[:, 0]
        grads = tape.gradient(loss, inp)[0]
        heatmap = tf.reduce_mean(tf.abs(grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)
    if float(max_val) != 0.0:
        heatmap = heatmap / max_val

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, output_size)
    heatmap = np.uint8(255 * heatmap)
    return heatmap


def colorize_heatmap(heatmap_gray: np.ndarray) -> np.ndarray:
    heatmap_bgr = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    base_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    blended = (1 - alpha) * base_rgb + alpha * heatmap_rgb
    return blended.astype(np.uint8)


def predict_score(model: tf.keras.Model, input_img: np.ndarray) -> float:
    pred = model.predict(input_img, verbose=0)
    return float(pred[0][0])


def get_binary_class_names(train_dir: str = "dataset/train") -> Tuple[str, str]:
    metadata = load_model_metadata(METADATA_PATH)
    class_names = metadata.get("class_names", [])
    if isinstance(class_names, list) and len(class_names) >= 2:
        return str(class_names[0]), str(class_names[1])

    # image_dataset_from_directory sorts class names alphabetically.
    if os.path.isdir(train_dir):
        names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        if len(names) >= 2:
            return names[0], names[1]
    # Fallback if dataset folders are unavailable at inference time.
    return "FAKE", "REAL"


def summarize_attention(heatmap_gray: np.ndarray) -> Dict[str, float]:
    hm = heatmap_gray.astype(np.float32) / 255.0
    h, w = hm.shape

    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    center = hm[y0:y1, x0:x1]

    border = int(min(h, w) * 0.18)
    edge_mask = np.ones_like(hm, dtype=bool)
    edge_mask[border:h - border, border:w - border] = False

    mean_all = float(np.mean(hm)) + 1e-6
    mean_center = float(np.mean(center))
    mean_edges = float(np.mean(hm[edge_mask]))

    threshold = float(np.percentile(hm, 85))
    hotspot_ratio = float(np.mean(hm >= threshold))

    return {
        "mean_all": mean_all,
        "mean_center": mean_center,
        "mean_edges": mean_edges,
        "center_focus_ratio": mean_center / mean_all,
        "edge_focus_ratio": mean_edges / mean_all,
        "hotspot_ratio": hotspot_ratio,
    }


def explain_prediction(
    label: str,
    confidence: float,
    heatmap_gray: np.ndarray,
) -> Tuple[str, List[str], Dict[str, float]]:
    stats = summarize_attention(heatmap_gray)
    center_focus = stats["center_focus_ratio"]
    edge_focus = stats["edge_focus_ratio"]
    hotspot_ratio = stats["hotspot_ratio"]

    if label.upper() == "FAKE":
        headline = "Model leaned toward FAKE mainly from atypical high-attention patterns."
    else:
        headline = "Model leaned toward REAL based on a more consistent facial-attention pattern."

    bullets: List[str] = []

    if center_focus >= 1.15:
        bullets.append("Attention is concentrated around the central facial region.")
    else:
        bullets.append("Attention is not strongly centered on the face region.")

    if edge_focus >= 1.05:
        bullets.append("Model also focused on edges/background, which can indicate artifact sensitivity.")
    else:
        bullets.append("Relatively lower attention on image borders/background.")

    if hotspot_ratio >= 0.20:
        bullets.append("Heatmap has broad high-activation zones (diffuse evidence).")
    elif hotspot_ratio <= 0.10:
        bullets.append("Heatmap has compact high-activation zones (localized evidence).")
    else:
        bullets.append("Heatmap shows moderately spread activation.")

    bullets.append(
        f"Decision strength was {confidence * 100:.1f}% confidence (model score threshold at 0.5)."
    )

    return headline, bullets, stats


def analyze_image(
    model: tf.keras.Model,
    image: Image.Image,
) -> Tuple[str, float, np.ndarray, np.ndarray, np.ndarray]:
    metadata = load_model_metadata(METADATA_PATH)
    img_size_raw = metadata.get("img_size", [IMG_SIZE[0], IMG_SIZE[1]])
    img_size = (int(img_size_raw[0]), int(img_size_raw[1]))
    threshold = float(metadata.get("threshold", 0.5))

    input_img = preprocess_pil_image(image, img_size=img_size)
    resized_rgb = (input_img[0] * 255).astype(np.uint8)

    score = predict_score(model, input_img)
    class0, class1 = get_binary_class_names("dataset/train")
    label = class1 if score >= threshold else class0
    confidence = score if label == class1 else (1.0 - score)

    heatmap_gray = generate_gradcam_heatmap(model, input_img, output_size=img_size)
    heatmap_rgb = colorize_heatmap(heatmap_gray)
    overlay_rgb = overlay_heatmap(resized_rgb, heatmap_rgb, alpha=0.4)

    return label, confidence, resized_rgb, heatmap_rgb, overlay_rgb


def analyze_image_detailed(
    model: tf.keras.Model,
    image: Image.Image,
) -> Dict[str, object]:
    metadata = load_model_metadata(METADATA_PATH)
    img_size_raw = metadata.get("img_size", [IMG_SIZE[0], IMG_SIZE[1]])
    img_size = (int(img_size_raw[0]), int(img_size_raw[1]))
    threshold = float(metadata.get("threshold", 0.5))

    input_img = preprocess_pil_image(image, img_size=img_size)
    resized_rgb = (input_img[0] * 255).astype(np.uint8)

    score = predict_score(model, input_img)
    class0, class1 = get_binary_class_names("dataset/train")
    label = class1 if score >= threshold else class0
    confidence = score if label == class1 else (1.0 - score)

    heatmap_gray = generate_gradcam_heatmap(model, input_img, output_size=img_size)
    heatmap_rgb = colorize_heatmap(heatmap_gray)
    overlay_rgb = overlay_heatmap(resized_rgb, heatmap_rgb, alpha=0.4)
    headline, bullets, attention_stats = explain_prediction(label, confidence, heatmap_gray)

    return {
        "label": label,
        "confidence": confidence,
        "score": score,
        "class0": class0,
        "class1": class1,
        "threshold": threshold,
        "img_size": [img_size[0], img_size[1]],
        "original_rgb": resized_rgb,
        "heatmap_gray": heatmap_gray,
        "heatmap_rgb": heatmap_rgb,
        "overlay_rgb": overlay_rgb,
        "explain_headline": headline,
        "explain_bullets": bullets,
        "attention_stats": attention_stats,
    }


def main() -> None:
    model = load_detector_model("ai_detector_model.h5")
    img_path = get_valid_image("dataset/test")
    print("Generating heatmap for:", img_path)

    with Image.open(img_path) as image:
        label, confidence, _, _, overlay_rgb = analyze_image(model, image)

    print(f"Prediction: {label} ({confidence * 100:.2f}%)")
    plt.imshow(overlay_rgb)
    plt.title("AI Manipulation Heatmap")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
