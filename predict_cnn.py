import os
import sys
from PIL import Image
from heatmap import analyze_image_detailed, load_detector_model


def clean_path(path: str) -> str:
    path = path.strip().replace('"', "")
    return os.path.normpath(path)


def main() -> None:
    # ✅ FIX: use correct model file
    model = load_detector_model("ai_detector_model.keras")

    if len(sys.argv) > 1:
        image_path = clean_path(sys.argv[1])
    else:
        image_path = clean_path(input("Drag image here and press ENTER:\n"))

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    with Image.open(image_path) as img:
        result = analyze_image_detailed(model, img.convert("RGB"))

    label = result["label"]
    confidence = float(result["confidence"]) * 100
    score = float(result["score"])
    class0 = result["class0"]
    class1 = result["class1"]
    threshold = float(result["threshold"])

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Score: {score:.4f}")
    print(f"Class mapping: 0={class0}, 1={class1}")
    print(f"Threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()