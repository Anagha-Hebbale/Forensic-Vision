import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
import os

print("Loading hybrid model...")

cnn = tf.keras.models.load_model("models/cnn_feature.keras")
rf = joblib.load("models/random_forest.pkl")


def clean_path(p):
    p = p.strip().replace('"','')
    return os.path.normpath(p)


def predict_image(img_path):

    img_path = clean_path(img_path)

    if not os.path.exists(img_path):
        print("\n❌ File not found:", img_path)
        return

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    features = cnn.predict(img, verbose=0)

    prediction = rf.predict(features)[0]
    confidence = rf.predict_proba(features)[0]

    if prediction == 1:
        label = "FAKE"
        conf = confidence[1]
    else:
        label = "REAL"
        conf = confidence[0]

    print(f"\nPrediction: {label}")
    print(f"Confidence: {conf*100:.2f}%\n")


if __name__ == "__main__":
    path = input("Drag image here and press ENTER:\n")
    predict_image(path)
