import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ===== LOAD MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ai_detector_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

# ===== LOAD IMAGE =====
IMAGE_PATH = os.path.join(BASE_DIR, "test.jpg")  # since it's in main folder

if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found!")
    exit()

img = image.load_img(IMAGE_PATH, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# ===== PREDICT =====
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("🔴 Prediction: FAKE")
else:
    print("🟢 Prediction: REAL")

print("Confidence:", float(prediction))