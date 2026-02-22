import os
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import joblib

# ==============================
# SETTINGS (CPU safe)
# ==============================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
MAX_IMAGES_PER_CLASS = 2000

TRAIN_DIR = "dataset/train"

CNN_MODEL_PATH = "ai_detector_model.keras"
RF_SAVE_PATH = "models/random_forest.pkl"

os.makedirs("models", exist_ok=True)


# ==============================
# LIMIT DATASET SIZE
# ==============================
def limit_dataset(directory):
    file_paths = []
    labels = []

    classes = sorted(os.listdir(directory))

    for label, class_name in enumerate(classes):
        class_path = os.path.join(directory, class_name)

        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)[:MAX_IMAGES_PER_CLASS]

        for img in images:
            file_paths.append(os.path.join(class_path, img))
            labels.append(label)

    return file_paths, labels


# ==============================
# CREATE TF DATASET
# ==============================
def create_dataset(file_paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds


# ==============================
# LOAD DATA
# ==============================
print("Loading limited dataset (~2000 per class)...")

train_files, train_labels = limit_dataset(TRAIN_DIR)
train_ds = create_dataset(train_files, train_labels)

print("Total training samples:", len(train_files))


# ==============================
# LOAD CNN
# ==============================
print("Loading trained CNN model...")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)


# 🔥 IMPORTANT FIX: Build model once so inputs exist
dummy = tf.random.normal((1, IMG_SIZE[0], IMG_SIZE[1], 3))
cnn_model(dummy)


# ==============================
# FEATURE EXTRACTOR
# ==============================
# remove last classification layer
feature_extractor = tf.keras.Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[-2].output
)


# ==============================
# EXTRACT FEATURES
# ==============================
print("Extracting deep features...")

X = []
y = []

for images, labels in train_ds:
    features = feature_extractor.predict(images, verbose=0)
    X.append(features)
    y.append(labels.numpy())

X = np.concatenate(X)
y = np.concatenate(y)

print("Feature shape:", X.shape)


# ==============================
# TRAIN RANDOM FOREST
# ==============================
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    n_jobs=-1
)

rf.fit(X, y)

joblib.dump(rf, RF_SAVE_PATH)

print("✅ Hybrid model saved at:", RF_SAVE_PATH)