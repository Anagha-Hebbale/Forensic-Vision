import os
import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# SETTINGS (CPU SAFE)
# ==============================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 8

MAX_IMAGES_PER_CLASS = 2000

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"

MODEL_SAVE_PATH = "ai_detector_model.keras"

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


def create_dataset(file_paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ==============================
# LOAD DATA
# ==============================
print("Loading limited dataset (~2000 per class)...")

train_files, train_labels = limit_dataset(TRAIN_DIR)
valid_files, valid_labels = limit_dataset(VALID_DIR)

train_ds = create_dataset(train_files, train_labels)
valid_ds = create_dataset(valid_files, valid_labels)

print("Training samples:", len(train_files))
print("Validation samples:", len(valid_files))

# ==============================
# BUILD MODEL
# ==============================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# TRAIN
# ==============================
print("Training model...")
model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS
)

# ==============================
# SAVE MODEL
# ==============================
model.save(MODEL_SAVE_PATH)

print("✅ Model saved successfully!")