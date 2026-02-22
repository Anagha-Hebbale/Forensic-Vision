import tensorflow as tf

IMG_SIZE = (224,224)
BATCH = 32

train = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="binary"
)

valid = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/valid",
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode="binary"
)

print("Dataset loaded successfully")
