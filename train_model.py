import tensorflow as tf
import tensorflow_datasets as tfds

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

IMG_SIZE = 128
BATCH_SIZE = 32

# Preprocess
def format_example(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

ds_train = ds_train.map(format_example).batch(BATCH_SIZE).shuffle(1000)
ds_test = ds_test.map(format_example).batch(BATCH_SIZE)

# Check data
for image_batch, label_batch in ds_train.take(1):
    print("Image batch shape:", image_batch.shape)
    print("Label batch:", label_batch.numpy())
    
from tensorflow.keras import layers, models

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(ds_train, validation_data=ds_test, epochs=5)

# Save the model
model.save("cat_dog_model.h5")
print("✅ Model saved as cat_dog_model.h5")
