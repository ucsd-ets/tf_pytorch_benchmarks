title = "Beginning Tensorflow Benchmark"
dashes = "-" * (len(title) + 10)  # Adding 10 extra dashes

formatted_text = f"{dashes}\n{title}\n{dashes}"
print(formatted_text)

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import time  # Required for benchmarking

# Checkpoint 0: Load the Fashion MNIST dataset
start_time = time.time()
(train_data, test_data), dataset_info = tfds.load(
    'fashion_mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)
end_time = time.time()
print(f"Time taken for Loading Data: {end_time - start_time:.4f} seconds")

# Checkpoint 1: Preprocess the dataset
start_time = time.time()
program_start = start_time
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

train_data = train_data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.cache()
train_data = train_data.shuffle(dataset_info.splits['train'].num_examples)
train_data = train_data.batch(128)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

test_data = test_data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
test_data = test_data.batch(128)
test_data = test_data.cache()
test_data = test_data.prefetch(tf.data.AUTOTUNE)
end_time = time.time()
print(f"Time taken for Preprocessing: {end_time - start_time:.4f} seconds")

# Checkpoint 2: Define the model
start_time = time.time()
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
end_time = time.time()
print(f"Time taken for Model Definition: {end_time - start_time:.4f} seconds")

# Checkpoint 3: Compile the model
start_time = time.time()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
end_time = time.time()
print(f"Time taken for Model Compilation: {end_time - start_time:.4f} seconds")

# Checkpoint 4: Train the model
start_time = time.time()
model.fit(train_data, epochs=5, validation_data=test_data)
end_time = time.time()
print(f"Time taken for Training: {end_time - start_time:.4f} seconds")

# Checkpoint 5: Evaluate the model
start_time = time.time()
test_loss, test_acc = model.evaluate(test_data, verbose=2)
end_time = time.time()
print(f"Time taken for Evaluation: {end_time - start_time:.4f} seconds")
print(f"Total time taken (without loading data): {end_time - program_start:.4f} seconds")
print('\nTest accuracy:', test_acc)
