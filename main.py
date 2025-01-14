import tensorflow as tf


number_of_classes = train_labels.max() + 1
number_of_classes
image_height = 28
image_width = 28

number_of_weights = image_height * image_width * number_of_classes
number_of_weights

# Check for GPU
print(tf.config.list_physical_devices('GPU'))

# Load the fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (valid_images, valid_labels) = fashion_mnist.load_data()

# Define the index and print the label of the indexed training data
data_idx = 0  # Define the index
print(train_labels[data_idx])

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(number_of_classes)
])

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    verbose=True,
    validation_data=(valid_images, valid_labels)
)
model.predict(train_images[0:10])