
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Create a simple model for demonstration purposes
model = Sequential([
	Flatten(input_shape=(28, 28)),
	Dense(128, activation='relu'),
	Dense(10, activation='softmax')
])

# Randomly initialize the model weights
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_images = np.random.rand(60000, 28, 28)
valid_images = np.random.rand(10000, 28, 28)
number_of_classes = 10 # Define the number of classes
train_labels = np.random.randint(0, number_of_classes, 60000)

data_idx = 8675 # The question number to study with. Feel free to change up to 59999.
number_of_classes = 10 # Define the number of classes

plt.figure()
plt.imshow(train_images[data_idx], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

x_values = range(number_of_classes)
plt.figure()
plt.bar(x_values, model.predict(train_images[data_idx:data_idx+1]).flatten())
plt.xticks(range(10))
plt.show()

print("correct answer:", train_labels[data_idx])