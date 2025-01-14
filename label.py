
import matplotlib.pyplot as plt
import numpy as np

# Dummy data for demonstration purposes
train_images = np.random.rand(60000, 28, 28)
valid_images = np.random.rand(10000, 28, 28)

# The question number to study with. Feel free to change up to 59999.
data_idx = 42

plt.figure()
plt.imshow(train_images[data_idx], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

data_idx = 6174

plt.figure()
plt.imshow(valid_images[data_idx], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()