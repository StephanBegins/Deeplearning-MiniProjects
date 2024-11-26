import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess Data
x_train = x_train / 255.0  # Normalize pixel values to [0, 1]
x_test = x_test / 255.0

# 3. Build the Model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into 1D vector
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    Dense(64, activation='relu'),   # Another hidden layer with 64 neurons
    Dense(10, activation='softmax') # Output layer for 10 classes
])

# 4. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the Model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# 6. Evaluate the Model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Plot training accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.show()
