# Step 1: Import required libraries
# TensorFlow is used for building and training the neural network
# Matplotlib is used to visualize the training progress
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Step 2: Download the dataset
# TensorFlow provides a smaller version of the Cats vs Dogs dataset
# It will be automatically extracted into folders for training and validation
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_file = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=url, extract=True)

# Define paths to training and validation folders
base_dir = os.path.join(os.path.dirname(zip_file), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Step 3: Preprocess the data
# Resize all images to 150x150 pixels and normalize pixel values (0–1)
IMG_SIZE = 150
BATCH_SIZE = 32

# Data generators automatically label the data from folder names and apply preprocessing
train_gen = ImageDataGenerator(rescale=1./255)  # Rescaling pixel values
val_gen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data
train_data = train_gen.flow_from_directory(
    train_dir,                      # Path to training images
    target_size=(IMG_SIZE, IMG_SIZE),  # Resize all images
    batch_size=BATCH_SIZE,
    class_mode='binary'            # Binary classification: cat or dog
)

# Load and preprocess validation data
val_data = val_gen.flow_from_directory(
    val_dir,                        # Path to validation images
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Step 4: Build the CNN model
# CNN uses layers to automatically detect features like edges, shapes, and textures
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # First convolution layer
    MaxPooling2D(2, 2),  # Reduces image size while keeping important features

    Conv2D(64, (3, 3), activation='relu'),  # Second convolution layer
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),  # Third convolution layer
    MaxPooling2D(2, 2),

    Flatten(),  # Flattens the feature maps into a 1D array for the dense layer
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    Dense(1, activation='sigmoid')  # Output layer for binary classification (0 or 1)
])

# Step 5: Compile the model
# 'adam' optimizer adjusts learning rate automatically
# 'binary_crossentropy' is used for 2-class problems
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
# Feed training data to the model and evaluate performance on validation data
history = model.fit(
    train_data,
    epochs=5,  # You can increase this number for better accuracy
    validation_data=val_data
)

# Step 7: Plot accuracy and loss curves to visualize training performance
plt.figure(figsize=(10, 4))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
