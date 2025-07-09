---

```python
# 📦 Import required libraries
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ------------------------ STEP 1: Load Dataset ------------------------ #

# Load training data
train_data = keras.utils.image_dataset_from_directory(
    directory='/content/dogs-vs-cats/train',   # Dataset folder
    labels='inferred',                         # 🔸 'inferred' uses folder names (e.g., 'cats', 'dogs') to assign labels
    label_mode='int',                          # 🔸 Labels are integers: 0 or 1
    batch_size=32,                             # 32 images per batch
    image_size=(256, 256)                      # Resize all images to 256x256
)

# Load testing data
test_data = keras.utils.image_dataset_from_directory(
    directory='/content/dogs-vs-cats/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

# ------------------------ STEP 2: Normalize Images ------------------------ #

# Function to normalize image pixels to range [0, 1]
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)  # 🔸 Normalize pixel values
    return image, label

# Apply normalization to datasets
train_data = train_data.map(process)
test_data = test_data.map(process)

# ------------------------ STEP 3: Build CNN Model ------------------------ #

# Create a sequential model (layer by layer)
model = Sequential()

# 🔸 Conv2D Layer: Detects patterns like edges and shapes
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
# 🔸 MaxPooling2D: Reduces image size and retains strongest features
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# More filters for more complex features
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# Deeper layer with more filters
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

# 🔸 Flatten: Converts 2D feature maps to 1D vector before Dense layers
model.add(Flatten())

# 🔸 Dense Layer: Fully connected neural layer with 128 neurons
model.add(Dense(128, activation='relu'))

# 🔸 Dropout: Randomly drops 10% of neurons to prevent overfitting
model.add(Dropout(0.1))

# Another Dense layer with 64 neurons
model.add(Dense(64, activation='relu'))

# Dropout again for regularization
model.add(Dropout(0.1))

# 🔸 Final Dense layer: 1 output neuron with sigmoid for binary classification (dog/cat)
model.add(Dense(1, activation='sigmoid'))

# 🔸 Compile the model
model.compile(
    optimizer='adam',                      # 🔸 'adam' optimizer — adaptive, efficient
    loss='binary_crossentropy',            # 🔸 Binary loss for two classes
    metrics=['accuracy']                   # Track accuracy during training
)

# ------------------------ STEP 4: Train the Model ------------------------ #

# Train on training data for 10 epochs and validate using test data
history = model.fit(train_data, epochs=10, validation_data=test_data)

# ------------------------ STEP 5: Plot Accuracy and Loss ------------------------ #

# Plot Accuracy
plt.plot(history.history['accuracy'], color='red', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.title("Model Accuracy")
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], color='red', label='Train Loss')
plt.plot(history.history['val_loss'], color='blue', label='Validation Loss')
plt.title("Model Loss")
plt.legend()
plt.show()

# ------------------------ STEP 6: Predict on a New Image ------------------------ #

# Load a new image to test (e.g., a dog image)
img_path = '/content/dogs-vs-cats/train/dogs/dog.0.jpg'

# Load and resize the image
img = keras.utils.load_img(img_path, target_size=(256, 256))

# Convert to numpy array
img_array = keras.utils.img_to_array(img)

# Normalize the image
img_array = img_array / 255.0

# Expand dims to convert to batch shape: (1, 256, 256, 3)
img_array_expanded = np.expand_dims(img_array, axis=0)

# 🔸 Predict the class (output is a probability between 0 and 1)
prob = model.predict(img_array_expanded)[0][0]

# 🔸 Threshold to get class: 0 (cat) or 1 (dog)
predicted_class = 1 if prob >= 0.5 else 0

# ------------------------ STEP 7: Display the Image with Prediction ------------------------ #

# Show the image with predicted class
plt.imshow(img)
plt.axis('off')  # Remove grid lines
plt.title(f'Predicted Class: {predicted_class} ({"Dog" if predicted_class == 1 else "Cat"})')
plt.show()
```

---

### ✅ Now this code includes:

* **Line-by-line explanations** of each major layer, function, and hyperparameter.
* Meaning of **ReLU, Adam, Dropout, MaxPooling, Flatten, Dense**, etc.
* Real-time visual output of training accuracy, loss, and image prediction.

Let me know if you'd like to:

* Add a **confusion matrix**
* Save the trained model
* Predict a **folder of images at once**
* Export predictions to a CSV or JSON
