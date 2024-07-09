A complete example of how to build and train an ANN to predict whether someone will play tennis based on some given features. We'll use the TensorFlow/Keras library along with Pandas for handling the data. Let's assume we have the following dataset:

```plaintext
Outlook  Temperature Humidity Windy PlayTennis
Sunny    Hot         High      False  No
Sunny    Hot         High      True   No
Overcast Hot         High      False  Yes
Rain     Mild        High      False  Yes
Rain     Cool        Normal    False  Yes
Rain     Cool        Normal    True   No
Overcast Cool        Normal    True   Yes
Sunny    Mild        High      False  No
Sunny    Cool        Normal    False  Yes
Rain     Mild        Normal    False  Yes
Sunny    Mild        Normal    True   Yes
Overcast Mild        High      True   Yes
Overcast Hot         Normal    False  Yes
Rain     Mild        High      True   No
```

### Step 1: Prepare the Data

First, we'll encode the categorical features and target variable. Then, we'll split the data into training and testing sets.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create the dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == object:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Split the dataset
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Step 2: Build and Train the ANN Model

```python
# Build the ANN model
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy:.4f}')
```

### Step 3: Making Predictions

```python
# Make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to class labels

# Compare with actual labels
print("Predicted labels:", y_pred.flatten())
print("Actual labels:", y_test.values)
```

### Explanation:

1. **Data Preparation:**
   - We encode categorical variables using `LabelEncoder`.
   - We split the data into training and testing sets.
   - We standardize the features using `StandardScaler`.

2. **Model Building:**
   - We build a Sequential model with two hidden layers of 10 neurons each using ReLU activation.
   - The output layer has a single neuron with a sigmoid activation function for binary classification.

3. **Model Training and Evaluation:**
   - We compile the model using the Adam optimizer and binary cross-entropy loss function.
   - We train the model for 50 epochs with a batch size of 5.
   - We evaluate the model on the test set and print the test accuracy.

4. **Making Predictions:**
   - We make predictions on the test set and convert probabilities to binary class labels.
   - We compare the predicted labels with the actual labels.

You can adjust the number of epochs, batch size, and network architecture as needed to improve performance.
