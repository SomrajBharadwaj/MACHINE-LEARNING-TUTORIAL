Multilayer Perceptron (MLP) on Iris Dataset
This repository demonstrates the implementation of a Multilayer Perceptron (MLP) using TensorFlow and Keras to classify the Iris dataset. The code covers data preprocessing, model building, training with early stopping, and visualization of the training and validation performance.

Requirements
To run this code, make sure you have the following libraries installed:

numpy
pandas
sklearn
tensorflow
matplotlib
You can install these dependencies via pip:

bash
Copy code
pip install numpy pandas scikit-learn tensorflow matplotlib
Code Walkthrough
1. Import Libraries
We begin by importing the necessary libraries for data manipulation, model building, training, and visualization.

python
Copy code
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
2. Load Dataset
The Iris dataset, a widely used dataset in machine learning, is loaded using load_iris() from sklearn.datasets. The dataset contains 150 samples with four features each, representing three classes of iris flowers.

python
Copy code
iris = load_iris()
X = iris.data
y = iris.target
3. Split Dataset into Training and Testing Sets
The dataset is split into training and testing sets using train_test_split(). We use 80% of the data for training and 20% for testing.

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
4. Standardize Features
We use StandardScaler to standardize the features of the dataset. This ensures that each feature has a mean of 0 and a standard deviation of 1.

python
Copy code
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
5. Build MLP Model
A simple MLP model is constructed with two hidden layers, each with 10 neurons and ReLU activation. The output layer has 3 neurons (since there are 3 classes in the Iris dataset) and uses softmax activation.

python
Copy code
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(10, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes for iris species
])
6. Compile Model
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function. We also specify accuracy as the evaluation metric.

python
Copy code
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
7. Implement Early Stopping
To prevent overfitting, early stopping is applied. The training will stop if the validation loss does not improve for 10 consecutive epochs.

python
Copy code
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
8. Train the Model
The model is trained using the training data (X_train and y_train). We specify a validation split of 20% to monitor performance during training.

python
Copy code
history = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1, validation_split=0.2, callbacks=[early_stopping])
9. Evaluate the Model
After training, the model is evaluated on the test set, and the accuracy is printed.

python
Copy code
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')
10. Plot Training & Validation Loss and Accuracy
The training and validation loss, as well as accuracy, are plotted to visually assess the model's performance during training.

python
Copy code
plt.figure(figsize=(12, 6))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
Results
The model’s performance can be evaluated using the test accuracy and by analyzing the loss and accuracy plots.

Conclusion
This code demonstrates the implementation of a Multilayer Perceptron for classification on the Iris dataset. By using early stopping and monitoring the validation loss, we prevent overfitting. You can experiment with different architectures, optimizers, and learning rates to further improve the model's performance.
