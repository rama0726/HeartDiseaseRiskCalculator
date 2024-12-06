import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
np.random.seed(42)

from google.colab import drive
drive.mount('/content/drive')

#Append the directory to your python path using sys
import sys
import os
prefix = '/content/drive/My Drive/'
customized_path_to_your_homework = 'Colab Notebooks/'
sys_path = prefix + customized_path_to_your_homework
sys.path.append(sys_path)

fn_train = os.path.join(sys_path, 'heart.csv')
print('Path to training data: {}'.format(fn_train))

#load dataset
data = pd.read_csv(fn_train)#load_data(fn_train)

#Inspect data
print("First 5 rows of the dataset:")
print(data.head())

print("\nSummary statistics:")
print(data.describe())

print("\nInfo about the dataset:")
print(data.info())

#Create correlation matrix and set threshold
threshold = 0.3

correlation_matrix = data.corr()

#Choose features based on threshold correlation
high_correlated_features = correlation_matrix.index[abs(correlation_matrix['target']) > threshold].tolist()
high_correlated_features.remove("target")

print("Features chosen based on high correlation:")
print(high_correlated_features)

X_selected = data[high_correlated_features]
y = data['target']

#Feature Scaling
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Scale the features
X_scaled = scaler.fit_transform(X_selected)

#Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Create and train model
model = LogisticRegression()  # You can customize hyperparameters here
model.fit(X_train, y_train)

#Create a Neural Network Using Keras
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)


# Define the model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 
    'precision', 'recall'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2) #use 20% of training data for validation


#Training accuracy
accuracy_values = history.history['accuracy']
epochs = range(1, len(accuracy_values) + 1)  # Epochs start from 1

plt.plot(epochs, accuracy_values, 'o-', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

#Validation accuracy
val_accuracy_values = history.history['val_accuracy']
plt.plot(epochs, val_accuracy_values, 'o-', label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


#Evaluate the model using the following metrics

from sklearn.metrics import accuracy_score, auc, roc_curve, precision_recall_fscore_support, classification_report

# Make predictions on the test data
y_pred_probs = model.predict(X_test)  # Get predicted probabilities
y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to class labels (0 or 1)

# Generate the classification report
report = classification_report(y_test, y_pred) 

# Print the report
print(report)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

# Calculate the AUC
roc_auc = auc(fpr, tpr)

# Print the AUC
print(f"AUC-ROC: {roc_auc}")

import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model

# Assuming your model is stored in the variable 'model'
plot_model(model, to_file='model_overview_detailed.png', show_shapes=True, 
           show_layer_names=True, show_layer_activations=True)


