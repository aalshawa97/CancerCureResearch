https://www.youtube.com/watch?v=z1PGJ9quPV8&t=284s

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the dataset
dataset = pd.read_csv('cancer.csv')

# Separate features (x) and target variable (y)
x = dataset.drop(["diagnosis(1=m, 0=b)"], axis=1)  # Specify axis=1 to drop the column
y = dataset["diagnosis(1=m, 0=b)"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Build the model
model = tf.keras.models.Sequential()

# Input layer with 256 neurons and input shape equal to the number of features
model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
# Hidden layer with 256 neurons
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
# Output layer with 1 neuron (binary classification)
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=1000)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
