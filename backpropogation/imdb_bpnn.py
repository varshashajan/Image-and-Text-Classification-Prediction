import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from Backpropogation import  BackPropogation
import tensorflow
from tensorflow import keras

# Load IMDB dataset
top_words = 5000
(X, y), (_, _) = tensorflow.keras.datasets.imdb.load_data(num_words=top_words)

# Pad sequences to a fixed length

max_review_length = 500
X = tensorflow.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_review_length)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert labels to binary (0 for negative, 1 for positive)
y_train_binary = (y_train == 0).astype(int)
y_test_binary = (y_test == 0).astype(int)

# Initialize and train the BackPropagation model
backpropagation = BackPropogation(learning_rate=0.01, epochs=100)
backpropagation.fit(X_train, y_train_binary)

# Predictions on the test set
pred = backpropagation.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test_binary, pred)}")
report = classification_report(y_test_binary, pred, digits=2)
print(report)
print(f"Predictions: {pred}")

import pickle
with open("imdb_back_prop.pkl",'wb') as file:
    pickle.dump(backpropagation,file)