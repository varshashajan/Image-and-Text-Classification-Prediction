import tensorflow
import pickle

top_words = 5000
(X_train, y_train), (X_test, y_test) = tensorflow.keras.datasets.imdb.load_data(num_words=top_words)

max_review_length = 500
X_train = tensorflow.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = tensorflow.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)

# Modelling a sample DNN
model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Embedding(input_dim=top_words, output_dim=24, input_length=max_review_length))
model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(64, activation='relu'))
model.add(tensorflow.keras.layers.Dense(32, activation='relu'))
model.add(tensorflow.keras.layers.Dense(16, activation='relu'))
model.add(tensorflow.keras.layers.Dense(1, activation='sigmoid'))

# opt=Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)
print("Training Started.")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=20,callbacks=[early_stop])

loss, acc = model.evaluate(X_test, y_test)
print("Training Finished.")

print(f'Test Accuracy: {round(acc * 100)}')

model.save(r'D:\3rd sem\deeplearning\Deep Learning Prediction\dnn.keras')



