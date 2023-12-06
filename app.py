import streamlit as st
import numpy as np
import tensorflow
import pickle
from PIL import Image

# Function to perform tumor detection
def tumor_detection(img, model):
    img = Image.open(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    return "Tumor Detected" if res else "No Tumor"

# Function to perform sentiment classification
def sentiment_classification(new_review_text, model):
    max_review_length = 500
    word_to_index = tensorflow.keras.datasets.imdb.get_word_index()
    new_review_tokens = [word_to_index.get(word, 0) for word in new_review_text.split()]
    new_review_tokens = tensorflow.keras.preprocessing.sequence.pad_sequences([new_review_tokens], maxlen=max_review_length)
    prediction = model.predict(new_review_tokens)
    if type(prediction) == list:
        prediction = prediction[0]
    return "Positive" if prediction > 0.5 else "Negative"

st.title('Predicting Deep Learning Models')

option=st.selectbox(
      "Choose the classifier:",
      ["Image Classification","Text Classification"]
      )

if option == "Text Classification":
      new_review_text = st.text_area("Enter a New Review:", value="")
      if st.button("Submit") and not new_review_text.strip():
            st.warning("Please enter a review.")

      if new_review_text.strip():
            st.subheader("Choose Model for Text Classification")
            model = st.radio("Pick the model:", ['Perceptron','BPNN','DNN','RNN','LSTM'])

            if model == "Perceptron":
                  with open('perceptron.pkl', 'rb') as file:
                        model = pickle.load(file)
            elif model == "BPNN":
                  with open('imdb_back_prop.pkl', 'rb') as file:
                        model = pickle.load(file)
            elif model == "DNN":
                  model = tensorflow.keras.models.load_model('dnn.keras') 
            elif model == "RNN":
                  model = tensorflow.keras.models.load_model('rnn.keras')
            elif model == "LSTM":
                  model = tensorflow.keras.models.load_model('lstm.keras')

            if st.button("Classify"):
                  result = sentiment_classification(new_review_text, model)
                  st.subheader("Text Classification Result")
                  st.write(f"**{result}**")

elif option == "Image Classification":
    st.subheader("Tumor Detection")
    uploaded_file = st.file_uploader("Choose a tumor image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the tumor detection model
        model = tensorflow.keras.models.load_model('CN.keras')
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False, width=200)
        st.write("")

        if st.button("Detect Tumor"):
            result = tumor_detection(uploaded_file, model)
            st.subheader("Tumor Detection Result")
            st.write(f"**{result}**")