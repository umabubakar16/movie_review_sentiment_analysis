import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.metrics import Precision, Recall

# Load your dataset
df = pd.read_csv('data_rt.csv')  # Replace 'your_dataset.csv' with your dataset file path

# Check the column names in the dataset
print(df.columns)

# Extract the appropriate column containing the review data
# Replace 'review_column_name' with the actual column name containing the review data
review_column_name = 'reviews'
if review_column_name not in df.columns:
    raise KeyError(f"Column '{review_column_name}' not found in the dataset.")

x = df[review_column_name].astype(str)  # Convert to string dtype

# Define the parameters
max_tokens = 30000
output_seq_len = 500
embedding_dim = 128
time_steps = 32

# Initialize the text vectorization layer
vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=output_seq_len
)

# Adapt the text vectorization layer to the dataset
vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(x))

# Save the vectorization layer using pickle
with open('vectorize_layer.pkl', 'wb') as f:
    pickle.dump(vectorize_layer, f)

# Load the vectorization layer
with open('vectorize_layer.pkl', 'rb') as f:
    vectorize_layer = pickle.load(f)

# Load the sentiment analysis model
model = tf.keras.models.load_model('rt_lstm_sentiment_classifier.h5', compile=False)

# Define the function to preprocess input text
def preprocess_text(text):
    text = tf.convert_to_tensor([text])
    text = vectorize_layer(text)
    return text

# Define the function to predict sentiment
def get_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)
    if prediction > 0.5:
        sentiment = 'Positive'
        confidence = prediction[0][0] * 100
    else:
        sentiment = 'Negative'
        confidence = (1 - prediction[0][0]) * 100
    return sentiment, confidence

# Streamlit UI
st.title("Sentiment Analysis")

review_input = st.text_input("Enter your review:")

if review_input:
    sentiment, confidence = get_sentiment(review_input)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Confidence: {confidence:.2f}%")
