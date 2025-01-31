import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import random

# Load the saved model, tokenizer, and label encoder
@st.cache_resource
def load_model():
    model = joblib.load("chatbot_model.pkl")
    return model

@st.cache_resource
def load_tokenizer():
    tokenizer = joblib.load("chatbot_tokenizer.pkl")
    return tokenizer

@st.cache_resource
def load_label_encoder():
    label_encoder = joblib.load("chatbot_label_encoder.pkl")
    return label_encoder

# Define the chatbot response function
def chatbot_response(user_input, model, tokenizer, label_encoder, max_length=20, confidence_threshold=0.5):
    # Preprocess user input
    input_sequence = tokenizer.texts_to_sequences([user_input])
    padded_input = pad_sequences(input_sequence, maxlen=max_length, padding="post")

    # Predict probabilities for each tag
    probabilities = model.predict(padded_input)[0]
    max_prob = max(probabilities)
    predicted_tag_index = probabilities.argmax()

    # Check if the confidence is above the threshold
    if max_prob < confidence_threshold:
        return "I'm not sure about that. Can you try rephrasing?"

    # If confidence is high enough, return a response for the predicted tag
    predicted_tag = label_encoder.inverse_transform([predicted_tag_index])[0]
    responses = intents_map.get(predicted_tag, ["I'm not sure how to respond to that."])
    return random.choice(responses)

# Load intents (you can modify this to load your intents.json if needed)
intents_map = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help?"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    # Add other tags and responses here
}

# Load resources
model = load_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()

# Streamlit UI
st.title("Chatbot Application")
st.write("Ask me anything! Type your message below:")

user_input = st.text_input("You:", key="user_input")
if st.button("Send"):
    if user_input.strip():
        response = chatbot_response(user_input, model, tokenizer, label_encoder)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Please enter a message to start the conversation.")
