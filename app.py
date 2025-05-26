import streamlit as st
import pickle
import numpy as np

# Load the vectorizer and model
@st.cache_resource
def load_model_and_vectorizer():
    with open("regional_model.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("tfidf_vectorizer.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

vectorizer, model = load_model_and_vectorizer()

# App title
st.title("🌍 Culture AI - Regional Slang Detector")
st.write("Enter a slang or phrase, and we'll tell you where it's likely from!")

# Input text
user_input = st.text_area("📝 Enter the slang/text:", "")

if st.button("🔍 Detect Region"):
    if user_input.strip():
        try:
            # Vectorize input and make prediction
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            st.success(f"🌐 Predicted Region: **{prediction}**")
        except Exception as e:
            st.error(f"❌ Error while predicting: {e}")
    else:
        st.warning("⚠️ Please enter some text.")
