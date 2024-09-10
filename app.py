import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import re
from googletrans import Translator

# Load model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Dictionary for emotions and their corresponding emojis
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", 
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Initialize the translator
translator = Translator()

# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove special characters and punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespaces
    text = text.strip()
    return text

# Main function
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here (Bahasa Indonesia)")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        # Clean the input text
        cleaned_text = clean_text(raw_text)
        # Translate cleaned text to English
        translated_text = translator.translate(cleaned_text, src='id', dest='en').text

        col1, col2 = st.columns(2)

        # Predict the emotion and its probability
        prediction = predict_emotions(translated_text)
        probability = get_prediction_proba(translated_text)

        with col1:
            st.success("Original Text (Translated)")
            st.write(translated_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"{prediction}: {emoji_icon}")
            st.write(f"Confidence: {np.max(probability)}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            # Visualize the prediction probabilities
            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions', 
                y='probability', 
                color='emotions'
            )
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()