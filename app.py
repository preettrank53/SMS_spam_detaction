import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import pandas as pd
import time

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Page config
st.set_page_config(
    page_title="📩 Spam Classifier",
    page_icon="🚫",
    layout="centered"
)

# Sidebar Theme Toggle
st.sidebar.title("🔧 Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .stTextInput > div > div > input {
            background-color: #262730;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: #FFFFFF;
            color: black;
        }
        </style>
    """, unsafe_allow_html=True)

# App Title
st.markdown("<h2 style='text-align: center; color: #6c63ff;'>📨 Email/SMS Spam Classifier</h2>", unsafe_allow_html=True)

# ----------------- Single Message Prediction ----------------- #
st.subheader("📝 Single Message Prediction")
input_sms = st.text_area("Enter your message")

if st.button("🔍 Predict"):
    with st.spinner('Analyzing your message...'):
        time.sleep(1)
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

    st.success("✅ Prediction complete!")
    if result == 1:
        st.error("🚫 This message is **Spam**.")
    else:
        st.success("💌 This message is **Not Spam**.")

# ----------------- Batch Message Upload ----------------- #
st.markdown("---")
st.subheader("📁 Upload File for Batch Prediction")

uploaded_file = st.file_uploader("Upload a `.txt` or `.csv` file", type=['txt', 'csv'])

if uploaded_file is not None:
    ext = uploaded_file.name.split('.')[-1]

    if ext == 'txt':
        lines = uploaded_file.read().decode('utf-8').splitlines()
        df = pd.DataFrame(lines, columns=["message"])
    elif ext == 'csv':
        df = pd.read_csv(uploaded_file)
        if 'message' not in df.columns:
            st.error("CSV must contain a 'message' column.")
            st.stop()
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.write("✅ File loaded successfully!")
    st.write(df.head())

    if st.button("📊 Run Batch Prediction"):
        predictions = []
        progress = st.progress(0, text="Processing messages...")

        for i, row in df.iterrows():
            transformed_row = transform_text(str(row['message']))
            vector_row = tfidf.transform([transformed_row])
            pred = model.predict(vector_row)[0]
            predictions.append("Spam" if pred == 1 else "Not Spam")
            progress.progress((i + 1) / len(df), text=f"Processing {i + 1}/{len(df)} messages")

        # Individual predictions
        df["Prediction"] = predictions
        st.success("✅ Line-by-line prediction completed!")
        st.write(df)

        # Final decision for entire message
        combined_message = " ".join(df["message"].dropna().astype(str).tolist())
        transformed_combined = transform_text(combined_message)
        vector_combined = tfidf.transform([transformed_combined])
        final_result = model.predict(vector_combined)[0]

        st.markdown("---")
        st.subheader("🧠 Final Decision for Entire Message")
        if final_result == 1:
            st.error("🟥 FINAL DECISION: The full message is **Spam**.")
        else:
            st.success("🟩 FINAL DECISION: The full message is **Not Spam**.")

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV with Predictions",
            data=csv,
            file_name='spam_predictions.csv',
            mime='text/csv'
        )
