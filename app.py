import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk


nltk.download("stopwords")

# Load model pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Text preprocessing
def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    words = text.split()
    return " ".join([stemmer.stem(word) for word in words if word not in stop_words])

# Streamlit app
st.title("🧪 Toxic Comment Classifier")
comment = st.text_area("Enter a comment to analyze:")

if st.button("Predict"):
    if comment.strip() == "":
        st.warning("Please enter a valid comment.")
    else:
        cleaned = preprocess(comment)
        prediction = model.predict([cleaned]) 
        prediction = prediction[0]  

        st.subheader("Prediction Results:")
        for label, is_present in zip(toxic_labels, prediction):
            st.write(f"**{label.capitalize()}**: {'✅ Detected' if is_present else '❌ Not Detected'}")



