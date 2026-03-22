import streamlit as st
import pickle
from nltk.corpus import stopwords
import re

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

# Clean text function
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\s+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# App title
st.title("IMDB Sentiment Analyzer")
st.write("Type a movie review and find out if it's positive or negative! ")

# Text input box
review = st.text_area("Enter your movie review here: ",height=150)

#Button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please eneter a review first!")
    else:
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = max(model.predict_proba(vectorized)[0]) *100

        if prediction == 'positive':
            st.success(f"POSITIVE - {round(confidence, 2)}% confidence")
        else:
            st.error(f"NEGATIVE - {round(confidence, 2)}% confidence")