import gradio as gr
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#load saved model and vectorizer
model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

#clean text function
def clean_text(text):
    text = re.sub(r'<.*?>' ,'', text)
    text = re.sub(r'http\S+|www\s+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Prediction function
def predict_sentiment(review):
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = max(model.predict_proba(vectorized)[0]) *100
    return f"{prediction.upper()} - {round(confidence, 2)}% confidence"

app = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter your movie review here.."),
    outputs=gr.Textbox(label="Sentiment"),
    title = "IMDB sentiment Analyzer",
    description="Type a movie review and find out if it's positive or negative!"
    
)

app.launch()