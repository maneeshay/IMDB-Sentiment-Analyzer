# IMDB Sentiment Analyzer 

A machine learning web app to classify movie reviews as positive or negative using Natural Language Processing (NLP).

##  Live Demo
[Try it here](https://huggingface.co/spaces/maneeshay/imdb-sentiment-analyzer)

##  Project Overview
Built a sentiment analysis model trained on 50,000 IMDB movie reviews that predicts whether a given review is positive or negative with **88% accuracy**.

##  Dataset
- **Source:** [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 reviews (25,000 positive, 25,000 negative)

##  Tech Stack
- Python
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Gradio

## Project Structure
```
IMDB-Sentiment-Analyzer/
│
├── sentiment.ipynb    # Main notebook (EDA, training, evaluation)
├── app.py             # Gradio web app
├── model.pkl          # Trained model
├── tfidf.pkl          # TF-IDF vectorizer
├── requirements.txt   # Dependencies
└── README.md
```

##  How It Works
1. Clean and preprocess raw movie reviews
2. Convert text to numbers using TF-IDF Vectorization
3. Train Logistic Regression model
4. Predict sentiment with confidence score

##  Author
Manisha Yadav