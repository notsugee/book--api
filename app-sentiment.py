from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import pickle
import time
from nltk.sentiment import SentimentIntensityAnalyzer  # Import for sentiment analysis
import nltk

nltk.download('vader_lexicon')  # Download necessary data for VADER

app = Flask(__name__)
CORS(app)

print("Loading data...")
start_time = time.time()

# Load pre-computed data
df = pd.read_csv('books.csv')
with open('book_embeddings.pkl', 'rb') as f:
    book_embeddings = pickle.load(f)
with open('similar_books.pkl', 'rb') as f:
    similar_books_dict = pickle.load(f)

# Create a fast lookup dictionary for books
book_lookup = {book.lower(): book for book in df['title']}

print(f"Data loaded in {time.time() - start_time:.2f} seconds")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def find_book_by_name(book_name):
    # Try exact match first (case-insensitive)
    lower_name = book_name.lower()
    if lower_name in book_lookup:
        return df[df['title'] == book_lookup[lower_name]].iloc[0]
    
    # If no exact match, use fuzzy matching
    matches = process.extractBests(book_name, book_lookup.values(), score_cutoff=80, limit=1)
    if matches:
        return df[df['title'] == matches[0][0]].iloc[0]
    return None

def get_books_by_sentiment(sentiment):
    if sentiment == 'positive':
        keywords = ['happy', 'joy', 'love', 'optimism', 'hope']
    elif sentiment == 'negative':
        keywords = ['sad', 'grief', 'anger', 'pain', 'despair']
    else:
        keywords = ['calm', 'neutral', 'balanced', 'ordinary']

    # Filter books based on categories or descriptions matching sentiment keywords
    filtered_books = df[df['categories'].str.contains('|'.join(keywords), case=False, na=False) |
                        df['description'].str.contains('|'.join(keywords), case=False, na=False)]
    
    # If no books found, return the top-rated ones as fallback
    if filtered_books.empty:
        filtered_books = df.sort_values('average_rating', ascending=False).head(10)

    return filtered_books.head(10)

def get_book_recommendations(book_name, top_n=10):
    start_time = time.time()

    # Perform sentiment analysis on the input
    sentiment_scores = sia.polarity_scores(book_name)
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    # Get books based on sentiment
    recommendations = get_books_by_sentiment(sentiment)
    
    print(f"Recommendations found in {time.time() - start_time:.2f} seconds")
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    start_time = time.time()
    data = request.json
    book_name = data.get('book_name')
    if not book_name:
        return jsonify({'error': 'No book name provided'}), 400

    recommendations = get_book_recommendations(book_name)
    
    results = []
    for _, book in recommendations.iterrows():
        results.append({
            'title': book['title'],
            'authors': book['authors'],
            'description': (str(book['description'])[:200] + '...') if book['description'] is not None and len(str(book['description'])) > 200 else (str(book['description']) if book['description'] is not None else ''),
            'categories': book['categories'],
            'average_rating': book['average_rating']
        })

    print(f"Total API response time: {time.time() - start_time:.2f} seconds")
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
