import streamlit as st
import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_client import search_movie, get_movie_reviews
from src.database import save_movie, save_reviews, load_reviews
from src.preprocess import clean_text

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_resource
def load_ai_models():
    vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
    model_path = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
    
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model

vectorizer, model = load_ai_models()

st.title("Movie sentiment analysis application")

movie_name = st.text_input("Enter movie name to analyze (e.g. Inception, Avatar):")

if st.button("Search and analyze"):
    if movie_name:
        with st.spinner(f"Searching for movie '{movie_name}'..."):
            movie_data = search_movie(movie_name)
            
            if not movie_data:
                st.error("Movie not found in the system.")
            else:
                st.success(f"Found: {movie_data['title']}")
                st.caption(movie_data['overview'])
                
                save_movie(movie_data)
                movie_id = movie_data['movie_id']
                
                df_reviews = load_reviews(movie_id)
                
                if df_reviews.empty:
                    st.info("New movie detected. Fetching data from API and analyzing. Please wait...")
                    
                    df_raw = get_movie_reviews(movie_id)
                    
                    if not df_raw.empty:
                        df_raw['clean_text'] = df_raw['text'].apply(clean_text)
                        X_tfidf = vectorizer.transform(df_raw['clean_text'])
                        df_raw['sentiment'] = model.predict(X_tfidf)
                        
                        save_reviews(df_raw)
                        df_reviews = load_reviews(movie_id)
                    else:
                        st.warning("This movie has no reviews on TMDB yet.")
                else:
                    st.info("Found data in database. Skipping API and AI processing.")

                if not df_reviews.empty:
                    positive_count = len(df_reviews[df_reviews['sentiment'] == 1])
                    negative_count = len(df_reviews[df_reviews['sentiment'] == 0])
                    
                    st.subheader("AI analysis results")
                    col1, col2 = st.columns(2)
                    col1.metric("Positive", positive_count)
                    col2.metric("Negative", negative_count)
                    
                    st.write("Detailed reviews:")
                    
                    # Create a copy specifically for UI formatting
                    display_df = df_reviews[['author', 'sentiment', 'text']].copy()
                    
                    # Shift index to start from 1 instead of 0
                    display_df.index = display_df.index + 1
                    
                    # Map numeric values to readable text
                    display_df['sentiment'] = display_df['sentiment'].map({1: 'Positive', 0: 'Negative'})
                    
                    # Display table with full container width to prevent column cutoff
                    st.dataframe(display_df, use_container_width=True)