import requests
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

def search_movie(query: str):
    """Search for a movie ID by user query."""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            return {
                "movie_id": results[0]["id"],
                "title": results[0]["title"],
                "overview": results[0]["overview"]
            }
    return None

def get_movie_reviews(movie_id: int):
    """Fetch reviews and attach movie_id to the data."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            df = pd.DataFrame(results)
            # Rename author's id column to review_id to avoid confusion with movie_id
            df = df.rename(columns={'id': 'review_id', 'content': 'text'})
            df['movie_id'] = movie_id
            
            # Keep only the core columns
            return df[['review_id', 'movie_id', 'author', 'text']]
    return pd.DataFrame()