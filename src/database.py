import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

def get_engine():
    return create_engine(DB_URL)

def save_movie(movie_data: dict):
    """Save movie metadata (Skip if it already exists)."""
    engine = get_engine()
    df = pd.DataFrame([movie_data])
    try:
        # Check if the movie already exists
        query = f"SELECT movie_id FROM movies WHERE movie_id = {movie_data['movie_id']}"
        existing = pd.read_sql(query, engine)
        if not existing.empty:
            return
    except Exception:
        pass # If an error occurs, the table might not exist yet; proceed
        
    df.to_sql('movies', engine, if_exists='append', index=False)

def save_reviews(df: pd.DataFrame):
    """Save new reviews. Only insert reviews that do not exist in the database."""
    if df.empty: 
        return
    
    engine = get_engine()
    movie_id = df['movie_id'].iloc[0]
    
    try:
        # Fetch the list of existing review IDs for this movie
        query = f"SELECT review_id FROM reviews WHERE movie_id = {movie_id}"
        existing_ids = pd.read_sql(query, engine)['review_id'].tolist()
        
        # Filter out only new reviews (IDs not currently in DB)
        df_new = df[~df['review_id'].isin(existing_ids)]
        
        if not df_new.empty:
            df_new.to_sql('reviews', engine, if_exists='append', index=False)
            print(f"Successfully updated {len(df_new)} new reviews.")
    except Exception:
        # If the reviews table does not exist (first run)
        df.to_sql('reviews', engine, if_exists='append', index=False)
        print(f"Initialized table and saved {len(df)} reviews.")

def load_reviews(movie_id: int) -> pd.DataFrame:
    """Fetch review data from the database for the UI."""
    engine = get_engine()
    try:
        return pd.read_sql(f"SELECT * FROM reviews WHERE movie_id = {movie_id}", engine)
    except Exception:
        return pd.DataFrame()