import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_movie_reviews(movie_id: int, max_pages: int = 1) -> pd.DataFrame:
    """
    Fetch movie reviews from TMDB API and return them as a Pandas DataFrame.
    """
    reviews_data = []
    
    for page in range(1, max_pages + 1):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": page
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"API Error: Status code {response.status_code}")
            break
            
        results = response.json().get("results", [])
        
        if not results:
            break
            
        for item in results:
            content = item.get("content")
            rating = item.get("author_details", {}).get("rating")
            
            # Only keep reviews that have text content and a valid rating
            if content and rating is not None:
                reviews_data.append({
                    "text": content,
                    "rating": rating
                })
                
    return pd.DataFrame(reviews_data)

if __name__ == "__main__":
    # List of popular movie IDs (e.g., Dune 2, Interstellar, The Dark Knight, Inception)
    movie_ids = [693134, 157336, 155, 27205]
    all_reviews = []

    print("Fetching data from TMDB API...")
    
    for mid in movie_ids:
        # Fetch up to 3 pages per movie to gather more data
        df = fetch_movie_reviews(mid, max_pages=3)
        if not df.empty:
            all_reviews.append(df)

    # Merge data from all movies into a single DataFrame
    if all_reviews:
        final_df = pd.concat(all_reviews, ignore_index=True)
        
        # 1. Create a Sentiment label column (1: Positive if rating >= 7, 0: Negative otherwise)
        final_df["sentiment"] = final_df["rating"].apply(lambda x: 1 if x >= 7 else 0)

        # 2. Ensure the destination directory exists
        os.makedirs("data/raw", exist_ok=True)

        # 3. Export to CSV file
        save_path = "data/raw/raw_reviews.csv"
        final_df.to_csv(save_path, index=False, encoding='utf-8')

        print(f"Success! Collected {len(final_df)} reviews.")
        print(f"File saved to: {save_path}")
        print(final_df[['rating', 'sentiment', 'text']].head())
    else:
        print("No data returned from the API.")