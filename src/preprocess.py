import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK datasets (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text: str) -> str:
    """
    Clean raw text: lowercase, remove HTML, punctuation, numbers, and stopwords.
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Convert to lowercase
    text = text.lower()

    # 1.5. Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # 2. Remove HTML tags (e.g., <br>)
    text = re.sub(r'<.*?>', ' ', text)
    
    # 3. Remove punctuation and numbers (keep only alphabet characters)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenize and remove English stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Join tokens back into a single string
    return " ".join(clean_tokens)

if __name__ == "__main__":
    print("Starting text preprocessing...")
    
    input_path = "data/raw/raw_reviews.csv"
    output_dir = "data/processed"
    output_path = os.path.join(output_dir, "cleaned_reviews.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}. Run api_client.py first.")
    else:
        # Read the raw data
        df = pd.read_csv(input_path)
        
        print(f"Cleaning {len(df)} reviews (this might take a few seconds)...")
        # Apply the cleaning function to create a new column
        df['clean_text'] = df['text'].apply(clean_text)
        
        # Ensure the processed directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the cleaned data
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"Success! Cleaned data saved to: {output_path}")
        print("\n--- Before vs After ---")
        print("RAW:", df['text'].iloc[0][:100], "...")
        print("CLEAN:", df['clean_text'].iloc[0][:100], "...")