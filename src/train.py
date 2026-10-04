import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def train_sentiment_model():
    data_path = "data/processed/cleaned_reviews.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    print("1. Loading processed data...")
    df = pd.read_csv(data_path)
    
    # Drop any rows where text might have become empty after cleaning
    df = df.dropna(subset=['clean_text', 'sentiment'])
    
    X = df['clean_text']
    y = df['sentiment']

    print("2. Splitting data into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Vectorizing text data (TF-IDF)...")
    # Convert text to a numerical matrix
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("4. Training XGBoost classifier...")
    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    model.fit(X_train_tfidf, y_train)

    print("5. Evaluating the model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"]))

    print("6. Saving model and vectorizer...")
    os.makedirs("models", exist_ok=True)
    
    # Save the TF-IDF vectorizer (needed to transform new reviews later)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    # Save the trained XGBoost model
    joblib.dump(model, "models/xgboost_model.pkl")
    
    print("Success! Artifacts saved in the 'models/' directory.")

if __name__ == "__main__":
    train_sentiment_model()