import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("mbti_1.csv")

# Binary label: 0 = Introvert (I), 1 = Extrovert (E)
df['label'] = df['type'].apply(lambda x: 0 if x[0] == 'I' else 1)

# Clean text (light clean, no stopword removal)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    return text

df['clean_posts'] = df['posts'].apply(clean_text)

# Balance the dataset
introverts = df[df['label'] == 0]
extroverts = df[df['label'] == 1]

extroverts_upsampled = resample(extroverts,
                                 replace=True,
                                 n_samples=len(introverts),
                                 random_state=42)

df_balanced = pd.concat([introverts, extroverts_upsampled])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['clean_posts'], df_balanced['label'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Naive Bayes model (better for text)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

# Accuracy
acc = model.score(X_test_tfidf, y_test)
print(f"✅ Model trained successfully!")
print(f"🎯 Accuracy: {acc:.4f}")
print("💾 model.pkl and tfidf.pkl saved.")
