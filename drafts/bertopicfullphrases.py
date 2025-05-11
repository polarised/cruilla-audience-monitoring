import pandas as pd
import re
import stanza
import spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

# ----------------------------
# 0. Configuración
# ----------------------------
hdbscan_model = HDBSCAN(
    min_cluster_size=40,
    min_samples=15,
    prediction_data=True
)

embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
nlp_spacy = spacy.load("es_core_news_sm")

df = pd.read_csv("tweets_dataset.csv")
tweets = df["tweet"].dropna().tolist()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\sÀ-ÿ]", "", text)
    return text.strip()

tweets_cleaned = [clean_text(t) for t in tweets]

def noun_tokenizer(text):
    doc = nlp_spacy(text)
    return [token.lemma_.lower() for token in doc if token.pos_ == "NOUN" and not token.is_stop and token.is_alpha]

# ----------------------------
# 1. Train BERTopic on full tweets
# ----------------------------
print("\n✅ Setting up BERTopic...")
vectorizer_model = CountVectorizer(tokenizer=noun_tokenizer, lowercase=True)

topic_model = BERTopic(
    language="multilingual",
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)

print("\n✅ Training BERTopic...")
topics, probs = topic_model.fit_transform(tweets_cleaned)

# ----------------------------
# 2. Save results cleanly
# ----------------------------
assigned_df = pd.DataFrame({
    "original_tweet_id": list(range(len(tweets_cleaned))),
    "original_tweet_text": tweets_cleaned,
    "Assigned_Topic_ID": topics,
    "Assignment_Confidence": [max(p) if p is not None else None for p in probs],
    "Assigned_Topic_Name": [topic_model.get_topic(t)[0][0] if t != -1 else "Unknown" for t in topics],
})

# ----------------------------
# 3. Save to CSV
# ----------------------------
assigned_df.to_csv("tweets_with_topics.csv", index=False)

print("\n✅ Full tweets and assigned topics saved to 'tweets_with_topics.csv'.")
print(assigned_df.head())
