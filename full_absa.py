import pandas as pd
import re
import stanza
import spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(
    min_cluster_size=40,  # BIGGER clusters
    min_samples=15,       # More strict density
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

def extract_small_phrases(text):
    doc = nlp_spacy(text)
    phrases = []
    for sent in doc.sents:
        current_phrase = []
        for token in sent:
            if token.text in [",", ";"] or token.pos_ == "CCONJ" or token.dep_ == "cc":  # <--- clean detection
                if current_phrase:
                    phrase = " ".join(current_phrase).strip()
                    if len(phrase.split()) >= 2:
                        phrases.append(phrase)
                    current_phrase = []
                continue  # skip "y", "pero", etc.
            current_phrase.append(token.text)
        if current_phrase:
            phrase = " ".join(current_phrase).strip()
            if len(phrase.split()) >= 2:
                phrases.append(phrase)
    return phrases



def noun_tokenizer(text):
    doc = nlp_spacy(text)
    return [token.lemma_.lower() for token in doc if token.pos_ == "NOUN" and not token.is_stop and token.is_alpha]

# ----------------------------
# 1. Extract subphrases
# ----------------------------
all_subphrases = []
tweet_ids = []

for idx, tweet in enumerate(tweets_cleaned):
    subphrases = extract_small_phrases(tweet)
    all_subphrases.extend(subphrases)
    tweet_ids.extend([idx] * len(subphrases))

print(f"\n✅ Extracted {len(all_subphrases)} subphrases from {len(tweets_cleaned)} tweets.")

print("\n✅ Sample of extracted subphrases and their original tweets:")

for idx in range(min(20, len(all_subphrases))):  # Print up to 20 examples
    subphrase = all_subphrases[idx]
    tweet_id = tweet_ids[idx]
    original_tweet = tweets_cleaned[tweet_id]
    
    print(f"\nSubphrase {idx+1}:")
    print(f"  - Subphrase: {subphrase}")
    print(f"  - From Tweet: {original_tweet}")


# ----------------------------
# 2. Train BERTopic (no KMeans)
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
topics, probs = topic_model.fit_transform(all_subphrases)

# ----------------------------
# 3. Save results cleanly
# ----------------------------
assigned_df = pd.DataFrame({
    "original_tweet_id": tweet_ids,
    "original_tweet_text": [tweets_cleaned[i] for i in tweet_ids],
    "subphrase": all_subphrases,
    "Assigned_Topic_ID": topics,
    "Assignment_Confidence": [max(p) if p is not None else None for p in probs],
    "Assigned_Topic_Name": [topic_model.get_topic(t)[0][0] if t != -1 else "Unknown" for t in topics],
})

# Filter meaningless subphrases (optional step)
assigned_df = assigned_df[assigned_df["subphrase"].str.split().str.len() >= 2]


# ----------------------------
# 4. Save to CSV
# ----------------------------
assigned_df.to_csv("subphrases_with_topics.csv", index=False)

print("\n✅ Subphrases and assigned topics saved to 'subphrases_with_topics.csv'.")
print(assigned_df.head())
