import pandas as pd
import re
import stanza
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True)


# ----------------------------
# 1. Load data
# ----------------------------
df = pd.read_csv("cruilla_es_absa_synthetic.csv")  # change filename if needed
tweets = df["tweet"].dropna().tolist()

# ----------------------------
# 2. Clean basic text
# ----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\sÀ-ÿ]", "", text)
    return text.strip()

tweets_cleaned = [clean_text(t) for t in tweets]

# ----------------------------
# 3. Load NLP models
# ----------------------------
nlp_stanza = stanza.Pipeline(lang='es', processors='tokenize,pos,constituency')
nlp_spacy = spacy.load("es_core_news_sm")

# ----------------------------
# 4. Define functions
# ----------------------------

# (A) Extract S constituents
def extract_S_phrases(text):
    doc = nlp_stanza(text)
    s_phrases = []
    for sentence in doc.sentences:
        tree = sentence.constituency
        queue = [tree]
        while queue:
            node = queue.pop(0)
            if not isinstance(node, tuple):
                if node.label == "S":
                    phrase = " ".join(node.leaf_labels()).strip()
                    s_phrases.append(phrase)
                queue.extend(child for child in node.children if not isinstance(child, tuple))
    # Remove redundant nested phrases
    to_remove = set()
    for i, phrase_i in enumerate(s_phrases):
        for j, phrase_j in enumerate(s_phrases):
            if i != j and phrase_j in phrase_i:
                to_remove.add(phrase_i)
    final_phrases = [p for p in s_phrases if p not in to_remove]
    return final_phrases

# (B) Tokenizer: only nouns
def noun_tokenizer(text):
    doc = nlp_spacy(text)
    return [token.lemma_.lower() for token in doc if token.pos_ == "NOUN" and not token.is_stop and token.is_alpha]

# ----------------------------
# 5. Extract all S constituents
# ----------------------------
all_extracted_S = []
tweet_ids = []  # Link back to tweets

for idx, tweet in enumerate(tweets_cleaned):
    s_phrases = extract_S_phrases(tweet)
    all_extracted_S.extend(s_phrases)
    tweet_ids.extend([idx] * len(s_phrases))

print(f"\n✅ Extracted {len(all_extracted_S)} S phrases from {len(tweets_cleaned)} tweets.")
print("All extracted S:\n")
for element in all_extracted_S:
    print(element + "\n")
# ----------------------------
# 6. Setup BERTopic with KMeans
# ----------------------------
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
vectorizer_model = CountVectorizer(tokenizer=noun_tokenizer, lowercase=True)
kmeans_model = KMeans(n_clusters=15, random_state=42)  # You can control the number
print("setting up embedding, vectorizer and kmeans..\n")
topic_model = BERTopic(
    language="multilingual",
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    hdbscan_model=hdbscan_model,
    calculate_probabilities=True,
    verbose=True
)


# ----------------------------
# 7. Train model on extracted S constituents
# ----------------------------
print("training bertopic")
topics, probs = topic_model.fit_transform(all_extracted_S)
topic_info = topic_model.get_topic_info()
# 7bis. Reduce number of topics to desired number
topic_model.reduce_topics(all_extracted_S, nr_topics=15)

# ----------------------------
# 8. Save results cleanly
# ----------------------------
topic_info = topic_model.get_topic_info()
assigned_df = pd.DataFrame({
    "original_tweet_id": tweet_ids,
    "original_tweet_text": [tweets_cleaned[i] for i in tweet_ids],
    "S_constituent": all_extracted_S,
    "Assigned_Topic_ID": topics,
    "Assignment_Confidence": [max(p) if p is not None else None for p in probs],
    "Assigned_Topic_Name": [topic_model.get_topic(t)[0][0] if t != -1 else "Unknown" for t in topics],
    "original_aspect": [df["aspect"].iloc[i] for i in tweet_ids],
    "original_sentiment": [df["sentiment"].iloc[i] for i in tweet_ids],
    "idioma": [df["idioma"].iloc[i] for i in tweet_ids]
})

print("\n✅ Example of assigned topics:")
print(assigned_df.head())

# ----------------------------
# 9. (Optional) Save everything
# ----------------------------
# topic_model.save("bertopic_model.csv")
# assigned_df.to_csv("assigned_S_constituents.csv", index=False)
