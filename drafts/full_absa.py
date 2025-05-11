import pandas as pd
import re
import spacy
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from hdbscan import HDBSCAN
from umap import UMAP

# ----------------------------
# 1. Cargar dataset completo
# ----------------------------
df = pd.read_csv("synthetic data/tweets_combinados.csv")
tweets = df["tweet"].dropna().tolist()

# ----------------------------
# 2. Limpiar tweets
# ----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\sÀ-ÿ]", "", text)
    return text.strip()

tweets_cleaned = [clean_text(t) for t in tweets]
tweets_cleaned = [t for t in tweets_cleaned if len(t.split()) > 2]  # ❗ Evitar tweets vacíos o muy cortos

# ----------------------------
# 3. Stopwords personalizadas
# ----------------------------
nlp = spacy.load("es_core_news_sm")
stopwords_spacy = list(nlp.Defaults.stop_words)
custom_garbage = [
    "brutal", "todooo", "flipanteee", "genialll", "biennn", "bno", "xq", "toa",
    "k", "vibraaa", "super", "bien", "lo", "dio", "todo", "cuando", "aunque",
    "porque", "ya", "que"
]
full_stopwords = list(set(stopwords_spacy + custom_garbage))

# ----------------------------
# 4. Componentes de BERTopic
# ----------------------------
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

umap_model = UMAP(
    n_neighbors=15,
    n_components=2,
    min_dist=0.1,
    metric="cosine"
)

hdbscan_model = HDBSCAN(
    min_cluster_size=45,
    min_samples=10,
    prediction_data=True
)

vectorizer_model = TfidfVectorizer(
    stop_words=full_stopwords,
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=3
)

# ----------------------------
# 5. Entrenar BERTopic
# ----------------------------
topic_model = BERTopic(
    language="multilingual",
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)

print("✅ Entrenando BERTopic...")
topics, probs = topic_model.fit_transform(tweets_cleaned)

# ----------------------------
# 6. Guardar resultados
# ----------------------------
assigned_df = pd.DataFrame({
    "tweet": tweets_cleaned,
    "Assigned_Topic_ID": topics,
    "Assignment_Confidence": [max(p) if p is not None else None for p in probs],
    "Assigned_Topic_Name": [
        topic_model.get_topic(t)[0][0] if t != -1 and topic_model.get_topic(t) else "Unknown"
        for t in topics
    ],
})

assigned_df.to_csv("tweets_topics_grandes.csv", index=False)
print("✅ Resultados guardados en 'tweets_topics_grandes.csv'")
print(assigned_df.head())
