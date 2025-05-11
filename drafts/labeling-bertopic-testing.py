import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
import spacy
import re

# ----------------------------
# 1. Carrega dades
# ----------------------------
df = pd.read_csv("cruilla_es_absa_synthetic.csv")
tweets = df["tweet"].dropna().tolist()

# ----------------------------
# 2. Neteja bàsica de text
# ----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\sÀ-ÿ]", "", text)
    return text.strip()

tweets = [clean_text(t) for t in tweets]

# ----------------------------
# 3. Tokenització: només noms i adjectius
# ----------------------------
nlp = spacy.load("es_core_news_sm")

def noun_adj_tokenizer(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN"] and not token.is_stop and token.is_alpha]

vectorizer_model = CountVectorizer(tokenizer=noun_adj_tokenizer, lowercase=True)

# ----------------------------
# 4. Configura BERTopic
# ----------------------------
embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=5)

topic_model = BERTopic(
    language="multilingual",
    embedding_model=embedding_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    verbose=True
)

# ----------------------------
# 5. Entrena el model
# ----------------------------
topics, _ = topic_model.fit_transform(tweets)

# ----------------------------
# 6. Mostra resultats
# ----------------------------
topics_info = topic_model.get_topic_info()

print("\n🟡 Topics detectats:\n")
for index, row in topics_info.iterrows():
    if row["Topic"] != -1:
        topic_id = row["Topic"]
        keywords = [word for word, _ in topic_model.get_topic(topic_id)]
        freq = row["Count"]

        print(f"🟢 Topic {topic_id} ({freq} tuits):")
        print(f"   → Keywords: {', '.join(keywords)}\n")
