import spacy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

nlp = spacy.load("es_core_news_md")

file_path = "synthetic data/tweets_combinados.csv"
df = pd.read_csv(file_path)

tweets = df['tweet'].tolist()

entities = []

for tweet in tweets:
    doc = nlp(tweet)
    for ent in doc.ents:
        # only keep PERSON and ORG
        if ent.label_ in ["PER", "ORG"]:
            entities.append(ent.text)

# freq of entities
entity_counts = Counter(entities)
top_entities = entity_counts.most_common(20)

# top 20
entities, counts = zip(*top_entities)
plt.figure(figsize=(12, 6))
plt.barh(entities, counts, color="teal")
plt.title("Top 20 PERSON and ORG Entities in Tweets")
plt.xlabel("Frequency")
plt.ylabel("Entities")
plt.gca().invert_yaxis()
plt.show()
