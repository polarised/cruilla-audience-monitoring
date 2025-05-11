""" # import plotly.express as px
# import pandas as pd

# # Initialize your dataframe
# csv_path = r"tweets_with_topics.csv"
# assigned_df = pd.read_csv(csv_path)

# print("Loaded assigned_df successfully!")

# def visualize_table(assigned_df, save_path="subphrases_topics_plot.html"):
#     fig = px.scatter(
#         assigned_df,
#         x="Assigned_Topic_ID",
#         y=[1]*len(assigned_df),
#         text="subphrase",
#         hover_data=["Assigned_Topic_Name", "original_tweet_text"],
#         color="Assigned_Topic_ID",
#         size_max=60
#     )
#     fig.update_traces(textposition='top center')
#     fig.update_layout(
#         title="🔎 Subphrases grouped by Topic",
#         xaxis_title="Topic ID",
#         yaxis_visible=False,
#         showlegend=False,
#         height=600
#     )
#     # Save plot to HTML
#     fig.write_html(save_path)
#     print(f"\n✅ Plot saved successfully as '{save_path}'")

# # Call the function
# visualize_table(assigned_df)





# SECOND VERSION /////////////////////////////////////////////
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from umap import UMAP

# 1. Cargar resultados de BERTopic
df = pd.read_csv("tweets_topics_grandes.csv")
texts = df["tweet"].tolist()
topic_ids = df["Assigned_Topic_ID"].tolist()
topic_names = df["Assigned_Topic_Name"].tolist()
confidences = df["Assignment_Confidence"].tolist()

# 2. Calcular embeddings (igual que en el modelo)
print("✅ Calculando embeddings...")
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True)

# 3. Usar mismo UMAP que en tu BERTopic
print("✅ Proyectando con UMAP...")
umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine")
umap_embeddings = umap_model.fit_transform(embeddings)

# 4. Crear DataFrame para visualización
umap_df = pd.DataFrame(umap_embeddings, columns=["x", "y"])
umap_df["Tweet"] = texts
umap_df["Topic_ID"] = topic_ids
umap_df["Topic_Name"] = topic_names
umap_df["Confidence"] = confidences

# 5. Visualizar con Plotly
print("✅ Generando visualización...")
fig = px.scatter(
    umap_df, x="x", y="y",
    color="Topic_ID",
    hover_data=["Tweet", "Topic_Name", "Confidence"],
    title="🔎 UMAP de tópicos generados por BERTopic (tweets)"
)

# 6. Guardar como HTML
fig.write_html("umap_clusters_tweets.html")
fig.show()
 """