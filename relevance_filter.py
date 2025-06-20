from sentence_transformers import SentenceTransformer, util
import re
import pandas as pd

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

reference_texts = [
    # Entradas
    "Compra de entradas para el Cruïlla",
    "Vendo entrada para el Cruïlla 2025",
    "Entradas agotadas del Cruïlla Festival",
    "Consigue tu abono para el Cruïlla",
    
    # Horarios y escenarios
    "Horarios de los conciertos del Cruïlla",
    "Escenarios del Cruïlla 2025",
    "Cambio de horario en el Cruïlla Festival",
    "Consulta la programación del Cruïlla",
    
    # Transporte
    "Cómo llegar al Cruïlla en Rodalies",
    "Tren oficial del Cruïlla Festival",
    "Problemas con el transporte al Cruïlla",
    "Estaciones cercanas al Cruïlla",

    # Opiniones y emociones
    "Conciertos inolvidables en el Cruïlla",
    "Críticas sobre actuaciones en el Cruïlla",
    "Opiniones sobre artistas en el Cruïlla Festival",
    "Experiencia en el Cruïlla 2025",

    # General festival context
    "Festival Cruïlla 2025 en el Parc del Fòrum",
    "Mejor festival del verano en Barcelona",
    "Ambiente en el Cruïlla Festival",
    "Line-up del Cruïlla 2025"
]

# Embeddings de les frases de referència
reference_embeddings = model.encode(reference_texts, convert_to_tensor=True)

#netejar text
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"@(\w+)", r"\1", text) 
    text = re.sub(r"http\S+|#[^\s]+", "", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_about_cruilla_festival(tweet_text, reply_to_user=None, threshold=0.40):
    if isinstance(reply_to_user, str) and reply_to_user.strip().lower() == 'cruillabcn':
        return True, 1.0
    cleaned = clean_tweet(tweet_text)
    tweet_embedding = model.encode([cleaned], convert_to_tensor=True)
    similarities = util.cos_sim(tweet_embedding, reference_embeddings)
    max_similarity = similarities[0].max().item()
    return max_similarity >= threshold, max_similarity

INPUT_CSV = "scrapped_data/annotatedevalfinal.csv"
OUTPUT_CSV = "scrapped_data/annotatedevalfinal_threshold0386.csv"

df = pd.read_csv(INPUT_CSV)
df = df[df['author_userName'].str.lower() != 'cruillabcn']
df = df.dropna(subset=['fullText'])

df['is_relevant'], df['similarity'] = zip(*df.apply(
    lambda row: is_about_cruilla_festival(row['fullText'], row.get('Reply To User')),
    axis=1
))

df.to_csv(OUTPUT_CSV, index=False)
print(f"csv amb llindar 0.35 desat a: {OUTPUT_CSV}")
