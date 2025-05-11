import spacy
from sentence_transformers import SentenceTransformer, util
import re

# Cargamos el modelo en español y el modelo de embeddings
nlp = spacy.load('es_core_news_sm')
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Tópicos predefinidos (sin listas de sinónimos)
topics = ["precios", "bebidas", "organización", "colas", "experiencia"]

# Convertimos los tópicos en embeddings para comparación
topic_embeddings = {label: embedding_model.encode(label) for label in topics}

# Umbral de confianza para asignación de tópicos
THRESHOLD = 0.4

# Ejemplos de tweets para probar
tweets = [
    "Las colas largas y los precios altos fueron un problema.",
    "La organización fue buena, pero los precios eran excesivos.",
    "Me encantó el concierto pero las colas eran interminables y los baños sucios.",
    "Los precios altos y las bebidas calientes arruinaron mi experiencia."
]

def clean_clause_for_embedding(clause):
    # Eliminamos puntuación innecesaria y conectores triviales
    clause = re.sub(r'[.,!?]', '', clause)  # Quitar puntuación básica
    # Eliminamos conjunciones triviales
    tokens = [token.text for token in nlp(clause) if token.text.lower() not in {"y", "o", "pero", "aunque"}]
    return " ".join(tokens)

def separate_and_classify_clauses(text):
    doc = nlp(text)
    clauses = []
    current_clause = []
    for token in doc:
        current_clause.append(token.text)
        # Dividimos en conjunciones que típicamente marcan subordinadas
        if token.dep_ in {"cc", "mark"}:
            if current_clause:
                clause = " ".join(current_clause).strip()
                clauses.append(clause)
                current_clause = []
    if current_clause:
        clauses.append(" ".join(current_clause).strip())
    
    # Clasificamos cada cláusula usando embeddings
    classified_clauses = []
    for clause in clauses:
        cleaned_clause = clean_clause_for_embedding(clause)
        print(f"\nCláusula original: '{clause}'")
        print(f"Cláusula limpia para embedding: '{cleaned_clause}'")
        clause_embedding = embedding_model.encode(cleaned_clause)
        best_topic = None
        best_score = -1
        for topic, topic_embedding in topic_embeddings.items():
            score = util.cos_sim(clause_embedding, topic_embedding).item()
            print(f"  → Comparando con tópico '{topic}': Coseno = {score:.4f}")
            if score > best_score:
                best_topic = topic
                best_score = score
        # Solo asignamos un tópico si supera el umbral
        if best_score >= THRESHOLD:
            print(f"  ✔️ Asignado a tópico: '{best_topic}' (score = {best_score:.4f})")
            classified_clauses.append((clause, best_topic))
        else:
            print(f"  ❌ Sin tópico (score = {best_score:.4f})")
            classified_clauses.append((clause, "sin tópico"))
    
    return classified_clauses

# Probamos con los ejemplos
test_results = {tweet: separate_and_classify_clauses(tweet) for tweet in tweets}
for tweet, clauses in test_results.items():
    print(f"\nTweet: {tweet}")
    print(f"Cláusulas clasificadas: {clauses}\n")
