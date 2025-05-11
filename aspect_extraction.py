import spacy
from transformers import pipeline
import re

nlp = spacy.load('es_core_news_sm')
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
topics = ["precios", "bebidas", "organización", "colas", "experiencia", "comida", "sonido", "baños", "seguridad", "personal", "ambiente"]
# topics provisionales

def separate_clauses(text):
    doc = nlp(text)
    clauses = []
    current_clause = []
    for token in doc:
        current_clause.append(token.text)
        # Dividimos solo en marcadores de subordinación y coordinación
        if token.dep_ in {"cc", "mark"}:
            if current_clause:
                clause = " ".join(current_clause).strip()
                clauses.append(clause)
                current_clause = []
    if current_clause:
        clauses.append(" ".join(current_clause).strip())
    return clauses


def classify_clauses_zero_shot(clauses, topics, threshold=0.4):
    classified_clauses = []
    for clause in clauses:
        print(f"\ncláusula: '{clause}'")
        # Limpiamos la cláusula para el clasificador
        cleaned_clause = re.sub(r'[.,!?]', '', clause)  # Quitar puntuación básica
        # Hacemos la clasificación zero-shot
        result = classifier(cleaned_clause, topics)
        best_topic = result['labels'][0]
        best_score = result['scores'][0]
        # Aplicamos el threshold para marcar como 'unknown'
        if best_score < threshold:
            best_topic = 'unknown'
            print(f"marcado como 'unknown' (score = {best_score:.4f})")
        else:
            print(f"asignado a tópico: '{best_topic}' (score = {best_score:.4f})")
        print(f"asignado a tópico: '{best_topic}' (score = {best_score:.4f})")
        classified_clauses.append((clause, best_topic))
    return classified_clauses
