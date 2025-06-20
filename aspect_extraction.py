
import re
import spacy
from transformers import pipeline
nlp = spacy.load('es_core_news_sm')
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli") el entrenado en corpus inglés, que todavía funciona
# en castellano y catalán pero peor.


""" 
def separate_clauses(text):
    # 1. Separació inicial per puntuació forta
    split_by_punctuation = re.split(r'.;', text)
    
    clauses = []
    for segment in split_by_punctuation:
        segment = segment.strip()
        if not segment:
            continue
        doc = nlp(segment)
        current_clause = []
        for token in doc:
            current_clause.append(token.text)
            if token.dep_ in {"cc", "mark"}:
                if current_clause:
                    clause = " ".join(current_clause).strip()
                    clauses.append(clause)
                    current_clause = []
        if current_clause:
            clauses.append(" ".join(current_clause).strip())
    
    return clauses """

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

SUBORDINATING_MARKERS = {
    # Spanish
    "porque", "aunque", "si", "cuando", "mientras", "ya que", "aun así",
    # Catalan
    "perquè", "encara que", "si", "quan", "mentre", "ja que", "tot i així",
    # English
    "because", "although", "if", "when", "while", "since", "even though"
}

COORDINATING_CONJUNCTIONS = {
    # Spanish
    "pero", "sino", "sin embargo",
    # Catalan
    "però", "sinó", "tanmateix",
    # English
    "but", "however", "yet"
}

def should_split(token):
    return token.text.lower() in SUBORDINATING_MARKERS or token.text.lower() in COORDINATING_CONJUNCTIONS

def separate_clauses(text):
    # First split by punctuation marks that strongly indicate clause boundaries
    split_by_punctuation = re.split(r'[.!?;]', text)

    clauses = []
    for segment in split_by_punctuation:
        segment = segment.strip()
        if not segment:
            continue

        doc = nlp(segment)
        current_clause = []

        for token in doc:
            current_clause.append(token.text)
            if should_split(token):
                clause = " ".join(current_clause).strip()
                if clause:
                    clauses.append(clause)
                current_clause = []

        if current_clause:
            clauses.append(" ".join(current_clause).strip())

    return clauses



def classify_clauses_zero_shot(clauses, topics, threshold=0.75):
    classified_clauses = []
    for clause in clauses:
        print(f"\ncláusula: '{clause}'")
        # Limpiamos la cláusula para el clasificador
        cleaned_clause = re.sub(r'[.,!?]', '', clause)  # Quitar puntuación básica
        #clasificación zero-shot
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
