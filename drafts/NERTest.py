import spacy
from collections import Counter

# Load the Spanish language model for NER
nlp = spacy.load("es_core_news_md")

def extract_entities(clause):
    """
    Extracts PERSON and ORG entities from a clause.
    
    Args:
        clause (str): The input text from which to extract entities.
        
    Returns:
        list: A sorted list of unique entities by frequency.
    """
    # Process the text using spaCy
    doc = nlp(clause)
    entities = []

    # Extract only PERSON and ORG entities
    for ent in doc.ents:
        if ent.label_ in ["PER", "ORG"]:
            entities.append(ent.text)

    # Count the frequency of each entity
    entity_counts = Counter(entities)

    # Sort entities by frequency in descending order
    sorted_entities = [entity for entity, count in entity_counts.most_common()]
    
    return sorted_entities

""" # Example usage
clause = "El concierto de Rosalía en el Cruïlla fue increíble, y la organización de Primavera Sound también destacó."
print(extract_entities(clause))
 """