import spacy

def extract_smaller_phrases(texts):
    nlp = spacy.load('es_core_news_md')  # Spanish medium model

    for text in texts:
        print(f"\n📜 Text: {text}")
        doc = nlp(text)

        phrases = []
        
        for sent in doc.sents:
            current_phrase = []
            for token in sent:
                # Split when:
                if token.text in [",", ";"] or token.dep_ in ("mark", "cc"):  # <--- added 'cc' for coordinating conjunctions
                    if current_phrase:
                        phrases.append(" ".join(current_phrase).strip())
                        current_phrase = []
                current_phrase.append(token.text)
            
            if current_phrase:
                phrases.append(" ".join(current_phrase).strip())

        if phrases:
            print("\n🔗 Extracted small phrases:")
            for p in phrases:
                print(f" - {p}")
        else:
            print("⚠️ No phrases found.")

# Example usage
texts = [
    "El servicio de atención al cliente fue excelente, pero el envío tardó demasiado y el embalaje estaba dañado."
]
extract_smaller_phrases(texts)
