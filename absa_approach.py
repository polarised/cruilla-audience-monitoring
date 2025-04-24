import stanza

nlp = stanza.Pipeline(lang='es', processors='tokenize,pos,constituency')

text = "La batería de portátil es increíble, pero el teclado es un poco malo y la música se oye mal."
doc = nlp(text)

s_phrases = []
for sentence in doc.sentences:
    tree = sentence.constituency
    queue = [tree]
    while queue:
        node = queue.pop(0)
        if not isinstance(node, tuple):
            if node.label == "S":
                phrase = " ".join(node.leaf_labels()).strip()
                s_phrases.append(phrase)
            queue.extend(child for child in node.children if not isinstance(child, tuple))


to_remove = set()
for i, phrase_i in enumerate(s_phrases):
    for j, phrase_j in enumerate(s_phrases):
        if i != j and phrase_j in phrase_i:
            to_remove.add(phrase_i)

final_phrases = [p for p in s_phrases if p not in to_remove]

# 3. Mostrar resultado final
print("\n✅ Frases S finales, limpias:")
for p in final_phrases:
    print(f" - {p}")
