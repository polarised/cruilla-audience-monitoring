from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

tweets = [
    "El festival ha sido una pasada este año! 🔥",
    "Aquest any el so no era gaire bo...",
    "¡Qué emoción! Espero que anuncien los horarios pronto.",
    "No m'agrada gens l'ambient d'enguany."
]

for tweet in tweets:
    result = sentiment_pipe(tweet)[0]
    print(f"{tweet} → {result['label']} ({round(result['score'], 3)})")
