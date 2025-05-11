from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, pipeline

# Función para análisis de sentimiento de una sola cláusula
def sentiment_of_clause(clause):
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = sentiment_pipe(clause)[0]
    return result['label'], round(result['score'], 3)
