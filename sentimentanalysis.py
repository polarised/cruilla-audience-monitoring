import re
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, pipeline

# Load model and tokenizer only once
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def sentiment_of_clause(clause):
    cleaned_clause = re.sub(r'@\w+', '', clause).strip()
    result = sentiment_pipe(cleaned_clause)[0]
    return result['label'], round(result['score'], 3)
