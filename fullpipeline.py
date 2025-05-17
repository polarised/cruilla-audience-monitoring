import pandas as pd
from aspect_extraction import separate_clauses, classify_clauses_zero_shot
from sentimentanalysis import sentiment_of_clause
import csv


# 1. Llegir el CSV amb els tweets
def read_tweets(file_path):
    df = pd.read_csv(file_path)
    return df['Post Text'].tolist()

# 2. Processar un sol tweet
def process_tweet(tweet, topics, threshold=0.4):
    clauses = separate_clauses(tweet)
    classified = classify_clauses_zero_shot(clauses, topics, threshold)
    
    results = []
    for clause, topic in classified:
        sentiment, score = sentiment_of_clause(clause)
        results.append({
            "clause": clause,
            "topic": topic,
            "sentiment": sentiment,
            "sentiment_score": score
        })
    return results

# 3. Processar tots els tweets
def process_all_tweets(tweets, topics):
    all_results = []
    for tweet in tweets:
        tweet_result = process_tweet(tweet, topics)
        all_results.append({
            "tweet": tweet,
            "analysis": tweet_result
        })
    return all_results


# 4. Executar el pipeline
if __name__ == "__main__":
    FILE_PATH = "scrapped_data/all_comments.csv"
    OUTPUT_CSV = "results.csv"
    TOPICS = ["precios", "bebidas", "organización", "colas", "experiencia", "comida", "sonido", "baños", "seguridad", "personal", "ambiente"]
    
    tweets = read_tweets(FILE_PATH)
    final_results = process_all_tweets(tweets, TOPICS)

    # Mostrar resultats
    for tweet_data in final_results:
        print(f"\nTWEET: {tweet_data['tweet']}")
        for analysis in tweet_data['analysis']:
            print(f"  - [{analysis['topic']}] ({analysis['sentiment']}, {analysis['sentiment_score']}) -> {analysis['clause']}")

    # Guardar resultats en CSV
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["tweet", "clause", "topic", "sentiment", "sentiment_score"])
        writer.writeheader()
        for tweet_data in final_results:
            tweet_text = tweet_data["tweet"]
            for analysis in tweet_data["analysis"]:
                writer.writerow({
                    "tweet": tweet_text,
                    "clause": analysis["clause"],
                    "topic": analysis["topic"],
                    "sentiment": analysis["sentiment"],
                    "sentiment_score": analysis["sentiment_score"]
                })
    
    print(f"\nresultats guardats a '{OUTPUT_CSV}'")
