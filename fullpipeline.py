import pandas as pd
from aspect_extraction import separate_clauses, classify_clauses_zero_shot
from sentimentanalysis import sentiment_of_clause
import csv
import time
import re

def clean_text(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    return text.strip()

def read_tweets(file_path):
    df = pd.read_csv(file_path)
    tweets = df['fullText'].tolist()
    return tweets

def process_tweet(tweet, topics, threshold=0.69):
    tweet = clean_text(tweet)
    clauses = separate_clauses(tweet)
    classified = classify_clauses_zero_shot(clauses, topics, threshold)
    
    results = []
    for clause, topic in classified:
        words = clause.split()
        if len(words) <= 2:
            continue  # nos gustan las clausulas grandes!

        sentiment, score = sentiment_of_clause(clause)
        results.append({
            "clause": clause,
            "topic": topic,
            "sentiment": sentiment,
            "sentiment_score": score
        })
    return results

# 3. processar tots els tweets
def process_all_tweets(tweets, topics):
    all_results = []
    total_start = time.time()
    
    for i, tweet in enumerate(tweets, start=1):
        start = time.time()
        print(f"\nprocessing tweet {i}/{len(tweets)}...")
        tweet_result = process_tweet(tweet, topics)
        elapsed = time.time() - start
        print(f"completed in {elapsed:.2f} segons.")
        
        all_results.append({
            "tweet": tweet,
            "analysis": tweet_result
        })

    total_elapsed = time.time() - total_start
    print(f"\nprocessing complete in {total_elapsed:.2f} seconds.")
    return all_results


# 4. executar el pipeline
if __name__ == "__main__":
    FILE_PATH = "scrapped_data/all_relevant_for_ABSA_annotation.csv"
    OUTPUT_CSV = "resultstoannotate.csv"
    TOPICS = ["precios", "transporte", "bebidas", "organización", "colas", "experiencia", "comida", "sonido", "baños", "seguridad", "personal", "ambiente", "pregunta", "información"]
    
    tweets = read_tweets(FILE_PATH)
    final_results = process_all_tweets(tweets, TOPICS)

    # Mostrar resultats
    for tweet_data in final_results:
        print(f"\nTWEET: {tweet_data['tweet']}")
        for analysis in tweet_data['analysis']:
            print(f"  - [{analysis['topic']}] ({analysis['sentiment']}, {analysis['sentiment_score']}) -> {analysis['clause']}")

    # Guardar resultats en CSV
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["tweet_id", "tweet", "clause", "topic", "sentiment", "sentiment_score"])
        writer.writeheader()
        for tweet_id, tweet_data in enumerate(final_results):
            tweet_text = tweet_data["tweet"]
            for analysis in tweet_data["analysis"]:
                writer.writerow({
                    "tweet_id": tweet_id,
                    "tweet": tweet_text,
                    "clause": analysis["clause"],
                    "topic": analysis["topic"],
                    "sentiment": analysis["sentiment"],
                    "sentiment_score": analysis["sentiment_score"]
                })

    print(f"\nresultats guardats a '{OUTPUT_CSV}'")
