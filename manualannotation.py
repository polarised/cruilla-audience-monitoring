import pandas as pd

file_path = "resultstoannotate.csv"  # Path to your file
output_path = "results_annotated.csv"

df = pd.read_csv(file_path)

# Add new column if needed
if "manual_sentiment" not in df.columns:
    df["manual_sentiment"] = ""

sentiment_map = {"0": "negative", "1": "neutral", "2": "positive"}

print("Begin annotation. Type 0=neg, 1=neu, 2=pos, or just Enter to skip.\n")

for idx, row in df.iterrows():
    if pd.notna(df.at[idx, "manual_sentiment"]) and df.at[idx, "manual_sentiment"] != "":
        continue  # Already annotated

    print(f"\nTweet ID {row['tweet_id']}")
    print(f"Full tweet: {row['tweet']}")
    print(f"Clause: {row['clause']}")

    while True:
        label = input("Sentiment (0=neg, 1=neu, 2=pos, Enter=skip): ").strip()
        if label == "":
            break
        if label in sentiment_map:
            df.at[idx, "manual_sentiment"] = sentiment_map[label]
            break
        print("invalid")

    df.to_csv(output_path, index=False)
    print("progress saved")

print("\nannotated file saved to:", output_path)
