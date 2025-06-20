import matplotlib.pyplot as plt

def create_sentiment_chart(category, positives, negatives):
    """
    Creates and saves a PNG chart showing positive and negative percentages for a given category.

    Args:
        category (str): Name of the category (e.g., "Music Festival")
        positives (float): Percentage of positive sentiment (0-100)
        negatives (float): Percentage of negative sentiment (0-100)
    """

    # Check if percentages are valid
    if positives + negatives > 100:
        raise ValueError("Positives and negatives together cannot exceed 100%.")
    
    neutrals = 100 - (positives + negatives)

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positives, negatives, neutrals]
    colors = ['#4CAF50', '#F44336', '#9E9E9E']  # green, red, grey

    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops={'color':"w"}
    )

    # Add title
    plt.title(f"Sentiment Analysis for {category}", fontsize=16)

    # Keep the pie chart circular
    ax.axis('equal')

    # Save to 'sentiment-chart.png' in the same folder
    plt.savefig("sentiment-chart.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved successfully as 'sentiment-chart.png'.")

# Example usage:
create_sentiment_chart(category="Music Festival", positives=73, negatives=27)
