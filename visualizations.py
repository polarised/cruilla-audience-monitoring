import plotly.express as px
import pandas as pd

# Initialize your dataframe
csv_path = r"subphrases_with_topics.csv"
assigned_df = pd.read_csv(csv_path)

print("Loaded assigned_df successfully!")

def visualize_table(assigned_df, save_path="subphrases_topics_plot.html"):
    fig = px.scatter(
        assigned_df,
        x="Assigned_Topic_ID",
        y=[1]*len(assigned_df),
        text="subphrase",
        hover_data=["Assigned_Topic_Name", "original_tweet_text"],
        color="Assigned_Topic_ID",
        size_max=60
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        title="🔎 Subphrases grouped by Topic",
        xaxis_title="Topic ID",
        yaxis_visible=False,
        showlegend=False,
        height=600
    )
    # Save plot to HTML
    fig.write_html(save_path)
    print(f"\n✅ Plot saved successfully as '{save_path}'")

# Call the function
visualize_table(assigned_df)
