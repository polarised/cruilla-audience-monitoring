import React from "react";
import "../styles/Readme.css";

function Readme() {
  return (
    <div className="readme">
      <h1 className="readmeTitle">Prototype Definition</h1>
      <div className="readmeContent">
        <h2> Overview</h2>
        <p>
          This section will serve the purpose of informing about the models and processes involved in the analysis. The project aims to understand 
          the sentiment of festival-goers by analyzing tweets related to the Cruïlla Festival. 
          The system performs data preprocessing, classification using trained AI models, and visualization 
          through an interactive dashboard. This enables stakeholders to monitor public opinion in near real-time and derive actionable insights.

        </p>

        <h2> Data</h2>
        <p>The dataset is composed of tweets collected during the Cruïlla Festival. These tweets are first cleaned by removing noise such as emojis, mentions, URLs, and hashtags. 
            Language filtering ensures only relevant text (Spanish and Catalan) is retained. The text is then tokenized, lemmatized, 
            and vectorized using TF-IDF to prepare it for sentiment analysis. Sentiments are labeled as positive, neutral, or negative, 
            and stored in a PostgreSQL database for further processing and visualization.
        </p>
        
        <h2> Models used</h2>
        <ul>
          <li>Here we will explain the models and how the make decisions.</li>
          <li>...</li>
        </ul>

        <h2> Website </h2>
        <p>
          The website acts as a user interface for accessing and exploring the sentiment data. 
          It is built using React and fetches visualizations from Grafana embedded in the frontend. 
          Users can filter tweets by topic (e.g., food, sound, security) and view summary statistics such as positive vs. 
          negative sentiment ratios, most common positive/negative terms, and overall sentiment trends. 
          The goal is to provide festival organizers with an intuitive tool for understanding attendee feedback.
        </p>

        <h2> Future Features</h2>
        <ul>
          <li>Live tweet scraping & classification: Automate the tweet collection process using Twitter’s API to continuously retrieve and classify new tweets in real time.</li>
          <li>Mobile optimization: Make the dashboard accessible and fully responsive for mobile devices, allowing on-the-go monitoring by event staff.</li>
        </ul>
      </div>
    </div>
  );
}

export default Readme;