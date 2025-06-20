import React from "react";
import BannerImage from "../assets/descarga.png";
import CarteleraImage from "../assets/cartelera.jpg";
import "../styles/About.css";

function About() {
  return (
    <div className="about">
      <div
        className="aboutTop"
        style={{ backgroundImage: `url(${BannerImage})` }}
      ></div>
      <div className="aboutBottom">
        <h1> ABOUT THE PROJECT </h1>
        <p>
          The Cruïlla Sentiment Dashboard is an interactive tool designed to
          explore real-time audience feedback from social media. By analyzing
          tweets related to the Cruïlla Festival, we categorize opinions,
          extract insights, and visualize engagement trends using advanced
          Natural Language Processing models like BERT.
        </p>
        <p>
          This platform allows festival organizers and researchers to understand
          the crowd’s sentiment during the event, identify highlights, and
          address issues as they unfold. With our dashboard, you can filter
          tweets by category, view sentiment evolution, and dive into the data
          behind the festival experience.
        </p>

        <h2> ABOUT THE CRUÏLLA FESTIVAL 2025 </h2>
        <div className="cartelera">
          <img src={CarteleraImage} alt="Cruïlla 2025 Lineup" />
        </div>
      </div>
    </div>
  );
}

export default About;