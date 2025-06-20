//import React from "react";
import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import Logo from "../assets/descarga.png";
import LogoCruilla from "../assets/Logo_Cruilla2019_Negre.png";
import LogoUAB from "../assets/UAB-logotip_0.png"; 
import colab from "../assets/colaboracion.png";
import "../styles/Homepage.css";

const festivalDate = new Date("2025-07-09T00:00:00");


function Home() {
  const [timeLeft, setTimeLeft] = useState({});

  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date();
      const difference = festivalDate - now;

      const days = Math.floor(difference / (1000 * 60 * 60 * 24));
      const hours = Math.floor((difference / (1000 * 60 * 60)) % 24);
      const minutes = Math.floor((difference / 1000 / 60) % 60);
      const seconds = Math.floor((difference / 1000) % 60);

      setTimeLeft({ days, hours, minutes, seconds });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="home">
      <img src={Logo} alt="Cruïlla Logo" className="logo" />
      <div className="headerContainer">
        <h1>Cruïlla Sentiment Dashboard</h1>
        <p>Discover real-time insights from festival-goers' tweets</p>

        {/* COUNTDOWN */}
        <div className="countdown">
          <h2>Festival starts in:</h2>
          <div className="countdown-timer">
            <div className="time-box">
              <span className="number">{timeLeft.days}</span>
              <span className="label">Days</span>
            </div>
            <div className="time-box">
              <span className="number">{timeLeft.hours}</span>
              <span className="label">Hours</span>
            </div>
            <div className="time-box">
              <span className="number">{timeLeft.minutes}</span>
              <span className="label">Minutes</span>
            </div>
            <div className="time-box">
              <span className="number">{timeLeft.seconds}</span>
              <span className="label">Seconds</span>
            </div>
          </div>
        </div>


        <Link to="/Data">
          <button>Explore Dashboard</button>
        </Link>

      <div className="image-row">
        <img src={LogoCruilla} alt="Cruïlla" className="partner-logo" />
        <img src={colab} alt="Colaboración" className="colab" />
        <img src={LogoUAB} alt="UAB Logo" className="partner-logo" />
      </div>
      </div>
    </div>
  );
}

export default Home;