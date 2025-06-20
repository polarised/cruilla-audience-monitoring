import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Login.css";

function Login() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const PASSWORD = "cruilla";

  const handleLogin = () => {
    if (password === PASSWORD) {
      localStorage.setItem("auth", "true");
      window.dispatchEvent(new Event("storageChanged"));
      navigate("/data");
    } else {
      setError("Incorrect Password");
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">
        <h2>Data Access</h2>
        <p>Enter the password to access the panel</p>
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {error && <span className="error">{error}</span>}
        <button onClick={handleLogin}>Entrar</button>
      </div>
    </div>
  );
}

export default Login;
