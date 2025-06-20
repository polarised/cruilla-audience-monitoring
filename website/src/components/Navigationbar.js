import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import ReorderIcon from '@mui/icons-material/Reorder';
import Logo from '../assets/descarga.png';
import '../styles/Navbar.css';

function Navbar() {
  const [openLinks, setOpenLinks] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(localStorage.getItem("auth") === "true");
  const navigate = useNavigate();

  useEffect(() => {
    function handleStorageChange() {
      setIsAuthenticated(localStorage.getItem("auth") === "true");
    }

    window.addEventListener("storageChanged", handleStorageChange);

    // Limpieza
    return () => {
      window.removeEventListener("storageChanged", handleStorageChange);
    };
  }, []);

  const toggleNavbar = () => {
    setOpenLinks(!openLinks);
  };

  const handleLogout = () => {
    localStorage.removeItem("auth");
    setIsAuthenticated(false);
    window.dispatchEvent(new Event("storageChanged"));  // Para avisar a Navbar y otros
    navigate("/login");
  };

  return (
    <div className="navbar">
      <div className="leftSide" id={openLinks ? "open" : "close"}>
        <img src={Logo} alt="logo" />
        <div className="hiddenLinks">
          <Link to="/">Home</Link>
          <Link to="/data">Data</Link>
          <Link to="/readme">Readme</Link>
          <Link to="/about">About</Link>
          {isAuthenticated ? (
            <button onClick={handleLogout}>Logout</button>
          ) : (
            <Link to="/login">Login</Link>
          )}
        </div>
      </div>
      <div className="rightSide">
        <Link to="/">Home</Link>
        <Link to="/data">Data</Link>
        <Link to="/readme">Readme</Link>
        <Link to="/about">About</Link>
        {isAuthenticated ? (
          <button onClick={handleLogout}>Logout</button>
        ) : (
          <Link to="/login">Login</Link>
        )}
        <button onClick={toggleNavbar}>
          <ReorderIcon />
        </button>
      </div>
    </div>
  );
}

export default Navbar;
