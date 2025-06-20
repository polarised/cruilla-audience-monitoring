import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Navbar from './components/Navigationbar';
import Footer from './components/Footer';
import PrivateRoute from './components/PrivateRoute';

import Home from './pages/Homepage';
import About from './pages/About';
import Data from './pages/Data';
import Readme from './pages/Readme';
import Login from './pages/Login';

function App() {
  return (
    <div className="App">
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/readme" element={<Readme />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/data"
            element={
              <PrivateRoute>
                <Data />
              </PrivateRoute>
            }
          />
        </Routes>
        <Footer />
      </Router>
    </div>
  );
}

export default App;
