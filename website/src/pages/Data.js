import { useState } from "react";
import "../styles/Data.css";

function Data() {
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [showFilters, setShowFilters] = useState(false);
  const [expandedPanel, setExpandedPanel] = useState(null);

  const categories = [
    "All",
    "Precios",
    "Bebidas",
    "Organización",
    "Colas",
    "Experiencia",
    "Comida",
    "Sonido",
    "Baños",
    "Seguridad",
    "Personal",
    "Ambiente",
    "Transporte",
    "Información",
    "Pregunta",
  ];

  const grafanaBase =
    "http://localhost:3000/d-solo/26b52e88-1766-449c-b33f-7d102c5da932/topicsall";
  const from = "now-6h";
  const to = "now";
  const panelIds = {
    sentiment: 1,
    positive: 2,
    negative: 3,
  };

  const getPanelUrl = (panelId) => {
    const topicValue =
      selectedCategory.toLowerCase() === "all"
        ? "%"
        : selectedCategory.toLowerCase();
    return `${grafanaBase}?orgId=1&from=${from}&to=${to}&panelId=${panelId}&theme=light&var-topic=${encodeURIComponent(
      topicValue
    )}`;
  };

  const publicDashboards = {
  all: "http://localhost:3000/public-dashboards/810dc73de49344c18cd9990f3b1e91c6",
  bebidas: "http://localhost:3000/public-dashboards/5052d13792b243e28ce9d9575af5aa85",
  comida: "http://localhost:3000/public-dashboards/bff7822dc22247adbb11458eafa5edb7",
  sonido: "http://localhost:3000/public-dashboards/0bb8ddc1416f4c9da8ef52db5d17e94f",
  experiencia: "http://localhost:3000/public-dashboards/6883bdade983464e85a2a1601b52e597",
  información: "http://localhost:3000/public-dashboards/5bb072274b5040fcb1b398618e14385a",
  organización: "http://localhost:3000/public-dashboards/ede2523e81854965a74c389d8110e31d",
  personal: "http://localhost:3000/public-dashboards/7ed08bcfd10f416498090ddb78d4e821",
  precios: "http://localhost:3000/public-dashboards/6766853f3e26427fad6b4a2b947f9de7",
  pregunta: "http://localhost:3000/public-dashboards/c9d760a46ce842fd8ef1fc6e215b376f",
  transporte: "http://localhost:3000/public-dashboards/d3c19ddd0fac4b39ba858896066b984d",
  //faltan: colas, baños, seguridad, ambiente:
  
  // Añade más según tus dashboards públicos
  };

  const getGeneralDashboardUrl = () => {
    const category = selectedCategory.toLowerCase();
    return publicDashboards[category] || publicDashboards["all"];
  };


  return (
    <div className="data">
      <h1 className="dataTitle">Cruïlla Festival Dashboard</h1>

      <div className="dataContent">
        <div className="filter-section">
          <button
            className="filter-button"
            onClick={() => setShowFilters(!showFilters)}
          >
            {showFilters ? "Hide Filters" : "Filter by Category"}
          </button>

          {showFilters && (
            <div className="category-filters">
              {categories.map((category) => (
                <button
                  key={category}
                  className={`category-button ${
                    selectedCategory === category ? "active" : ""
                  }`}
                  onClick={() => {
                    setSelectedCategory(category);
                    setShowFilters(false);
                  }}
                >
                  {category}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="selected-category">
          Currently showing: <span>{selectedCategory}</span>
        </div>

        <div className="charts-grid">
          {/* Sentiment Summary */}
          <div className="chart-card">
            <h2>Sentiment Summary</h2>
            <iframe
              src={getPanelUrl(panelIds.sentiment)}
              width="100%"
              height="200"
              frameBorder="0"
              title="Sentiment Summary"
            />
            <button
              className="expand-button small"
              onClick={() => setExpandedPanel("sentiment")}
            >
              Extend
            </button>
            <p>Positive vs Negative sentiment ratio</p>
          </div>

          {/* Positive Terms */}
          <div className="chart-card">
            <h2>Top Positive Terms</h2>
            <iframe
              src={getPanelUrl(panelIds.positive)}
              width="100%"
              height="200"
              frameBorder="0"
              title="Top Positive Terms"
            />
            <button
              className="expand-button small"
              onClick={() => setExpandedPanel("positive")}
            >
              Extend
            </button>
            <p>Most repeated positive terms</p>
          </div>

          {/* Negative Terms */}
          <div className="chart-card">
            <h2>Top Negative Terms</h2>
            <iframe
              src={getPanelUrl(panelIds.negative)}
              width="100%"
              height="200"
              frameBorder="0"
              title="Top Negative Terms"
            />
            <button
              className="expand-button small"
              onClick={() => setExpandedPanel("negative")}
            >
              Extend
            </button>
            <p>Most repeated negative terms</p>
          </div>
        </div>

        <div className="full-dashboard">
          <h2>General Dashboard</h2>
          <iframe
            src={getGeneralDashboardUrl()}
            width="100%"
            height="900"
            frameBorder="0"
            title="Filtered Dashboard"
          />
          <button
            className="expand-button"
            onClick={() => setExpandedPanel("full")}
          >
            View full screen
          </button>
        </div>

      </div>

      {/* Panel expandido */}
      {expandedPanel && (
        <div
          className="modal-overlay"
          onClick={() => setExpandedPanel(null)}
        >
          <div
            className={`modal-panel ${
              expandedPanel === "full" ? "fullscreen" : "popup"
            }`}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="close-modal"
              onClick={() => setExpandedPanel(null)}
            >
              ✕ Close
            </button>
            <iframe
              title="Expanded Dashboard"
              src={
                expandedPanel === "full"
                  ? getGeneralDashboardUrl()
                  : getPanelUrl(panelIds[expandedPanel])
              }
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default Data;
