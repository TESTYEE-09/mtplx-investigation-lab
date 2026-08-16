import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/index.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error("MTPLX dashboard mount point #root is missing from index.html");
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
