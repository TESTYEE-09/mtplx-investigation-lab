import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";

const MTPLX_BACKEND = process.env.MTPLX_BACKEND_URL ?? "http://127.0.0.1:8000";

const proxyTarget = {
  target: MTPLX_BACKEND,
  changeOrigin: true,
  secure: false,
  ws: true,
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  // Vite is invoked from `dashboard/`; bundle into the python package
  // so `python -m build` ships the SPA inside the wheel.
  build: {
    outDir: path.resolve(__dirname, "../mtplx/dashboard/_static"),
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
  // When served through `mtplx serve`, the SPA lives under `/dashboard/`.
  // `base` keeps asset URLs relative-friendly so the same bundle works at
  // either the dev origin (`/`) or the mounted prefix (`/dashboard/`).
  base: "./",
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      "/v1": proxyTarget,
      "/health": proxyTarget,
      "/metrics": proxyTarget,
      "/admin": proxyTarget,
    },
  },
});
