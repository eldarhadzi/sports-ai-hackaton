import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: the React app runs on :5199 and proxies /api to the FastAPI server (api/server.py) on :8765.
// Ports are deliberately uncommon; override with WEB_PORT / API_PORT.
const API_PORT = process.env.API_PORT ?? "8765";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: Number(process.env.WEB_PORT ?? 5199),
    strictPort: true,
    proxy: { "/api": { target: `http://127.0.0.1:${API_PORT}`, changeOrigin: true } },
  },
  build: { chunkSizeWarningLimit: 1200 },
});
