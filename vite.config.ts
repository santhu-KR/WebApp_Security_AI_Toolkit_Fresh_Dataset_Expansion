import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    // Convenience for a local-only demo. This embeds the key in browser assets.
    // Never use this approach for a public production deployment.
    define: { "process.env.GEMINI_API_KEY": JSON.stringify(env.GEMINI_API_KEY || "") }
  };
});