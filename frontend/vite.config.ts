import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// The orchestrator's FastAPI app serves the built dashboard from src/static.
// Vite output goes there so a docker COPY can keep it co-located with the
// Python app, and the dev server proxies /api calls to the orchestrator
// process listening on port 9000 (see src/main.py uvicorn.run).
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../src/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:9000',
    },
  },
})
