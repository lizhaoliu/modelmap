import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    // served by the FastAPI app (modelmap serve), and shipped inside the wheel
    outDir: '../src/modelmap/web',
    emptyOutDir: true,
  },
  server: {
    proxy: { '/api': 'http://127.0.0.1:7860' },
  },
})
