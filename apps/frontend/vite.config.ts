import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const cacheDir = process.env.VITE_CACHE_DIR || 'node_modules/.vite';

export default defineConfig({
  plugins: [react()],
  cacheDir,
  server: {
    port: 5173,
    host: '0.0.0.0'
  }
});
