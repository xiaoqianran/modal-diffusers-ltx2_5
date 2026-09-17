import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://127.0.0.1:8000'
  return {
    server: {
      host: '127.0.0.1',
      port: 5187,
      strictPort: true,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/outputs': { target, changeOrigin: true },
      },
    },
  }
})
