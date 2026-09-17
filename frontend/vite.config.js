import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'https://weiranzhiqian--ltx25-nvfp4-ltx25server-web.modal.run'
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
