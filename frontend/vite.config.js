import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://127.0.0.1:8000'
  // VITE_ALLOWED_HOSTS: comma-separated extra hostnames for cloud dev tunnels
  // (Vite 7 rejects unknown Host headers by default). Defaults to allowing all
  // so the dev server works behind any tunnel/proxy without extra configuration.
  const allowedHosts = env.VITE_ALLOWED_HOSTS
    ? env.VITE_ALLOWED_HOSTS.split(',').map(value => value.trim()).filter(Boolean)
    : true
  return {
    server: {
      host: '0.0.0.0',
      port: 5187,
      strictPort: true,
      allowedHosts,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/outputs': { target, changeOrigin: true },
      },
    },
  }
})
