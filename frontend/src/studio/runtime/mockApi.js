/**
 * Browser-only mock API for frontend development.
 *
 * Enabled with VITE_MOCK_API=1. It performs no HTTP requests, starts no Python
 * service, and never touches Modal or CUDA. The shape mirrors runtime/api.js.
 */

const SESSION_NUMBER = 260920

function svgData(title, subtitle, accent = '#b8ff72') {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="640" viewBox="0 0 1024 640">
    <defs>
      <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop stop-color="#10120f"/>
        <stop offset="1" stop-color="#202b18"/>
      </linearGradient>
    </defs>
    <rect width="1024" height="640" fill="url(#g)"/>
    <circle cx="790" cy="145" r="180" fill="${accent}" opacity=".14"/>
    <circle cx="160" cy="520" r="260" fill="${accent}" opacity=".08"/>
    <text x="64" y="500" fill="#f2f5ef" font-family="system-ui,sans-serif" font-size="46" font-weight="650">${title}</text>
    <text x="66" y="548" fill="#a9b0a4" font-family="system-ui,sans-serif" font-size="22">${subtitle}</text>
  </svg>`
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

const now = Date.now()
const iso = offsetMinutes => new Date(now - offsetMinutes * 60_000).toISOString()

function request(mode, prompt, overrides = {}) {
  return {
    mode,
    prompt,
    negative_prompt: 'blurry, jittery, distorted',
    width: 768,
    height: 512,
    num_frames: mode === 't2i' || mode === 'ref2i' ? 49 : 121,
    fps: 24,
    steps: 8,
    guidance_scale: 3,
    seed: 42,
    upscale: true,
    upscale_method: 'latent',
    decoder: 'vae',
    ...overrides,
  }
}

let warm = true
let jobs = [
  {
    id: 'mock-running-001',
    session_number: SESSION_NUMBER,
    status: 'running',
    progress: 0.64,
    error: null,
    video_url: null,
    image_url: null,
    request: request('t2av', 'A quiet cinematic garden after rain, soft morning light'),
    created_at: iso(1),
    updated_at: iso(0),
    generation_seconds: null,
    peak_vram_gb: null,
  },
  {
    id: 'mock-queued-002',
    session_number: SESSION_NUMBER,
    status: 'queued',
    progress: 0,
    error: null,
    video_url: null,
    image_url: null,
    request: request('i2v', 'Slow camera push through a glass greenhouse at sunrise'),
    created_at: iso(3),
    updated_at: iso(3),
    generation_seconds: null,
    peak_vram_gb: null,
  },
  {
    id: 'mock-image-003',
    session_number: SESSION_NUMBER,
    status: 'completed',
    progress: 1,
    error: null,
    video_url: null,
    image_url: svgData('Botanical Study', 'T2I · mock output · no backend'),
    request: request('t2i', 'Minimal botanical still life on dark stone, editorial lighting'),
    created_at: iso(8),
    updated_at: iso(7),
    generation_seconds: 3.51,
    peak_vram_gb: 46.84,
  },
  {
    id: 'mock-image-004',
    session_number: SESSION_NUMBER,
    status: 'completed',
    progress: 1,
    error: null,
    video_url: null,
    image_url: svgData('Studio Reference', 'REF2I · mock output · browser only', '#8dc7ff'),
    request: request('ref2i', 'Architectural product shot, restrained materials, soft shadows', {
      upscale: false,
      num_frames: 49,
    }),
    created_at: iso(16),
    updated_at: iso(15),
    generation_seconds: 4.08,
    peak_vram_gb: 39.7,
  },
  {
    id: 'mock-failed-005',
    session_number: SESSION_NUMBER,
    status: 'failed',
    progress: 0.37,
    error: 'Mock failure: input validation example',
    video_url: null,
    image_url: null,
    request: request('a2v', 'Portrait performance driven by a short vocal clip'),
    created_at: iso(25),
    updated_at: iso(24),
    generation_seconds: null,
    peak_vram_gb: null,
  },
]

function clone(value) {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value))
}

function nextId() {
  return `mock-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

function assetKind(file) {
  const type = file?.type || ''
  if (type.startsWith('image/')) return 'image'
  if (type.startsWith('video/')) return 'video'
  if (type.startsWith('audio/')) return 'audio'
  return 'unknown'
}

export const mockApi = {
  async health() {
    return {
      ok: true,
      service: 'ltx25-browser-mock',
      transformer_precision: 'mock',
      warmup: {
        state: warm ? 'ready' : 'cold',
        gpu: warm ? 'MOCK' : null,
        engines: warm ? ['ltx', 'qwen'] : [],
      },
    }
  },

  async warm() {
    warm = true
    return { ok: true, mock: true }
  },

  async unload() {
    warm = false
    return { ok: true, mock: true }
  },

  async loras() {
    return [
      { id: 'mock-cinematic', name: 'Mock Cinematic', filename: 'mock-cinematic.safetensors' },
      { id: 'mock-product', name: 'Mock Product', filename: 'mock-product.safetensors' },
    ]
  },

  async createSession() {
    return { session_number: SESSION_NUMBER }
  },

  async listJobs(sessionNumber) {
    return clone(jobs.filter(job => Number(job.session_number) === Number(sessionNumber)))
  },

  async getJob(id) {
    const job = jobs.find(item => item.id === id)
    if (!job) throw new Error('Mock job not found')
    return clone(job)
  },

  async submitJob(body) {
    const id = nextId()
    const timestamp = new Date().toISOString()
    jobs.unshift({
      id,
      session_number: Number(body.session_number || SESSION_NUMBER),
      status: 'queued',
      progress: 0,
      error: null,
      video_url: null,
      image_url: null,
      request: clone(body),
      created_at: timestamp,
      updated_at: timestamp,
      generation_seconds: null,
      peak_vram_gb: null,
    })
    return { id }
  },

  async interrupt(jobId) {
    const job = jobs.find(item => item.id === jobId)
    if (job) {
      job.status = 'interrupted'
      job.progress = 0
      job.updated_at = new Date().toISOString()
    }
    return { ok: true }
  },

  async deleteJob(id) {
    jobs = jobs.filter(job => job.id !== id)
    return { ok: true }
  },

  async uploadAsset(file) {
    return {
      id: nextId(),
      filename: file.name,
      kind: assetKind(file),
      size: file.size,
      url: URL.createObjectURL(file),
    }
  },

  async reuseOutput(jobId) {
    const job = jobs.find(item => item.id === jobId)
    if (!job?.video_url && !job?.image_url && !job?.audio_url) throw new Error('Mock job has no reusable output')
    const kind = job.video_url ? 'video' : job.image_url ? 'image' : 'audio'
    const extension = kind === 'video' ? 'mp4' : kind === 'image' ? 'png' : 'wav'
    return {
      id: nextId(),
      filename: `${jobId}.${extension}`,
      kind,
      size: 0,
      url: job.video_url || job.image_url || job.audio_url,
    }
  },
}
