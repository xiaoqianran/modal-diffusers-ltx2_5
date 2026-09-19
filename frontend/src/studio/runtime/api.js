/**
 * HTTP boundary.
 *
 * The only module that knows about URLs, verbs and response shapes. Everything
 * above it deals in plain values, which keeps the runtime free of fetch details
 * and makes the endpoints easy to stub in tests.
 *
 * No Vue, no DOM, no business rules.
 */

const JSON_HEADERS = { 'Content-Type': 'application/json' }

/**
 * Resolve a site-relative path against the current origin.
 *
 * The dev server proxies `/api` and `/outputs`, so every call uses a relative
 * path. Browsers resolve those against the document implicitly, but other
 * environments (tests, workers) require an absolute URL, so resolution happens
 * here rather than being assumed.
 */
function absolute(path) {
  if (/^https?:\/\//i.test(path)) return path
  const base = globalThis.window?.location?.href || 'http://localhost/'
  return new URL(path, base).href
}

/** Error carrying the server's `detail` so the UI can show something useful. */
export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function parse(response) {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return { detail: text.slice(0, 400) }
  }
}

async function send(path, { method = 'GET', body, signal, keepalive = false } = {}) {
  const init = { method, signal, keepalive }
  if (body !== undefined) {
    init.headers = JSON_HEADERS
    init.body = JSON.stringify(body)
  }
  const response = await fetch(absolute(path), init)
  const payload = await parse(response)
  if (!response.ok) {
    const detail = payload?.detail
    const message = typeof detail === 'string'
      ? detail
      : detail ? JSON.stringify(detail).slice(0, 300) : `${response.status} ${response.statusText}`
    throw new ApiError(message, response.status, detail)
  }
  return payload
}

export const api = {
  health: () => send('/api/health'),

  warm: () => send('/api/admin/warm', { method: 'POST' }),

  unload: ({ keepalive = false } = {}) => send('/api/admin/unload', { method: 'POST', keepalive }),

  loras: () => send('/api/loras'),

  createSession: () => send('/api/sessions', { method: 'POST' }),

  listJobs: (sessionNumber, limit = 100) => send(
    `/api/jobs?session_number=${encodeURIComponent(sessionNumber)}&limit=${limit}`
  ),

  getJob: id => send(`/api/jobs/${id}`),

  submitJob: body => send('/api/jobs', { method: 'POST', body }),

  interrupt: jobId => send('/api/interrupt', { method: 'POST', body: { job_id: jobId } }),

  deleteJob: id => send(`/api/jobs/${id}`, { method: 'DELETE' }),

  /** Multipart upload; the browser sets the boundary so no JSON header here. */
  async uploadAsset(file) {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch(absolute('/api/assets'), { method: 'POST', body: form })
    const payload = await parse(response)
    if (!response.ok) {
      throw new ApiError(typeof payload?.detail === 'string' ? payload.detail : '上传失败',
        response.status, payload?.detail)
    }
    return payload
  },

  /**
   * Fetch a rendered output as a Blob, used when re-uploading for Retake/Extend.
   *
   * The server returns site-relative URLs like `/outputs/x.mp4`, which `fetch`
   * cannot parse on its own in every environment, so they are resolved against
   * the current origin first.
   */
  async fetchOutput(url) {
    const response = await fetch(absolute(url))
    if (!response.ok) {
      throw new ApiError(`无法读取输出 (${response.status})`, response.status)
    }
    return response.blob()
  },
}
