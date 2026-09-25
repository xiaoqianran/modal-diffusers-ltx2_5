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
const API_TARGET = String(import.meta.env?.VITE_API_TARGET || '').replace(/\/+$/, '')
const MEDIA_URL_KEYS = new Set(['video_url', 'image_url', 'audio_url', 'hdr_exr_url'])

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
  if (API_TARGET && /^\/(?:api|outputs)\//.test(path)) {
    return new URL(path, `${API_TARGET}/`).href
  }
  const base = globalThis.window?.location?.href || 'http://localhost/'
  return new URL(path, base).href
}

function resolveMediaUrls(value) {
  if (Array.isArray(value)) return value.map(resolveMediaUrls)
  if (!value || typeof value !== 'object') return value
  const resolved = { ...value }
  for (const key of MEDIA_URL_KEYS) {
    if (typeof resolved[key] === 'string' && resolved[key]) {
      resolved[key] = absolute(resolved[key])
    }
  }
  const primaryMedia = resolved.video_url || resolved.image_url || resolved.audio_url
  if (primaryMedia) {
    const url = new URL(primaryMedia)
    url.searchParams.set('download', '1')
    resolved.download_url = url.href
  }
  if (resolved.hdr_exr_url) {
    const url = new URL(resolved.hdr_exr_url)
    url.searchParams.set('download', '1')
    resolved.hdr_exr_download_url = url.href
  }
  return resolved
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
    return resolveMediaUrls(JSON.parse(text))
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

async function putBinary(url, body, headers = {}) {
  const response = await fetch(absolute(url), { method: 'PUT', body, headers })
  if (!response.ok) {
    const payload = await parse(response)
    throw new ApiError(
      typeof payload?.detail === 'string' ? payload.detail : `上传失败 (${response.status})`,
      response.status,
      payload?.detail
    )
  }
  return response
}

async function putBinaryWithRetry(url, body, headers = {}, attempts = 3) {
  let lastError = null
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await putBinary(url, body, headers)
    } catch (error) {
      lastError = error
      const retryable = !(error instanceof ApiError) || error.status >= 500
      if (!retryable || attempt + 1 >= attempts) break
      await new Promise(resolve => setTimeout(resolve, 150 * (2 ** attempt)))
    }
  }
  throw lastError
}

async function uploadMultipart(file, plan, concurrency = null) {
  const queue = plan.parts.slice()
  const completed = []
  const parallelism = Math.max(1, Math.min(16, Number(concurrency || plan.concurrency || 4)))

  async function worker() {
    while (queue.length) {
      const part = queue.shift()
      const start = (part.part_number - 1) * plan.part_size
      const end = Math.min(start + plan.part_size, file.size)
      const response = await putBinaryWithRetry(part.url, file.slice(start, end))
      const etag = response.headers.get('ETag') || response.headers.get('etag')
      if (!etag) {
        throw new ApiError(
          '对象存储没有暴露 ETag；请检查 S3/MinIO CORS 的 ExposeHeaders',
          502
        )
      }
      completed.push({ part_number: part.part_number, etag })
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(parallelism, queue.length) }, () => worker())
  )
  return completed.sort((a, b) => a.part_number - b.part_number)
}

async function finalizeUpload(assetId, body) {
  let lastError = null
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      return await send(`/api/assets/${encodeURIComponent(assetId)}/complete`, {
        method: 'POST',
        body,
      })
    } catch (error) {
      lastError = error
      if (error instanceof ApiError && error.status < 500) throw error
    }
  }
  throw lastError
}

export const api = {
  health: () => send('/api/health'),

  warm: () => send('/api/admin/warm', { method: 'POST' }),

  unload: ({ keepalive = false } = {}) => send('/api/admin/unload', { method: 'POST', keepalive }),

  loras: () => send('/api/loras'),

  createSession: () => send('/api/sessions', { method: 'POST' }),

  listJobs: (sessionNumber, limit = 100) => send(
    `/api/jobs/status?session_number=${encodeURIComponent(sessionNumber)}&limit=${limit}`
  ),

  listGeneratedAssets: (sessionNumber, mediaKind = 'image', page = 1, pageSize = 24) => send(
    `/api/assets/generated?session_number=${encodeURIComponent(sessionNumber)}&media_kind=${encodeURIComponent(mediaKind)}&page=${page}&page_size=${pageSize}`
  ),

  getJob: id => send(`/api/jobs/${id}`),

  submitJob: body => send('/api/jobs', { method: 'POST', body }),

  interrupt: jobId => send('/api/interrupt', { method: 'POST', body: { job_id: jobId } }),

  deleteJob: id => send(`/api/jobs/${id}`, { method: 'DELETE' }),

  /**
   * Unified media upload.
   *
   * Volume mode returns a site-local proxy PUT. S3 mode returns either one
   * presigned PUT or a set of presigned multipart URLs. In object-storage mode
   * media bytes never pass through the local FastAPI process.
   */
  async uploadAsset(file) {
    let plan = await send('/api/assets/prepare', {
      method: 'POST',
      body: {
        filename: file.name,
        size: file.size,
        content_type: file.type || 'application/octet-stream',
      },
    })

    async function abortPlan(currentPlan) {
      try {
        await send(`/api/assets/${encodeURIComponent(currentPlan.asset_id)}/upload`, {
          method: 'DELETE',
        })
      } catch {
        // Best effort: compatible object stores can also expire incomplete multipart uploads by lifecycle.
      }
    }

    async function transfer(currentPlan) {
      const uploadStarted = performance.now()
      let parts = []
      if (currentPlan.mode === 'multipart') {
        parts = await uploadMultipart(file, currentPlan)
      } else {
        await putBinaryWithRetry(currentPlan.url, file, currentPlan.headers || {})
      }
      return {
        parts,
        clientUploadSeconds: (performance.now() - uploadStarted) / 1000,
      }
    }

    let transferred = false
    try {
      let result
      try {
        result = await transfer(plan)
      } catch (primaryError) {
        try {
          const fallbackPlan = await send(`/api/assets/${encodeURIComponent(plan.asset_id)}/fallback`, {
            method: 'POST',
          })
          if (!fallbackPlan || !fallbackPlan.mode) throw new Error('Fallback upload plan is unavailable')
          plan = fallbackPlan
        } catch {
          await abortPlan(plan)
          throw primaryError
        }
        result = await transfer(plan)
      }

      transferred = true
      return await finalizeUpload(plan.asset_id, {
        parts: result.parts,
        client_upload_seconds: result.clientUploadSeconds,
      })
    } catch (error) {
      if (!transferred) await abortPlan(plan)
      throw error
    }
  },

  reuseOutput: jobId => send(`/api/assets/from-job/${encodeURIComponent(jobId)}`, { method: 'POST' }),

}
