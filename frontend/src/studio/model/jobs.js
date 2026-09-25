/**
 * Job domain model — status vocabulary, grouping and the synthetic records used
 * for optimistic submissions.
 *
 * Pure module: no Vue, no DOM, no network.
 */

/** Server statuses, plus `submitting` which only ever exists client-side. */
export const JOB_STATUS = {
  SUBMITTING: 'submitting',
  QUEUED: 'queued',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  INTERRUPTED: 'interrupted',
}

export const ACTIVE_STATUSES = [JOB_STATUS.SUBMITTING, JOB_STATUS.QUEUED, JOB_STATUS.RUNNING]
export const TERMINAL_STATUSES = [JOB_STATUS.COMPLETED, JOB_STATUS.FAILED, JOB_STATUS.INTERRUPTED]

/** Filter buckets used by the library's status segmented control. */
export const STATUS_FILTERS = {
  all: null,
  active: ACTIVE_STATUSES,
  completed: [JOB_STATUS.COMPLETED],
  failed: [JOB_STATUS.FAILED, JOB_STATUS.INTERRUPTED],
}

export const STATUS_LABELS = {
  [JOB_STATUS.SUBMITTING]: '提交中',
  [JOB_STATUS.QUEUED]: '等待',
  [JOB_STATUS.RUNNING]: '生成中',
  [JOB_STATUS.COMPLETED]: '完成',
  [JOB_STATUS.FAILED]: '失败',
  [JOB_STATUS.INTERRUPTED]: '已取消',
}

export function isActive(job) {
  return ACTIVE_STATUSES.includes(job?.status)
}

export function isTerminal(job) {
  return TERMINAL_STATUSES.includes(job?.status)
}

export function hasMedia(job) {
  return Boolean(job?.video_url || job?.image_url || job?.audio_url)
}

export function isPending(job) {
  return Boolean(job?.pending) || job?.status === JOB_STATUS.SUBMITTING
}

export function statusLabel(job) {
  if (job?.status === JOB_STATUS.RUNNING) {
    return `${Math.round((job.progress || 0) * 100)}%`
  }
  return STATUS_LABELS[job?.status] || job?.status || ''
}

/**
 * Build the placeholder record shown the instant a user submits, before the
 * server has assigned an id. It mirrors the server's JobResponse shape so every
 * selector and component can treat it identically.
 *
 * @param {string} token      Client-side identity, unique per submission.
 * @param {object} request    The generation request that is being sent.
 * @param {number} sessionNumber
 * @param {number} [createdAt] Epoch milliseconds.
 */
export function makePendingJob(token, request, sessionNumber, createdAt = Date.now()) {
  const timestamp = new Date(createdAt).toISOString()
  return {
    id: token,
    session_number: sessionNumber,
    status: JOB_STATUS.SUBMITTING,
    progress: 0,
    error: null,
    video_url: null,
    image_url: null,
    audio_url: null,
    hdr_exr_url: null,
    request,
    created_at: timestamp,
    updated_at: timestamp,
    generation_seconds: null,
    peak_vram_gb: null,
    pending: true,
  }
}

/**
 * Merge optimistic placeholders in front of server records.
 *
 * The runtime decides when a placeholder is retired (it owns the job count and
 * the TTL); this function simply renders whatever is currently pending. Keeping
 * it that way avoids a second source of truth for "has this job landed".
 */
export function mergePendingJobs(pendingEntries, serverJobs) {
  if (!pendingEntries.length) return serverJobs
  const placeholders = pendingEntries
    .map(entry => makePendingJob(entry.token, entry.request, entry.sessionNumber, entry.createdAt))
  return [...placeholders, ...serverJobs]
}

/** Queue position per job id, counting waiting jobs oldest-first. */
export function buildQueueIndex(jobs) {
  const waiting = jobs
    .filter(job => job.status === JOB_STATUS.QUEUED)
    .slice()
    .reverse()
  return new Map(waiting.map((job, index) => [job.id, index + 1]))
}
