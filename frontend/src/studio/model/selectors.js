/**
 * Selectors — derive every view of a job list from one source of truth.
 *
 * The whole studio is a single generation session, so Stage, Takes, Queue and
 * Library are all projections of the same array. Nothing here mutates its input
 * or caches a second copy: callers are expected to wrap these in a computed.
 *
 * Pure module: no Vue, no DOM, no network.
 */

import {
  ACTIVE_STATUSES,
  JOB_STATUS,
  STATUS_FILTERS,
  hasMedia,
  isActive,
  isPending,
  statusLabel,
} from './jobs.js'
import { getModeCapabilities, modeTag } from './modes.js'

/** The job the stage should show: an explicit pick, else running, else newest done. */
export function selectStageJob(jobs, selectedId = null) {
  if (!jobs.length) return null
  if (selectedId) {
    const pinned = jobs.find(job => job.id === selectedId)
    if (pinned) return pinned
  }
  const running = jobs.find(job => job.status === JOB_STATUS.RUNNING)
  if (running) return running
  const waiting = jobs.filter(job => job.status === JOB_STATUS.QUEUED)
  if (waiting.length) return waiting[waiting.length - 1]
  const done = jobs.find(hasMedia)
  return done || jobs[0]
}

/**
 * Takes are rendered results only, newest first (the API returns newest first,
 * so no reordering is needed). Active jobs deliberately stay in the queue so the
 * two views never duplicate each other.
 */
export function selectTakes(jobs) {
  return jobs.filter(hasMedia)
}

/** Jobs the queue panel owns: one running card plus a compact waiting list. */
export function selectQueue(jobs) {
  return {
    running: jobs.filter(job => job.status === JOB_STATUS.RUNNING),
    upcoming: [
      ...jobs.filter(job => job.status === JOB_STATUS.QUEUED).slice().reverse(),
      ...jobs.filter(isPending),
    ],
  }
}

export function selectCounts(jobs) {
  const count = statuses => jobs.filter(job => statuses.includes(job.status)).length
  return {
    all: jobs.length,
    active: count(ACTIVE_STATUSES),
    completed: count([JOB_STATUS.COMPLETED]),
    failed: count(STATUS_FILTERS.failed),
  }
}

/** Filter options for the library, derived from what actually exists. */
export function selectAvailableModes(jobs) {
  return [...new Set(jobs.map(job => job.request?.mode).filter(Boolean))].sort()
}

/** Apply the library's status / mode / text filters. */
export function filterJobs(jobs, filter = {}) {
  const { status = 'all', mode = 'all', query = '' } = filter
  const allowed = STATUS_FILTERS[status]
  const needle = query.trim().toLowerCase()
  return jobs.filter(job => {
    if (allowed && !allowed.includes(job.status)) return false
    if (mode !== 'all' && job.request?.mode !== mode) return false
    if (needle) {
      const haystack = `${job.request?.prompt || ''} ${job.id} ${job.request?.mode || ''}`.toLowerCase()
      if (!haystack.includes(needle)) return false
    }
    return true
  })
}

/**
 * Sort for display. Active work always floats to the top regardless of the
 * chosen order, because it is what the user is waiting on.
 */
export function sortJobs(jobs, order = 'newest') {
  const direction = order === 'oldest' ? 1 : -1
  return jobs.slice().sort((a, b) => {
    const rank = job => (isActive(job) ? 0 : 1)
    if (rank(a) !== rank(b)) return rank(a) - rank(b)
    const left = new Date(a.created_at).getTime() || 0
    const right = new Date(b.created_at).getTime() || 0
    return (left - right) * direction
  })
}

/** The library list: filtered then sorted. */
export function selectLibraryJobs(jobs, filter, order) {
  return sortJobs(filterJobs(jobs, filter), order)
}

/** One-line spec of what a job asked for ("T2AV · 768×512 · 121f · 24fps"). */
export function describeJob(job) {
  const request = job?.request || {}
  const parts = [modeTag(request.mode)]
  const scale = request.upscale ? 2 : 1
  if (request.width && request.height) {
    parts.push(`${request.width * scale}×${request.height * scale}`)
  }
  if (request.num_frames) parts.push(`${request.num_frames}f`)
  if (request.fps) parts.push(`${request.fps}fps`)
  if (request.steps) parts.push(`${request.steps} steps`)
  return parts.join(' · ')
}

/** Runtime metadata strings, shown only where there is room for them. */
export function describePerformance(job) {
  const parts = []
  if (job?.generation_seconds != null) parts.push(`${job.generation_seconds.toFixed(2)}s`)
  if (job?.peak_vram_gb != null) parts.push(`${job.peak_vram_gb.toFixed(1)} GiB peak`)
  return parts
}

/** Clip length in seconds, derived from frame count and fps. */
export function clipDuration(job) {
  const frames = Number(job?.request?.num_frames) || 0
  const fps = Number(job?.request?.fps) || 24
  if (!frames) return ''
  return `${((frames - 1) / fps).toFixed(1)}s`
}

/**
 * Actions a job offers. Retake / Extend are stage actions gated on a finished
 * video, not a mode the user picks up front.
 */
export function selectJobActions(job) {
  if (!job || isPending(job)) {
    return { canInterrupt: false, canDelete: false, canDownload: false, canReuse: false, canDerive: false, canEmbed: false }
  }
  const canEmbed = getModeCapabilities(job.request?.mode).output === 'video'
  return {
    canInterrupt: ACTIVE_STATUSES.includes(job.status),
    canDelete: [JOB_STATUS.COMPLETED, JOB_STATUS.FAILED, JOB_STATUS.INTERRUPTED].includes(job.status),
    canDownload: hasMedia(job),
    canReuse: true,
    canDerive: canEmbed && job.status === JOB_STATUS.COMPLETED && Boolean(job.video_url),
    canEmbed,
  }
}

/** Small label used on take tiles. */
export function takeLabel(job) {
  return job?.generation_seconds != null
    ? `${job.generation_seconds.toFixed(1)}s`
    : statusLabel(job)
}
