/**
 * Formatting helpers with no business meaning.
 *
 * Pure module: no Vue, no DOM, no network.
 */

/** Escape a string for interpolation into HTML. */
export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char])
}

export function formatBytes(bytes) {
  const value = Number(bytes)
  if (!Number.isFinite(value) || value <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let index = 0
  let size = value
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024
    index += 1
  }
  return `${size.toFixed(size >= 10 || index === 0 ? 0 : 1)} ${units[index]}`
}

export function formatSeconds(value) {
  const seconds = Number(value)
  if (!Number.isFinite(seconds)) return ''
  return `${seconds.toFixed(2)}s`
}

/** "1.2 GB" style VRAM figure. */
export function formatVram(gibibytes) {
  const value = Number(gibibytes)
  if (!Number.isFinite(value)) return ''
  return `${value.toFixed(1)} GiB`
}

/** Milliseconds as `mm:ss`, for progress readouts. */
export function formatClock(milliseconds) {
  const total = Math.max(0, Math.floor(Number(milliseconds || 0) / 1000))
  const minutes = String(Math.floor(total / 60)).padStart(2, '0')
  const seconds = String(total % 60).padStart(2, '0')
  return `${minutes}:${seconds}`
}

/** Shorten a prompt for single-line display. */
export function truncate(text, max = 60) {
  const value = String(text ?? '')
  return value.length <= max ? value : `${value.slice(0, max - 1)}…`
}
