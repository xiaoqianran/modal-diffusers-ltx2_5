/**
 * Polling scheduler with adaptive interval.
 *
 * Extracted so the timing policy lives in one place and can be swapped in tests
 * without touching the runtime: the runtime only says "something is running" or
 * "everything is idle" and this decides when to fire next.
 *
 * No Vue, no DOM, no network - the callback is injected.
 */

export const POLL_ACTIVE_MS = 1800
export const POLL_IDLE_MS = 6000
export const POLL_ERROR_MS = 5000
export const POLL_HEALTH_MS = 30000
export const POLL_HEALTH_WARMING_MS = 1500

/**
 * A self-rescheduling timer.
 *
 * @param {() => Promise<boolean>} task Resolves true when work is still active,
 *   which selects the fast interval. Rejections select the error interval.
 * @returns {{start: () => void, stop: () => void, kick: () => void, isRunning: () => boolean}}
 */
export function createPoller(task, {
  activeMs = POLL_ACTIVE_MS,
  idleMs = POLL_IDLE_MS,
  errorMs = POLL_ERROR_MS,
} = {}) {
  let timer = null
  let stopped = true
  let inFlight = false

  async function tick() {
    // Skip overlapping requests: a slow backend should not queue up calls.
    if (inFlight || stopped) return
    inFlight = true
    let delay = idleMs
    try {
      const active = await task()
      delay = active ? activeMs : idleMs
    } catch {
      delay = errorMs
    } finally {
      inFlight = false
    }
    if (!stopped) {
      timer = setTimeout(tick, delay)
    }
  }

  return {
    start() {
      if (!stopped) return
      stopped = false
      tick()
    },
    stop() {
      stopped = true
      if (timer) clearTimeout(timer)
      timer = null
    },
    /** Run now and reschedule, used after a mutation that changes the list. */
    kick() {
      if (stopped) return
      if (timer) clearTimeout(timer)
      tick()
    },
    isRunning() {
      return !stopped
    },
  }
}
