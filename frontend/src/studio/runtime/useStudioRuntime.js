/**
 * Studio runtime - the imperative shell.
 *
 * Owns every side effect in the studio and exposes state plus actions. Views read
 * state and call actions; they never fetch, poll or build requests themselves.
 *
 * Three kinds of state are kept deliberately separate:
 *   server  - session, jobs, health, loras (owned by the backend)
 *   draft   - the composer's in-progress generation
 *   ui      - selection, active view, panel state
 * Everything else is a computed projection via the pure model.
 *
 * The pure model (`studio/model`) is a frozen dependency of this module: it is
 * imported, never reimplemented.
 */

import { computed, reactive, ref, shallowRef } from 'vue'

import { api, ApiError } from './api.js'
import { createPoller, POLL_HEALTH_MS, POLL_HEALTH_WARMING_MS } from './polling.js'
import { uploadFile } from './uploads.js'

import { DEFAULT_MODE, STILL_MODES, getModeCapabilities } from '../model/modes.js'
import { isActive, isPending, makePendingJob, mergePendingJobs } from '../model/jobs.js'
import {
  selectStageJob, selectTakes, selectQueue, selectCounts, selectAvailableModes,
  selectLibraryJobs, selectJobActions,
} from '../model/selectors.js'
import {
  validateGenerationDraft, buildBatchRequests, parseSize,
} from '../model/generationRequest.js'

const SESSION_KEY = 'ltx25.session'
const PENDING_TTL_MS = 30000
const DEFAULT_FILTER = { status: 'all', mode: 'all', query: '' }
const LTX_I2V_SIZES = Object.freeze([
  '704x704',
  '768x576',
  '576x768',
  '768x512',
  '512x768',
  '896x512',
  '512x896',
])

function closestLtxI2vSize(job) {
  const width = Number(job?.request?.width)
  const height = Number(job?.request?.height)
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return '768x512'
  }
  const target = width / height
  return LTX_I2V_SIZES.reduce((best, candidate) => {
    const [cw, ch] = candidate.split('x').map(Number)
    const [bw, bh] = best.split('x').map(Number)
    return Math.abs(cw / ch - target) < Math.abs(bw / bh - target) ? candidate : best
  }, LTX_I2V_SIZES[0])
}
const DEFAULT_RANGE = {
  retakeStart: 0,
  retakeEnd: 3,
  extendDirection: 'end',
  extendSeconds: 5,
  extendContext: 3,
  strength: 1,
  framePosition: 'last',
}

/** Create the default draft. Exported so tests and storybook-style previews can use it. */
export function createDraft(overrides = {}) {
  return {
    mode: DEFAULT_MODE,
    engine: 'auto',
    prompt: '',
    negativePrompt: 'worst quality, inconsistent motion, blurry, jittery, distorted',
    size: '768x512',
    numFrames: 121,
    fps: 24,
    steps: 8,
    guidanceScale: 3,
    seed: 42,
    batch: 1,
    upscale: true,
    upscaleMethod: 'latent',
    decoder: 'vae',
    modalityScale: null,
    audioGuidanceScale: null,
    audioStgScale: null,
    audioRescaleScale: null,
    audioSkipStep: null,
    audioStgBlocks: [],
    autoDuration: false,
    minSeconds: 1,
    maxSeconds: 8,
    keyframeFrames: Array(10).fill(null),
    dfrTemporalUpscalings: 0,
    dfrSpatialUpscalings: 1,
    hdrColorSpace: null,
    qwenTrueCfgScale: 1,
    qwenUseKvCache: true,
    transparentBackground: false,
    loraId: null,
    loraStrength: 1,
    range: { ...DEFAULT_RANGE },
    ...overrides,
  }
}

/** Read a persisted session number, tolerating a disabled/blocked localStorage. */
function readStoredSession() {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY)
    return raw ? Number(raw) : null
  } catch {
    return null
  }
}

function storeSession(value) {
  try {
    window.localStorage.setItem(SESSION_KEY, String(value))
  } catch {
    /* private mode or blocked storage: session simply is not persisted */
  }
}

/**
 * @param {{ apiOverride?: object }} [options] Injection point for tests.
 * @returns Studio runtime: refs, computeds and actions.
 */
export function useStudioRuntime(options = {}) {
  const client = options.apiOverride || api

  // ---------------------------------------------------------------- state

  // server state
  const sessionNumber = ref(readStoredSession())
  const jobs = shallowRef([])
  const health = ref(null)
  const loras = shallowRef([])
  const pending = ref(new Map())

  // draft state
  const draft = reactive(createDraft())
  const attachments = reactive({
    first: null,
    last: null,
    source: null,
    audio: null,
    ...Object.fromEntries(Array.from({ length: 10 }, (_, index) => [`reference${index}`, null])),
  })

  // ui state
  const selectedJobId = ref(null)
  const view = ref('generate')
  const queueCollapsed = ref(false)
  const filter = reactive({ ...DEFAULT_FILTER })
  const sortOrder = ref('newest')
  const busy = reactive({ submitting: false, uploading: {}, preparing: false })
  const notice = reactive({ text: '', state: '' })

  let pendingSeq = 0
  let disposed = false
  let warmLeaseTimer = null

  // ------------------------------------------------------------ computeds

  /** Every job the UI renders: optimistic placeholders ahead of server records. */
  const allJobs = computed(() => mergePendingJobs([...pending.value.values()], jobs.value))

  const stageJob = computed(() => selectStageJob(allJobs.value, selectedJobId.value))
  const takes = computed(() => selectTakes(allJobs.value))
  const queue = computed(() => selectQueue(allJobs.value))
  const counts = computed(() => selectCounts(allJobs.value))
  const availableModes = computed(() => selectAvailableModes(allJobs.value))
  const libraryJobs = computed(() => selectLibraryJobs(allJobs.value, filter, sortOrder.value))
  const stageActions = computed(() => selectJobActions(stageJob.value))
  const hasActiveWork = computed(() => allJobs.value.some(isActive))

  const activeCapability = computed(() => getModeCapabilities(draft.mode))
  const isStillMode = computed(() => STILL_MODES.includes(draft.mode))
  const problems = computed(() => validateGenerationDraft(draft, attachments, loras.value))
  const canSubmit = computed(() => problems.value.length === 0 && !busy.submitting)

  const gpuState = computed(() => health.value?.warmup?.state || 'unknown')
  const gpuLabel = computed(() => {
    if (!health.value) return '离线'
    const warm = health.value.warmup || {}
    if (warm.state === 'ready') return warm.gpu || 'READY'
    if (warm.state === 'warming') return '预热中…'
    return '离线'
  })

  // -------------------------------------------------------------- polling

  const jobPoller = createPoller(async () => {
    await refreshJobs()
    return hasActiveWork.value
  })

  const healthPoller = createPoller(async () => {
    await refreshHealth()
    return gpuState.value === 'warming'
  }, {
    activeMs: POLL_HEALTH_WARMING_MS,
    idleMs: POLL_HEALTH_MS,
    errorMs: POLL_HEALTH_MS,
  })

  // -------------------------------------------------------------- actions

  function setNotice(text, state = 'ok') {
    notice.text = text
    notice.state = state
  }

  function clearNotice() {
    notice.text = ''
    notice.state = ''
  }

  /**
   * Retire placeholders the server list has caught up with.
   *
   * Matching is by identity, not by a token echoed from the server: the backend
   * declares no `extra="forbid"`, so `client_token` is silently dropped and never
   * comes back. Instead a placeholder is retired when the server reports at least
   * as many jobs in this session as we have placeholders plus previously known
   * jobs - i.e. the count grew by the number we submitted.
   *
   * A TTL backstop covers the case where a submission fails outright before the
   * server records anything.
   */
  function sweepPending() {
    if (!pending.value.size) return
    const now = Date.now()
    const settled = jobs.value.length - baselineJobCount
    const next = new Map(pending.value)
    let retired = 0

    for (const [token, entry] of next) {
      if (retired < settled) {
        next.delete(token)
        retired += 1
      } else if (now - entry.createdAt > PENDING_TTL_MS) {
        next.delete(token)
      }
    }

    if (next.size !== pending.value.size) pending.value = next
  }

  /** Job count as of the last refresh, used to detect how many placeholders settled. */
  let baselineJobCount = 0

  async function refreshHealth() {
    try {
      health.value = await client.health()
    } catch {
      health.value = null
    }
  }

  async function refreshJobs() {
    if (!sessionNumber.value) {
      jobs.value = []
      return
    }
    try {
      const list = await client.listJobs(sessionNumber.value)
      jobs.value = list
      sweepPending()
      // Only advance the baseline once the sweep has consumed the difference,
      // otherwise the same records would settle placeholders twice.
      baselineJobCount = list.length + pending.value.size
    } catch (error) {
      // A transient failure should not wipe the list; the poller backs off.
      if (error instanceof ApiError && error.status === 404) jobs.value = []
    }
  }

  async function ensureSession() {
    if (sessionNumber.value) return sessionNumber.value
    const created = await client.createSession()
    sessionNumber.value = created.session_number
    storeSession(created.session_number)
    return sessionNumber.value
  }

  async function refreshLoras() {
    try {
      loras.value = await client.loras()
    } catch {
      loras.value = []
    }
  }

  async function warmGpu() {
    if (disposed) return false
    try {
      if (client.warm) await client.warm()
      await refreshHealth()
      return true
    } catch {
      health.value = null
      return false
    }
  }

  async function releaseGpu({ keepalive = false } = {}) {
    dispose()
    try {
      if (client.unload) await client.unload({ keepalive })
      return true
    } catch {
      return false
    }
  }

  // ---- draft editing -----------------------------------------------------

  function setMode(mode) {
    const capability = getModeCapabilities(mode)
    draft.mode = mode
    if (capability.qwenOnly) draft.engine = 'qwen'
    else if (!['t2i', 'image_edit'].includes(mode) && draft.engine === 'qwen') draft.engine = 'auto'
    // Mirror the backend's normalisation so the controls reflect what is sent.
    if (!capability.supportsUpscale) {
      draft.upscale = false
      draft.upscaleMethod = 'latent'
    } else if (capability.forcesUpscale) {
      draft.upscale = true
    }
    if (capability.forcesDecoder) draft.decoder = capability.forcesDecoder
    if (capability.fixedFrames) draft.numFrames = capability.fixedFrames
    if (!capability.supportsPixelUpscale) draft.upscaleMethod = 'latent'
  }

  function updateDraft(patch) {
    Object.assign(draft, patch)
  }

  function updateRange(patch) {
    draft.range = { ...draft.range, ...patch }
  }

  function resetDraftInputs() {
    draft.prompt = ''
    // The seed advances so a repeat submit produces a different take, matching
    // the backend's own practice of varying the seed across a batch.
    draft.seed = Number(draft.seed || 0) + 1
    clearAttachments()
  }

  function clearAttachments() {
    for (const slot of Object.keys(attachments)) attachments[slot] = null
  }

  async function attach(slot, file) {
    const capability = activeCapability.value
    const expected = slot === 'audio' ? 'audio'
      : slot === 'source' ? (capability.sourceIsVideo ? 'video' : null)
        : 'image'
    busy.uploading = { ...busy.uploading, [slot]: true }
    try {
      attachments[slot] = await uploadFile(file, { kind: expected, client })
      setNotice('素材已上传')
    } catch (error) {
      setNotice(`上传失败：${error.message}`, 'error')
    } finally {
      const next = { ...busy.uploading }
      delete next[slot]
      busy.uploading = next
    }
  }

  function detach(slot) {
    attachments[slot] = null
  }

  // ---- selection ---------------------------------------------------------

  function selectJob(id) {
    selectedJobId.value = id
  }

  function clearSelection() {
    selectedJobId.value = null
  }

  // ---- view --------------------------------------------------------------

  function showView(name) {
    view.value = name
  }

  function setFilter(patch) {
    Object.assign(filter, patch)
  }

  function setSortOrder(order) {
    sortOrder.value = order
  }

  function toggleQueue() {
    queueCollapsed.value = !queueCollapsed.value
  }

  // ---- submission --------------------------------------------------------

  /**
   * Submit the current draft.
   *
   * Returns as soon as the placeholders are registered: the network round trips
   * continue in the background so the button is never held hostage by a slow
   * backend. Callers therefore get feedback immediately.
   */
  async function submit() {
    clearNotice()
    const blocking = validateGenerationDraft(draft, attachments, loras.value)
    if (blocking.length) {
      setNotice(blocking[0], 'error')
      return { ok: false, reason: blocking[0] }
    }

    // The guard only spans the synchronous part of a submission (validation,
    // placeholder registration). Once the placeholders exist the round trips run
    // in the background, so it is released immediately and the composer stays
    // usable for the next idea - that is the whole point of the optimistic path.
    if (busy.submitting) return { ok: false, reason: '正在提交' }
    busy.submitting = true
    clearSelection()

    try {
      const session = await ensureSession()
      const requests = buildBatchRequests({ ...draft, ...parseSizeFields(), ...{ batch: Number(draft.batch) || 1 } }, attachments, session)
      const tokens = []

      for (const request of requests) {
        const token = `pending-${++pendingSeq}`
        tokens.push(token)
        const next = new Map(pending.value)
        next.set(token, {
          token,
          createdAt: Date.now(),
          sessionNumber: session,
          // The request sent to the server is the model's output verbatim. No
          // client-side marker is injected: the backend ignores unknown fields,
          // so anything added here would never be observable again.
          request,
        })
        pending.value = next
      }

      const count = requests.length
      setNotice(count === 1
        ? `已加入队列 · ${draft.mode.toUpperCase()}`
        : `已加入 ${count} 个任务`)
      resetDraftInputs()

      // Release the guard now: the placeholders are registered and the user can
      // compose the next shot while these requests are still in flight.
      busy.submitting = false

      // Background the round trips so the caller is never blocked.
      void (async () => {
        let succeeded = 0
        let failure = null
        for (const request of requests) {
          try {
            await client.submitJob(request)
            succeeded += 1
          } catch (error) {
            failure = error
          }
        }
        if (failure) {
          setNotice(succeeded
            ? `部分成功：${succeeded}/${count} 已入队 · ${failure.message}`
            : `提交失败：${failure.message}`, 'error')
        }
        // Placeholders are retired by the refresh that follows, not here: the
        // server's own record replaces them, and a failed submission is covered
        // by the notice plus the TTL sweep.
        await refreshJobs()
        jobPoller.kick()
      })()

      return { ok: true, count }
    } catch (error) {
      setNotice(`提交失败：${error.message}`, 'error')
      busy.submitting = false
      return { ok: false, reason: error.message }
    }
  }

  /** The model expects width/height; the composer stores a `size` string. */
  function parseSizeFields() {
    const { width, height } = parseSize(draft.size)
    return { width, height }
  }

  // ---- job actions -------------------------------------------------------

  async function cancelJob(jobId) {
    try {
      await client.interrupt(jobId)
      await refreshJobs()
      jobPoller.kick()
    } catch (error) {
      setNotice(`取消失败：${error.message}`, 'error')
    }
  }

  async function deleteJob(jobId) {
    try {
      await client.deleteJob(jobId)
      if (selectedJobId.value === jobId) clearSelection()
      await refreshJobs()
    } catch (error) {
      setNotice(`删除失败：${error.message}`, 'error')
    }
  }

  /**
   * Start a new workflow from the staged artifact without a browser round-trip.
   *
   * Target slot is derived from the target mode capability:
   * - Qwen image edit -> reference0
   * - image-to-video/refine/ref2i -> first
   * - retake/extend -> source
   */
  async function deriveFromOutput(mode) {
    const job = stageJob.value
    const capability = getModeCapabilities(mode)
    const hasImage = Boolean(job?.image_url)
    const hasVideo = Boolean(job?.video_url)
    const needsVideo = capability.needsSource && capability.sourceIsVideo
    const needsImage = capability.qwenOnly || capability.input === 'image'

    if (!job || (needsVideo && !hasVideo) || (needsImage && !hasImage)) {
      setNotice(needsVideo ? '需要先选择一个已完成的视频' : '需要先选择一个已完成的图片', 'error')
      return { ok: false }
    }
    busy.preparing = true
    setMode(mode)
    // A completed Qwen image handed to I2V must enter the LTX path explicitly.
    // Do not leave this as `auto`: the handoff itself is a concrete workflow
    // transition from still-image generation/editing to LTX video generation.
    if (mode === 'i2v') {
      draft.engine = 'ltx'
      // Qwen still-image presets can be much larger than the LTX video base
      // contract (for example 1024x1024 or 1536x1024). Do not carry that size
      // across the workflow boundary: LTX I2V validates its own base render
      // size and may apply a later 2x upscale. Start from the verified default.
      draft.size = closestLtxI2vSize(job)
    }
    try {
      setNotice('准备素材：云端复用输出')
      const asset = await client.reuseOutput(job.id)
      if (capability.multiReference) {
        attachments.reference0 = { ...asset, kind: asset.kind || 'image' }
      } else if (capability.needsSource) {
        attachments.source = { ...asset, kind: asset.kind || (needsVideo ? 'video' : null) }
      } else if (capability.input === 'image') {
        attachments.first = { ...asset, kind: asset.kind || 'image' }
      }
      setNotice(`已载入输出，可继续 ${capability.label}`)
      return { ok: true }
    } catch (error) {
      if (capability.multiReference) attachments.reference0 = null
      if (capability.needsSource) attachments.source = null
      if (capability.input === 'image') attachments.first = null
      setNotice(`载入输出失败：${error.message}`, 'error')
      return { ok: false }
    } finally {
      busy.preparing = false
    }
  }

  /** Copy a past job's parameters back into the composer. */
  async function reuseJob(jobId) {
    let job = jobs.value.find(item => item.id === jobId)
    try {
      job = await client.getJob(jobId)
    } catch (error) {
      setNotice(`载入参数失败：${error.message}`, 'error')
      return
    }
    if (!job?.request) return
    const request = job.request
    const capability = getModeCapabilities(request.mode)
    const scale = request.upscale ? 2 : 1
    Object.assign(draft, {
      mode: request.mode,
      engine: request.engine || 'auto',
      prompt: request.prompt || '',
      negativePrompt: request.negative_prompt || '',
      size: request.width ? `${request.width}x${request.height}` : draft.size,
      numFrames: request.num_frames || draft.numFrames,
      fps: request.fps || draft.fps,
      steps: request.steps || draft.steps,
      guidanceScale: request.guidance_scale ?? draft.guidanceScale,
      qwenTrueCfgScale: request.qwen_true_cfg_scale ?? draft.qwenTrueCfgScale,
      qwenUseKvCache: request.qwen_use_kv_cache ?? draft.qwenUseKvCache,
      transparentBackground: Boolean(request.transparent_background),
      audioStgScale: request.audio_stg_scale ?? draft.audioStgScale,
      audioRescaleScale: request.audio_rescale_scale ?? draft.audioRescaleScale,
      audioSkipStep: request.audio_skip_step ?? draft.audioSkipStep,
      audioStgBlocks: request.audio_stg_blocks || [],
      autoDuration: request.mode === 't2a' && request.num_frames == null,
      minSeconds: request.min_seconds ?? draft.minSeconds,
      maxSeconds: request.max_seconds ?? draft.maxSeconds,
      dfrTemporalUpscalings: request.dfr_temporal_upscalings ?? draft.dfrTemporalUpscalings,
      dfrSpatialUpscalings: request.dfr_spatial_upscalings ?? draft.dfrSpatialUpscalings,
      hdrColorSpace: request.hdr_color_space ?? null,
      seed: request.seed ?? draft.seed,
      upscale: Boolean(request.upscale),
      upscaleMethod: request.upscale_method || 'latent',
      decoder: request.decoder || 'vae',
      loraId: request.loras?.[0]?.id || null,
      loraStrength: request.loras?.[0]?.strength ?? 1,
      range: {
        ...draft.range,
        retakeStart: request.retake_start ?? draft.range.retakeStart,
        retakeEnd: request.retake_end ?? draft.range.retakeEnd,
        extendDirection: request.extend_direction || draft.range.extendDirection,
        extendSeconds: request.extend_seconds ?? draft.range.extendSeconds,
        extendContext: request.extend_context_seconds ?? draft.range.extendContext,
        strength: request.strength ?? draft.range.strength,
        framePosition: request.frame_position || draft.range.framePosition,
      },
    })
    void capability
    void scale
    setNotice(`已载入 ${request.mode.toUpperCase()} 的参数`)
  }

  // ---- lifecycle ---------------------------------------------------------

  async function start() {
    if (disposed) return
    await warmGpu()
    if (disposed) return
    warmLeaseTimer = globalThis.setInterval(() => {
      if (!disposed && client.warm) void Promise.resolve(client.warm()).catch(() => {})
    }, 30000)
    healthPoller.start()
    jobPoller.start()
    await Promise.all([ensureSession().catch(() => null), refreshLoras(), refreshHealth()])
    if (disposed) return
    await refreshJobs()
  }

  function dispose() {
    if (disposed) return
    disposed = true
    if (warmLeaseTimer !== null) {
      globalThis.clearInterval(warmLeaseTimer)
      warmLeaseTimer = null
    }
    jobPoller.stop()
    healthPoller.stop()
  }

  return {
    // state
    sessionNumber, jobs, health, loras, pending,
    draft, attachments,
    selectedJobId, view, queueCollapsed, filter, sortOrder, busy, notice,

    // derived
    allJobs, stageJob, takes, queue, counts, availableModes, libraryJobs,
    stageActions, hasActiveWork, activeCapability, isStillMode,
    problems, canSubmit, gpuState, gpuLabel,

    // actions
    start, dispose, warmGpu, releaseGpu,
    refreshJobs, refreshHealth, refreshLoras, ensureSession,
    setMode, updateDraft, updateRange, resetDraftInputs, clearAttachments,
    attach, detach,
    selectJob, clearSelection,
    showView, setFilter, setSortOrder, toggleQueue,
    submit, cancelJob, deleteJob, deriveFromOutput, reuseJob,
    setNotice, clearNotice,
  }
}
