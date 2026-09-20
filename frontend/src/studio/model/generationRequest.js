/**
 * Draft validation and request construction.
 *
 * This is where the mode rules actually bite: which attachments a mode requires,
 * which latent index each one lands on, and which optional fields apply. Both
 * are pure functions of a draft so they can be unit tested and reused by the
 * Vue layer without modification.
 *
 * Pure module: no Vue, no DOM, no network.
 */

import { getModeCapabilities, isStillMode, modeLabel } from './modes.js'

/** Latent index each condition occupies, per the backend contract. */
const CONDITION_INDEX = {
  first: 0,
  last: -1,
  source: 0,
  reference: 1,
}

/**
 * @typedef {object} Draft
 * @property {string} mode
 * @property {string} engine
 * @property {string} prompt
 * @property {string} negativePrompt
 * @property {number} width
 * @property {number} height
 * @property {number} numFrames
 * @property {number} fps
 * @property {number} steps
 * @property {number} guidanceScale
 * @property {number} seed
 * @property {number} batch
 * @property {boolean} upscale
 * @property {string} upscaleMethod
 * @property {string} decoder
 * @property {number|null} modalityScale
 * @property {number|null} audioGuidanceScale
 * @property {string|null} loraId
 * @property {number} loraStrength
 * @property {object} range   Mode-specific extras (retakeStart, extendSeconds, …).
 */

/**
 * Validate a draft against the capabilities of its mode.
 * @returns {string[]} Human-readable problems; empty means valid.
 */
export function validateGenerationDraft(draft, assets = {}, loras = []) {
  const problems = []
  const capability = getModeCapabilities(draft.mode)

  if (!String(draft.prompt || '').trim()) problems.push('Prompt 不能为空')

  if (draft.engine === 'qwen' && draft.mode !== 't2i') {
    problems.push('Qwen-Image 2.1 当前只支持 Text → Image')
  }
  if (draft.engine === 'qwen' && draft.loraId) {
    problems.push('Qwen-Image 2.1 暂未接入 LoRA 路由')
  }

  if (capability.input === 'image' && !assets.first) {
    problems.push(`${modeLabel(draft.mode)} 需要先添加参考图片`)
  }
  if (capability.needsLast && !assets.last) {
    problems.push(`${modeLabel(draft.mode)} 需要末帧图片`)
  }
  if (capability.needsAudio && !assets.audio) {
    problems.push(`${modeLabel(draft.mode)} 需要先添加音频`)
  }
  if (capability.needsSource && !assets.source) {
    problems.push(`${modeLabel(draft.mode)} 需要先添加参考素材`)
  }
  if (capability.sourceIsVideo && assets.source && assets.source.kind !== 'video') {
    problems.push(`${modeLabel(draft.mode)} 的参考素材必须是视频`)
  }

  if (capability.requiresLora) {
    if (!draft.loraId) {
      problems.push('IC-LoRA 模式需要选择一个 IC-LoRA')
    } else {
      const lora = loras.find(item => item.id === draft.loraId)
      if (lora && !lora.generic_iclora_compatible) {
        problems.push('该 LoRA 不支持通用 IC-LoRA 输入')
      }
    }
  }

  if (draft.mode === 'retake') {
    const start = Number(draft.range?.retakeStart)
    const end = Number(draft.range?.retakeEnd)
    if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) {
      problems.push('重拍起点必须小于终点')
    }
  }

  // ref2i only accepts a fixed set of frame counts.
  if (capability.validFrames && !capability.validFrames.includes(Number(draft.numFrames))) {
    problems.push(`该模式仅支持 ${capability.validFrames.join(' / ')} 帧`)
  }

  const wantsUpscale = capability.forcesUpscale ? true
    : capability.supportsUpscale && Boolean(draft.upscale)

  if (wantsUpscale) {
    if (draft.width * draft.height > 960 * 544) {
      problems.push('2× 放大的基础分辨率不能超过 960×544')
    }
    if (draft.upscaleMethod === 'pixel' && !capability.supportsPixelUpscale) {
      problems.push(`${modeLabel(draft.mode)} 不支持 Pixel 放大`)
    }
  }

  // The diffusion decoder needs a latent upscale or refine stage to run at all.
  const effectiveDecoder = capability.forcesDecoder || draft.decoder
  if (effectiveDecoder === 'diffusion' && !wantsUpscale) {
    problems.push('Diffusion 解码器需要开启 2× 放大')
  }

  return problems
}

/** Conditions a draft contributes, in the order the backend expects. */
export function buildConditions(draft, assets = {}) {
  const capability = getModeCapabilities(draft.mode)
  const conditions = []

  if (capability.input === 'image' && assets.first) {
    conditions.push({
      asset_id: assets.first.id,
      kind: 'image',
      index: CONDITION_INDEX.first,
      strength: 1,
    })
  }
  if (capability.needsLast && assets.last) {
    conditions.push({
      asset_id: assets.last.id,
      kind: 'image',
      index: CONDITION_INDEX.last,
      strength: 1,
    })
  }
  if (capability.needsSource && assets.source) {
    conditions.push({
      asset_id: assets.source.id,
      kind: capability.sourceIsVideo ? 'video' : assets.source.kind,
      // IC-LoRA takes its reference at latent index 1; everything else at 0.
      index: capability.singleReference ? CONDITION_INDEX.reference : CONDITION_INDEX.source,
      strength: 1,
    })
  }

  return conditions
}

/** LoRA list for a draft. IC-LoRA requires exactly one; others allow none. */
export function buildLoras(draft) {
  if (!draft.loraId) return []
  return [{ id: draft.loraId, strength: Number(draft.loraStrength ?? 1) }]
}

/**
 * Build the request body for one job in a batch.
 *
 * @param {Draft} draft
 * @param {object} assets
 * @param {number} sessionNumber
 * @param {number} [seedOffset] Added to the draft seed for batch index.
 */
export function buildGenerationRequest(draft, assets, sessionNumber, seedOffset = 0) {
  const capability = getModeCapabilities(draft.mode)
  const still = isStillMode(draft.mode)
  const range = draft.range || {}

  // The backend force-disables upscale for several modes and pins the decoder to
  // VAE; mirroring that here means the request we send already matches what the
  // server will normalise it to.
  const upscale = capability.supportsUpscale
    ? (capability.forcesUpscale ? true : Boolean(draft.upscale))
    : false
  const decoder = capability.forcesDecoder || draft.decoder || 'vae'

  const body = {
    session_number: sessionNumber,
    mode: draft.mode,
    engine: draft.engine || 'auto',
    prompt: String(draft.prompt || '').trim(),
    negative_prompt: String(draft.negativePrompt || '').trim() || null,
    width: draft.width,
    height: draft.height,
    upscale,
    upscale_method: still || !capability.supportsPixelUpscale ? 'latent' : draft.upscaleMethod || 'latent',
    decoder,
    seed: Number(draft.seed || 0) + seedOffset,
    steps: Number(draft.steps),
    guidance_scale: Number(draft.guidanceScale),
    num_frames: capability.fixedFrames ?? Number(draft.numFrames),
    fps: Number(draft.fps),
    conditions: buildConditions(draft, assets),
    loras: buildLoras(draft),
    audio_asset_id: capability.needsAudio && assets.audio ? assets.audio.id : null,

    // These two are genuinely Optional on the backend, so null is meaningful
    // ("not a retake") rather than a missing value.
    retake_start: draft.mode === 'retake' ? Number(range.retakeStart) : null,
    retake_end: draft.mode === 'retake' ? Number(range.retakeEnd) : null,

    // The remaining range fields are NOT optional in schemas.py - they carry
    // defaults and reject null. Send their documented defaults when the mode does
    // not use them, otherwise the request fails validation outright.
    extend_direction: draft.mode === 'extend' ? range.extendDirection : 'end',
    extend_seconds: draft.mode === 'extend' ? Number(range.extendSeconds) : 5,
    extend_context_seconds: draft.mode === 'extend' ? Number(range.extendContext) : 3,
    strength: draft.mode === 'refine_image' ? Number(range.strength) : 1,
    frame_position: draft.mode === 'ref2i' ? range.framePosition : 'last',

    regenerate_video: true,
    regenerate_audio: true,
    modality_scale: draft.modalityScale ?? null,
    audio_guidance_scale: draft.audioGuidanceScale ?? null,
  }

  return body
}

/** Build every request in a batch, honouring the batch size. */
export function buildBatchRequests(draft, assets, sessionNumber) {
  const count = Math.max(1, Math.min(20, Number(draft.batch) || 1))
  return Array.from({ length: count }, (_, index) =>
    buildGenerationRequest(draft, assets, sessionNumber, index)
  )
}

/**
 * Parse a `"768x512"` size value.
 * @returns {{width:number, height:number}}
 */
export function parseSize(value) {
  const [width, height] = String(value || '').split('x').map(Number)
  return { width: width || 768, height: height || 512 }
}
