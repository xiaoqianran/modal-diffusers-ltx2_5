/**
 * Mode capabilities — the single source of truth for how each generation mode
 * behaves.
 *
 * Every `if (mode === 'i2v')` style branch in the UI should read from this table
 * instead of hard-coding mode names. That keeps the rules in one place and makes
 * adding a mode a data change rather than a hunt through components.
 *
 * Pure module: no Vue, no DOM, no network.
 */

/** @typedef {'image'|'video'|'audio'|null} MediaKind */

/**
 * @typedef {object} ModeCapability
 * @property {string}   label          Human label used in the UI.
 * @property {string}   intentLabel    Product-facing action label.
 * @property {'video'|'image'|'audio'} output What the mode produces.
 * @property {MediaKind} input         Primary visual input, if any.
 * @property {boolean}  needsLast      Requires a distinct last-frame image.
 * @property {boolean}  needsAudio     Requires an audio asset.
 * @property {boolean}  needsSource    Accepts a generic reference (image or video).
 * @property {boolean}  sourceIsVideo  Reference must specifically be a video.
 * @property {boolean}  singleReference Uses exactly one reference at index 1 (IC-LoRA).
 * @property {boolean}  multiReference Accepts ordered image references (Qwen, max 10).
 * @property {boolean}  orderedKeyframes References also expose timeline positions.
 * @property {boolean}  qwenOnly       This mode has no LTX execution path.
 * @property {boolean}  supportsAutoDuration Uses the LTX-2.5 DurationHead.
 * @property {boolean}  supportsHdr     Supports HDR/EXR input/output contract.
 * @property {boolean}  fixedSchedule   Uses a fixed distilled denoising schedule.
 * @property {boolean}  supportsUpscale Whether 2x spatial upscale applies.
 * @property {boolean}  supportsPixelUpscale Whether the pixel IC-LoRA path applies.
 * @property {boolean}  forcesUpscale  Always upsamples regardless of the toggle.
 * @property {number}   [fixedFrames]  Overrides the frame count.
 * @property {string[]} [rangeFields]  Extra numeric fields this mode needs.
 * @property {string[]} allowedEngines Engines the composer may expose.
 * @property {string}   group          Rail group this mode belongs to.
 */

/** Ordered group definitions, used by the navigation rail. */
export const MODE_GROUPS = [
  {
    id: 'create',
    label: '创建',
    sections: [
      { label: '视频', modes: ['t2av', 'i2v', 'flf2v', 'a2v', 'keyframe_interpolation', 'dfr'] },
      { label: '图片', modes: ['t2i', 'image_edit', 'ref2i'] },
      { label: '音频', modes: ['t2a'] },
    ],
  },
  {
    id: 'edit',
    label: '编辑',
    sections: [{ label: '基于输出', modes: ['retake', 'extend', 'refine_image'] }],
  },
  {
    id: 'control',
    label: '控制',
    sections: [{ label: '条件与适配', modes: ['condition', 'iclora'] }],
  },
]

const BASE = {
  label: '',
  intentLabel: '',
  output: 'video',
  input: null,
  needsLast: false,
  needsAudio: false,
  needsSource: false,
  sourceIsVideo: false,
  singleReference: false,
  multiReference: false,
  orderedKeyframes: false,
  qwenOnly: false,
  supportsAutoDuration: false,
  supportsHdr: false,
  fixedSchedule: false,
  supportsUpscale: true,
  supportsPixelUpscale: true,
  forcesUpscale: false,
  rangeFields: [],
  allowedEngines: ['auto', 'ltx'],
  group: 'create',
}

/** @type {Record<string, ModeCapability>} */
export const MODE_CAPABILITIES = {
  t2av: { ...BASE, label: '文本 → 视频', intentLabel: 'Generate video', input: null, group: 'create' },

  i2v: { ...BASE, label: '图片 → 视频', intentLabel: 'Animate image', input: 'image', group: 'create' },

  flf2v: { ...BASE, label: '首尾帧 → 视频', intentLabel: 'First & last frame', input: 'image', needsLast: true, group: 'create' },

  a2v: { ...BASE, label: '音频 → 视频', intentLabel: 'Audio-driven video', needsAudio: true, group: 'create' },

  t2a: {
    ...BASE,
    label: '文本 → 音频',
    intentLabel: 'Generate audio',
    output: 'audio',
    supportsAutoDuration: true,
    supportsUpscale: false,
    supportsPixelUpscale: false,
    forcesDecoder: 'vae',
    group: 'create',
  },

  keyframe_interpolation: {
    ...BASE,
    label: '关键帧插值',
    intentLabel: 'Keyframe interpolation',
    output: 'video',
    multiReference: true,
    orderedKeyframes: true,
    supportsHdr: true,
    supportsUpscale: false,
    supportsPixelUpscale: false,
    forcesDecoder: 'vae',
    group: 'create',
  },

  dfr: {
    ...BASE,
    label: 'DFR 生产质量',
    intentLabel: 'DFR production',
    output: 'video',
    multiReference: true,
    orderedKeyframes: true,
    supportsAutoDuration: true,
    fixedSchedule: true,
    supportsUpscale: false,
    supportsPixelUpscale: false,
    forcesDecoder: 'vae',
    group: 'create',
  },

  t2i: {
    ...BASE,
    label: '文本 → 图片',
    intentLabel: 'Generate image',
    output: 'image',
    allowedEngines: ['auto', 'ltx', 'qwen'],
    forcesUpscale: true,
    group: 'create',
  },

  image_edit: {
    ...BASE,
    label: 'Qwen 图片编辑',
    intentLabel: 'Edit image',
    output: 'image',
    multiReference: true,
    qwenOnly: true,
    allowedEngines: ['qwen'],
    supportsPixelUpscale: false,
    forcesUpscale: true,
    group: 'create',
  },

  ref2i: {
    ...BASE,
    label: '图片 → 图片',
    intentLabel: 'Reference to image',
    output: 'image',
    input: 'image',
    // ref2i is single-stage at base resolution: the backend force-disables
    // upscale and temporal upscale (schemas.py, STILL_IMAGE_MODES branch).
    supportsUpscale: false,
    supportsPixelUpscale: false,
    forcesUpscale: false,
    fixedFrames: 49,
    validFrames: [25, 41, 49],
    rangeFields: ['framePosition'],
    group: 'create',
  },

  refine_image: {
    ...BASE,
    label: '精修图片',
    intentLabel: 'Refine image',
    output: 'image',
    input: 'image',
    supportsPixelUpscale: false,
    rangeFields: ['strength'],
    group: 'edit',
  },

  retake: {
    ...BASE,
    label: '重拍片段',
    intentLabel: 'Retake',
    needsSource: true,
    sourceIsVideo: true,
    // Single-stage masked denoise at source resolution; decoder is pinned to VAE.
    supportsUpscale: false,
    supportsPixelUpscale: false,
    forcesDecoder: 'vae',
    rangeFields: ['retakeStart', 'retakeEnd'],
    group: 'edit',
  },

  extend: {
    ...BASE,
    label: '延长片段',
    intentLabel: 'Extend',
    needsSource: true,
    sourceIsVideo: true,
    supportsUpscale: false,
    supportsPixelUpscale: false,
    forcesDecoder: 'vae',
    rangeFields: ['extendDirection', 'extendSeconds', 'extendContext'],
    group: 'edit',
  },

  condition: {
    ...BASE,
    label: '多条件引导',
    intentLabel: 'Conditions',
    needsSource: true,
    group: 'control',
  },

  iclora: {
    ...BASE,
    label: 'IC-LoRA 参考',
    intentLabel: 'IC-LoRA reference',
    needsSource: true,
    singleReference: true,
    // The backend force-disables upscale, temporal upscale and switches the
    // decoder to VAE so the reference stays attached through stage 1.
    supportsUpscale: false,
    supportsPixelUpscale: false,
    forcesDecoder: 'vae',
    requiresLora: true,
    group: 'control',
  },
}

export const DEFAULT_MODE = 't2av'

/** Modes that produce a still image rather than a clip. */
export const STILL_MODES = Object.entries(MODE_CAPABILITIES)
  .filter(([, capability]) => capability.output === 'image')
  .map(([mode]) => mode)

/** Modes that edit an existing output rather than creating from scratch. */
export const DERIVED_MODES = ['retake', 'extend', 'refine_image']

export function getModeCapabilities(mode) {
  return MODE_CAPABILITIES[mode] || MODE_CAPABILITIES[DEFAULT_MODE]
}

export function isKnownMode(mode) {
  return Object.prototype.hasOwnProperty.call(MODE_CAPABILITIES, mode)
}

export function modeLabel(mode) {
  return getModeCapabilities(mode).label
}

export function modeIntentLabel(mode) {
  const capability = getModeCapabilities(mode)
  return capability.intentLabel || capability.label
}

/** Short uppercase tag shown on cards ("T2AV", "I2V", …). */
export function modeTag(mode) {
  return String(mode || DEFAULT_MODE).toUpperCase()
}

export function isStillMode(mode) {
  return getModeCapabilities(mode).output === 'image'
}

export function isDerivedMode(mode) {
  return DERIVED_MODES.includes(mode)
}

/** Which rail group owns a mode, so selecting one can expand the right section. */
export function groupOfMode(mode) {
  return getModeCapabilities(mode).group
}

export function modesInGroup(groupId) {
  return Object.entries(MODE_CAPABILITIES)
    .filter(([, capability]) => capability.group === groupId)
    .map(([mode]) => mode)
}
