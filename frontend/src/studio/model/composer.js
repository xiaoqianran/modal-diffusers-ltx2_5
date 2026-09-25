import { getModeCapabilities } from './modes.js'
import { expectedEngine } from './workflows.js'

export const LTX_RATIO_OPTIONS = Object.freeze([
  { value: '704x704', label: '1:1', aspect: '1 / 1' },
  { value: '768x576', label: '4:3', aspect: '4 / 3' },
  { value: '576x768', label: '3:4', aspect: '3 / 4' },
  { value: '768x512', label: '3:2', aspect: '3 / 2' },
  { value: '512x768', label: '2:3', aspect: '2 / 3' },
  { value: '896x512', label: 'Wide', aspect: '7 / 4' },
  { value: '512x896', label: 'Portrait', aspect: '4 / 7' },
  { value: '704x480', label: '22:15', aspect: '22 / 15' },
  { value: '640x384', label: '5:3', aspect: '5 / 3' },
  { value: '512x320', label: '8:5', aspect: '8 / 5' },
])

export const QWEN_RATIO_OPTIONS = Object.freeze([
  { value: '1024x1024', label: '1:1 · 2K', aspect: '1 / 1' },
  { value: '1200x896', label: '4:3 · 2K', aspect: '4 / 3' },
  { value: '896x1200', label: '3:4 · 2K', aspect: '3 / 4' },
  { value: '1264x848', label: '3:2 · 2K', aspect: '3 / 2' },
  { value: '848x1264', label: '2:3 · 2K', aspect: '2 / 3' },
  { value: '1376x768', label: '16:9 · 2K', aspect: '16 / 9' },
  { value: '768x1376', label: '9:16 · 2K', aspect: '9 / 16' },
])

const DEFAULT_FRAME_OPTIONS = Object.freeze([
  { value: 49, label: '2s', detail: '49f' },
  { value: 121, label: '5s', detail: '121f' },
  { value: 241, label: '10s', detail: '241f' },
  { value: 481, label: '20s', detail: '481f' },
])

export function usesQwen(draft, capability = getModeCapabilities(draft.mode)) {
  return expectedEngine(draft) === 'qwen' || capability.qwenOnly
}

export function ratioOptionsForDraft(draft, capability = getModeCapabilities(draft.mode)) {
  return usesQwen(draft, capability) ? QWEN_RATIO_OPTIONS : LTX_RATIO_OPTIONS
}

export function frameOptionsForCapability(capability) {
  const fixed = capability.validFrames
  if (Array.isArray(fixed) && fixed.length) {
    return fixed.map(value => ({ value, label: String(value), detail: 'frames' }))
  }
  return DEFAULT_FRAME_OPTIONS
}

export function attachmentSlotsForDraft(capability, draft, attachments) {
  const result = []

  if (capability.multiReference) {
    const imageAccept = capability.supportsHdr ? 'image/*,.exr' : 'image/*'
    const used = []
    for (let index = 0; index < 10; index += 1) {
      if (attachments[`reference${index}`]) used.push(index)
    }
    const next = Array.from({ length: 10 }, (_, index) => index)
      .find(index => !attachments[`reference${index}`])

    for (const index of used) {
      result.push({
        slot: `reference${index}`,
        icon: 'IMAGE',
        title: index === 0 && draft.mode === 'image_edit'
          ? 'Primary image'
          : `Reference ${index + 1}`,
        accept: imageAccept,
      })
    }
    if (next !== undefined) {
      result.push({
        slot: `reference${next}`,
        icon: 'IMAGE',
        title: `Add reference ${used.length + 1}/10`,
        accept: imageAccept,
      })
    }
    return result
  }

  if (capability.input === 'image') {
    result.push({
      slot: 'first',
      icon: 'IMAGE',
      title: capability.needsLast ? 'First frame' : 'Source image',
      accept: 'image/*',
    })
  }
  if (capability.needsLast) {
    result.push({ slot: 'last', icon: 'IMAGE', title: 'Last frame', accept: 'image/*' })
  }
  if (capability.needsAudio) {
    result.push({ slot: 'audio', icon: 'AUDIO', title: 'Audio', accept: 'audio/*' })
  }
  if (capability.needsSource) {
    result.push({
      slot: 'source',
      icon: 'MEDIA',
      title: 'Source',
      accept: capability.sourceIsVideo ? 'video/*' : 'image/*,video/*',
    })
  }
  return result
}

export function keyframeSlotsForDraft(capability, attachments) {
  if (!capability.orderedKeyframes) return []
  const result = []
  for (let index = 0; index < 10; index += 1) {
    if (attachments[`reference${index}`]) {
      result.push({ index, label: `Keyframe ${index + 1}` })
    }
  }
  return result
}

export function engineStateFromHealth(health, engine) {
  if (!health) return { state: 'offline', label: 'Offline' }
  const warmup = health.warmup || {}

  if (warmup.state === 'warming') return { state: 'loading', label: 'Loading' }
  if (warmup.state === 'error') return { state: 'error', label: 'Error' }
  if (warmup.state !== 'ready') return { state: 'idle', label: 'Idle' }

  const engines = Array.isArray(warmup.engines) ? warmup.engines : []
  if (!engines.length) {
    return { state: 'unknown', label: 'Unknown' }
  }
  if (engines.includes(engine)) return { state: 'ready', label: 'Ready' }
  return { state: 'unavailable', label: 'Unavailable' }
}
