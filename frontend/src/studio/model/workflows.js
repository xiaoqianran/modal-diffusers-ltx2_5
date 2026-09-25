import { getModeCapabilities } from './modes.js'

export const WORKFLOW_GROUPS = Object.freeze([
  { id: 'image', label: 'Image', modes: ['t2i', 'image_edit', 'ref2i'] },
  { id: 'video', label: 'Video', modes: ['t2av', 'i2v', 'flf2v', 'a2v'] },
  { id: 'edit', label: 'Edit', modes: ['retake', 'extend', 'refine_image'] },
  { id: 'advanced', label: 'Advanced', modes: ['keyframe_interpolation', 'dfr', 'condition', 'iclora', 't2a'] },
])

export function workflowGroupForMode(mode) {
  return WORKFLOW_GROUPS.find(group => group.modes.includes(mode))?.id || 'video'
}

export function expectedEngine(draft) {
  if (draft.engine === 'qwen') return 'qwen'
  if (draft.engine === 'ltx') return 'ltx'
  const capability = getModeCapabilities(draft.mode)
  if (capability.qwenOnly) return 'qwen'
  if (draft.mode === 't2i' && !draft.loraId) return 'qwen'
  return 'ltx'
}

export function engineLabel(engine) {
  if (engine === 'qwen') return 'Qwen-Image 2.1'
  if (engine === 'ltx') return 'LTX-2.5'
  return 'Director'
}
