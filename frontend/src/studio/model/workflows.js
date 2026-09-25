import { getModeCapabilities } from './modes.js'

export const CREATION_SURFACES = Object.freeze([
  {
    id: 'image',
    label: 'Image',
    description: 'Generate and transform still images',
    defaultMode: 't2i',
    modes: ['t2i', 'image_edit', 'ref2i'],
  },
  {
    id: 'video',
    label: 'Video',
    description: 'Create motion, scenes and clips',
    defaultMode: 't2av',
    modes: ['t2av', 'i2v', 'flf2v', 'a2v'],
  },
  {
    id: 'audio',
    label: 'Audio',
    description: 'Generate sound from a prompt',
    defaultMode: 't2a',
    modes: ['t2a'],
  },
])

export const ADVANCED_WORKFLOWS = Object.freeze([
  { mode: 'keyframe_interpolation', label: 'Keyframes' },
  { mode: 'dfr', label: 'DFR' },
  { mode: 'condition', label: 'Conditions' },
  { mode: 'iclora', label: 'IC-LoRA' },
])

// Picker groups contain only workflows that make sense before an artifact exists.
// Artifact-derived actions (Retake / Extend / Refine) are deliberately absent:
// they are exposed next to the selected output instead.
export const WORKFLOW_GROUPS = Object.freeze([
  { id: 'image', label: 'Image', modes: ['t2i', 'image_edit', 'ref2i'] },
  { id: 'video', label: 'Video', modes: ['t2av', 'i2v', 'flf2v', 'a2v'] },
  { id: 'audio', label: 'Audio', modes: ['t2a'] },
  { id: 'advanced', label: 'Advanced', modes: ADVANCED_WORKFLOWS.map(item => item.mode) },
])

export function workflowGroupForMode(mode) {
  const direct = WORKFLOW_GROUPS.find(group => group.modes.includes(mode))?.id
  if (direct) return direct

  // Derived workflows stay attached to the artifact kind they operate on rather
  // than becoming top-level navigation categories.
  const output = getModeCapabilities(mode).output
  return output === 'image' || output === 'audio' ? output : 'video'
}

export function creationSurfaceForMode(mode) {
  return CREATION_SURFACES.find(surface => surface.modes.includes(mode))?.id
    || workflowGroupForMode(mode)
}

export function defaultModeForSurface(surfaceId) {
  return CREATION_SURFACES.find(surface => surface.id === surfaceId)?.defaultMode || 't2av'
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
