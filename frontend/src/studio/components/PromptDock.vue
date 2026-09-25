<script setup>
import { computed } from 'vue'
import { modeIntentLabel } from '../model/modes.js'
import { engineLabel, expectedEngine, WORKFLOW_GROUPS } from '../model/workflows.js'
import ReferenceTray from './ReferenceTray.vue'

const props = defineProps({
  draft: { type: Object, required: true },
  attachmentSlots: { type: Array, default: () => [] },
  attachments: { type: Object, required: true },
  uploading: { type: Object, default: () => ({}) },
  canSubmit: { type: Boolean, default: false },
  notice: { type: Object, default: () => ({ text: '', state: '' }) },
})
const emit = defineEmits(['submit', 'file', 'remove', 'mode'])
const expected = computed(() => engineLabel(expectedEngine(props.draft)))
const sourceHandoff = computed(() => (
  props.draft.mode === 'i2v' && Boolean(props.attachments.first)
))
const durationLabel = computed(() => {
  if (props.draft.autoDuration) return 'Auto duration'
  const frames = Number(props.draft.numFrames || 0)
  const fps = Number(props.draft.fps || 0)
  if (!frames || !fps) return null
  const seconds = Math.max(1, Math.round(((frames - 1) / fps) * 10) / 10)
  return `${seconds}s · ${fps}fps`
})
</script>

<template>
  <form
    class="prompt-dock prompt-dock-v3"
    :class="{ 'has-input-slot': sourceHandoff || attachmentSlots.length }"
    @submit.prevent="emit('submit')"
  >
    <div v-if="sourceHandoff || attachmentSlots.length" class="composer-input-slot">
      <div v-if="sourceHandoff" class="workflow-handoff" aria-label="Source image to LTX video workflow">
        <span class="handoff-source">Source image</span>
        <span class="handoff-arrow" aria-hidden="true">→</span>
        <span class="handoff-target">Animate with LTX-2.5</span>
      </div>
      <ReferenceTray
        v-else
        :slots="attachmentSlots"
        :attachments="attachments"
        :uploading="uploading"
        @file="(...args) => emit('file', ...args)"
        @remove="slot => emit('remove', slot)"
      />
    </div>
    <div class="prompt-row">
      <textarea
        v-model="draft.prompt"
        rows="2"
        required
        :placeholder="sourceHandoff ? 'Describe how this image should move…' : 'Describe the scene, motion, camera, lighting, material, mood…'"
        @keydown.ctrl.enter.prevent="canSubmit && emit('submit')"
      />
      <button class="generate-button" type="submit" :disabled="!canSubmit">
        <span>Generate</span><span class="generate-arrow" aria-hidden="true">↑</span>
      </button>
    </div>
    <div class="prompt-summary">
      <label class="composer-chip prompt-workflow workflow-chip">
        <span class="sr-only">Creative intent</span>
        <select :value="draft.mode" @change="emit('mode', $event.target.value)">
          <optgroup v-for="group in WORKFLOW_GROUPS" :key="group.id" :label="group.label">
            <option v-for="mode in group.modes" :key="mode" :value="mode">{{ modeIntentLabel(mode) }}</option>
          </optgroup>
        </select>
      </label>
      <span class="composer-chip prompt-route">
        <small>{{ draft.engine === 'auto' ? 'Director' : 'Engine' }}</small>
        <strong>{{ draft.engine === 'auto' ? 'Auto → ' + expected : expected }}</strong>
      </span>
      <span class="composer-chip">{{ draft.size }}</span>
      <span v-if="durationLabel" class="composer-chip">{{ durationLabel }}</span>
      <span class="prompt-shortcut">Ctrl ↵</span>
    </div>
    <p v-if="notice.text" class="dock-notice" :data-state="notice.state">{{ notice.text }}</p>
  </form>
</template>
