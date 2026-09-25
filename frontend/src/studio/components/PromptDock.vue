<script setup>
import { computed } from 'vue'
import { modeLabel } from '../model/modes.js'
import { engineLabel, expectedEngine } from '../model/workflows.js'
import ReferenceTray from './ReferenceTray.vue'

const props = defineProps({
  draft: { type: Object, required: true },
  attachmentSlots: { type: Array, default: () => [] },
  attachments: { type: Object, required: true },
  uploading: { type: Object, default: () => ({}) },
  canSubmit: { type: Boolean, default: false },
  notice: { type: Object, default: () => ({ text: '', state: '' }) },
})
const emit = defineEmits(['submit', 'file', 'remove'])
const expected = computed(() => engineLabel(expectedEngine(props.draft)))
</script>

<template>
  <form class="prompt-dock prompt-dock-v3" @submit.prevent="emit('submit')">
    <ReferenceTray
      :slots="attachmentSlots"
      :attachments="attachments"
      :uploading="uploading"
      @file="(...args) => emit('file', ...args)"
      @remove="slot => emit('remove', slot)"
    />
    <div class="prompt-row">
      <textarea
        v-model="draft.prompt"
        rows="2"
        required
        placeholder="Describe the scene, motion, camera, lighting, material, mood…"
        @keydown.ctrl.enter.prevent="canSubmit && emit('submit')"
      />
      <button class="generate-button" type="submit" :disabled="!canSubmit">
        <span>Generate</span><span class="generate-arrow" aria-hidden="true">↑</span>
      </button>
    </div>
    <div class="prompt-summary">
      <span class="prompt-workflow">{{ modeLabel(draft.mode) }}</span>
      <span class="prompt-summary-separator">·</span>
      <span v-if="draft.engine === 'auto'" class="prompt-route">Auto → {{ expected }}</span>
      <span v-else class="prompt-route">{{ expected }}</span>
      <span class="prompt-shortcut">Ctrl ↵</span>
    </div>
    <p v-if="notice.text" class="dock-notice" :data-state="notice.state">{{ notice.text }}</p>
  </form>
</template>
