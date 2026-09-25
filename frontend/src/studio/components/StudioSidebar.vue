<script setup>
import { computed } from 'vue'

import { modeIntentLabel } from '../model/modes.js'
import {
  ADVANCED_WORKFLOWS,
  CREATION_SURFACES,
  creationSurfaceForMode,
  defaultModeForSurface,
} from '../model/workflows.js'

const props = defineProps({
  currentMode: { type: String, required: true },
  historyActive: { type: Boolean, default: false },
})
const emit = defineEmits(['select-mode', 'history'])

const activeSurface = computed(() => creationSurfaceForMode(props.currentMode))
const advancedActive = computed(() => (
  ADVANCED_WORKFLOWS.some(item => item.mode === props.currentMode)
))

function chooseSurface(surfaceId) {
  emit('select-mode', defaultModeForSurface(surfaceId))
}
</script>

<template>
  <nav class="workflow-sidebar workflow-sidebar-v4" aria-label="Creative workspace">
    <div class="workflow-sidebar-head">
      <span class="eyebrow">CREATE</span>
    </div>

    <div class="creation-surfaces">
      <button
        v-for="surface in CREATION_SURFACES"
        :key="surface.id"
        class="creation-surface"
        :class="{ active: !historyActive && !advancedActive && activeSurface === surface.id }"
        type="button"
        @click="chooseSurface(surface.id)"
      >
        <span class="creation-surface-icon" aria-hidden="true">
          {{ surface.id === 'image' ? '□' : surface.id === 'video' ? '▷' : '⌁' }}
        </span>
        <span class="creation-surface-copy">
          <strong>{{ surface.label }}</strong>
          <small>{{ surface.description }}</small>
        </span>
      </button>
    </div>

    <div class="workflow-tool-section">
      <span class="rail-caption">TOOLS</span>
      <button
        v-for="item in ADVANCED_WORKFLOWS"
        :key="item.mode"
        class="rail-tool"
        :class="{ active: !historyActive && currentMode === item.mode }"
        type="button"
        @click="emit('select-mode', item.mode)"
      >
        <span>{{ item.label }}</span>
        <small>{{ modeIntentLabel(item.mode) }}</small>
      </button>
    </div>

    <div class="workflow-sidebar-foot">
      <button
        class="workflow-link history-link"
        :class="{ active: historyActive }"
        type="button"
        @click="emit('history')"
      >
        <span>History</span>
      </button>
    </div>
  </nav>
</template>
