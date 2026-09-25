<script setup>
import { modeLabel } from '../model/modes.js'
import { WORKFLOW_GROUPS } from '../model/workflows.js'

defineProps({
  currentMode: { type: String, required: true },
  historyActive: { type: Boolean, default: false },
})
const emit = defineEmits(['select-mode', 'history'])
</script>

<template>
  <nav class="workflow-sidebar" aria-label="Create workflows">
    <div class="workflow-sidebar-head"><span class="eyebrow">CREATE</span></div>
    <div class="workflow-groups">
      <section v-for="group in WORKFLOW_GROUPS" :key="group.id" class="workflow-group">
        <h3>{{ group.label }}</h3>
        <button
          v-for="mode in group.modes"
          :key="mode"
          class="workflow-link"
          :class="{ active: !historyActive && currentMode === mode }"
          type="button"
          @click="emit('select-mode', mode)"
        >
          <span>{{ modeLabel(mode) }}</span>
        </button>
      </section>
    </div>
    <div class="workflow-sidebar-foot">
      <button class="workflow-link history-link" :class="{ active: historyActive }" type="button" @click="emit('history')">
        <span>History</span>
      </button>
    </div>
  </nav>
</template>
