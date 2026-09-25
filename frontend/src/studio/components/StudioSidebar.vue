<script setup>
import { WORKFLOW_GROUPS, workflowGroupForMode } from '../model/workflows.js'

const props = defineProps({
  currentMode: { type: String, required: true },
  historyActive: { type: Boolean, default: false },
})
const emit = defineEmits(['select-mode', 'history'])

const createModes = WORKFLOW_GROUPS
  .filter(group => ['image', 'video'].includes(group.id))
  .flatMap(group => group.modes)
const editModes = WORKFLOW_GROUPS.find(group => group.id === 'edit').modes
const controlModes = WORKFLOW_GROUPS.find(group => group.id === 'advanced').modes

function sectionForMode(mode) {
  const group = workflowGroupForMode(mode)
  if (group === 'edit') return 'edit'
  if (group === 'advanced') return 'control'
  return 'create'
}

function chooseSection(section) {
  if (section === sectionForMode(props.currentMode)) return
  if (section === 'create') emit('select-mode', 't2i')
  if (section === 'edit') emit('select-mode', 'retake')
  if (section === 'control') emit('select-mode', 'keyframe_interpolation')
}
</script>

<template>
  <nav class="workflow-sidebar" aria-label="Create workflows">
    <div class="workflow-sidebar-head">
      <span class="eyebrow">WORKSPACE</span>
    </div>
    <div class="workflow-groups rail-sections">
      <button
        class="workflow-link rail-section-link"
        :class="{ active: !historyActive && sectionForMode(currentMode) === 'create' }"
        type="button"
        @click="chooseSection('create')"
      >
        <span>Create</span><small>{{ createModes.length }} workflows</small>
      </button>
      <button
        class="workflow-link rail-section-link"
        :class="{ active: !historyActive && sectionForMode(currentMode) === 'edit' }"
        type="button"
        @click="chooseSection('edit')"
      >
        <span>Edit</span><small>{{ editModes.length }} workflows</small>
      </button>
      <button
        class="workflow-link rail-section-link"
        :class="{ active: !historyActive && sectionForMode(currentMode) === 'control' }"
        type="button"
        @click="chooseSection('control')"
      >
        <span>Control</span><small>{{ controlModes.length }} workflows</small>
      </button>
    </div>
    <div class="workflow-sidebar-foot">
      <button class="workflow-link history-link" :class="{ active: historyActive }" type="button" @click="emit('history')">
        <span>History</span>
      </button>
    </div>
  </nav>
</template>
