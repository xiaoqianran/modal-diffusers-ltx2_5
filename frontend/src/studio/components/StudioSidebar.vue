<script setup>
import { modeLabel } from '../model/modes.js'
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

function modesForSection(section) {
  if (section === 'create') return createModes
  if (section === 'edit') return editModes
  return controlModes
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
      <div
        v-for="section in [
          { id: 'create', label: 'Create', count: createModes.length },
          { id: 'edit', label: 'Edit', count: editModes.length },
          { id: 'control', label: 'Control', count: controlModes.length },
        ]"
        :key="section.id"
        class="rail-section"
        :class="{ expanded: !historyActive && sectionForMode(currentMode) === section.id }"
      >
        <button
          class="workflow-link rail-section-link"
          :class="{ active: !historyActive && sectionForMode(currentMode) === section.id }"
          type="button"
          @click="chooseSection(section.id)"
        >
          <span>{{ section.label }}</span><small>{{ section.count }} workflows</small>
        </button>
        <div
          v-if="!historyActive && sectionForMode(currentMode) === section.id"
          class="rail-workflows"
        >
          <button
            v-for="mode in modesForSection(section.id)"
            :key="mode"
            class="rail-workflow"
            :class="{ active: currentMode === mode }"
            type="button"
            @click="emit('select-mode', mode)"
          >
            <span>{{ modeLabel(mode) }}</span>
            <i aria-hidden="true">›</i>
          </button>
        </div>
      </div>
    </div>
    <div class="workflow-sidebar-foot">
      <button class="workflow-link history-link" :class="{ active: historyActive }" type="button" @click="emit('history')">
        <span>History</span>
      </button>
    </div>
  </nav>
</template>
