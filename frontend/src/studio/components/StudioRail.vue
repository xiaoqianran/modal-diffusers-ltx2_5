<script setup>
import { MODE_GROUPS, modeLabel } from '../model/modes.js'

const props = defineProps({
  mode: { type: String, required: true },
  activeGroup: { type: String, default: null },
  view: { type: String, default: 'generate' },
})

const emit = defineEmits(['group', 'mode', 'library'])
</script>

<template>
  <nav class="rail" aria-label="模式导航">
    <div v-for="group in MODE_GROUPS" :key="group.id" class="rail-group" :data-group="group.id">
      <button
        class="rail-item"
        :class="{ active: props.view === 'generate' && props.activeGroup === group.id }"
        type="button"
        @click="emit('group', group.id)"
      >{{ group.label }}</button>

      <div v-if="props.view === 'generate' && props.activeGroup === group.id" class="rail-tree">
        <template v-for="section in group.sections" :key="section.label">
          <span class="rail-label">{{ section.label }}</span>
          <button
            v-for="mode in section.modes"
            :key="mode"
            class="mode"
            :class="{ active: props.mode === mode }"
            type="button"
            @click="emit('mode', mode)"
          >{{ modeLabel(mode) }}</button>
        </template>
      </div>
    </div>

    <div class="rail-group">
      <button
        class="rail-item"
        :class="{ active: props.view === 'library' }"
        type="button"
        @click="emit('library')"
      >Library</button>
    </div>
  </nav>
</template>
