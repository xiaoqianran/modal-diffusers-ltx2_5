<script setup>
import { modeLabel } from '../model/modes.js'
import { takeLabel } from '../model/selectors.js'

const props = defineProps({
  takes: { type: Array, default: () => [] },
  selectedId: { type: String, default: null },
})

const emit = defineEmits(['select'])
</script>

<template>
  <section v-if="props.takes.length" class="filmstrip-wrap">
    <div class="filmstrip-head">
      <div><span>TAKES</span><small>Recent outputs</small></div>
      <b>{{ props.takes.length }}</b>
    </div>
    <div class="filmstrip">
      <button
        v-for="job in props.takes"
        :key="job.id"
        class="take"
        :class="{ active: props.selectedId === job.id }"
        type="button"
        :title="job.request?.prompt || ''"
        @click="emit('select', job.id)"
      >
        <span class="take-media">
          <video v-if="job.video_url" :src="job.video_url" preload="metadata" muted playsinline />
          <img v-else-if="job.image_url" :src="job.image_url" alt="" loading="lazy" />
          <span v-else-if="job.audio_url" class="thumb-blank">AUDIO</span>
          <span v-else class="thumb-blank">{{ modeLabel(job.request?.mode) }}</span>
        </span>
        <span class="take-meta">
          <span class="take-mode"><i aria-hidden="true" />{{ modeLabel(job.request?.mode) }}</span>
          <span class="take-time">{{ takeLabel(job) }}</span>
        </span>
      </button>
    </div>
  </section>
</template>
