<script setup>
import { computed } from 'vue'

import { modeLabel } from '../model/modes.js'
import { describeJob, describePerformance, selectJobActions } from '../model/selectors.js'

const props = defineProps({
  assets: { type: Object, required: true },
})

const emit = defineEmits(['back', 'kind', 'page', 'select', 'reuse', 'remove'])

const pageNumbers = computed(() => {
  const current = Number(props.assets.page || 1)
  const pages = Number(props.assets.pages || 1)
  const start = Math.max(1, Math.min(current - 2, pages - 4))
  const end = Math.min(pages, start + 4)
  return Array.from({ length: Math.max(0, end - start + 1) }, (_, index) => start + index)
})

function prompt(job) {
  return String(job.request?.prompt || '').trim() || 'Untitled generation'
}

function meta(job) {
  return describePerformance(job).join(' · ') || job.id.slice(0, 8)
}
</script>

<template>
  <div class="assets-view">
    <header class="assets-head">
      <div class="assets-title">
        <span class="eyebrow">ASSETS</span>
        <h1>Generated assets</h1>
        <small>Your completed image and video outputs.</small>
      </div>
      <button class="ghost library-back" type="button" @click="emit('back')">← Generate</button>
    </header>

    <div class="assets-toolbar">
      <div class="asset-kind-tabs" role="tablist" aria-label="Asset type">
        <button
          class="asset-kind-tab"
          :class="{ active: assets.kind === 'image' }"
          type="button"
          role="tab"
          :aria-selected="assets.kind === 'image'"
          @click="emit('kind', 'image')"
        >
          <span class="asset-tab-icon">□</span>
          Images
        </button>
        <button
          class="asset-kind-tab"
          :class="{ active: assets.kind === 'video' }"
          type="button"
          role="tab"
          :aria-selected="assets.kind === 'video'"
          @click="emit('kind', 'video')"
        >
          <span class="asset-tab-icon">▷</span>
          Videos
        </button>
      </div>
      <span class="assets-count">{{ assets.total }} {{ assets.kind === 'image' ? 'images' : 'videos' }}</span>
    </div>

    <div class="assets-scroll">
      <div v-if="assets.loading" class="assets-state">Loading assets…</div>
      <div v-else-if="!assets.items.length" class="assets-state">
        No generated {{ assets.kind === 'image' ? 'images' : 'videos' }} yet.
      </div>

      <div v-else class="asset-grid">
        <article v-for="job in assets.items" :key="job.id" class="asset-card">
          <button class="asset-preview" type="button" @click="emit('select', job.id)">
            <img v-if="assets.kind === 'image'" :src="job.image_url" alt="" loading="lazy" />
            <video v-else :src="job.video_url" preload="metadata" muted playsinline />
            <span class="asset-preview-overlay">Open</span>
          </button>

          <div class="asset-card-body">
            <div class="asset-card-head">
              <span class="mode-chip">{{ modeLabel(job.request?.mode) }}</span>
              <small>{{ meta(job) }}</small>
            </div>
            <p :title="prompt(job)">{{ prompt(job) }}</p>
            <small class="asset-spec">{{ describeJob(job) }}</small>
          </div>

          <div class="asset-card-actions">
            <button type="button" class="job-action" @click="emit('select', job.id)">Open</button>
            <button
              v-if="selectJobActions(job).canReuse"
              type="button"
              class="job-action"
              @click="emit('reuse', job.id)"
            >Reuse</button>
            <a
              class="job-action"
              :href="job.download_url || job.video_url || job.image_url"
              download
            >Download</a>
            <button
              type="button"
              class="job-action danger"
              @click="emit('remove', job.id)"
            >Delete</button>
          </div>
        </article>
      </div>
    </div>

    <footer v-if="assets.pages > 1" class="assets-pagination">
      <button
        type="button"
        class="page-button"
        :disabled="assets.page <= 1 || assets.loading"
        @click="emit('page', assets.page - 1)"
      >←</button>
      <button
        v-for="page in pageNumbers"
        :key="page"
        type="button"
        class="page-button"
        :class="{ active: page === assets.page }"
        :disabled="assets.loading"
        @click="emit('page', page)"
      >{{ page }}</button>
      <button
        type="button"
        class="page-button"
        :disabled="assets.page >= assets.pages || assets.loading"
        @click="emit('page', assets.page + 1)"
      >→</button>
      <small>Page {{ assets.page }} / {{ assets.pages }}</small>
    </footer>
  </div>
</template>
