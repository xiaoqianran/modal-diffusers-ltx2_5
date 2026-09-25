<script setup>
import { computed, ref, watch } from 'vue'

import { modeLabel } from '../model/modes.js'
import { describeJob, describePerformance, selectJobActions } from '../model/selectors.js'

const props = defineProps({
  assets: { type: Object, required: true },
})

const emit = defineEmits(['back', 'kind', 'page', 'filter', 'select', 'reuse', 'remove'])

const search = ref(props.assets.query || '')
let searchTimer = null

watch(() => props.assets.query, value => {
  if (value !== search.value) search.value = value || ''
})

watch(search, value => {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(() => emit('filter', { query: value }), 250)
})

const pageNumbers = computed(() => {
  const current = Number(props.assets.page || 1)
  const pages = Number(props.assets.pages || 1)
  const start = Math.max(1, Math.min(current - 2, Math.max(1, pages - 4)))
  const end = Math.min(pages, start + 4)
  return Array.from({ length: Math.max(0, end - start + 1) }, (_, index) => start + index)
})

function prompt(job) {
  return String(job.request?.prompt || '').trim() || 'Untitled generation'
}

function meta(job) {
  return describePerformance(job).join(' · ') || job.id.slice(0, 8)
}

function previewStyle(job) {
  const scale = job.request?.upscale ? 2 : 1
  const width = Number(job.request?.width || 1) * scale
  const height = Number(job.request?.height || 1) * scale
  return { aspectRatio: `${width} / ${height}` }
}

function duration(job) {
  const frames = Number(job.request?.num_frames || 0)
  const fps = Number(job.request?.fps || 0)
  if (!frames || !fps) return ''
  return `${(frames / fps).toFixed(frames / fps >= 10 ? 0 : 1)}s`
}

function playPreview(event) {
  const video = event.currentTarget?.querySelector?.('video')
  if (!video) return
  video.play().catch(() => {})
}

function stopPreview(event) {
  const video = event.currentTarget?.querySelector?.('video')
  if (!video) return
  video.pause()
  try { video.currentTime = 0 } catch {}
}

function clearFilters() {
  search.value = ''
  emit('filter', { query: '', aspect: 'all', size: 'all', sort: 'newest' })
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

      <div class="assets-filterbar">
        <label class="assets-search">
          <span aria-hidden="true">⌕</span>
          <input v-model="search" type="search" placeholder="Search prompt or ID" aria-label="Search assets" />
        </label>

        <select
          :value="assets.aspect"
          aria-label="Aspect ratio"
          @change="emit('filter', { aspect: $event.target.value })"
        >
          <option value="all">All ratios</option>
          <option value="landscape">Landscape</option>
          <option value="portrait">Portrait</option>
          <option value="square">Square</option>
        </select>

        <select
          :value="assets.size"
          aria-label="Output size"
          @change="emit('filter', { size: $event.target.value })"
        >
          <option value="all">All sizes</option>
          <option value="small">&lt; 1 MP</option>
          <option value="medium">1–2 MP</option>
          <option value="large">2 MP+</option>
        </select>

        <select
          :value="assets.sort"
          aria-label="Sort assets"
          @change="emit('filter', { sort: $event.target.value })"
        >
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
        </select>
      </div>

      <span class="assets-count">{{ assets.total }} {{ assets.kind === 'image' ? 'images' : 'videos' }}</span>
    </div>

    <div class="assets-scroll">
      <div v-if="assets.loading" class="assets-state">Loading assets…</div>
      <div v-else-if="!assets.items.length" class="assets-state assets-empty">
        <strong>No matching {{ assets.kind === 'image' ? 'images' : 'videos' }}</strong>
        <small>Try another filter, or generate something new.</small>
        <button type="button" class="ghost" @click="clearFilters">Clear filters</button>
      </div>

      <div
        v-else
        class="asset-grid"
        :class="{ 'asset-grid-images': assets.kind === 'image', 'asset-grid-videos': assets.kind === 'video' }"
      >
        <article v-for="job in assets.items" :key="job.id" class="asset-card">
          <button
            class="asset-preview"
            type="button"
            :style="previewStyle(job)"
            @click="emit('select', job.id)"
            @mouseenter="assets.kind === 'video' && playPreview($event)"
            @mouseleave="assets.kind === 'video' && stopPreview($event)"
          >
            <img v-if="assets.kind === 'image'" :src="job.image_url" alt="" loading="lazy" />
            <video v-else :src="job.video_url" preload="metadata" muted loop playsinline />
            <span v-if="assets.kind === 'video' && duration(job)" class="asset-duration">{{ duration(job) }}</span>
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
            <button type="button" class="job-action danger" @click="emit('remove', job.id)">Delete</button>
          </div>
        </article>
      </div>
    </div>

    <footer class="assets-pagination">
      <div class="assets-page-controls" :class="{ invisible: assets.pages <= 1 }">
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
      </div>
      <small>{{ assets.total ? `Page ${assets.page} / ${assets.pages}` : 'No assets' }}</small>
    </footer>
  </div>
</template>
