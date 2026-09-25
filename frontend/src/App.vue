<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import PromptDock from './studio/components/PromptDock.vue'
import StudioInspector from './studio/components/StudioInspector.vue'
import StudioLibrary from './studio/components/StudioLibrary.vue'
import StudioQueue from './studio/components/StudioQueue.vue'
import StudioSidebar from './studio/components/StudioSidebar.vue'
import StudioStage from './studio/components/StudioStage.vue'
import StudioTakes from './studio/components/StudioTakes.vue'
import { buildQueueIndex } from './studio/model/jobs.js'
import { modeLabel } from './studio/model/modes.js'
import { mockApi } from './studio/runtime/mockApi.js'
import { useStudioRuntime } from './studio/runtime/useStudioRuntime.js'

const useMockApi = ['1', 'true'].includes(String(import.meta.env.VITE_MOCK_API || '').toLowerCase())
const studio = useStudioRuntime({ apiOverride: useMockApi ? mockApi : undefined })

const {
  sessionNumber,
  health,
  loras,
  draft,
  attachments,
  selectedJobId,
  view,
  filter,
  sortOrder,
  busy,
  notice,
  allJobs,
  stageJob,
  takes,
  queue,
  counts,
  availableModes,
  libraryJobs,
  stageActions,
  activeCapability,
  canSubmit,
  gpuState,
  gpuLabel,
  start,
  dispose,
  warmGpu,
  releaseGpu,
  refreshJobs,
  refreshHealth,
  refreshLoras,
  setMode,
  setFilter,
  setSortOrder,
  attach,
  detach,
  selectJob,
  showView,
  submit,
  cancelJob,
  deleteJob,
  deriveFromOutput,
  reuseJob,
} = studio

const jobsOpen = ref(false)
const moreOpen = ref(false)

const queueIndex = computed(() => buildQueueIndex(allJobs.value))
const stageQueuePosition = computed(() => (
  stageJob.value ? (queueIndex.value.get(stageJob.value.id) || 0) : 0
))
const queueCount = computed(() => (
  (queue.value?.running?.length || 0) + (queue.value?.upcoming?.length || 0)
))
const isMock = computed(() => health.value?.service?.includes('mock'))
const healthText = computed(() => {
  if (isMock.value) return 'Mock data'
  return health.value ? gpuLabel.value : 'Offline'
})
const activityText = computed(() => {
  const job = stageJob.value
  if (job?.status === 'running') return `Generating · ${Math.round((job.progress || 0) * 100)}%`
  if (job?.status === 'queued') return 'Queued'
  if (busy.preparing) return 'Preparing media'
  if (gpuState.value === 'ready') return 'LTX ready · Qwen ready'
  if (gpuState.value === 'warming') return 'Loading models'
  return 'Models offline'
})
const sectionTitle = computed(() => view.value === 'library' ? 'History' : modeLabel(draft.mode))

const LTX_RATIO_OPTIONS = Object.freeze([
  { value: '704x704', label: '1:1', aspect: '1 / 1' },
  { value: '768x576', label: '4:3', aspect: '4 / 3' },
  { value: '576x768', label: '3:4', aspect: '3 / 4' },
  { value: '768x512', label: '3:2', aspect: '3 / 2' },
  { value: '512x768', label: '2:3', aspect: '2 / 3' },
  { value: '896x512', label: 'Wide', aspect: '7 / 4' },
  { value: '512x896', label: 'Portrait', aspect: '4 / 7' },
  { value: '704x480', label: '22:15', aspect: '22 / 15' },
  { value: '640x384', label: '5:3', aspect: '5 / 3' },
  { value: '512x320', label: '8:5', aspect: '8 / 5' },
])
const QWEN_RATIO_OPTIONS = Object.freeze([
  { value: '1024x1024', label: '1:1 · 2K', aspect: '1 / 1' },
  { value: '1200x896', label: '4:3 · 2K', aspect: '4 / 3' },
  { value: '896x1200', label: '3:4 · 2K', aspect: '3 / 4' },
  { value: '1264x848', label: '3:2 · 2K', aspect: '3 / 2' },
  { value: '848x1264', label: '2:3 · 2K', aspect: '2 / 3' },
  { value: '1376x768', label: '16:9 · 2K', aspect: '16 / 9' },
  { value: '768x1376', label: '9:16 · 2K', aspect: '9 / 16' },
])
const qwenActive = computed(() => (
  activeCapability.value.qwenOnly
  || draft.engine === 'qwen'
  || (draft.mode === 't2i' && draft.engine === 'auto' && !draft.loraId)
))
const ratioOptions = computed(() => qwenActive.value ? QWEN_RATIO_OPTIONS : LTX_RATIO_OPTIONS)
const frameOptions = computed(() => {
  const fixed = activeCapability.value.validFrames
  if (Array.isArray(fixed) && fixed.length) {
    return fixed.map(value => ({ value, label: String(value), detail: 'frames' }))
  }
  return [
    { value: 49, label: '2s', detail: '49f' },
    { value: 121, label: '5s', detail: '121f' },
    { value: 241, label: '10s', detail: '241f' },
    { value: 481, label: '20s', detail: '481f' },
  ]
})

const attachmentSlots = computed(() => {
  const capability = activeCapability.value
  const result = []

  if (capability.multiReference) {
    const imageAccept = capability.supportsHdr ? 'image/*,.exr' : 'image/*'
    const used = []
    for (let index = 0; index < 10; index += 1) {
      if (attachments[`reference${index}`]) used.push(index)
    }
    const next = Array.from({ length: 10 }, (_, index) => index)
      .find(index => !attachments[`reference${index}`])
    for (const index of used) {
      result.push({ slot: `reference${index}`, icon: 'IMAGE', title: index === 0 && draft.mode === 'image_edit' ? 'Primary / reference 1' : `Reference ${index + 1}`, accept: imageAccept })
    }
    if (next !== undefined) {
      result.push({ slot: `reference${next}`, icon: 'IMAGE', title: `Add reference ${used.length + 1}/10`, accept: imageAccept })
    }
    return result
  }

  if (capability.input === 'image') {
    result.push({
      slot: 'first',
      icon: 'IMAGE',
      title: capability.needsLast ? 'First frame' : 'Source image',
      accept: 'image/*',
    })
  }
  if (capability.needsLast) result.push({ slot: 'last', icon: 'IMAGE', title: 'Last frame', accept: 'image/*' })
  if (capability.needsAudio) result.push({ slot: 'audio', icon: 'AUDIO', title: 'Audio', accept: 'audio/*' })
  if (capability.needsSource) {
    result.push({
      slot: 'source',
      icon: 'MEDIA',
      title: 'Source',
      accept: capability.sourceIsVideo ? 'video/*' : 'image/*,video/*',
    })
  }
  return result
})

const keyframeSlots = computed(() => {
  if (!['keyframe_interpolation', 'dfr'].includes(draft.mode)) return []
  const result = []
  for (let index = 0; index < 10; index += 1) {
    if (attachments[`reference${index}`]) result.push({ index, label: `Keyframe ${index + 1}` })
  }
  return result
})

function chooseMode(mode) {
  setMode(mode)
  showView('generate')
}

function openHistory() {
  jobsOpen.value = false
  showView('library')
}

function backToGenerate() {
  showView('generate')
}

async function onAttachment(slot, file) {
  await attach(slot, file)
}

function dropFile(file) {
  const capability = activeCapability.value
  let slot = null
  if (capability.needsAudio) slot = 'audio'
  else if (capability.needsSource) slot = 'source'
  else if (capability.multiReference) {
    slot = Array.from({ length: 10 }, (_, index) => `reference${index}`)
      .find(key => !attachments[key]) || null
  } else if (capability.input === 'image' && capability.needsLast && attachments.first) slot = 'last'
  else if (capability.input === 'image') slot = 'first'
  if (slot) void attach(slot, file)
}

async function derive(mode) {
  const result = await deriveFromOutput(mode)
  if (result.ok) showView('generate')
}

function reuse(id) {
  void reuseJob(id)
  showView('generate')
}

function focus(id) {
  selectJob(id)
  showView('generate')
}

function onUpscaleChange() {
  setMode(draft.mode)
  if (!draft.upscale) draft.upscaleMethod = 'latent'
}

function onEngineChange() {
  if (draft.engine !== 'qwen') return
  draft.steps = 40
  draft.guidanceScale = 1
  draft.numFrames = 9
  draft.fps = 24
  draft.decoder = 'vae'
  draft.loraId = null
  draft.qwenTrueCfgScale = draft.qwenTrueCfgScale ?? 1
  draft.qwenUseKvCache = true
}

async function refreshAll() {
  moreOpen.value = false
  await Promise.all([refreshJobs(), refreshHealth(), refreshLoras()])
}

async function copySession() {
  moreOpen.value = false
  if (!sessionNumber.value) return
  try {
    await navigator.clipboard.writeText(String(sessionNumber.value))
  } catch {
    // Clipboard may be unavailable; the session remains visible in the menu.
  }
}

function closeMenus() {
  moreOpen.value = false
}

function handleKeydown(event) {
  if (event.key !== 'Escape') return
  if (!jobsOpen.value && !moreOpen.value) return
  event.preventDefault()
  jobsOpen.value = false
  moreOpen.value = false
}

function handleOnline() {
  void warmGpu()
  void refreshJobs()
}

function handlePageHide(event) {
  if (event.persisted) return
  void releaseGpu({ keepalive: true })
}

onMounted(async () => {
  document.addEventListener('click', closeMenus)
  window.addEventListener('keydown', handleKeydown)
  window.addEventListener('online', handleOnline)
  window.addEventListener('pagehide', handlePageHide)
  await start()
})

onBeforeUnmount(() => {
  document.removeEventListener('click', closeMenus)
  window.removeEventListener('keydown', handleKeydown)
  window.removeEventListener('online', handleOnline)
  window.removeEventListener('pagehide', handlePageHide)
  dispose()
})
</script>

<template>
  <div class="app studio-v3">
    <header class="topbar studio-topbar">
      <div class="topbar-left">
        <div class="brand" aria-label="Modal Studio">
          <span class="brand-mark">Modal</span>
          <span class="brand-sub">Studio</span>
        </div>
        <span class="topbar-separator" />
        <span class="project-button"><span class="project-dot" /> Creative workspace</span>
      </div>

      <div class="topbar-center">
        <span class="workspace-title">{{ sectionTitle }}</span>
      </div>

      <div class="header-actions">
        <button class="topbar-pill jobs-pill" :class="{ active: jobsOpen }" type="button" @click.stop="jobsOpen = !jobsOpen">
          <span class="status-dot" :class="{ live: queueCount > 0 }" />
          Jobs <b>{{ queueCount }}</b>
        </button>
        <button class="topbar-pill health studio-status" type="button" :data-state="gpuState === 'ready' ? 'online' : 'offline'" @click="warmGpu">
          <span class="status-dot" :class="{ live: gpuState === 'ready' }" />
          <span class="status-copy"><strong>{{ healthText }}</strong><small>{{ activityText }}</small></span>
        </button>
        <div class="menu-wrap topbar-menu" @click.stop>
          <button class="topbar-icon" type="button" title="More" aria-label="More" @click="moreOpen = !moreOpen">•••</button>
          <div v-if="moreOpen" class="popover topbar-popover">
            <button class="popover-item" type="button" @click="refreshAll">Refresh workspace</button>
            <button class="popover-item" type="button" @click="copySession">Copy session ID</button>
            <span class="popover-note">Session {{ sessionNumber || '—' }}</span>
          </div>
        </div>
      </div>
    </header>

    <section class="studio-shell studio-shell-v3" :class="{ 'history-mode': view === 'library' }">
      <StudioSidebar
        :current-mode="draft.mode"
        :history-active="view === 'library'"
        @select-mode="chooseMode"
        @history="openHistory"
      />

      <main class="studio-workspace studio-workspace-v3">
        <StudioLibrary
          v-if="view === 'library'"
          :jobs="libraryJobs"
          :counts="counts"
          :available-modes="availableModes"
          :filter="filter"
          :sort-order="sortOrder"
          @back="backToGenerate"
          @filter="setFilter"
          @sort="setSortOrder"
          @select="focus"
          @reuse="reuse"
          @remove="deleteJob"
        />

        <template v-else>
          <section class="creation-space creation-space-v3">
            <div class="canvas-shell">
              <StudioStage
                :job="stageJob"
                :actions="stageActions"
                :queue-position="stageQueuePosition"
                :preparing="busy.preparing"
                @drop="dropFile"
                @reuse="reuse"
                @remove="deleteJob"
                @derive="derive"
              />
            </div>

            <StudioTakes
              :takes="takes"
              :selected-id="selectedJobId"
              @select="selectJob"
            />
          </section>

          <PromptDock
            :draft="draft"
            :attachment-slots="attachmentSlots"
            :attachments="attachments"
            :uploading="busy.uploading"
            :can-submit="canSubmit"
            :notice="notice"
            @submit="submit"
            @mode="chooseMode"
            @file="onAttachment"
            @remove="detach"
          />
        </template>
      </main>

      <StudioInspector
        v-if="view !== 'library'"
        :draft="draft"
        :capability="activeCapability"
        :loras="loras"
        :ratio-options="ratioOptions"
        :frame-options="frameOptions"
        :keyframe-slots="keyframeSlots"
        @engine-change="onEngineChange"
        @upscale-change="onUpscaleChange"
      />
    </section>

    <button v-if="jobsOpen" class="drawer-scrim" type="button" aria-label="Close jobs" @click="jobsOpen = false" />
    <aside v-if="jobsOpen" class="floating-drawer jobs-drawer">
      <div class="drawer-heading">
        <div><span class="eyebrow">ACTIVITY</span><h2>Jobs</h2></div>
        <button class="drawer-close" type="button" @click="jobsOpen = false">×</button>
      </div>
      <StudioQueue :queue="queue" :collapsed="false" @toggle="jobsOpen = false" @cancel="cancelJob" />
    </aside>
  </div>
</template>
