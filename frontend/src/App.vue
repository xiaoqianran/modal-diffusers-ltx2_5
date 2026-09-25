<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import PromptDock from './studio/components/PromptDock.vue'
import StudioAssets from './studio/components/StudioAssets.vue'
import StudioInspector from './studio/components/StudioInspector.vue'
import StudioLibrary from './studio/components/StudioLibrary.vue'
import StudioQueue from './studio/components/StudioQueue.vue'
import StudioSidebar from './studio/components/StudioSidebar.vue'
import StudioStage from './studio/components/StudioStage.vue'
import StudioTakes from './studio/components/StudioTakes.vue'
import {
  attachmentSlotsForDraft,
  frameOptionsForCapability,
  keyframeSlotsForDraft,
  ratioOptionsForDraft,
} from './studio/model/composer.js'
import { buildQueueIndex } from './studio/model/jobs.js'
import { modeIntentLabel } from './studio/model/modes.js'
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
  assets,
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
  engineStates,
  start,
  dispose,
  warmGpu,
  releaseGpu,
  refreshJobs,
  refreshHealth,
  refreshLoras,
  refreshAssets,
  setMode,
  setEngine,
  setFilter,
  setSortOrder,
  setAssetKind,
  setAssetPage,
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
const runtimeOpen = ref(false)

const queueIndex = computed(() => buildQueueIndex(allJobs.value))
const stageQueuePosition = computed(() => (
  stageJob.value ? (queueIndex.value.get(stageJob.value.id) || 0) : 0
))
const queueCount = computed(() => (
  (queue.value?.running?.length || 0) + (queue.value?.upcoming?.length || 0)
))
const queueCapacity = computed(() => (
  Number(health.value?.queue?.pending_capacity || health.value?.queue?.capacity || 256)
))
const executionCapacity = computed(() => (
  Number(health.value?.queue?.execution_capacity || 1)
))
const queueMeta = computed(() => ({
  active: queueCount.value,
  capacity: queueCapacity.value,
  executionCapacity: executionCapacity.value,
  available: health.value?.queue?.available !== false,
}))
const runtimeState = computed(() => {
  const job = stageJob.value
  if (job?.status === 'running') return 'generating'
  if (job?.status === 'queued') return 'queued'
  if (busy.preparing) return 'preparing'
  if (gpuState.value === 'warming') return 'warming'
  if (gpuState.value === 'ready') return 'ready'
  return 'offline'
})
const runtimePrimary = computed(() => {
  if (runtimeState.value === 'generating') return `${Math.round((stageJob.value?.progress || 0) * 100)}%`
  if (runtimeState.value === 'queued') return `#${stageQueuePosition.value || 1}`
  if (runtimeState.value === 'warming') return 'Warming'
  if (runtimeState.value === 'preparing') return 'Preparing'
  if (runtimeState.value === 'ready') return gpuLabel.value
  return 'Offline'
})
const sectionTitle = computed(() => {
  if (view.value === 'library') return 'History'
  if (view.value === 'assets') return 'Assets'
  return modeIntentLabel(draft.mode)
})
const ratioOptions = computed(() => ratioOptionsForDraft(draft, activeCapability.value))
const frameOptions = computed(() => frameOptionsForCapability(activeCapability.value))
const attachmentSlots = computed(() => (
  attachmentSlotsForDraft(activeCapability.value, draft, attachments)
))
const keyframeSlots = computed(() => (
  keyframeSlotsForDraft(activeCapability.value, attachments)
))

function chooseMode(mode) {
  setMode(mode)
  showView('generate')
}

function openHistory() {
  jobsOpen.value = false
  showView('library')
}

async function openAssets() {
  jobsOpen.value = false
  showView('assets')
  await refreshAssets({ kind: assets.kind, page: assets.page })
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

async function removeAsset(id) {
  await deleteJob(id)
  await refreshAssets({ kind: assets.kind, page: assets.page })
}

function onUpscaleChange() {
  setMode(draft.mode)
  if (!draft.upscale) draft.upscaleMethod = 'latent'
}

async function refreshAll() {
  moreOpen.value = false
  runtimeOpen.value = false
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
  runtimeOpen.value = false
}

function handleKeydown(event) {
  if (event.key !== 'Escape') return
  if (!jobsOpen.value && !moreOpen.value && !runtimeOpen.value) return
  event.preventDefault()
  jobsOpen.value = false
  moreOpen.value = false
  runtimeOpen.value = false
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
          <span class="jobs-copy"><span>Jobs</span><small v-if="queueCount">{{ queueCount }}/{{ queueCapacity }}</small></span>
          <b>{{ queueCount }}</b>
        </button>
        <div class="menu-wrap runtime-menu" @click.stop>
          <button
            class="topbar-pill health studio-status runtime-pill"
            type="button"
            :data-state="runtimeState"
            :aria-expanded="runtimeOpen"
            @click="runtimeOpen = !runtimeOpen"
          >
            <span class="runtime-dot" />
            <span>Runtime</span>
            <strong>{{ runtimeState === 'ready' ? 'Ready' : runtimePrimary }}</strong>
          </button>
          <div v-if="runtimeOpen" class="popover runtime-popover">
            <div class="runtime-popover-head">
              <div>
                <span class="eyebrow">RUNTIME</span>
                <strong>{{ runtimeState === 'ready' ? 'Ready' : runtimePrimary }}</strong>
              </div>
              <button class="runtime-refresh" type="button" @click="warmGpu">Refresh</button>
            </div>
            <div class="runtime-detail-row">
              <span><i class="runtime-state-dot" :data-state="gpuState" />GPU</span>
              <strong>{{ gpuLabel }}</strong>
            </div>
            <div class="runtime-detail-row">
              <span><i class="runtime-state-dot" :data-state="engineStates.ltx.state" />LTX-2.5</span>
              <strong>{{ engineStates.ltx.label }}</strong>
            </div>
            <div class="runtime-detail-row">
              <span><i class="runtime-state-dot" :data-state="engineStates.qwen.state" />Qwen Image 2.1</span>
              <strong>{{ engineStates.qwen.label }}</strong>
            </div>
          </div>
        </div>
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

    <section class="studio-shell studio-shell-v3" :class="{ 'history-mode': view !== 'generate' }">
      <StudioSidebar
        :current-mode="draft.mode"
        :history-active="view === 'library'"
        :assets-active="view === 'assets'"
        @select-mode="chooseMode"
        @assets="openAssets"
        @history="openHistory"
      />

      <main class="studio-workspace studio-workspace-v3">
        <StudioAssets
          v-if="view === 'assets'"
          :assets="assets"
          @back="backToGenerate"
          @kind="setAssetKind"
          @page="setAssetPage"
          @select="focus"
          @reuse="reuse"
          @remove="removeAsset"
        />

        <StudioLibrary
          v-else-if="view === 'library'"
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
        v-if="view === 'generate'"
        :draft="draft"
        :capability="activeCapability"
        :loras="loras"
        :ratio-options="ratioOptions"
        :frame-options="frameOptions"
        :keyframe-slots="keyframeSlots"
        @engine-change="setEngine"
        @upscale-change="onUpscaleChange"
      />
    </section>

    <button v-if="jobsOpen" class="drawer-scrim" type="button" aria-label="Close jobs" @click="jobsOpen = false" />
    <aside v-if="jobsOpen" class="floating-drawer jobs-drawer">
      <div class="drawer-heading">
        <div><span class="eyebrow">ACTIVITY</span><h2>Jobs</h2></div>
        <button class="drawer-close" type="button" @click="jobsOpen = false">×</button>
      </div>
      <StudioQueue
        :queue="queue"
        :queue-meta="queueMeta"
        :collapsed="false"
        @toggle="jobsOpen = false"
        @cancel="cancelJob"
      />
    </aside>
  </div>
</template>
