<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import StudioLibrary from './studio/components/StudioLibrary.vue'
import StudioQueue from './studio/components/StudioQueue.vue'
import StudioStage from './studio/components/StudioStage.vue'
import StudioTakes from './studio/components/StudioTakes.vue'
import { buildQueueIndex } from './studio/model/jobs.js'
import { MODE_GROUPS, groupOfMode, modeLabel, modeTag } from './studio/model/modes.js'
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
  editFromOutput,
  reuseJob,
} = studio

const workspaceSection = ref(groupOfMode(draft.mode) === 'edit' ? 'edit' : 'generate')
const jobsOpen = ref(false)
const controlsOpen = ref(false)
const moreOpen = ref(false)
const modePickerOpen = ref(false)

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

const ratioOptions = Object.freeze([
  { value: '768x512', label: '3:2', aspect: '3 / 2' },
  { value: '704x480', label: '22:15', aspect: '22 / 15' },
  { value: '640x384', label: '5:3', aspect: '5 / 3' },
  { value: '512x320', label: '8:5', aspect: '8 / 5' },
])

const fpsOptions = Object.freeze([16, 24, 30])

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
  if (capability.input === 'image') {
    result.push({
      slot: 'first',
      icon: 'IMAGE',
      title: capability.needsLast ? 'First frame' : 'Reference',
      accept: 'image/*',
    })
  }
  if (capability.needsLast) {
    result.push({ slot: 'last', icon: 'IMAGE', title: 'Last frame', accept: 'image/*' })
  }
  if (capability.needsAudio) {
    result.push({ slot: 'audio', icon: 'AUDIO', title: 'Audio', accept: 'audio/*' })
  }
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

const sectionTitle = computed(() => {
  if (view.value === 'library') return 'History'
  if (workspaceSection.value === 'assets') return 'Assets'
  if (workspaceSection.value === 'edit') return 'Edit'
  return 'Generate'
})

function chooseMode(mode) {
  setMode(mode)
  modePickerOpen.value = false
  showView('generate')
  workspaceSection.value = groupOfMode(mode) === 'edit' ? 'edit' : 'generate'
}

function openSection(section) {
  jobsOpen.value = false
  controlsOpen.value = false

  if (section === 'history') {
    workspaceSection.value = 'history'
    showView('library')
    return
  }

  showView('generate')
  workspaceSection.value = section

  if (section === 'generate' && groupOfMode(draft.mode) === 'edit') {
    setMode('t2av')
  }
  if (section === 'edit' && groupOfMode(draft.mode) !== 'edit') {
    setMode('retake')
  }
}

function backToGenerate() {
  openSection('generate')
}

function attachmentMeta(slot) {
  const asset = attachments[slot]
  if (!asset) {
    if (slot === 'audio') return 'WAV / MP3 / M4A'
    if (slot === 'source') return activeCapability.value.sourceIsVideo ? 'Choose video' : 'Image / video'
    return 'Choose image'
  }
  const size = Number(asset.size || 0)
  const sizeLabel = size ? (size / 1024 / 1024).toFixed(1) + ' MB' : ''
  return [asset.kind, sizeLabel].filter(Boolean).join(' · ')
}

function attachmentTitle(slot, fallback) {
  return attachments[slot]?.filename || fallback
}

async function onAttachmentChange(slot, event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (file) await attach(slot, file)
}

function dropFile(file) {
  const capability = activeCapability.value
  let slot = null
  if (capability.needsAudio) slot = 'audio'
  else if (capability.needsSource) slot = 'source'
  else if (capability.input === 'image' && capability.needsLast && attachments.first) slot = 'last'
  else if (capability.input === 'image') slot = 'first'
  if (slot) void attach(slot, file)
}

async function derive(mode) {
  const result = await editFromOutput(mode)
  if (result.ok) {
    workspaceSection.value = 'edit'
    showView('generate')
    controlsOpen.value = true
  }
}

function reuse(id) {
  reuseJob(id)
  workspaceSection.value = groupOfMode(draft.mode) === 'edit' ? 'edit' : 'generate'
  showView('generate')
}

function focus(id) {
  selectJob(id)
  workspaceSection.value = 'generate'
  showView('generate')
}

function onUpscaleChange() {
  setMode(draft.mode)
  if (!draft.upscale) draft.upscaleMethod = 'latent'
}

function onEngineChange() {
  if (draft.engine !== 'qwen') return
  // Probe-validated Qwen baseline. Keep LTX defaults untouched when switching back.
  draft.steps = 40
  draft.guidanceScale = 1
  draft.numFrames = 9
  draft.fps = 24
  draft.decoder = 'vae'
  draft.loraId = null
}

function setOptionalNumber(key, raw) {
  draft[key] = raw === '' ? null : Number(raw)
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
    // Clipboard may be unavailable; the session stays visible in this menu.
  }
}

function closeMenus() {
  moreOpen.value = false
  modePickerOpen.value = false
}

function handleKeydown(event) {
  if (event.key !== 'Escape') return
  if (!controlsOpen.value && !jobsOpen.value && !modePickerOpen.value && !moreOpen.value) return

  event.preventDefault()
  controlsOpen.value = false
  jobsOpen.value = false
  modePickerOpen.value = false
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
  <div class="app studio-v2">
    <header class="topbar studio-topbar">
      <div class="topbar-left">
        <div class="brand" aria-label="LTX Studio">
          <span class="brand-mark">LTX</span>
          <span class="brand-sub">Studio</span>
        </div>
        <span class="topbar-separator" />
        <button class="project-button" type="button">
          <span class="project-dot" />
          Untitled project
        </button>
      </div>

      <div class="topbar-center">
        <span class="workspace-title">{{ sectionTitle }}</span>
      </div>

      <div class="header-actions">
        <button
          class="topbar-pill jobs-pill"
          :class="{ active: jobsOpen }"
          type="button"
          @click.stop="jobsOpen = !jobsOpen; controlsOpen = false"
        >
          <span class="status-dot" :class="{ live: queueCount > 0 }" />
          Jobs
          <b>{{ queueCount }}</b>
        </button>

        <button
          class="topbar-pill health"
          type="button"
          :data-state="health ? 'online' : 'offline'"
          @click="warmGpu"
        >{{ healthText }}</button>

        <div class="menu-wrap topbar-menu" @click.stop>
          <button
            class="topbar-icon"
            type="button"
            title="More"
            aria-label="More"
            @click="moreOpen = !moreOpen"
          >•••</button>
          <div v-if="moreOpen" class="popover topbar-popover">
            <button class="popover-item" type="button" @click="refreshAll">Refresh workspace</button>
            <button class="popover-item" type="button" @click="copySession">Copy session ID</button>
            <span class="popover-note">Session {{ sessionNumber || '—' }}</span>
          </div>
        </div>
      </div>
    </header>

    <section class="studio-shell">
      <nav class="studio-sidebar" aria-label="Workspace">
        <div class="sidebar-main">
          <button
            class="sidebar-item"
            :class="{ active: view === 'generate' && workspaceSection === 'generate' }"
            type="button"
            title="Generate"
            @click="openSection('generate')"
          >
            <span class="sidebar-icon" aria-hidden="true">＋</span>
            <span>Generate</span>
          </button>

          <button
            class="sidebar-item"
            :class="{ active: view === 'generate' && workspaceSection === 'edit' }"
            type="button"
            title="Edit"
            @click="openSection('edit')"
          >
            <span class="sidebar-icon" aria-hidden="true">◇</span>
            <span>Edit</span>
          </button>

          <button
            class="sidebar-item"
            :class="{ active: view === 'generate' && workspaceSection === 'assets' }"
            type="button"
            title="Assets"
            @click="openSection('assets')"
          >
            <span class="sidebar-icon" aria-hidden="true">▧</span>
            <span>Assets</span>
          </button>

          <button
            class="sidebar-item"
            :class="{ active: view === 'library' }"
            type="button"
            title="History"
            @click="openSection('history')"
          >
            <span class="sidebar-icon" aria-hidden="true">◷</span>
            <span>History</span>
          </button>
        </div>

        <button
          class="sidebar-item sidebar-bottom"
          :class="{ active: controlsOpen }"
          type="button"
          title="Controls"
          @click.stop="controlsOpen = !controlsOpen; jobsOpen = false"
        >
          <span class="sidebar-icon" aria-hidden="true">⌘</span>
          <span>Controls</span>
        </button>
      </nav>

      <main class="studio-workspace">
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

        <section v-else-if="workspaceSection === 'assets'" class="assets-view">
          <div class="section-heading">
            <div>
              <span class="eyebrow">PROJECT MEDIA</span>
              <h1>Assets</h1>
              <p>References stay attached to the current generation mode.</p>
            </div>

            <label class="mode-switch">
              <span>Mode</span>
              <select :value="draft.mode" @change="setMode($event.target.value)">
                <optgroup v-for="group in MODE_GROUPS" :key="group.id" :label="group.label">
                  <template v-for="section in group.sections" :key="section.label">
                    <option v-for="mode in section.modes" :key="mode" :value="mode">
                      {{ modeLabel(mode) }}
                    </option>
                  </template>
                </optgroup>
              </select>
            </label>
          </div>

          <div v-if="attachmentSlots.length" class="asset-slot-grid">
            <label
              v-for="item in attachmentSlots"
              :key="item.slot"
              class="asset-slot-card"
              :class="{ ready: Boolean(attachments[item.slot]) }"
            >
              <input
                type="file"
                :accept="item.accept"
                hidden
                @change="onAttachmentChange(item.slot, $event)"
              />
              <span class="asset-slot-type">{{ item.icon }}</span>
              <strong>{{ attachmentTitle(item.slot, item.title) }}</strong>
              <small>{{ busy.uploading[item.slot] ? 'Uploading…' : attachmentMeta(item.slot) }}</small>
              <button
                v-if="attachments[item.slot]"
                class="asset-remove"
                type="button"
                @click.prevent.stop="detach(item.slot)"
              >Remove</button>
            </label>
          </div>

          <div v-else class="assets-empty">
            <div class="empty-orbit">＋</div>
            <h2>No references needed</h2>
            <p>{{ modeLabel(draft.mode) }} can start from a prompt. Switch mode to attach images, video, or audio.</p>
            <button class="secondary-action" type="button" @click="openSection('generate')">Back to Generate</button>
          </div>
        </section>

        <template v-else>
          <section class="creation-space">
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

          <form class="prompt-dock" @submit.prevent="submit">
            <div class="reference-strip" :class="{ 'is-empty': !attachmentSlots.length }">
              <template v-if="attachmentSlots.length">
                <label
                  v-for="item in attachmentSlots"
                  :key="item.slot"
                  class="reference-chip"
                  :class="{
                    ready: Boolean(attachments[item.slot]),
                    uploading: Boolean(busy.uploading[item.slot]),
                  }"
                >
                  <input
                    type="file"
                    :accept="item.accept"
                    hidden
                    @change="onAttachmentChange(item.slot, $event)"
                  />
                  <span class="reference-add">＋</span>
                  <span>
                    <strong>{{ attachmentTitle(item.slot, item.title) }}</strong>
                    <small>{{ busy.uploading[item.slot] ? 'Uploading…' : attachmentMeta(item.slot) }}</small>
                  </span>
                  <button
                    v-if="attachments[item.slot]"
                    type="button"
                    class="reference-remove"
                    aria-label="Remove asset"
                    @click.prevent.stop="detach(item.slot)"
                  >×</button>
                </label>
              </template>
              <span v-else class="reference-placeholder">Prompt-only generation</span>
            </div>

            <div class="prompt-row">
              <textarea
                v-model="draft.prompt"
                rows="2"
                required
                placeholder="Describe the shot, motion, camera, lighting, mood…"
              />
              <button class="generate-button" type="submit" :disabled="!canSubmit">
                <span>Generate</span>
                <span class="generate-arrow" aria-hidden="true">↑</span>
              </button>
            </div>

            <div class="prompt-toolbar">
              <div class="toolbar-controls">
                <div class="mode-picker-wrap" @click.stop>
                  <button
                    class="mode-trigger"
                    :class="{ open: modePickerOpen }"
                    type="button"
                    aria-haspopup="menu"
                    :aria-expanded="modePickerOpen"
                    @click="modePickerOpen = !modePickerOpen"
                  >
                    <span class="control-caption">Mode</span>
                    <span class="mode-trigger-value">{{ modeLabel(draft.mode) }}</span>
                    <span class="mode-chevron" aria-hidden="true">⌄</span>
                  </button>

                  <div v-if="modePickerOpen" class="mode-picker" role="menu">
                    <div class="mode-picker-head">
                      <span>Generation mode</span>
                      <small>{{ modeTag(draft.mode) }}</small>
                    </div>

                    <div
                      v-for="group in MODE_GROUPS"
                      :key="group.id"
                      class="mode-picker-group"
                    >
                      <span class="mode-picker-group-label">{{ group.label }}</span>
                      <div class="mode-picker-grid">
                        <template v-for="section in group.sections" :key="section.label">
                          <button
                            v-for="mode in section.modes"
                            :key="mode"
                            class="mode-option"
                            :class="{ active: draft.mode === mode }"
                            type="button"
                            role="menuitem"
                            @click="chooseMode(mode)"
                          >
                            <span>{{ modeLabel(mode) }}</span>
                            <span v-if="draft.mode === mode" class="mode-check" aria-hidden="true">✓</span>
                          </button>
                        </template>
                      </div>
                    </div>
                  </div>
                </div>

                <button class="model-chip" type="button" tabindex="-1">
                  <span class="model-spark">✦</span>
                  <span>LTX-2.5</span>
                </button>

                <div class="parameter-picker ratio-picker">
                  <span class="parameter-label">Ratio</span>
                  <div class="parameter-options" aria-label="Aspect ratio">
                    <button
                      v-for="item in ratioOptions"
                      :key="item.value"
                      class="parameter-option ratio-option"
                      :class="{ active: draft.size === item.value }"
                      :aria-pressed="draft.size === item.value"
                      type="button"
                      @click="draft.size = item.value"
                    >
                      <span class="ratio-shape" :style="{ aspectRatio: item.aspect }" aria-hidden="true" />
                      <span class="parameter-value">{{ item.label }}</span>
                    </button>
                  </div>
                </div>

                <div class="parameter-picker duration-picker">
                  <span class="parameter-label">Duration</span>
                  <div class="parameter-options" aria-label="Duration">
                    <button
                      v-for="item in frameOptions"
                      :key="item.value"
                      class="parameter-option duration-option"
                      :class="{ active: draft.numFrames === item.value }"
                      :aria-pressed="draft.numFrames === item.value"
                      type="button"
                      @click="draft.numFrames = item.value"
                    >
                      <span class="parameter-value">{{ item.label }}</span>
                      <small>{{ item.detail }}</small>
                    </button>
                  </div>
                </div>

                <div class="parameter-picker fps-picker">
                  <span class="parameter-label">FPS</span>
                  <div class="parameter-options" aria-label="FPS">
                    <button
                      v-for="value in fpsOptions"
                      :key="value"
                      class="parameter-option fps-option"
                      :class="{ active: draft.fps === value }"
                      :aria-pressed="draft.fps === value"
                      type="button"
                      @click="draft.fps = value"
                    >
                      <span class="parameter-value">{{ value }}</span>
                      <small>fps</small>
                    </button>
                  </div>
                </div>
                <button
                  class="controls-button"
                  :class="{ active: controlsOpen }"
                  type="button"
                  @click.stop="controlsOpen = !controlsOpen; jobsOpen = false"
                >
                  <span>Controls</span>
                  <span aria-hidden="true">⌘</span>
                </button>
              </div>
            </div>

            <p v-if="notice.text" class="dock-notice" :data-state="notice.state">{{ notice.text }}</p>
          </form>
        </template>
      </main>
    </section>

    <button
      v-if="jobsOpen || controlsOpen"
      class="drawer-scrim"
      type="button"
      aria-label="Close panel"
      @click="jobsOpen = false; controlsOpen = false"
    />

    <aside v-if="jobsOpen" class="floating-drawer jobs-drawer">
      <div class="drawer-heading">
        <div>
          <span class="eyebrow">ACTIVITY</span>
          <h2>Jobs</h2>
        </div>
        <button class="drawer-close" type="button" @click="jobsOpen = false">×</button>
      </div>
      <StudioQueue
        :queue="queue"
        :collapsed="false"
        @toggle="jobsOpen = false"
        @cancel="cancelJob"
      />
    </aside>

    <aside v-if="controlsOpen" class="floating-drawer controls-drawer">
      <div class="drawer-heading">
        <div>
          <span class="eyebrow">{{ modeTag(draft.mode) }}</span>
          <h2>Controls</h2>
        </div>
        <button class="drawer-close" type="button" @click="controlsOpen = false">×</button>
      </div>

      <div class="controls-scroll">
        <section v-if="activeCapability.rangeFields?.length" class="control-section">
          <div class="control-section-title">Mode controls</div>

          <div class="drawer-fields two-col">
            <label v-if="activeCapability.rangeFields.includes('retakeStart')" class="field">
              <span>Retake start</span>
              <input v-model.number="draft.range.retakeStart" type="number" min="0" step="0.1" />
            </label>
            <label v-if="activeCapability.rangeFields.includes('retakeEnd')" class="field">
              <span>Retake end</span>
              <input v-model.number="draft.range.retakeEnd" type="number" min="0.1" step="0.1" />
            </label>
            <label v-if="activeCapability.rangeFields.includes('extendDirection')" class="field">
              <span>Direction</span>
              <select v-model="draft.range.extendDirection">
                <option value="end">Forward</option>
                <option value="start">Backward</option>
              </select>
            </label>
            <label v-if="activeCapability.rangeFields.includes('extendSeconds')" class="field">
              <span>Extend seconds</span>
              <input v-model.number="draft.range.extendSeconds" type="number" min="1" max="20" step="0.5" />
            </label>
            <label v-if="activeCapability.rangeFields.includes('extendContext')" class="field">
              <span>Context seconds</span>
              <input v-model.number="draft.range.extendContext" type="number" min="0.5" max="20" step="0.5" />
            </label>
            <label v-if="activeCapability.rangeFields.includes('strength')" class="field">
              <span>Reference strength</span>
              <input v-model.number="draft.range.strength" type="number" min="0.1" max="1" step="0.05" />
            </label>
            <label v-if="activeCapability.rangeFields.includes('framePosition')" class="field">
              <span>Frame position</span>
              <select v-model="draft.range.framePosition">
                <option value="last">Last</option>
                <option value="center">Center</option>
              </select>
            </label>
          </div>
        </section>

        <section class="control-section">
          <div class="control-section-title">Generation</div>
          <div class="drawer-fields two-col">
            <label class="field">
              <span>Model engine</span>
              <select v-model="draft.engine" @change="onEngineChange">
                <option value="auto">Auto · Director</option>
                <option value="ltx">LTX-2.5 NVFP4</option>
                <option value="qwen" :disabled="draft.mode !== 't2i'">Qwen-Image 2.1 BF16</option>
              </select>
            </label>
            <label class="field">
              <span>Frames</span>
              <input v-model.number="draft.numFrames" type="number" min="9" max="481" step="8" :disabled="draft.engine === 'qwen'" />
            </label>
            <label class="field">
              <span>Steps</span>
              <input v-model.number="draft.steps" type="number" min="1" max="100" />
            </label>
            <label class="field">
              <span>Guidance</span>
              <input v-model.number="draft.guidanceScale" type="number" min="0" max="20" step="0.1" :disabled="draft.engine === 'qwen'" />
            </label>
            <label class="field">
              <span>Seed</span>
              <input v-model.number="draft.seed" type="number" min="0" />
            </label>
            <label class="field">
              <span>Batch</span>
              <input v-model.number="draft.batch" type="number" min="1" max="20" />
            </label>
            <label class="field">
              <span>FPS</span>
              <input v-model.number="draft.fps" type="number" min="8" max="60" :disabled="draft.engine === 'qwen'" />
            </label>
          </div>
        </section>

        <section class="control-section">
          <div class="control-section-title">Upscale</div>
          <label class="switch-row">
            <span>
              <strong>2× spatial upscale</strong>
              <small>Increase final spatial resolution.</small>
            </span>
            <input
              v-model="draft.upscale"
              type="checkbox"
              :disabled="!activeCapability.supportsUpscale"
              @change="onUpscaleChange"
            />
          </label>

          <label class="field">
            <span>Upscale method</span>
            <select
              v-model="draft.upscaleMethod"
              :disabled="!draft.upscale || !activeCapability.supportsPixelUpscale"
            >
              <option value="latent">Latent</option>
              <option value="pixel">Pixel IC-LoRA</option>
            </select>
          </label>
        </section>

        <section class="control-section">
          <div class="control-section-title">Advanced</div>

          <label class="field">
            <span>Negative prompt</span>
            <textarea v-model="draft.negativePrompt" rows="3" />
          </label>

          <div class="drawer-fields two-col">
            <label class="field">
              <span>Video modality</span>
              <input
                :value="draft.modalityScale ?? ''"
                type="number"
                min="0"
                max="15"
                step="0.1"
                placeholder="1.0"
                @input="setOptionalNumber('modalityScale', $event.target.value)"
              />
            </label>
            <label class="field">
              <span>Audio guidance</span>
              <input
                :value="draft.audioGuidanceScale ?? ''"
                type="number"
                min="0"
                max="15"
                step="0.1"
                placeholder="Default"
                @input="setOptionalNumber('audioGuidanceScale', $event.target.value)"
              />
            </label>
          </div>

          <label class="field">
            <span>Decoder</span>
            <select v-model="draft.decoder" :disabled="Boolean(activeCapability.forcesDecoder) || draft.engine === 'qwen'">
              <option value="vae">VAE</option>
              <option value="diffusion">Diffusion</option>
            </select>
          </label>
        </section>

        <section class="control-section">
          <div class="control-section-title">LoRA</div>
          <label class="field">
            <span>Adapter</span>
            <select v-model="draft.loraId" :disabled="draft.engine === 'qwen'">
              <option :value="null">None</option>
              <option v-for="item in loras" :key="item.id" :value="item.id">
                {{ item.name || item.id }}
              </option>
            </select>
          </label>
          <label class="field">
            <span>Strength</span>
            <input v-model.number="draft.loraStrength" type="number" min="-2" max="2" step="0.1" :disabled="draft.engine === 'qwen'" />
          </label>
        </section>
      </div>
    </aside>
  </div>
</template>
