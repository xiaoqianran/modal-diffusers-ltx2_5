<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

import StudioLibrary from './studio/components/StudioLibrary.vue'
import StudioQueue from './studio/components/StudioQueue.vue'
import StudioRail from './studio/components/StudioRail.vue'
import StudioStage from './studio/components/StudioStage.vue'
import StudioTakes from './studio/components/StudioTakes.vue'
import { buildQueueIndex } from './studio/model/jobs.js'
import { groupOfMode, modeTag } from './studio/model/modes.js'
import { useStudioRuntime } from './studio/runtime/useStudioRuntime.js'

const studio = useStudioRuntime()
const {
  sessionNumber,
  health,
  loras,
  draft,
  attachments,
  selectedJobId,
  view,
  queueCollapsed,
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
  toggleQueue,
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

const activeGroup = ref(groupOfMode(draft.mode))
const settingsOpen = ref(false)
const moreOpen = ref(false)

const queueIndex = computed(() => buildQueueIndex(allJobs.value))
const stageQueuePosition = computed(() => (
  stageJob.value ? (queueIndex.value.get(stageJob.value.id) || 0) : 0
))

const healthText = computed(() => {
  const precision = health.value?.transformer_precision?.toUpperCase() || 'GPU'
  return health.value ? precision + ' · ' + gpuLabel.value : '离线'
})

const frameOptions = computed(() => {
  const fixed = activeCapability.value.validFrames
  if (Array.isArray(fixed) && fixed.length) {
    return fixed.map(value => ({ value, label: value + ' frames' }))
  }
  return [
    { value: 49, label: '2.0s' },
    { value: 121, label: '5.0s' },
    { value: 241, label: '10.0s' },
    { value: 481, label: '20.0s' },
  ]
})

const attachmentSlots = computed(() => {
  const capability = activeCapability.value
  const result = []
  if (capability.input === 'image') {
    result.push({ slot: 'first', icon: '🖼', title: capability.needsLast ? '首帧' : '参考图片', accept: 'image/*' })
  }
  if (capability.needsLast) {
    result.push({ slot: 'last', icon: '🖼', title: '末帧', accept: 'image/*' })
  }
  if (capability.needsAudio) {
    result.push({ slot: 'audio', icon: '♫', title: '音频', accept: 'audio/*' })
  }
  if (capability.needsSource) {
    result.push({
      slot: 'source',
      icon: '🎞',
      title: '参考素材',
      accept: capability.sourceIsVideo ? 'video/*' : 'image/*,video/*',
    })
  }
  return result
})

function chooseGroup(groupId) {
  showView('generate')
  activeGroup.value = activeGroup.value === groupId ? null : groupId
}

function chooseMode(mode) {
  setMode(mode)
  activeGroup.value = groupOfMode(mode)
  showView('generate')
}

function openLibrary() {
  activeGroup.value = null
  showView('library')
}

function backToGenerate() {
  showView('generate')
  activeGroup.value = groupOfMode(draft.mode)
}

function attachmentMeta(slot) {
  const asset = attachments[slot]
  if (!asset) {
    if (slot === 'audio') return 'WAV / MP3 / M4A'
    if (slot === 'source') return activeCapability.value.sourceIsVideo ? '选择视频' : '图片 / 视频'
    return '选择图片'
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
    activeGroup.value = 'edit'
    showView('generate')
  }
}

function reuse(id) {
  reuseJob(id)
  activeGroup.value = groupOfMode(draft.mode)
  showView('generate')
}

function focus(id) {
  selectJob(id)
  showView('generate')
  activeGroup.value = groupOfMode(draft.mode)
}

function onUpscaleChange() {
  setMode(draft.mode)
  if (!draft.upscale) draft.upscaleMethod = 'latent'
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
    // Clipboard may be blocked; the session id remains visible in the menu.
  }
}

function closeMenus() {
  settingsOpen.value = false
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
  window.addEventListener('online', handleOnline)
  window.addEventListener('pagehide', handlePageHide)
  await start()
})

onBeforeUnmount(() => {
  document.removeEventListener('click', closeMenus)
  window.removeEventListener('online', handleOnline)
  window.removeEventListener('pagehide', handlePageHide)
  dispose()
})
</script>

<template>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark">LTX</span>
        <span class="brand-sub">Generate</span>
      </div>

      <div class="header-actions">
        <button
          class="health"
          type="button"
          :data-state="health ? 'online' : 'offline'"
          @click="warmGpu"
        >{{ healthText }}</button>

        <div class="menu-wrap" @click.stop>
          <button
            class="ghost icon"
            type="button"
            title="更多"
            aria-label="更多"
            @click="moreOpen = !moreOpen"
          >···</button>
          <div v-if="moreOpen" class="popover">
            <button class="popover-item" type="button" @click="refreshAll">刷新队列</button>
            <button class="popover-item" type="button" @click="copySession">复制会话 ID</button>
            <span class="popover-note">SESSION {{ sessionNumber || '—' }}</span>
          </div>
        </div>
      </div>
    </header>

    <section class="workspace">
      <StudioRail
        :mode="draft.mode"
        :active-group="activeGroup"
        :view="view"
        @group="chooseGroup"
        @mode="chooseMode"
        @library="openLibrary"
      />

      <div class="stage-column">
        <div v-if="view === 'generate'" class="generate-view">
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

          <StudioTakes
            :takes="takes"
            :selected-id="selectedJobId"
            @select="selectJob"
          />

          <form class="composer" @submit.prevent="submit">
            <div v-if="attachmentSlots.length" class="attachments">
              <template v-for="(item, index) in attachmentSlots" :key="item.slot">
                <span v-if="index > 0 && activeCapability.needsLast" class="attach-arrow" aria-hidden="true">→</span>
                <label
                  class="attach"
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
                  <span class="attach-icon" aria-hidden="true">{{ item.icon }}</span>
                  <span class="attach-text">
                    <strong>{{ attachmentTitle(item.slot, item.title) }}</strong>
                    <small>{{ busy.uploading[item.slot] ? '上传中…' : attachmentMeta(item.slot) }}</small>
                  </span>
                  <button
                    v-if="attachments[item.slot]"
                    class="attach-clear"
                    type="button"
                    aria-label="移除"
                    @click.prevent.stop="detach(item.slot)"
                  >×</button>
                </label>
              </template>
            </div>

            <div v-if="activeCapability.rangeFields?.length" class="mode-options">
              <label v-if="activeCapability.rangeFields.includes('retakeStart')" class="mode-field">
                <span>重拍起点 (s)</span>
                <input v-model.number="draft.range.retakeStart" type="number" min="0" step="0.1" />
              </label>
              <label v-if="activeCapability.rangeFields.includes('retakeEnd')" class="mode-field">
                <span>重拍终点 (s)</span>
                <input v-model.number="draft.range.retakeEnd" type="number" min="0.1" step="0.1" />
              </label>
              <label v-if="activeCapability.rangeFields.includes('extendDirection')" class="mode-field">
                <span>延长方向</span>
                <select v-model="draft.range.extendDirection">
                  <option value="end">向后</option>
                  <option value="start">向前</option>
                </select>
              </label>
              <label v-if="activeCapability.rangeFields.includes('extendSeconds')" class="mode-field">
                <span>延长时长 (s)</span>
                <input v-model.number="draft.range.extendSeconds" type="number" min="1" max="20" step="0.5" />
              </label>
              <label v-if="activeCapability.rangeFields.includes('extendContext')" class="mode-field">
                <span>上下文 (s)</span>
                <input v-model.number="draft.range.extendContext" type="number" min="0.5" max="20" step="0.5" />
              </label>
              <label v-if="activeCapability.rangeFields.includes('strength')" class="mode-field">
                <span>参考强度</span>
                <input v-model.number="draft.range.strength" type="number" min="0.1" max="1" step="0.05" />
              </label>
              <label v-if="activeCapability.rangeFields.includes('framePosition')" class="mode-field">
                <span>取帧位置</span>
                <select v-model="draft.range.framePosition">
                  <option value="last">末帧</option>
                  <option value="center">中间帧</option>
                </select>
              </label>
            </div>

            <div class="dock-main">
              <textarea
                v-model="draft.prompt"
                rows="2"
                required
                placeholder="描述你的镜头：主体、动作、镜头语言、光线、氛围…"
              />
              <button class="primary" type="submit" :disabled="!canSubmit">
                生成 <span aria-hidden="true">↑</span>
              </button>
            </div>

            <div class="dock-bar">
              <span class="dock-mode">{{ modeTag(draft.mode) }}</span>

              <div class="quick-params">
                <label class="chip-field">
                  <select v-model="draft.size" aria-label="分辨率">
                    <option value="768x512">16:9</option>
                    <option value="704x480">3:2</option>
                    <option value="640x384">5:3</option>
                    <option value="512x320">8:5</option>
                  </select>
                </label>

                <label class="chip-field">
                  <select v-model.number="draft.numFrames" aria-label="时长">
                    <option v-for="item in frameOptions" :key="item.value" :value="item.value">{{ item.label }}</option>
                  </select>
                </label>

                <label class="chip-field">
                  <select v-model.number="draft.fps" aria-label="帧率">
                    <option :value="16">16 fps</option>
                    <option :value="24">24 fps</option>
                    <option :value="30">30 fps</option>
                  </select>
                </label>

                <label class="chip-field">
                  <select v-model.number="draft.steps" aria-label="步数">
                    <option :value="4">4 steps</option>
                    <option :value="8">8 steps</option>
                    <option :value="16">16 steps</option>
                  </select>
                </label>

                <div class="menu-wrap" @click.stop>
                  <button
                    class="chip"
                    :class="{ active: settingsOpen }"
                    type="button"
                    title="生成设置"
                    aria-label="生成设置"
                    @click="settingsOpen = !settingsOpen"
                  >⚙</button>

                  <div v-if="settingsOpen" class="popover popover-settings">
                    <span class="popover-title">生成设置</span>
                    <div class="fields">
                      <label class="field">
                        <span>Frames</span>
                        <input v-model.number="draft.numFrames" type="number" min="9" max="481" step="8" />
                      </label>
                      <label class="field">
                        <span>Steps</span>
                        <input v-model.number="draft.steps" type="number" min="1" max="100" />
                      </label>
                      <label class="field">
                        <span>Guidance</span>
                        <input v-model.number="draft.guidanceScale" type="number" min="0" max="20" step="0.1" />
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
                        <input v-model.number="draft.fps" type="number" min="8" max="60" />
                      </label>
                    </div>

                    <span class="popover-title">放大</span>
                    <label class="check">
                      <input
                        v-model="draft.upscale"
                        type="checkbox"
                        :disabled="!activeCapability.supportsUpscale"
                        @change="onUpscaleChange"
                      />
                      <span>2× 空间放大</span>
                    </label>
                    <label class="field">
                      <span>放大方式</span>
                      <select
                        v-model="draft.upscaleMethod"
                        :disabled="!draft.upscale || !activeCapability.supportsPixelUpscale"
                      >
                        <option value="latent">Latent</option>
                        <option value="pixel">Pixel IC-LoRA</option>
                      </select>
                    </label>

                    <span class="popover-title">高级</span>
                    <label class="field">
                      <span>Negative Prompt</span>
                      <input v-model="draft.negativePrompt" />
                    </label>
                    <div class="fields">
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
                          placeholder="default"
                          @input="setOptionalNumber('audioGuidanceScale', $event.target.value)"
                        />
                      </label>
                      <label class="field">
                        <span>Decoder</span>
                        <select v-model="draft.decoder" :disabled="Boolean(activeCapability.forcesDecoder)">
                          <option value="vae">VAE</option>
                          <option value="diffusion">Diffusion</option>
                        </select>
                      </label>
                    </div>

                    <div class="lora-row">
                      <label class="field">
                        <span>LoRA</span>
                        <select v-model="draft.loraId">
                          <option :value="null">不使用</option>
                          <option v-for="item in loras" :key="item.id" :value="item.id">
                            {{ item.name || item.id }}
                          </option>
                        </select>
                      </label>
                      <label class="field">
                        <span>Strength</span>
                        <input v-model.number="draft.loraStrength" type="number" min="-2" max="2" step="0.1" />
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <p class="message" :data-state="notice.state">{{ notice.text }}</p>
          </form>
        </div>

        <StudioLibrary
          v-else
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
      </div>

      <StudioQueue
        :queue="queue"
        :collapsed="queueCollapsed"
        @toggle="toggleQueue"
        @cancel="cancelJob"
      />
    </section>
  </div>
</template>
