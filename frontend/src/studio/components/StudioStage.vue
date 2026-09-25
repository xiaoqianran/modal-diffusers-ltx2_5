<script setup>
import { ref } from 'vue'

import { JOB_STATUS, isPending } from '../model/jobs.js'
import { isStillMode } from '../model/modes.js'
import { describeDirectorPlan, describeJob, describePerformance } from '../model/selectors.js'
import ArtifactActions from './ArtifactActions.vue'

const props = defineProps({
  job: { type: Object, default: null },
  actions: { type: Object, default: () => ({}) },
  queuePosition: { type: Number, default: 0 },
  preparing: { type: Boolean, default: false },
})

const emit = defineEmits(['reuse', 'remove', 'derive', 'drop'])
const infoOpen = ref(false)

const percent = job => Math.round((job?.progress || 0) * 100)

function generationLabel(job) {
  return isStillMode(job?.request?.mode) ? 'Generating image' : 'Generating video'
}

function generationDetail(job) {
  const request = job?.request || {}
  if (isStillMode(request.mode)) {
    return request.width && request.height ? `${request.width}×${request.height}` : 'LTX-2.5'
  }

  const parts = []
  if (request.num_frames) parts.push(`${request.num_frames} frames`)
  if (request.fps) parts.push(`${request.fps} fps`)
  return parts.join(' · ') || 'LTX-2.5'
}

function onDrop(event) {
  const file = event.dataTransfer?.files?.[0]
  if (file) emit('drop', file)
}
</script>

<template>
  <section class="stage" aria-live="polite" @dragover.prevent @drop.prevent="onDrop">
    <div class="canvas">
      <div class="canvas-meta">
        <span>{{ props.job ? describeJob(props.job) : 'LTX 2.5' }}</span>
        <span v-if="props.job && describeDirectorPlan(props.job)" class="resolved-model-badge">
          {{ describeDirectorPlan(props.job) }}
        </span>
      </div>

      <div class="viewport-slot">
        <div v-if="!props.job" class="viewport viewport-empty">
          <div class="stage-state">
            <span class="stage-empty-kicker">CREATE</span>
            <p class="stage-hint-title">Start with an idea</p>
            <small>Write a prompt below, or drop image / video media here.</small>
          </div>
        </div>

        <div
          v-else
          class="viewport"
          :class="{ 'viewport-image': Boolean(props.job.image_url) }"
        >
          <video
            v-if="props.job.video_url"
            class="stage-media"
            controls
            preload="metadata"
            :src="props.job.video_url"
          />
          <img
            v-else-if="props.job.image_url"
            class="stage-media"
            :src="props.job.image_url"
            alt="生成结果"
            loading="lazy"
          />
          <audio
            v-else-if="props.job.audio_url"
            class="stage-media stage-audio"
            controls
            preload="metadata"
            :src="props.job.audio_url"
          />
          <div v-else-if="isPending(props.job)" class="stage-state">
            <p>正在提交</p>
            <small>任务已加入队列</small>
          </div>
          <div v-else-if="props.job.status === JOB_STATUS.QUEUED" class="stage-state">
            <p>排队中</p>
            <small>队列位置 #{{ props.queuePosition }}</small>
          </div>
          <template v-else-if="props.job.status === JOB_STATUS.RUNNING">
            <div class="canvas-run-head" aria-hidden="true">
              <span class="run-status">
                <i class="run-dot" />
                Generating
              </span>
              <span class="run-percent-small">{{ percent(props.job) }}%</span>
            </div>

            <div class="stage-state generation-hud">
              <strong class="stage-percent">{{ percent(props.job) }}%</strong>
              <p>{{ generationLabel(props.job) }}</p>
              <small>{{ generationDetail(props.job) }}</small>
            </div>

            <div class="canvas-progress-rail" aria-hidden="true">
              <div
                class="canvas-progress-fill"
                :style="{ width: percent(props.job) + '%' }"
              />
            </div>
          </template>
          <div v-else-if="props.job.status === JOB_STATUS.FAILED" class="stage-state stage-failed">
            <p>生成失败</p>
            <small>{{ props.job.error || '' }}</small>
          </div>
          <div v-else class="stage-state">
            <p>已取消</p>
          </div>
        </div>
      </div>

      <div v-if="props.job" class="canvas-foot">
        <p class="canvas-prompt" :title="props.job.request?.prompt || ''">
          {{ props.job.request?.prompt || '' }}
        </p>
        <div class="canvas-actions">
          <ArtifactActions
            :job="props.job"
            :actions="props.actions"
            :preparing="props.preparing"
            @derive="mode => emit('derive', mode)"
            @reuse="id => emit('reuse', id)"
            @remove="id => emit('remove', id)"
          />

          <div class="menu-wrap" @click.stop>
            <button class="stage-act" type="button" title="详细信息" @click="infoOpen = !infoOpen">···</button>
            <div v-if="infoOpen" class="popover">
              <span class="popover-note">{{ describeJob(props.job) }}</span>
              <span v-if="describeDirectorPlan(props.job)" class="popover-note">{{ describeDirectorPlan(props.job) }}</span>
              <span class="popover-note">{{ isPending(props.job) ? '提交中' : 'ID ' + props.job.id }}</span>
              <span
                v-for="item in describePerformance(props.job)"
                :key="item"
                class="popover-note"
              >{{ item }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
