<script setup>
import { ref } from 'vue'

import { JOB_STATUS, isPending } from '../model/jobs.js'
import { describeJob, describePerformance } from '../model/selectors.js'

const props = defineProps({
  job: { type: Object, default: null },
  actions: { type: Object, default: () => ({}) },
  queuePosition: { type: Number, default: 0 },
  preparing: { type: Boolean, default: false },
})

const emit = defineEmits(['reuse', 'remove', 'derive', 'drop'])
const infoOpen = ref(false)

const percent = job => Math.round((job?.progress || 0) * 100)

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
      </div>

      <div v-if="!props.job" class="viewport viewport-empty">
        <div class="stage-state">
          <p class="stage-hint-title">准备就绪</p>
          <small>在下方描述一个镜头，或把图片 / 视频拖到这里开始。</small>
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
        <div v-else-if="isPending(props.job)" class="stage-state">
          <p>正在提交</p>
          <small>任务已加入队列</small>
        </div>
        <div v-else-if="props.job.status === JOB_STATUS.QUEUED" class="stage-state">
          <p>排队中</p>
          <small>队列位置 #{{ props.queuePosition }}</small>
        </div>
        <div v-else-if="props.job.status === JOB_STATUS.RUNNING" class="stage-state stage-running">
          <p>生成中</p>
          <strong class="stage-percent">{{ percent(props.job) }}%</strong>
          <div class="progress">
            <div class="progress-bar" :style="{ width: percent(props.job) + '%' }" />
          </div>
        </div>
        <div v-else-if="props.job.status === JOB_STATUS.FAILED" class="stage-state stage-failed">
          <p>生成失败</p>
          <small>{{ props.job.error || '' }}</small>
        </div>
        <div v-else class="stage-state">
          <p>已取消</p>
        </div>
      </div>

      <div v-if="props.job" class="canvas-foot">
        <p class="canvas-prompt" :title="props.job.request?.prompt || ''">
          {{ props.job.request?.prompt || '' }}
        </p>
        <div class="canvas-actions">
          <template v-if="props.actions.canDerive">
            <button
              class="stage-act"
              type="button"
              :disabled="props.preparing"
              @click="emit('derive', 'retake')"
            >
              {{ props.preparing ? '准备中…' : '重拍' }}
            </button>
            <button
              class="stage-act"
              type="button"
              :disabled="props.preparing"
              @click="emit('derive', 'extend')"
            >
              延长
            </button>
          </template>

          <a
            v-if="props.actions.canDownload"
            class="stage-act"
            :href="props.job.video_url || props.job.image_url"
            download
          >下载</a>
          <button
            v-if="props.actions.canReuse"
            class="stage-act"
            type="button"
            @click="emit('reuse', props.job.id)"
          >复用</button>
          <button
            v-if="props.actions.canDelete"
            class="stage-act danger"
            type="button"
            @click="emit('remove', props.job.id)"
          >删除</button>

          <div class="menu-wrap" @click.stop>
            <button class="stage-act" type="button" title="详细信息" @click="infoOpen = !infoOpen">···</button>
            <div v-if="infoOpen" class="popover">
              <span class="popover-note">{{ describeJob(props.job) }}</span>
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
