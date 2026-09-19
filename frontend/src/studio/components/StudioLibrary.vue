<script setup>
import { computed } from 'vue'

import { isPending, statusLabel } from '../model/jobs.js'
import { modeLabel } from '../model/modes.js'
import { clipDuration, describePerformance, selectJobActions } from '../model/selectors.js'

const props = defineProps({
  jobs: { type: Array, default: () => [] },
  counts: { type: Object, default: () => ({ all: 0, active: 0, completed: 0, failed: 0 }) },
  availableModes: { type: Array, default: () => [] },
  filter: { type: Object, required: true },
  sortOrder: { type: String, default: 'newest' },
})

const emit = defineEmits(['back', 'filter', 'sort', 'select', 'reuse', 'remove'])
const visibleJobs = computed(() => props.jobs.filter(job => !isPending(job)))

function meta(job) {
  return [...describePerformance(job), clipDuration(job)].filter(Boolean).join(' · ') || job.id.slice(0, 8)
}
</script>

<template>
  <div class="library-view">
    <div class="library-head">
      <span>HISTORY</span>
      <button class="ghost" type="button" @click="emit('back')">Back to Generate</button>
    </div>

    <div class="library-filters">
      <div class="segmented" role="group" aria-label="按状态筛选">
        <button
          v-for="item in [
            ['all', '全部'],
            ['active', '进行中'],
            ['completed', '已完成'],
            ['failed', '失败'],
          ]"
          :key="item[0]"
          type="button"
          class="seg"
          :class="{ active: props.filter.status === item[0] }"
          @click="emit('filter', { status: item[0] })"
        >
          {{ item[1] }} <b>{{ props.counts[item[0]] ?? 0 }}</b>
        </button>
      </div>

      <div class="library-controls">
        <select
          aria-label="按模式筛选"
          :value="props.filter.mode"
          @change="emit('filter', { mode: $event.target.value })"
        >
          <option value="all">全部模式</option>
          <option v-for="mode in props.availableModes" :key="mode" :value="mode">{{ modeLabel(mode) }}</option>
        </select>
        <input
          type="search"
          placeholder="搜索 prompt 或 ID"
          aria-label="搜索"
          :value="props.filter.query"
          @input="emit('filter', { query: $event.target.value })"
        />
        <select
          aria-label="排序"
          :value="props.sortOrder"
          @change="emit('sort', $event.target.value)"
        >
          <option value="newest">最新优先</option>
          <option value="oldest">最早优先</option>
        </select>
      </div>
    </div>

    <div class="gallery">
      <div v-if="!visibleJobs.length" class="empty">
        {{ props.jobs.length ? '没有匹配的任务' : '还没有记录。' }}
      </div>

      <figure
        v-for="job in visibleJobs"
        :key="job.id"
        class="tile"
        :class="'tile-' + job.status"
      >
        <div class="tile-media">
          <video v-if="job.video_url" :src="job.video_url" preload="metadata" muted playsinline />
          <img v-else-if="job.image_url" :src="job.image_url" alt="" loading="lazy" />
          <div v-else class="thumb-blank">{{ modeLabel(job.request?.mode) }}</div>
        </div>
        <figcaption>
          <div class="tile-head">
            <span class="mode-chip">{{ modeLabel(job.request?.mode) }}</span>
            <small class="tile-status">{{ statusLabel(job) }}</small>
          </div>
          <p>{{ job.request?.prompt || '' }}</p>
          <small class="tile-meta">{{ meta(job) }}</small>
        </figcaption>
        <div class="tile-actions">
          <button class="job-action" type="button" @click="emit('select', job.id)">查看</button>
          <button
            v-if="selectJobActions(job).canReuse"
            class="job-action"
            type="button"
            @click="emit('reuse', job.id)"
          >复用</button>
          <a
            v-if="selectJobActions(job).canDownload"
            class="job-action"
            :href="job.video_url || job.image_url"
            download
          >下载</a>
          <button
            v-if="selectJobActions(job).canDelete"
            class="job-action danger"
            type="button"
            @click="emit('remove', job.id)"
          >删除</button>
        </div>
      </figure>
    </div>
  </div>
</template>
