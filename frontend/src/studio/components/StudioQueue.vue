<script setup>
import { isPending } from '../model/jobs.js'
import { modeLabel } from '../model/modes.js'
import { describeJob } from '../model/selectors.js'

const props = defineProps({
  queue: {
    type: Object,
    default: () => ({ running: [], upcoming: [] }),
  },
  collapsed: { type: Boolean, default: false },
})

const emit = defineEmits(['toggle', 'cancel'])
const percent = job => Math.round((job?.progress || 0) * 100)
</script>

<template>
  <aside class="queue-panel" :class="{ collapsed: props.collapsed }">
    <div class="panel-head">
      <span>QUEUE</span>
      <div class="head-actions">
        <small>{{ props.queue.running.length + props.queue.upcoming.length }}</small>
        <button
          class="ghost icon"
          type="button"
          :title="props.collapsed ? '展开队列' : '折叠队列'"
          :aria-label="props.collapsed ? '展开队列' : '折叠队列'"
          @click="emit('toggle')"
        >{{ props.collapsed ? '‹' : '›' }}</button>
      </div>
    </div>

    <div class="queue-body">
      <div class="queue">
        <div v-if="!props.queue.running.length && !props.queue.upcoming.length" class="empty">队列空闲</div>

        <article
          v-for="job in props.queue.running"
          :key="job.id"
          class="qcard qcard-running"
        >
          <div class="qcard-head">
            <span class="dot" />
            <span>生成中</span>
            <small>{{ percent(job) }}%</small>
          </div>
          <p class="qcard-title">{{ job.request?.prompt || '' }}</p>
          <small class="qcard-spec">{{ describeJob(job) }}</small>
          <div class="progress">
            <div class="progress-bar" :style="{ width: percent(job) + '%' }" />
          </div>
          <div class="qcard-actions">
            <button class="job-action danger" type="button" @click="emit('cancel', job.id)">取消</button>
          </div>
        </article>

        <template v-if="props.queue.upcoming.length">
          <div class="queue-subhead">UP NEXT · {{ props.queue.upcoming.length }}</div>
          <div class="queue-rows">
            <div
              v-for="(job, index) in props.queue.upcoming"
              :key="job.id"
              class="qrow"
              :class="{ 'qrow-pending': isPending(job) }"
            >
              <span class="qrow-index">{{ index + 1 }}</span>
              <span class="qrow-title" :title="job.request?.prompt || ''">{{ job.request?.prompt || '' }}</span>
              <span class="mode-chip">{{ modeLabel(job.request?.mode) }}</span>
              <small v-if="isPending(job)">提交中</small>
              <button
                v-else
                class="qrow-cancel"
                type="button"
                title="取消"
                @click="emit('cancel', job.id)"
              >×</button>
            </div>
          </div>
        </template>
      </div>
    </div>
  </aside>
</template>
