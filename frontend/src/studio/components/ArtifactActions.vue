<script setup>
defineProps({
  actions: { type: Object, default: () => ({}) },
  job: { type: Object, required: true },
  preparing: { type: Boolean, default: false },
})
const emit = defineEmits(['derive', 'reuse', 'remove'])
</script>

<template>
  <div class="artifact-actions">
    <button v-if="actions.canEditImage" class="stage-act primary" type="button" :disabled="preparing" @click="emit('derive', 'image_edit')">Edit</button>
    <button v-if="actions.canAnimateImage" class="stage-act primary" type="button" :disabled="preparing" @click="emit('derive', 'i2v')">Animate</button>
    <button v-if="actions.canRefineImage" class="stage-act" type="button" :disabled="preparing" @click="emit('derive', 'refine_image')">Refine</button>
    <button v-if="actions.canRetakeVideo" class="stage-act" type="button" :disabled="preparing" @click="emit('derive', 'retake')">{{ preparing ? 'Preparing…' : 'Retake' }}</button>
    <button v-if="actions.canExtendVideo" class="stage-act" type="button" :disabled="preparing" @click="emit('derive', 'extend')">Extend</button>
    <a v-if="actions.canDownload" class="stage-act" :href="job.download_url || job.video_url || job.image_url || job.audio_url" download>Download</a>
    <a v-if="job.hdr_exr_url" class="stage-act" :href="job.hdr_exr_download_url || job.hdr_exr_url" download>EXR ZIP</a>
    <button v-if="actions.canReuse" class="stage-act" type="button" @click="emit('reuse', job.id)">Reuse</button>
    <button v-if="actions.canDelete" class="stage-act danger" type="button" @click="emit('remove', job.id)">Delete</button>
  </div>
</template>
