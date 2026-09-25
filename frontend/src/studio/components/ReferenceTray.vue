<script setup>
defineProps({
  slots: { type: Array, default: () => [] },
  attachments: { type: Object, required: true },
  uploading: { type: Object, default: () => ({}) },
})
const emit = defineEmits(['file', 'remove'])

function titleFor(item, attachments) {
  return attachments[item.slot]?.filename || item.title
}
function isPrimary(item) {
  return item.title?.toLowerCase().startsWith('primary')
}
function detailFor(item, attachments, uploading) {
  if (uploading[item.slot]) return 'Uploading…'
  const asset = attachments[item.slot]
  if (!asset) return item.slot === 'audio' ? 'WAV / MP3 / M4A' : 'Choose media'
  const size = Number(asset.size || 0)
  const sizeLabel = size ? (size / 1024 / 1024).toFixed(1) + ' MB' : ''
  return [asset.kind, sizeLabel].filter(Boolean).join(' · ')
}
function onChange(item, event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (file) emit('file', item.slot, file)
}
</script>

<template>
  <div class="reference-tray" :class="{ empty: !slots.length }">
    <span v-if="!slots.length" class="reference-placeholder">Prompt-only workflow</span>
    <template v-else>
      <label
        v-for="item in slots"
        :key="item.slot"
        class="reference-card"
        :class="{ ready: Boolean(attachments[item.slot]), primary: isPrimary(item) }"
      >
        <input type="file" :accept="item.accept" hidden @change="onChange(item, $event)" />
        <span class="reference-card-type">{{ isPrimary(item) ? 'PRIMARY' : item.icon }}</span>
        <span class="reference-card-copy">
          <strong>{{ titleFor(item, attachments) }}</strong>
          <small>{{ detailFor(item, attachments, uploading) }}</small>
        </span>
        <button
          v-if="attachments[item.slot]"
          type="button"
          class="reference-card-remove"
          aria-label="Remove attachment"
          @click.prevent.stop="emit('remove', item.slot)"
        >×</button>
      </label>
    </template>
  </div>
</template>
