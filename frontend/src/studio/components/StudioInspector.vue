<script setup>
import { computed } from 'vue'
import { modeLabel, modeTag } from '../model/modes.js'
import { engineLabel, expectedEngine } from '../model/workflows.js'

const props = defineProps({
  draft: { type: Object, required: true },
  capability: { type: Object, required: true },
  loras: { type: Array, default: () => [] },
  ratioOptions: { type: Array, default: () => [] },
  frameOptions: { type: Array, default: () => [] },
  keyframeSlots: { type: Array, default: () => [] },
})
const emit = defineEmits(['engine-change', 'upscale-change'])

const qwenActive = computed(() => (
  props.capability.qwenOnly
  || props.draft.engine === 'qwen'
  || (props.draft.mode === 't2i' && props.draft.engine === 'auto' && !props.draft.loraId)
))
const expected = computed(() => engineLabel(expectedEngine(props.draft)))
const isImageOutput = computed(() => props.capability.output === 'image')
const selectedRatio = computed(() => (
  props.ratioOptions.find(item => item.value === props.draft.size)
  || props.ratioOptions[0]
  || { label: props.draft.size, aspect: '16 / 9' }
))

function setOptionalNumber(key, raw) {
  props.draft[key] = raw === '' ? null : Number(raw)
}
</script>

<template>
  <aside class="studio-inspector">
    <div class="inspector-scroll">
      <section class="inspector-block inspector-identity">
        <span class="eyebrow">{{ modeTag(draft.mode) }}</span>
        <h2>{{ modeLabel(draft.mode) }}</h2>
      </section>

      <section class="inspector-block">
        <div class="inspector-heading">
          <span>Routing</span>
          <small>Director</small>
        </div>
        <label class="field">
          <span>Strategy</span>
          <select v-model="draft.engine" @change="emit('engine-change')">
            <option value="auto">Auto · Director</option>
            <option value="ltx" :disabled="capability.qwenOnly">Manual · LTX-2.5 NVFP4</option>
            <option value="qwen" :disabled="!['t2i', 'image_edit'].includes(draft.mode)">Manual · Qwen-Image 2.1 BF16</option>
          </select>
        </label>
        <div class="resolved-card">
          <span>{{ draft.engine === 'auto' ? 'Expected' : 'Engine' }}</span>
          <strong>{{ expected }}</strong>
          <small v-if="draft.engine === 'auto'">The completed job shows the authoritative Director route.</small>
        </div>
      </section>

      <section class="inspector-block">
        <div class="inspector-heading"><span>Output</span></div>
        <details class="ratio-picker">
          <summary class="ratio-picker-trigger">
            <span class="ratio-picker-label">
              <span class="ratio-shape" :style="{ aspectRatio: selectedRatio.aspect }" />
              <span><small>Aspect ratio</small><strong>{{ selectedRatio.label }}</strong></span>
            </span>
            <span class="ratio-picker-size">{{ draft.size }}</span>
            <span class="ratio-picker-chevron">⌄</span>
          </summary>
          <div class="ratio-popover">
            <button
              v-for="item in ratioOptions"
              :key="item.value"
              type="button"
              class="ratio-choice"
              :class="{ active: draft.size === item.value }"
              @click="draft.size = item.value"
            >
              <span class="ratio-shape" :style="{ aspectRatio: item.aspect }" />
              <span>{{ item.label }}</span>
            </button>
          </div>
        </details>

        <div v-if="!isImageOutput" class="compact-options">
          <label class="field">
            <span>Frames / duration</span>
            <select v-model.number="draft.numFrames" :disabled="capability.supportsAutoDuration && draft.autoDuration">
              <option v-for="item in frameOptions" :key="item.value" :value="item.value">{{ item.label }} · {{ item.detail }}</option>
            </select>
          </label>
          <label class="field">
            <span>FPS</span>
            <input v-model.number="draft.fps" type="number" min="8" max="60" />
          </label>
        </div>

        <template v-if="capability.supportsAutoDuration">
          <label class="switch-row">
            <span><strong>Auto duration</strong><small>Use the LTX-2.5 duration head.</small></span>
            <input v-model="draft.autoDuration" type="checkbox" />
          </label>
          <div v-if="draft.autoDuration" class="drawer-fields two-col">
            <label class="field"><span>Min seconds</span><input v-model.number="draft.minSeconds" type="number" min="1" max="19" step="0.5" /></label>
            <label class="field"><span>Max seconds</span><input v-model.number="draft.maxSeconds" type="number" min="2" max="20" step="0.5" /></label>
          </div>
        </template>
      </section>

      <section v-if="capability.rangeFields?.length" class="inspector-block">
        <div class="inspector-heading"><span>Workflow</span></div>
        <div class="drawer-fields two-col">
          <label v-if="capability.rangeFields.includes('retakeStart')" class="field">
            <span>Retake start</span>
            <input v-model.number="draft.range.retakeStart" type="number" min="0" step="0.1" />
          </label>
          <label v-if="capability.rangeFields.includes('retakeEnd')" class="field">
            <span>Retake end</span>
            <input v-model.number="draft.range.retakeEnd" type="number" min="0.1" step="0.1" />
          </label>
          <label v-if="capability.rangeFields.includes('extendDirection')" class="field">
            <span>Direction</span>
            <select v-model="draft.range.extendDirection"><option value="end">Forward</option><option value="start">Backward</option></select>
          </label>
          <label v-if="capability.rangeFields.includes('extendSeconds')" class="field">
            <span>Extend seconds</span>
            <input v-model.number="draft.range.extendSeconds" type="number" min="1" max="20" step="0.5" />
          </label>
          <label v-if="capability.rangeFields.includes('extendContext')" class="field">
            <span>Context seconds</span>
            <input v-model.number="draft.range.extendContext" type="number" min="0.5" max="20" step="0.5" />
          </label>
          <label v-if="capability.rangeFields.includes('strength')" class="field">
            <span>Reference strength</span>
            <input v-model.number="draft.range.strength" type="number" min="0.1" max="1" step="0.05" />
          </label>
          <label v-if="capability.rangeFields.includes('framePosition')" class="field">
            <span>Frame position</span>
            <select v-model="draft.range.framePosition"><option value="last">Last</option><option value="center">Center</option></select>
          </label>
        </div>
      </section>

      <details class="inspector-block inspector-advanced">
        <summary>Advanced</summary>

        <div class="inspector-subheading first">Generation</div>
        <div class="drawer-fields two-col">
          <label class="field"><span>Steps</span><input v-model.number="draft.steps" type="number" min="1" max="100" :disabled="capability.fixedSchedule" /></label>
          <label class="field"><span>Guidance</span><input v-model.number="draft.guidanceScale" type="number" min="0" max="20" step="0.1" :disabled="qwenActive || capability.fixedSchedule" /></label>
          <label class="field"><span>Seed</span><input v-model.number="draft.seed" type="number" min="0" /></label>
          <label class="field"><span>Batch</span><input v-model.number="draft.batch" type="number" min="1" max="20" /></label>
        </div>

        <div v-if="draft.mode === 'dfr'" class="drawer-fields two-col">
          <label class="field"><span>DFR spatial</span><select v-model.number="draft.dfrSpatialUpscalings"><option :value="1">1 · half → full</option><option :value="2">2 · quarter → full</option></select></label>
          <label class="field"><span>DFR temporal</span><select v-model.number="draft.dfrTemporalUpscalings"><option :value="0">0 · base FPS</option><option :value="1">1 · 2× FPS</option><option :value="2">2 · 4× FPS</option></select></label>
        </div>

        <label v-if="capability.supportsHdr" class="field">
          <span>HDR / EXR colour space</span>
          <select v-model="draft.hdrColorSpace"><option :value="null">SDR</option><option value="SRGB_LINEAR">Linear Rec.709 / sRGB</option><option value="ACESCG">ACEScg</option><option value="ACESCCT">ACEScct</option></select>
        </label>

        <div v-if="keyframeSlots.length" class="drawer-fields two-col">
          <label v-for="item in keyframeSlots" :key="item.index" class="field">
            <span>{{ item.label }} frame</span>
            <input v-model.number="draft.keyframeFrames[item.index]" type="number" min="0" :max="Math.max(0, Number(draft.numFrames || 1) - 1)" step="1" placeholder="Auto" />
          </label>
        </div>

        <div class="inspector-subheading">Processing</div>
        <label class="switch-row">
          <span><strong>2× spatial upscale</strong><small>Increase final spatial resolution.</small></span>
          <input v-model="draft.upscale" type="checkbox" :disabled="!capability.supportsUpscale" @change="emit('upscale-change')" />
        </label>
        <label class="field">
          <span>Upscale method</span>
          <select v-model="draft.upscaleMethod" :disabled="!draft.upscale || !capability.supportsPixelUpscale">
            <option value="latent">Latent</option><option value="pixel">Pixel IC-LoRA</option>
          </select>
        </label>

        <label v-if="!capability.fixedSchedule" class="field">
          <span>Negative prompt</span>
          <textarea v-model="draft.negativePrompt" rows="3" />
        </label>

        <div v-if="!capability.fixedSchedule" class="drawer-fields two-col">
          <label class="field">
            <span>Video modality</span>
            <input :value="draft.modalityScale ?? ''" type="number" min="0" max="15" step="0.1" placeholder="1.0" @input="setOptionalNumber('modalityScale', $event.target.value)" />
          </label>
          <label class="field">
            <span>Audio guidance</span>
            <input :value="draft.audioGuidanceScale ?? ''" type="number" min="0" max="15" step="0.1" placeholder="Default" @input="setOptionalNumber('audioGuidanceScale', $event.target.value)" />
          </label>
        </div>

        <div v-if="draft.mode === 't2a'" class="drawer-fields two-col">
          <label class="field"><span>Audio STG</span><input :value="draft.audioStgScale ?? ''" type="number" min="0" max="15" step="0.1" placeholder="Default" @input="setOptionalNumber('audioStgScale', $event.target.value)" /></label>
          <label class="field"><span>Audio rescale</span><input :value="draft.audioRescaleScale ?? ''" type="number" min="0" max="15" step="0.1" placeholder="Default" @input="setOptionalNumber('audioRescaleScale', $event.target.value)" /></label>
          <label class="field"><span>Audio skip step</span><input :value="draft.audioSkipStep ?? ''" type="number" min="0" max="100" step="1" placeholder="Default" @input="setOptionalNumber('audioSkipStep', $event.target.value)" /></label>
        </div>

        <template v-if="qwenActive">
          <div class="drawer-fields two-col">
            <label class="field"><span>Qwen True CFG</span><input v-model.number="draft.qwenTrueCfgScale" type="number" min="0" max="20" step="0.1" /></label>
            <label class="switch-row compact-switch"><span><strong>KV cache</strong><small>Reuse attention KV during denoising.</small></span><input v-model="draft.qwenUseKvCache" type="checkbox" /></label>
          </div>
          <label class="switch-row"><span><strong>Transparent RGBA</strong><small>Request a true alpha-channel PNG.</small></span><input v-model="draft.transparentBackground" type="checkbox" /></label>
        </template>

        <label class="field">
          <span>Decoder</span>
          <select v-model="draft.decoder" :disabled="Boolean(capability.forcesDecoder) || qwenActive">
            <option value="vae">VAE</option><option value="diffusion">Diffusion</option>
          </select>
        </label>

        <div class="inspector-subheading">LoRA</div>
        <label class="field">
          <span>Adapter</span>
          <select v-model="draft.loraId" :disabled="draft.engine === 'qwen'">
            <option :value="null">None</option>
            <option v-for="item in loras" :key="item.id" :value="item.id">{{ item.name || item.id }}</option>
          </select>
        </label>
        <label class="field"><span>Strength</span><input v-model.number="draft.loraStrength" type="number" min="-2" max="2" step="0.1" :disabled="draft.engine === 'qwen'" /></label>
      </details>
    </div>
  </aside>
</template>
