/**
 * Tests for the pure model layer.
 *
 * Run with:  node src/studio/model/model.test.js
 *
 * The model has no framework dependencies, so this needs no test runner. Keeping
 * it dependency-free means it can also run in CI without installing anything.
 */

import assert from 'node:assert/strict'
import {
  MODE_CAPABILITIES, MODE_GROUPS, DEFAULT_MODE, STILL_MODES, DERIVED_MODES,
  getModeCapabilities, modeLabel, modeTag, isStillMode, groupOfMode, modesInGroup,
} from './modes.js'
import {
  JOB_STATUS, ACTIVE_STATUSES, STATUS_FILTERS, isActive, isTerminal, hasMedia,
  isPending, statusLabel, makePendingJob, mergePendingJobs, buildQueueIndex,
} from './jobs.js'
import {
  selectStageJob, selectTakes, selectQueue, selectCounts, selectAvailableModes,
  filterJobs, sortJobs, selectLibraryJobs, describeJob, describePerformance,
  clipDuration, selectJobActions, takeLabel,
} from './selectors.js'
import {
  validateGenerationDraft, buildConditions, buildLoras, buildGenerationRequest,
  buildBatchRequests, parseSize,
} from './generationRequest.js'

let passed = 0
let failed = 0

function test(name, fn) {
  try {
    fn()
    passed += 1
    console.log(`  ok   ${name}`)
  } catch (error) {
    failed += 1
    console.log(`  FAIL ${name}`)
    console.log(`       ${error.message}`)
  }
}

function group(title) {
  console.log(`\n${title}`)
}

// ---------------------------------------------------------------- helpers

const job = (overrides = {}) => ({
  id: overrides.id || Math.random().toString(16).slice(2, 10),
  status: 'completed',
  progress: 1,
  error: null,
  video_url: '/outputs/a.mp4',
  image_url: null,
  request: { mode: 't2av', prompt: 'test', width: 768, height: 512, num_frames: 121, fps: 24, steps: 8, upscale: true },
  created_at: '2026-09-20T10:00:00.000Z',
  generation_seconds: 8.1,
  peak_vram_gb: 46.8,
  ...overrides,
})

const draft = (overrides = {}) => ({
  mode: 't2av',
  prompt: 'ocean sunset',
  negativePrompt: 'worst quality',
  width: 768, height: 512, numFrames: 121, fps: 24, steps: 8,
  guidanceScale: 3, seed: 42, batch: 1,
  upscale: true, upscaleMethod: 'latent', decoder: 'vae',
  modalityScale: null, audioGuidanceScale: null,
  loraId: null, loraStrength: 1,
  range: {},
  ...overrides,
})

const asset = (id, kind) => ({ id, kind, filename: `${id}.bin`, size: 1024 })

// ------------------------------------------------------------ modes.js

group('modes')

test('every mode has a label and a group', () => {
  for (const [mode, capability] of Object.entries(MODE_CAPABILITIES)) {
    assert.ok(capability.label, `${mode} missing label`)
    assert.ok(capability.group, `${mode} missing group`)
    assert.ok(['video', 'image'].includes(capability.output), `${mode} bad output`)
  }
})

test('every mode appears in exactly one rail group', () => {
  const seen = []
  for (const g of MODE_GROUPS) {
    for (const section of g.sections) {
      for (const mode of section.modes) {
        assert.ok(!seen.includes(mode), `${mode} listed twice`)
        seen.push(mode)
      }
    }
  }
  assert.deepEqual(seen.sort(), Object.keys(MODE_CAPABILITIES).sort())
})

test('unknown modes fall back to the default without throwing', () => {
  assert.equal(getModeCapabilities('nope').label, MODE_CAPABILITIES[DEFAULT_MODE].label)
})

test('still modes are exactly the image outputs', () => {
  assert.deepEqual(STILL_MODES.sort(), ['ref2i', 'refine_image', 't2i'].sort())
  assert.equal(isStillMode('t2i'), true)
  assert.equal(isStillMode('t2av'), false)
})

test('modeTag upper-cases', () => {
  assert.equal(modeTag('i2v'), 'I2V')
  assert.equal(modeTag(null), 'T2AV')
})

test('groupOfMode matches the rail definitions', () => {
  assert.equal(groupOfMode('i2v'), 'create')
  assert.equal(groupOfMode('retake'), 'edit')
  assert.equal(groupOfMode('iclora'), 'control')
})

test('modesInGroup returns members', () => {
  assert.deepEqual(modesInGroup('control').sort(), ['condition', 'iclora'])
})

test('only retake and extend require a video source', () => {
  const videoSources = Object.entries(MODE_CAPABILITIES)
    .filter(([, c]) => c.sourceIsVideo).map(([m]) => m).sort()
  assert.deepEqual(videoSources, ['extend', 'retake'])
})

test('only t2i force-enables upscale', () => {
  // Verified against schemas.py: t2i/refine_image are always two-stage, while
  // ref2i is explicitly single-stage at base resolution (upscale forced false).
  assert.equal(MODE_CAPABILITIES.t2i.forcesUpscale, true)
  assert.equal(MODE_CAPABILITIES.ref2i.forcesUpscale, false)
  assert.equal(MODE_CAPABILITIES.t2av.forcesUpscale, false)
})

test('modes that cannot upscale are marked as such', () => {
  const noUpscale = Object.entries(MODE_CAPABILITIES)
    .filter(([, c]) => !c.supportsUpscale).map(([m]) => m).sort()
  assert.deepEqual(noUpscale, ['extend', 'iclora', 'ref2i', 'retake'])
})

test('iclora, retake and extend pin the decoder to VAE', () => {
  for (const mode of ['iclora', 'retake', 'extend']) {
    assert.equal(MODE_CAPABILITIES[mode].forcesDecoder, 'vae', `${mode} should pin decoder`)
  }
  assert.equal(MODE_CAPABILITIES.t2av.forcesDecoder, undefined)
})

test('ref2i pins its frame count and whitelist', () => {
  assert.equal(MODE_CAPABILITIES.ref2i.fixedFrames, 49)
  assert.deepEqual(MODE_CAPABILITIES.ref2i.validFrames, [25, 41, 49])
})

// -------------------------------------------------------------- jobs.js

group('jobs')

test('status predicates', () => {
  assert.equal(isActive({ status: 'running' }), true)
  assert.equal(isActive({ status: 'completed' }), false)
  assert.equal(isTerminal({ status: 'failed' }), true)
  assert.equal(isPending({ status: 'submitting' }), true)
  assert.equal(isPending({ pending: true, status: 'queued' }), true)
})

test('hasMedia detects either output', () => {
  assert.equal(hasMedia({ video_url: '/x.mp4' }), true)
  assert.equal(hasMedia({ image_url: '/x.png' }), true)
  assert.equal(hasMedia({ video_url: null, image_url: null }), false)
})

test('statusLabel shows percent while running', () => {
  assert.equal(statusLabel({ status: 'running', progress: 0.71 }), '71%')
  assert.equal(statusLabel({ status: 'queued' }), '等待')
  assert.equal(statusLabel({ status: 'submitting' }), '提交中')
})

test('makePendingJob produces a server-shaped record', () => {
  const pending = makePendingJob('pending-1', { mode: 't2av' }, 12345, 1000)
  assert.equal(pending.id, 'pending-1')
  assert.equal(pending.status, JOB_STATUS.SUBMITTING)
  assert.equal(pending.pending, true)
  assert.equal(pending.session_number, 12345)
  // Must expose every key the selectors read.
  for (const key of ['progress', 'error', 'video_url', 'image_url', 'created_at', 'generation_seconds']) {
    assert.ok(key in pending, `missing ${key}`)
  }
})

test('mergePendingJobs renders placeholders in front of server records', () => {
  const entries = [{ token: 'pending-1', request: { mode: 't2av' }, sessionNumber: 1, createdAt: 5 }]
  const merged = mergePendingJobs(entries, [job()])
  assert.equal(merged.length, 2)
  assert.equal(merged[0].id, 'pending-1')
  assert.equal(merged[0].pending, true)
})

test('mergePendingJobs ordering is stable across several placeholders', () => {
  const entries = [
    { token: 'pending-1', request: { mode: 't2av' }, sessionNumber: 1, createdAt: 5 },
    { token: 'pending-2', request: { mode: 'i2v' }, sessionNumber: 1, createdAt: 6 },
  ]
  const merged = mergePendingJobs(entries, [job({ id: 'real' })])
  assert.deepEqual(merged.map(j => j.id), ['pending-1', 'pending-2', 'real'])
})

test('mergePendingJobs is a no-op without placeholders', () => {
  const server = [job()]
  assert.equal(mergePendingJobs([], server), server)
})

test('buildQueueIndex numbers waiting jobs oldest first', () => {
  const jobs = [
    job({ id: 'newest', status: 'queued', created_at: '2026-09-20T10:03:00Z' }),
    job({ id: 'oldest', status: 'queued', created_at: '2026-09-20T10:01:00Z' }),
    job({ id: 'done', status: 'completed' }),
  ]
  // Input order newest-first; the index reverses it so the oldest is #1.
  const index = buildQueueIndex(jobs)
  assert.equal(index.get('oldest'), 1)
  assert.equal(index.get('newest'), 2)
  assert.equal(index.has('done'), false)
})

// ---------------------------------------------------------- selectors.js

group('selectors')

test('selectStageJob returns null for an empty list', () => {
  assert.equal(selectStageJob([]), null)
})

test('selectStageJob prefers running over everything', () => {
  const jobs = [job({ id: 'done' }), job({ id: 'run', status: 'running', video_url: null })]
  assert.equal(selectStageJob(jobs).id, 'run')
})

test('selectStageJob honours an explicit selection', () => {
  const jobs = [job({ id: 'a' }), job({ id: 'b' }), job({ id: 'run', status: 'running' })]
  assert.equal(selectStageJob(jobs, 'a').id, 'a')
})

test('selectStageJob falls back to the oldest queued', () => {
  const jobs = [
    job({ id: 'q-new', status: 'queued', video_url: null }),
    job({ id: 'q-old', status: 'queued', video_url: null }),
  ]
  assert.equal(selectStageJob(jobs).id, 'q-old')
})

test('selectStageJob ignores a stale selection', () => {
  const jobs = [job({ id: 'a' }), job({ id: 'run', status: 'running' })]
  assert.equal(selectStageJob(jobs, 'gone').id, 'run')
})

test('selectTakes returns only rendered results, newest first', () => {
  // The API already returns newest first, so the newest take must stay first.
  const jobs = [
    job({ id: 'new', created_at: '2026-09-20T10:03:00Z' }),
    job({ id: 'run', status: 'running', video_url: null, image_url: null }),
    job({ id: 'old', created_at: '2026-09-20T10:01:00Z' }),
  ]
  assert.deepEqual(selectTakes(jobs).map(t => t.id), ['new', 'old'])
})

test('selectQueue splits running from upcoming without overlap', () => {
  const jobs = [
    job({ id: 'run', status: 'running' }),
    job({ id: 'q1', status: 'queued', video_url: null }),
    job({ id: 'q2', status: 'queued', video_url: null }),
    job({ id: 'pend', status: 'submitting', pending: true }),
    job({ id: 'done' }),
  ]
  const { running, upcoming } = selectQueue(jobs)
  assert.deepEqual(running.map(j => j.id), ['run'])
  // Queued first (oldest last in input becomes first), then pending.
  assert.deepEqual(upcoming.map(j => j.id).sort(), ['pend', 'q1', 'q2'])
  assert.equal(selectQueue(jobs).running.some(j => upcoming.includes(j)), false)
})

test('selectCounts matches the filter buckets', () => {
  const jobs = [
    job({ status: 'completed' }),
    job({ status: 'failed' }),
    job({ status: 'interrupted' }),
    job({ status: 'running' }),
    job({ status: 'queued' }),
  ]
  const counts = selectCounts(jobs)
  assert.equal(counts.all, 5)
  assert.equal(counts.active, 2)
  assert.equal(counts.completed, 1)
  // "failed" bucket intentionally includes interrupted.
  assert.equal(counts.failed, 2)
})

test('filterJobs by status bucket', () => {
  const jobs = [job({ id: 'a', status: 'completed' }), job({ id: 'b', status: 'failed' })]
  assert.deepEqual(filterJobs(jobs, { status: 'completed' }).map(j => j.id), ['a'])
  assert.deepEqual(filterJobs(jobs, { status: 'failed' }).map(j => j.id), ['b'])
  assert.equal(filterJobs(jobs, { status: 'all' }).length, 2)
})

test('filterJobs by mode and query', () => {
  const jobs = [
    job({ id: 'a', request: { mode: 't2av', prompt: 'ocean sunset' } }),
    job({ id: 'b', request: { mode: 'i2v', prompt: 'lighthouse' } }),
  ]
  assert.deepEqual(filterJobs(jobs, { mode: 'i2v' }).map(j => j.id), ['b'])
  assert.deepEqual(filterJobs(jobs, { query: 'OCEAN' }).map(j => j.id), ['a'])
  assert.deepEqual(filterJobs(jobs, { query: 'lighthouse' }).map(j => j.id), ['b'])
  assert.equal(filterJobs(jobs, { query: 'nothing' }).length, 0)
})

test('filterJobs query also matches id', () => {
  const jobs = [job({ id: 'deadbeef' })]
  assert.equal(filterJobs(jobs, { query: 'dead' }).length, 1)
})

test('sortJobs floats active work to the top in both orders', () => {
  const jobs = [
    job({ id: 'done-new', created_at: '2026-09-20T10:05:00Z' }),
    job({ id: 'run', status: 'running', created_at: '2026-09-20T10:01:00Z' }),
    job({ id: 'done-old', created_at: '2026-09-20T10:00:00Z' }),
  ]
  assert.equal(sortJobs(jobs, 'newest')[0].id, 'run')
  assert.equal(sortJobs(jobs, 'oldest')[0].id, 'run')
  const newest = sortJobs(jobs, 'newest').map(j => j.id)
  assert.deepEqual(newest, ['run', 'done-new', 'done-old'])
})

test('sortJobs does not mutate its input', () => {
  const jobs = [job({ id: 'a' }), job({ id: 'b' })]
  const before = jobs.map(j => j.id)
  sortJobs(jobs, 'oldest')
  assert.deepEqual(jobs.map(j => j.id), before)
})

test('selectLibraryJobs composes filter and sort', () => {
  const jobs = [
    job({ id: 'a', request: { mode: 't2av', prompt: 'ocean' } }),
    job({ id: 'b', request: { mode: 'i2v', prompt: 'ocean' } }),
  ]
  assert.deepEqual(selectLibraryJobs(jobs, { mode: 't2av' }, 'newest').map(j => j.id), ['a'])
})

test('selectAvailableModes lists distinct modes, sorted', () => {
  const jobs = [
    job({ request: { mode: 't2av' } }),
    job({ request: { mode: 'i2v' } }),
    job({ request: { mode: 't2av' } }),
  ]
  assert.deepEqual(selectAvailableModes(jobs), ['i2v', 't2av'])
})

test('describeJob renders a compact spec and doubles upscaled dimensions', () => {
  const spec = describeJob(job())
  assert.match(spec, /^T2AV/)
  assert.match(spec, /1536×1024/)
  assert.match(spec, /121f/)
  assert.match(spec, /24fps/)
})

test('describeJob omits upscale scaling when disabled', () => {
  const plain = job({ request: { mode: 't2av', width: 768, height: 512, upscale: false } })
  assert.match(describeJob(plain), /768×512/)
})

test('describePerformance returns both metrics', () => {
  assert.deepEqual(describePerformance(job()), ['8.10s', '46.8 GiB peak'])
})

test('describePerformance is empty without metrics', () => {
  assert.deepEqual(describePerformance({ generation_seconds: null, peak_vram_gb: null }), [])
})

test('clipDuration derives seconds from frames and fps', () => {
  assert.equal(clipDuration(job()), '5.0s')
  assert.equal(clipDuration({ request: { num_frames: 0 } }), '')
})

test('selectJobActions gates retake on a finished video', () => {
  const done = selectJobActions(job())
  assert.equal(done.canDerive, true)
  assert.equal(done.canDownload, true)
  assert.equal(done.canInterrupt, false)

  const running = selectJobActions(job({ status: 'running', video_url: null }))
  assert.equal(running.canDerive, false)
  assert.equal(running.canInterrupt, true)

  const image = selectJobActions(job({ video_url: null, image_url: '/x.png', request: { mode: 't2i' } }))
  assert.equal(image.canDerive, false)
  assert.equal(image.canDownload, true)
})

test('selectJobActions offers nothing for a placeholder', () => {
  const actions = selectJobActions(makePendingJob('pending-1', {}, 1, 0))
  assert.deepEqual(Object.values(actions), [false, false, false, false, false, false])
})

test('takeLabel prefers duration then status', () => {
  assert.equal(takeLabel(job()), '8.1s')
  assert.equal(takeLabel({ generation_seconds: null, status: 'queued' }), '等待')
})

// --------------------------------------------------- generationRequest.js

group('generationRequest')

test('valid t2av draft passes', () => {
  assert.deepEqual(validateGenerationDraft(draft(), {}), [])
})

test('empty prompt is rejected', () => {
  const problems = validateGenerationDraft(draft({ prompt: '   ' }), {})
  assert.equal(problems.length, 1)
  assert.match(problems[0], /Prompt/)
})

test('i2v requires an image', () => {
  const problems = validateGenerationDraft(draft({ mode: 'i2v' }), {})
  assert.match(problems.join(' '), /参考图片/)
  assert.deepEqual(validateGenerationDraft(draft({ mode: 'i2v' }), { first: asset('a', 'image') }), [])
})

test('flf2v requires both frames', () => {
  const first = { first: asset('a', 'image') }
  assert.match(validateGenerationDraft(draft({ mode: 'flf2v' }), first).join(' '), /末帧/)
  const both = { ...first, last: asset('b', 'image') }
  assert.deepEqual(validateGenerationDraft(draft({ mode: 'flf2v' }), both), [])
})

test('a2v requires audio', () => {
  assert.match(validateGenerationDraft(draft({ mode: 'a2v' }), {}).join(' '), /音频/)
  assert.deepEqual(validateGenerationDraft(draft({ mode: 'a2v' }), { audio: asset('a', 'audio') }), [])
})

test('retake and extend require a video source', () => {
  const image = { source: asset('a', 'image') }
  assert.match(validateGenerationDraft(draft({ mode: 'retake', range: { retakeStart: 0, retakeEnd: 3 } }), image).join(' '), /必须是视频/)
  const video = { source: asset('a', 'video') }
  assert.deepEqual(validateGenerationDraft(draft({ mode: 'retake', range: { retakeStart: 0, retakeEnd: 3 } }), video), [])
})

test('retake range must be ordered', () => {
  const video = { source: asset('a', 'video') }
  const problems = validateGenerationDraft(
    draft({ mode: 'retake', range: { retakeStart: 5, retakeEnd: 2 } }), video
  )
  assert.match(problems.join(' '), /起点必须小于终点/)
})

test('iclora requires a compatible LoRA', () => {
  const source = { source: asset('a', 'video') }
  const base = draft({ mode: 'iclora' })
  assert.match(validateGenerationDraft(base, source, []).join(' '), /需要选择一个 IC-LoRA/)

  const incompatible = [{ id: 'x', generic_iclora_compatible: false }]
  assert.match(validateGenerationDraft(draft({ mode: 'iclora', loraId: 'x' }), source, incompatible).join(' '), /不支持/)

  const compatible = [{ id: 'x', generic_iclora_compatible: true }]
  assert.deepEqual(validateGenerationDraft(draft({ mode: 'iclora', loraId: 'x' }), source, compatible), [])
})

test('upscale beyond the base limit is rejected', () => {
  const problems = validateGenerationDraft(draft({ width: 1280, height: 720 }), {})
  assert.match(problems.join(' '), /960×544/)
})

test('pixel upscale is rejected for modes that can upscale but not via pixel', () => {
  // refine_image supports latent upscale but not the pixel IC-LoRA path, so it
  // is the mode that actually exercises this rule.
  const problems = validateGenerationDraft(
    draft({ mode: 'refine_image', upscale: true, upscaleMethod: 'pixel' }),
    { first: asset('a', 'image') }
  )
  assert.match(problems.join(' '), /不支持 Pixel/)
})

test('pixel upscale is simply ignored for modes that cannot upscale', () => {
  // retake force-disables upscale, so the impossible combination never reaches
  // the user as an error - the request just sends latent.
  const body = buildGenerationRequest(
    draft({ mode: 'retake', upscale: true, upscaleMethod: 'pixel', range: { retakeStart: 0, retakeEnd: 3 } }),
    { source: asset('s', 'video') }, 1
  )
  assert.equal(body.upscale, false)
  assert.equal(body.upscale_method, 'latent')
})

test('buildConditions for i2v uses index 0', () => {
  const conditions = buildConditions(draft({ mode: 'i2v' }), { first: asset('a', 'image') })
  assert.deepEqual(conditions, [{ asset_id: 'a', kind: 'image', index: 0, strength: 1 }])
})

test('buildConditions for flf2v emits first and last at 0 and -1', () => {
  const conditions = buildConditions(draft({ mode: 'flf2v' }), {
    first: asset('a', 'image'), last: asset('b', 'image'),
  })
  const indices = conditions.map(c => c.index).sort((x, y) => x - y)
  assert.deepEqual(indices, [-1, 0])
})

test('buildConditions for iclora uses reference index 1', () => {
  const conditions = buildConditions(draft({ mode: 'iclora' }), { source: asset('s', 'video') })
  assert.equal(conditions[0].index, 1)
})

test('buildConditions for condition mode uses index 0', () => {
  const conditions = buildConditions(draft({ mode: 'condition' }), { source: asset('s', 'image') })
  assert.equal(conditions[0].index, 0)
})

test('buildConditions for retake always sends kind=video', () => {
  const conditions = buildConditions(draft({ mode: 'retake' }), { source: asset('s', 'video') })
  assert.equal(conditions[0].kind, 'video')
})

test('buildConditions for t2av sends nothing', () => {
  assert.deepEqual(buildConditions(draft(), {}), [])
})

test('buildLoras returns one entry with strength', () => {
  assert.deepEqual(buildLoras(draft({ loraId: 'x', loraStrength: 0.8 })), [{ id: 'x', strength: 0.8 }])
  assert.deepEqual(buildLoras(draft()), [])
})

test('buildGenerationRequest carries the seed offset for batches', () => {
  const first = buildGenerationRequest(draft({ seed: 10 }), {}, 99, 0)
  const third = buildGenerationRequest(draft({ seed: 10 }), {}, 99, 2)
  assert.equal(first.seed, 10)
  assert.equal(third.seed, 12)
  assert.equal(first.session_number, 99)
})

test('buildGenerationRequest forces upscale for t2i', () => {
  const body = buildGenerationRequest(draft({ mode: 't2i', upscale: false }), {}, 1)
  assert.equal(body.upscale, true)
  assert.equal(body.upscale_method, 'latent')
})

test('buildGenerationRequest pins ref2i frames and frame_position', () => {
  const body = buildGenerationRequest(
    draft({ mode: 'ref2i', numFrames: 121, range: { framePosition: 'center' } }),
    { first: asset('a', 'image') },
    1
  )
  assert.equal(body.num_frames, 49)
  assert.equal(body.frame_position, 'center')
})

test('buildGenerationRequest nulls the optional retake range', () => {
  // retake_start / retake_end are genuinely Optional on the backend, so null
  // means "not a retake" rather than a missing value.
  const body = buildGenerationRequest(draft(), {}, 1)
  assert.equal(body.retake_start, null)
  assert.equal(body.retake_end, null)
  assert.equal(body.audio_asset_id, null)
})

test('buildGenerationRequest sends defaults for non-nullable range fields', () => {
  // Regression: these fields reject null in schemas.py and carry defaults
  // instead, so sending null made every video request fail with 422.
  const body = buildGenerationRequest(draft(), {}, 1)
  assert.equal(body.extend_direction, 'end')
  assert.equal(body.extend_seconds, 5)
  assert.equal(body.extend_context_seconds, 3)
  assert.equal(body.strength, 1)
  assert.equal(body.frame_position, 'last')
})

test('no request field is null except the documented optionals', () => {
  const nullable = new Set([
    'negative_prompt', 'retake_start', 'retake_end',
    'modality_scale', 'audio_guidance_scale', 'audio_asset_id',
  ])
  for (const mode of Object.keys(MODE_CAPABILITIES)) {
    const body = buildGenerationRequest(draft({ mode }), {
      first: asset('a'.repeat(32), 'image'),
      last: asset('b'.repeat(32), 'image'),
      source: asset('c'.repeat(32), 'video'),
      audio: asset('d'.repeat(32), 'audio'),
    }, 1)
    for (const [key, value] of Object.entries(body)) {
      if (value === null) {
        assert.ok(nullable.has(key), `${mode}: ${key} must not be null`)
      }
    }
  }
})

test('buildGenerationRequest always sets the regenerate flags', () => {
  // retake requires at least one of these to be true, or the backend rejects it.
  const body = buildGenerationRequest(draft({ mode: 'retake', range: { retakeStart: 0, retakeEnd: 3 } }), { source: asset('s', 'video') }, 1)
  assert.equal(body.regenerate_video, true)
  assert.equal(body.regenerate_audio, true)
})

test('buildGenerationRequest fills retake range when relevant', () => {
  const body = buildGenerationRequest(
    draft({ mode: 'retake', range: { retakeStart: 1.5, retakeEnd: 4 } }),
    { source: asset('s', 'video') }, 1
  )
  assert.equal(body.retake_start, 1.5)
  assert.equal(body.retake_end, 4)
})

test('buildGenerationRequest blanks an empty negative prompt', () => {
  assert.equal(buildGenerationRequest(draft({ negativePrompt: '  ' }), {}, 1).negative_prompt, null)
})

test('buildGenerationRequest forces upscale off for ref2i', () => {
  // schemas.py sets upscale=False for ref2i regardless of the toggle.
  const body = buildGenerationRequest(
    draft({ mode: 'ref2i', upscale: true }), { first: asset('a', 'image') }, 1
  )
  assert.equal(body.upscale, false)
  assert.equal(body.upscale_method, 'latent')
})

test('buildGenerationRequest pins the decoder to VAE for editing modes', () => {
  const retake = buildGenerationRequest(
    draft({ mode: 'retake', decoder: 'diffusion', upscale: true, range: { retakeStart: 0, retakeEnd: 3 } }),
    { source: asset('s', 'video') }, 1
  )
  assert.equal(retake.decoder, 'vae')
  assert.equal(retake.upscale, false)

  const iclora = buildGenerationRequest(
    draft({ mode: 'iclora', decoder: 'diffusion', loraId: 'l' }),
    { source: asset('s', 'video') }, 1
  )
  assert.equal(iclora.decoder, 'vae')
})

test('ref2i rejects a frame count outside its whitelist', () => {
  const problems = validateGenerationDraft(
    draft({ mode: 'ref2i', numFrames: 121 }), { first: asset('a', 'image') }
  )
  assert.match(problems.join(' '), /25 \/ 41 \/ 49/)
})

test('diffusion decoder without any upscale stage is rejected', () => {
  const problems = validateGenerationDraft(draft({ decoder: 'diffusion', upscale: false }), {})
  assert.match(problems.join(' '), /Diffusion 解码器需要开启/)
  // With upscale on, it is allowed.
  assert.deepEqual(validateGenerationDraft(draft({ decoder: 'diffusion', upscale: true }), {}), [])
})

test('diffusion decoder is fine for modes that always upscale', () => {
  assert.deepEqual(validateGenerationDraft(draft({ mode: 't2i', decoder: 'diffusion', upscale: false }), {}), [])
})

test('buildBatchRequests honours batch size and clamps it', () => {
  assert.equal(buildBatchRequests(draft({ batch: 3 }), {}, 1).length, 3)
  assert.equal(buildBatchRequests(draft({ batch: 0 }), {}, 1).length, 1)
  assert.equal(buildBatchRequests(draft({ batch: 99 }), {}, 1).length, 20)
  const seeds = buildBatchRequests(draft({ batch: 3, seed: 5 }), {}, 1).map(b => b.seed)
  assert.deepEqual(seeds, [5, 6, 7])
})

test('parseSize reads a size value with a fallback', () => {
  assert.deepEqual(parseSize('704x480'), { width: 704, height: 480 })
  assert.deepEqual(parseSize(''), { width: 768, height: 512 })
  assert.deepEqual(parseSize('bad'), { width: 768, height: 512 })
})

// ------------------------------------------------------------------ report

console.log(`\n${passed}/${passed + failed} passed`)
if (failed) {
  console.log(`${failed} FAILED`)
  process.exit(1)
}
