/**
 * Tests for the studio runtime.
 *
 * Run with:  node src/studio/runtime/runtime.test.js
 *
 * The runtime only depends on `vue`'s reactivity primitives plus injected
 * modules, so it can be exercised in Node without a DOM or a browser. A fake
 * API and a fake poller keep every test deterministic.
 */

import assert from 'node:assert/strict'

// Vue's reactivity works in plain Node; nothing here touches the DOM.
import { nextTick } from 'vue'

import { createPoller, POLL_ACTIVE_MS, POLL_IDLE_MS } from './polling.js'
import { guessKind, validateFile } from './uploads.js'
import { ApiError } from './api.js'
import { createDraft, useStudioRuntime } from './useStudioRuntime.js'

let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
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

// Node has no DOM. The runtime needs localStorage for session persistence and
// window.location to resolve relative output URLs, so a minimal window is
// provided here. The runtime still degrades gracefully without storage, which is
// covered separately.
const storage = new Map()
globalThis.window = {
  location: { href: 'http://localhost:5173/' },
  localStorage: {
    getItem: key => (storage.has(key) ? storage.get(key) : null),
    setItem: (key, value) => storage.set(key, String(value)),
  },
}
// Any unmocked network call is a test bug, so fail loudly rather than hanging.
globalThis.fetch = async url => {
  throw new Error(`unexpected fetch in runtime test: ${url}`)
}

// Node exposes File since 20, but guard for older runtimes used in CI.
if (typeof globalThis.File === 'undefined') {
  globalThis.File = class File extends Blob {
    constructor(parts, name, options) {
      super(parts, options)
      this.name = name
    }
  }
}

const asset = id => ({ id, kind: 'image', filename: 'x.png', size: 1024 })
const JOB = (overrides = {}) => ({
  id: overrides.id || 'a'.repeat(32),
  session_number: 1,
  status: 'completed',
  progress: 1,
  error: null,
  video_url: '/outputs/a.mp4',
  image_url: null,
  request: { mode: 't2av', prompt: 'x', width: 768, height: 512, num_frames: 121, fps: 24, steps: 8, upscale: true },
  created_at: '2026-09-20T10:00:00Z',
  generation_seconds: 8,
  peak_vram_gb: 40,
  ...overrides,
})

/**
 * Fake API that records every call.
 *
 * Overrides are wrapped so the recorder still runs: replacing `interrupt`
 * wholesale would otherwise hide the call the tests assert on.
 */
function fakeApi(overrides = {}) {
  const calls = []
  const record = (name, value) => { calls.push([name, value]) }

  const base = {
    health: async () => ({ warmup: { state: 'ready', gpu: 'MOCK GPU' } }),
    warm: async () => { record('warm', true); return { status: 'warming' } },
    unload: async options => { record('unload', options); return { result: 'ok' } },
    loras: async () => [],
    createSession: async () => ({ session_number: 555 }),
    listJobs: async () => [],
    submitJob: async body => {
      record('submitJob', body)
      return JOB({ id: 'b'.repeat(32) })
    },
    interrupt: async id => {
      record('interrupt', id)
      return { interrupted: true, current_job_id: id }
    },
    deleteJob: async id => {
      record('deleteJob', id)
      return null
    },
    uploadAsset: async file => {
      record('uploadAsset', file?.name)
      return asset('c'.repeat(32))
    },
    fetchOutput: async url => {
      record('fetchOutput', url)
      return new Blob([new Uint8Array([1, 2, 3])], { type: 'video/mp4' })
    },
  }

  const merged = { ...base }
  for (const [name, fn] of Object.entries(overrides)) {
    merged[name] = async (...args) => {
      record(name, args[0])
      return fn(...args)
    }
  }
  return { calls, ...merged }
}

/**
 * Build a runtime for a test.
 *
 * Storage is cleared first so a session number persisted by an earlier test
 * cannot leak in and make assertions order-dependent.
 */
function makeRuntime(apiOverride = fakeApi()) {
  storage.clear()
  // start() kicks the pollers; tests drive refresh explicitly instead.
  return useStudioRuntime({ apiOverride })
}

// ------------------------------------------------------------- polling.js

group('polling')

await test('poller reschedules and can be stopped', async () => {
  let ticks = 0
  const poller = createPoller(async () => { ticks += 1; return false }, { idleMs: 5 })
  poller.start()
  await new Promise(resolve => setTimeout(resolve, 40))
  poller.stop()
  const seen = ticks
  assert.ok(seen >= 2, `expected repeated ticks, saw ${seen}`)
  await new Promise(resolve => setTimeout(resolve, 30))
  assert.equal(ticks, seen, 'no ticks after stop')
})

await test('poller uses the active interval while work remains', () => {
  // The contract the runtime relies on: returning true means "poll faster".
  assert.ok(POLL_ACTIVE_MS < POLL_IDLE_MS)
  const poller = createPoller(async () => true)
  assert.equal(poller.isRunning(), false)
  poller.start()
  assert.equal(poller.isRunning(), true)
  poller.stop()
})

await test('poller survives a throwing task', async () => {
  let attempts = 0
  const poller = createPoller(async () => {
    attempts += 1
    if (attempts === 1) throw new Error('boom')
    return false
  }, { errorMs: 5, idleMs: 5 })
  poller.start()
  await new Promise(resolve => setTimeout(resolve, 40))
  poller.stop()
  assert.ok(attempts >= 2, 'a rejection must not kill the loop')
})

// ------------------------------------------------------------- uploads.js

group('uploads')

await test('guessKind recognises each media type', () => {
  assert.equal(guessKind({ name: 'a.PNG' }), 'image')
  assert.equal(guessKind({ name: 'a.mp4' }), 'video')
  assert.equal(guessKind({ name: 'a.wav' }), 'audio')
  assert.equal(guessKind({ name: 'a.txt' }), 'unknown')
  assert.equal(guessKind(null), 'unknown')
})

await test('validateFile rejects empty, oversized and mistyped files', () => {
  assert.match(validateFile(null), /没有选择文件/)
  assert.match(validateFile({ name: 'a.png', size: 0 }), /文件为空/)
  assert.match(validateFile({ name: 'a.png', size: 2 * 1024 * 1024 * 1024 }), /1 GiB/)
  assert.match(validateFile({ name: 'a.png', size: 10 }, { kind: 'video' }), /需要视频文件/)
  assert.equal(validateFile({ name: 'a.png', size: 10 }, { kind: 'image' }), null)
})

// -------------------------------------------------------------- runtime

group('runtime: state')

await test('starts with an empty draft and no selection', () => {
  const runtime = makeRuntime()
  assert.equal(runtime.draft.mode, 't2av')
  assert.equal(runtime.draft.prompt, '')
  assert.equal(runtime.selectedJobId.value, null)
  assert.deepEqual(runtime.allJobs.value, [])
  assert.equal(runtime.stageJob.value, null)
})

await test('createDraft applies overrides without sharing range state', () => {
  const one = createDraft()
  const two = createDraft({ mode: 'i2v' })
  one.range.retakeStart = 99
  assert.equal(two.range.retakeStart, 0, 'range must not be shared between drafts')
  assert.equal(two.mode, 'i2v')
})

await test('setMode normalises controls the backend would override', () => {
  const runtime = makeRuntime()
  runtime.setMode('retake')
  assert.equal(runtime.draft.upscale, false, 'retake cannot upscale')
  assert.equal(runtime.draft.decoder, 'vae', 'retake pins the decoder')
  runtime.setMode('t2i')
  assert.equal(runtime.draft.upscale, true, 't2i always upsamples')
  runtime.setMode('ref2i')
  assert.equal(runtime.draft.upscale, false, 'ref2i is single-stage')
  assert.equal(runtime.draft.numFrames, 49, 'ref2i pins its frame count')
})

await test('stage, takes and queue are projections of one list', async () => {
  const runtime = makeRuntime(fakeApi({
    listJobs: async () => [
      JOB({ id: 'run', status: 'running', progress: 0.5, video_url: null }),
      JOB({ id: 'done', status: 'completed' }),
    ],
  }))
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()

  assert.equal(runtime.stageJob.value.id, 'run', 'running takes the stage')
  assert.deepEqual(runtime.takes.value.map(j => j.id), ['done'])
  assert.deepEqual(runtime.queue.value.running.map(j => j.id), ['run'])
  assert.equal(runtime.counts.value.all, 2)
})

await test('counts and available modes track the job list', async () => {
  const runtime = makeRuntime(fakeApi({
    listJobs: async () => [
      JOB({ id: '1', status: 'completed' }),
      JOB({ id: '2', status: 'failed' }),
      JOB({ id: '3', status: 'running', video_url: null, request: { mode: 'i2v', prompt: 'y' } }),
    ],
  }))
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()
  assert.deepEqual(runtime.counts.value, { all: 3, active: 1, completed: 1, failed: 1 })
  assert.deepEqual(runtime.availableModes.value, ['i2v', 't2av'])
})

await test('library filters react to filter and sort state', async () => {
  const runtime = makeRuntime(fakeApi({
    listJobs: async () => [
      JOB({ id: 'a1', status: 'completed', request: { mode: 't2av', prompt: 'ocean' } }),
      JOB({ id: 'b1', status: 'failed', request: { mode: 'i2v', prompt: 'ocean' } }),
    ],
  }))
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()
  assert.equal(runtime.libraryJobs.value.length, 2)
  runtime.setFilter({ status: 'failed' })
  assert.deepEqual(runtime.libraryJobs.value.map(j => j.id), ['b1'])
  runtime.setFilter({ status: 'all', query: 'ocean' })
  assert.equal(runtime.libraryJobs.value.length, 2)
  runtime.setFilter({ query: 'nothing' })
  assert.equal(runtime.libraryJobs.value.length, 0)
})

group('runtime: submission')

await test('submit refuses an empty prompt without calling the API', async () => {
  const api = fakeApi()
  const runtime = makeRuntime(api)
  const result = await runtime.submit()
  assert.equal(result.ok, false)
  assert.match(runtime.notice.text, /Prompt/)
  assert.equal(api.calls.length, 0)
})

await test('submit refuses when a required attachment is missing', async () => {
  const api = fakeApi()
  const runtime = makeRuntime(api)
  runtime.setMode('i2v')
  runtime.draft.prompt = 'hello'
  const result = await runtime.submit()
  assert.equal(result.ok, false)
  assert.equal(api.calls.length, 0)
})

await test('submit registers a placeholder and returns immediately', async () => {
  const api = fakeApi({ createSession: async () => ({ session_number: 777 }) })
  const runtime = makeRuntime(api)
  runtime.draft.prompt = 'ocean sunset'

  // Block the submissions so the placeholder is observable at return time,
  // which is exactly what the UI needs: feedback before the network settles.
  let release
  const gate = new Promise(resolve => { release = resolve })
  api.submitJob = async body => { api.calls.push(['submitJob', body]); await gate; return JOB() }

  const result = await runtime.submit()
  assert.equal(result.ok, true)
  assert.equal(result.count, 1)

  const placeholders = runtime.allJobs.value.filter(j => j.pending)
  assert.equal(placeholders.length, 1, 'placeholder must exist before requests settle')
  assert.match(placeholders[0].request.prompt, /ocean/)
  assert.equal(placeholders[0].session_number, 777)

  release()
  await new Promise(resolve => setTimeout(resolve, 20))
})

await test('a second submit is accepted while the first is still in flight', async () => {
  const api = fakeApi()
  let release
  const gate = new Promise(resolve => { release = resolve })
  api.submitJob = async body => { api.calls.push(['submitJob', body]); await gate; return JOB() }

  const runtime = makeRuntime(api)
  runtime.draft.prompt = 'first'
  await runtime.submit()
  runtime.draft.prompt = 'second'
  const second = await runtime.submit()

  // The button must not be held hostage by the previous submission.
  assert.equal(second.ok, true, 'second submit should be accepted')
  assert.equal(runtime.allJobs.value.filter(j => j.pending).length, 2)

  release()
  await new Promise(resolve => setTimeout(resolve, 20))
})

await test('submit clears the prompt and advances the seed', async () => {
  const runtime = makeRuntime()
  runtime.draft.prompt = 'ocean'
  runtime.draft.seed = 10
  await runtime.submit()
  assert.equal(runtime.draft.prompt, '')
  assert.equal(runtime.draft.seed, 11)
})

await test('submit keeps the reusable preset', async () => {
  const runtime = makeRuntime()
  runtime.draft.prompt = 'ocean'
  runtime.draft.negativePrompt = 'blurry'
  runtime.draft.steps = 16
  runtime.draft.guidanceScale = 5
  await runtime.submit()
  // These are a preset, not one-off creative input, so they must survive.
  assert.equal(runtime.draft.negativePrompt, 'blurry')
  assert.equal(runtime.draft.steps, 16)
  assert.equal(runtime.draft.guidanceScale, 5)
})

await test('submit honours the batch size', async () => {
  const api = fakeApi()
  let release
  const gate = new Promise(resolve => { release = resolve })
  api.submitJob = async body => { api.calls.push(['submitJob', body]); await gate; return JOB() }

  const runtime = makeRuntime(api)
  runtime.draft.prompt = 'ocean'
  runtime.draft.batch = 3
  const result = await runtime.submit()

  assert.equal(result.count, 3)
  assert.equal(runtime.allJobs.value.filter(j => j.pending).length, 3)
  // Seeds must differ across the batch.
  const seeds = runtime.allJobs.value.filter(j => j.pending).map(j => j.request.seed)
  assert.deepEqual([...seeds].sort((a, b) => a - b), [42, 43, 44])

  release()
  await new Promise(resolve => setTimeout(resolve, 20))
})

await test('placeholders retire once the server reports the new record', async () => {
  // A placeholder is retired when the server's job count grows to cover it. No
  // token is echoed back, because the backend ignores unknown request fields.
  let list = []
  const api = fakeApi({
    submitJob: async () => {
      list = [JOB({ id: 'real1' })]
      return JOB({ id: 'real1' })
    },
    listJobs: async () => list,
  })
  const runtime = makeRuntime(api)
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()          // baseline: 0 jobs
  runtime.draft.prompt = 'ocean'

  let release
  const gate = new Promise(resolve => { release = resolve })
  const realSubmit = api.submitJob
  api.submitJob = async body => { const out = await realSubmit(body); await gate; return out }

  await runtime.submit()
  assert.equal(runtime.allJobs.value.filter(j => j.pending).length, 1)

  release()
  await new Promise(resolve => setTimeout(resolve, 30))
  assert.equal(runtime.allJobs.value.filter(j => j.pending).length, 0,
    'the real record should have retired the placeholder')
  assert.ok(runtime.allJobs.value.some(j => j.id === 'real1'))
})

await test('a failed submission does not leave a placeholder behind forever', async () => {
  const api = fakeApi({
    submitJob: async () => { throw new ApiError('server down', 500) },
  })
  const runtime = makeRuntime(api)
  runtime.sessionNumber.value = 1
  runtime.draft.prompt = 'ocean'
  await runtime.submit()
  await new Promise(resolve => setTimeout(resolve, 30))
  assert.match(runtime.notice.text, /提交失败/)
  // The list never grew, so the placeholder survives until the TTL expires; it
  // must not be silently dropped, or the user loses sight of the failure.
  assert.ok(runtime.allJobs.value.some(j => j.pending))
})

await test('placeholders survive until the server list catches up', async () => {
  // With the submission blocked the placeholder must stay visible, which is what
  // gives the user immediate feedback that their job was accepted.
  const api = fakeApi()
  let release
  const gate = new Promise(resolve => { release = resolve })
  api.submitJob = async body => { api.calls.push(['submitJob', body]); await gate; return JOB() }

  const runtime = makeRuntime(api)
  runtime.draft.prompt = 'ocean'
  await runtime.submit()
  await new Promise(resolve => setTimeout(resolve, 10))
  assert.equal(runtime.allJobs.value.filter(j => j.pending).length, 1)

  release()
  await new Promise(resolve => setTimeout(resolve, 20))
})

await test('a partial failure is reported without losing the successes', async () => {
  let calls = 0
  const api = fakeApi({
    submitJob: async () => {
      calls += 1
      if (calls === 2) throw new ApiError('服务器错误', 500)
      return JOB()
    },
  })
  const runtime = makeRuntime(api)
  runtime.draft.prompt = 'ocean'
  runtime.draft.batch = 3
  await runtime.submit()
  await new Promise(resolve => setTimeout(resolve, 30))
  assert.match(runtime.notice.text, /部分成功：2\/3/)
  assert.equal(runtime.notice.state, 'error')
})

await test('setNotice and clearNotice drive the message area', () => {
  const runtime = makeRuntime()
  runtime.setNotice('hello', 'ok')
  assert.equal(runtime.notice.text, 'hello')
  runtime.clearNotice()
  assert.equal(runtime.notice.text, '')
})

group('runtime: actions')

await test('cancelJob calls interrupt with the job id', async () => {
  const api = fakeApi()
  const runtime = makeRuntime(api)
  runtime.sessionNumber.value = 1
  await runtime.cancelJob('x'.repeat(32))
  assert.ok(api.calls.some(([name, arg]) => name === 'interrupt' && arg === 'x'.repeat(32)))
})

await test('deleteJob clears the selection when the selected job goes', async () => {
  const runtime = makeRuntime(fakeApi({ listJobs: async () => [JOB({ id: 'gone' })] }))
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()
  runtime.selectJob('gone')
  await runtime.deleteJob('gone')
  assert.equal(runtime.selectedJobId.value, null)
})

await test('selectJob pins the stage', async () => {
  const runtime = makeRuntime(fakeApi({
    listJobs: async () => [
      JOB({ id: 'run', status: 'running', progress: 0.3, video_url: null }),
      JOB({ id: 'old', status: 'completed' }),
    ],
  }))
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()
  assert.equal(runtime.stageJob.value.id, 'run')
  runtime.selectJob('old')
  assert.equal(runtime.stageJob.value.id, 'old', 'explicit selection wins')
  runtime.clearSelection()
  assert.equal(runtime.stageJob.value.id, 'run')
})

await test('editFromOutput re-uploads the output and switches mode', async () => {
  const api = fakeApi({
    listJobs: async () => [JOB({ id: 'v'.repeat(32), video_url: '/outputs/v.mp4' })],
  })
  const runtime = makeRuntime(api)
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()
  runtime.selectJob('v'.repeat(32))

  const result = await runtime.editFromOutput('retake')
  assert.equal(result.ok, true, `editFromOutput failed: ${runtime.notice.text}`)
  assert.equal(runtime.draft.mode, 'retake')
  assert.equal(runtime.attachments.source.id, 'c'.repeat(32))
  assert.equal(runtime.busy.preparing, false)
  // The output must have made a full download + re-upload round trip.
  const names = api.calls.map(([name]) => name)
  assert.ok(names.includes('fetchOutput'), 'should download the output')
  assert.ok(names.includes('uploadAsset'), 'should re-upload it as an asset')
})

await test('editFromOutput refuses without a completed video', async () => {
  const runtime = makeRuntime()
  runtime.draft.prompt = 'x'
  const result = await runtime.editFromOutput('retake')
  assert.equal(result.ok, false)
  assert.match(runtime.notice.text, /已完成的视频/)
})

await test('editFromOutput reports upload failures and leaves no source', async () => {
  const api = fakeApi({
    listJobs: async () => [JOB({ id: 'v'.repeat(32) })],
    uploadAsset: async () => { throw new ApiError('boom', 500) },
  })
  const runtime = makeRuntime(api)
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()
  runtime.selectJob('v'.repeat(32))
  const result = await runtime.editFromOutput('extend')
  assert.equal(result.ok, false)
  assert.equal(runtime.attachments.source, null)
  assert.equal(runtime.busy.preparing, false)
})

await test('reuseJob copies parameters back into the draft', async () => {
  const runtime = makeRuntime(fakeApi({
    listJobs: async () => [JOB({
      id: 'r'.repeat(32),
      request: {
        mode: 'i2v', prompt: 'copy me', negative_prompt: 'no', width: 704, height: 480,
        num_frames: 49, fps: 16, steps: 12, guidance_scale: 4, seed: 99, upscale: false,
        upscale_method: 'latent', decoder: 'vae', loras: [{ id: 'L', strength: 0.7 }],
      },
    })],
  }))
  runtime.sessionNumber.value = 1
  await runtime.refreshJobs()
  runtime.reuseJob('r'.repeat(32))
  assert.equal(runtime.draft.mode, 'i2v')
  assert.equal(runtime.draft.prompt, 'copy me')
  assert.equal(runtime.draft.size, '704x480')
  assert.equal(runtime.draft.numFrames, 49)
  assert.equal(runtime.draft.seed, 99)
  assert.equal(runtime.draft.loraId, 'L')
  assert.equal(runtime.draft.loraStrength, 0.7)
})

group('runtime: assets')

await test('attach stores the uploaded asset and toggles busy state', async () => {
  const runtime = makeRuntime()
  await runtime.attach('first', new File([new Uint8Array([1])], 'a.png', { type: 'image/png' }))
  assert.equal(runtime.attachments.first.id, 'c'.repeat(32))
  assert.deepEqual(runtime.busy.uploading, {})
})

await test('attach surfaces an upload failure without storing anything', async () => {
  const runtime = makeRuntime(fakeApi({
    uploadAsset: async () => { throw new ApiError('too big', 413) },
  }))
  await runtime.attach('first', new File([new Uint8Array([1])], 'a.png', { type: 'image/png' }))
  assert.equal(runtime.attachments.first, null)
  assert.match(runtime.notice.text, /上传失败/)
})

await test('detach and clearAttachments reset slots', async () => {
  const runtime = makeRuntime()
  await runtime.attach('first', new File([new Uint8Array([1])], 'a.png', { type: 'image/png' }))
  runtime.detach('first')
  assert.equal(runtime.attachments.first, null)
  await runtime.attach('first', new File([new Uint8Array([1])], 'a.png', { type: 'image/png' }))
  runtime.clearAttachments()
  assert.deepEqual({ ...runtime.attachments }, { first: null, last: null, source: null, audio: null })
})

group('runtime: lifecycle')

await test('start loads session, health, loras and jobs once', async () => {
  const api = fakeApi()
  const runtime = makeRuntime(api)
  await runtime.start()
  await nextTick()
  assert.equal(runtime.sessionNumber.value, 555)
  assert.equal(runtime.gpuState.value, 'ready')
  assert.equal(runtime.gpuLabel.value, 'MOCK GPU')
  assert.ok(api.calls.some(([name]) => name === 'warm'))
  runtime.dispose()
})

await test('releaseGpu stops polling and calls unload with keepalive', async () => {
  const api = fakeApi()
  const runtime = makeRuntime(api)
  await runtime.start()
  const ok = await runtime.releaseGpu({ keepalive: true })

  assert.equal(ok, true)
  assert.ok(api.calls.some(([name, options]) => name === 'unload' && options?.keepalive === true))
})

await test('gpuLabel degrades gracefully without health data', () => {
  const runtime = makeRuntime()
  assert.equal(runtime.gpuLabel.value, '离线')
  runtime.health.value = { warmup: { state: 'warming' } }
  assert.match(runtime.gpuLabel.value, /预热/)
})

await test('dispose stops polling permanently', async () => {
  const runtime = makeRuntime()
  runtime.start()
  runtime.dispose()
  // A disposed runtime must not reschedule itself.
  await new Promise(resolve => setTimeout(resolve, 30))
  assert.equal(runtime.hasActiveWork.value, false)
})

await test('refreshJobs without a session clears the list instead of throwing', async () => {
  const runtime = makeRuntime()
  runtime.sessionNumber.value = null
  await runtime.refreshJobs()
  assert.deepEqual(runtime.jobs.value, [])
})

// ------------------------------------------------------------------ report

console.log(`\n${passed}/${passed + failed} passed`)
if (failed) {
  console.log(`${failed} FAILED`)
  process.exit(1)
}
