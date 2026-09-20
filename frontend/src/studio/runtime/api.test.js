import assert from 'node:assert/strict'

import { api } from './api.js'

if (typeof globalThis.File === 'undefined') {
  globalThis.File = class File extends Blob {
    constructor(parts, name, options = {}) {
      super(parts, options)
      this.name = name
    }
  }
}

globalThis.window = { location: { href: 'http://localhost:5187/' } }

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

async function testSinglePut() {
  const calls = []
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init })
    if (calls.length === 1) {
      return json({
        asset_id: 'a'.repeat(32),
        filename: 'frame.png',
        kind: 'image',
        size: 4,
        backend: 'r2',
        mode: 'single',
        method: 'PUT',
        url: 'https://r2.test/input',
        headers: { 'Content-Type': 'image/png' },
        parts: [],
      }, 201)
    }
    if (calls.length === 2) return new Response('ok', { status: 200 })
    return json({ id: 'a'.repeat(32), kind: 'image', filename: 'frame.png', size: 4 })
  }

  const file = new File([new Uint8Array([1, 2, 3, 4])], 'frame.png', { type: 'image/png' })
  const asset = await api.uploadAsset(file)

  assert.equal(asset.id, 'a'.repeat(32))
  assert.match(calls[0].url, /\/api\/assets\/prepare$/)
  assert.equal(calls[1].url, 'https://r2.test/input')
  assert.equal(calls[1].init.method, 'PUT')
  assert.equal(calls[1].init.headers['Content-Type'], 'image/png')
  const completed = JSON.parse(calls[2].init.body)
  assert.deepEqual(completed.parts, [])
  assert.equal(typeof completed.client_upload_seconds, 'number')
}

async function testMultipartPut() {
  const calls = []
  const etags = ['"one"', '"two"', '"three"']
  let partIndex = 0
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init })
    if (calls.length === 1) {
      return json({
        asset_id: 'b'.repeat(32),
        filename: 'clip.mp4',
        kind: 'video',
        size: 12,
        backend: 'r2',
        mode: 'multipart',
        method: 'PUT',
        upload_id: 'u1',
        part_size: 4,
        headers: {},
        parts: [
          { part_number: 1, url: 'https://r2.test/p1' },
          { part_number: 2, url: 'https://r2.test/p2' },
          { part_number: 3, url: 'https://r2.test/p3' },
        ],
      }, 201)
    }
    if (String(url).startsWith('https://r2.test/p')) {
      const etag = etags[partIndex++]
      return new Response('ok', { status: 200, headers: { ETag: etag } })
    }
    return json({ id: 'b'.repeat(32), kind: 'video', filename: 'clip.mp4', size: 12 })
  }

  const file = new File([new Uint8Array(12)], 'clip.mp4', { type: 'video/mp4' })
  const asset = await api.uploadAsset(file)

  assert.equal(asset.kind, 'video')
  const puts = calls.filter(call => call.url.startsWith('https://r2.test/p'))
  assert.equal(puts.length, 3)
  assert.deepEqual(puts.map(call => call.init.body.size).sort((a, b) => a - b), [4, 4, 4])
  const completeCall = calls.find(call => /\/complete$/.test(call.url))
  const completed = JSON.parse(completeCall.init.body)
  assert.deepEqual(completed.parts.map(part => part.part_number), [1, 2, 3])
  assert.deepEqual(completed.parts.map(part => part.etag).sort(), etags.sort())
}

async function testAbortOnUploadFailure() {
  const calls = []
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init })
    if (calls.length === 1) {
      return json({
        asset_id: 'c'.repeat(32),
        filename: 'bad.png',
        kind: 'image',
        size: 1,
        backend: 'r2',
        mode: 'single',
        method: 'PUT',
        url: 'https://r2.test/fail',
        headers: {},
        parts: [],
      }, 201)
    }
    if (String(url) === 'https://r2.test/fail') {
      return json({ detail: 'storage failed' }, 500)
    }
    return new Response(null, { status: 204 })
  }

  const file = new File([new Uint8Array([1])], 'bad.png', { type: 'image/png' })
  await assert.rejects(() => api.uploadAsset(file), /storage failed/)
  assert.ok(calls.some(call => call.init.method === 'DELETE' && call.url.includes('/upload')))
}

async function testFinalizeFailureDoesNotDeleteTransferredObject() {
  const calls = []
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init })
    if (calls.length === 1) {
      return json({
        asset_id: 'd'.repeat(32),
        filename: 'done.png',
        kind: 'image',
        size: 1,
        backend: 'r2',
        mode: 'single',
        method: 'PUT',
        url: 'https://r2.test/done',
        headers: {},
        parts: [],
      }, 201)
    }
    if (String(url) === 'https://r2.test/done') return new Response('ok', { status: 200 })
    return json({ detail: 'finalize response lost' }, 503)
  }

  const file = new File([new Uint8Array([1])], 'done.png', { type: 'image/png' })
  await assert.rejects(() => api.uploadAsset(file), /finalize response lost/)
  assert.ok(!calls.some(call => call.init.method === 'DELETE'))
}

async function testFinalizeRetriesWithoutReuploadingBytes() {
  const calls = []
  let finalizeAttempts = 0
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init })
    if (calls.length === 1) {
      return json({
        asset_id: 'e'.repeat(32),
        filename: 'retry.png',
        kind: 'image',
        size: 1,
        backend: 'r2',
        mode: 'single',
        method: 'PUT',
        url: 'https://r2.test/retry',
        headers: {},
        parts: [],
      }, 201)
    }
    if (String(url) === 'https://r2.test/retry') return new Response('ok', { status: 200 })
    finalizeAttempts += 1
    if (finalizeAttempts === 1) return json({ detail: 'temporary' }, 503)
    return json({ id: 'e'.repeat(32), kind: 'image', filename: 'retry.png', size: 1 })
  }

  const file = new File([new Uint8Array([1])], 'retry.png', { type: 'image/png' })
  const asset = await api.uploadAsset(file)

  assert.equal(asset.id, 'e'.repeat(32))
  assert.equal(calls.filter(call => call.url === 'https://r2.test/retry').length, 1)
  assert.equal(finalizeAttempts, 2)
  assert.ok(!calls.some(call => call.init.method === 'DELETE'))
}

await testSinglePut()
console.log('  ok   single presigned PUT')
await testMultipartPut()
console.log('  ok   multipart presigned PUT')
await testAbortOnUploadFailure()
console.log('  ok   upload failure abort')
await testFinalizeFailureDoesNotDeleteTransferredObject()
console.log('  ok   finalize failure preserves transferred object')
await testFinalizeRetriesWithoutReuploadingBytes()
console.log('  ok   finalize retries without retransmitting bytes')
console.log('\n5/5 passed')
