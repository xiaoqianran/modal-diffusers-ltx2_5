import './style.css'

const app = document.querySelector('#app')
const SESSION_KEY = 'ltx25.session'
let sessionNumber = Number(localStorage.getItem(SESSION_KEY)) || null
let pollTimer = null
let healthTimer = null
let loras = []
const assets = { first: null, last: null, source: null, audio: null }

app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div>
        <p class="eyebrow">LTX-2.5 · RTX PRO 6000</p>
        <h1>Queue Studio</h1>
      </div>
      <div class="header-actions">
        <span id="sessionBadge" class="badge">SESSION —</span>
        <button id="health" class="health" type="button">连接中…</button>
      </div>
    </header>

    <section class="grid">
      <form id="generate" class="panel form-panel">
        <div class="section-title">
          <span>GENERATE</span>
          <small>连续提交 · 单 GPU 串行队列</small>
        </div>

        <div class="mode-tabs" role="tablist">
          <button type="button" data-mode="t2av" class="mode active">T2AV</button>
          <button type="button" data-mode="i2v" class="mode">I2V</button>
          <button type="button" data-mode="flf2v" class="mode">FLF2V</button>
          <button type="button" data-mode="a2v" class="mode">A2V</button>
          <button type="button" data-mode="t2i" class="mode">T2I</button>
          <button type="button" data-mode="refine_image" class="mode">REFINE</button>
          <button type="button" data-mode="ref2i" class="mode">REF2I</button>
          <button type="button" data-mode="condition" class="mode">CONDITION</button>
          <button type="button" data-mode="iclora" class="mode">IC-LORA</button>
          <button type="button" data-mode="retake" class="mode">RETAKE</button>
          <button type="button" data-mode="extend" class="mode">EXTEND</button>
        </div>
        <input id="mode" name="mode" type="hidden" value="t2av" />

        <div id="visualInputs" class="asset-grid" hidden>
          <label id="firstAsset" class="asset-drop">
            <input type="file" accept="image/*" hidden />
            <span>FIRST FRAME</span><strong>选择图片</strong><small>PNG / JPG / WEBP</small>
          </label>
          <label id="lastAsset" class="asset-drop" hidden>
            <input type="file" accept="image/*" hidden />
            <span>LAST FRAME</span><strong>选择图片</strong><small>FLF2V 终帧</small>
          </label>
        </div>
        <div id="audioInputs" class="asset-grid" hidden>
          <label id="audioAsset" class="asset-drop">
            <input type="file" accept="audio/*" hidden />
            <span>AUDIO</span><strong>选择音频</strong><small>WAV / MP3 / M4A / FLAC</small>
          </label>
        </div>
        <div id="sourceInputs" class="asset-grid" hidden>
          <label id="sourceAsset" class="asset-drop">
            <input type="file" accept="image/*,video/*" hidden />
            <span>SOURCE</span><strong>选择参考素材</strong><small>图片 / 视频</small>
          </label>
        </div>
        <div id="modeOptions" class="fields mode-options" hidden>
          <label id="retakeStartField" class="field" hidden><span>Retake start (s)</span><input name="retake_start" type="number" min="0" step="0.1" value="0" /></label>
          <label id="retakeEndField" class="field" hidden><span>Retake end (s)</span><input name="retake_end" type="number" min="0.1" step="0.1" value="3" /></label>
          <label id="extendDirectionField" class="field" hidden><span>Extend direction</span><select name="extend_direction"><option value="end">End</option><option value="start">Start</option></select></label>
          <label id="extendSecondsField" class="field" hidden><span>Extend seconds</span><input name="extend_seconds" type="number" min="1" max="20" step="0.5" value="5" /></label>
          <label id="extendContextField" class="field" hidden><span>Context seconds</span><input name="extend_context_seconds" type="number" min="0.5" max="20" step="0.5" value="3" /></label>
          <label id="strengthField" class="field" hidden><span>Reference strength</span><input name="strength" type="number" min="0.1" max="1" step="0.05" value="1" /></label>
          <label id="framePositionField" class="field" hidden><span>Frame position</span><select name="frame_position"><option value="last">Last</option><option value="center">Center</option></select></label>
        </div>

        <label class="field field-wide">
          <span>Prompt</span>
          <textarea name="prompt" rows="6" required placeholder="主体、动作、镜头、光线、氛围…"></textarea>
        </label>
        <label class="field field-wide compact">
          <span>Negative Prompt</span>
          <input name="negative_prompt" value="worst quality, inconsistent motion, blurry, jittery, distorted" />
        </label>

        <div class="fields">
          <label class="field"><span>尺寸</span>
            <select name="size">
              <option value="768x512">768 × 512</option>
              <option value="704x480">704 × 480</option>
              <option value="640x384">640 × 384</option>
              <option value="512x320">512 × 320</option>
            </select>
          </label>
          <label class="field"><span>Frames</span><input name="num_frames" type="number" min="9" max="481" step="8" value="121" /></label>
          <label class="field"><span>FPS</span><input name="fps" type="number" min="8" max="60" value="24" /></label>
          <label class="field"><span>Steps</span><input name="steps" type="number" min="1" max="100" value="8" /></label>
          <label class="field"><span>Guidance</span><input name="guidance_scale" type="number" min="0" max="20" step="0.1" value="3" /></label>
          <label class="field"><span>Seed</span><input name="seed" type="number" min="0" value="42" /></label>
          <label class="field"><span>Batch</span><input name="batch" type="number" min="1" max="20" value="1" /></label>
        </div>

        <details class="advanced">
          <summary>高级参数</summary>
          <div class="fields advanced-fields">
            <label class="field"><span>Video modality</span><input name="modality_scale" type="number" min="0" max="15" step="0.1" placeholder="1.0" /></label>
            <label class="field"><span>Audio guidance</span><input name="audio_guidance_scale" type="number" min="0" max="15" step="0.1" placeholder="default" /></label>
            <label class="field"><span>Decoder</span><select name="decoder"><option value="vae">VAE</option><option value="diffusion">Diffusion</option></select></label>
          </div>
          <div class="lora-row">
            <label class="field"><span>LoRA</span><select id="loraSelect"><option value="">不使用</option></select></label>
            <label class="field"><span>Strength</span><input id="loraStrength" type="number" min="-2" max="2" step="0.1" value="1" /></label>
          </div>
        </details>

        <label class="check"><input name="upscale" type="checkbox" checked /> <span>2× latent upscale</span></label>
        <button id="submit" class="primary" type="submit">加入队列</button>
        <p id="submitMessage" class="message"></p>
      </form>

      <section class="panel queue-panel">
        <div class="section-title">
          <div><span>QUEUE</span><small id="queueSummary">0 active</small></div>
          <button id="refresh" class="ghost" type="button">刷新</button>
        </div>
        <div id="queue" class="queue"></div>
      </section>
    </section>
  </main>
`

const el = (selector) => document.querySelector(selector)
const all = (selector) => [...document.querySelectorAll(selector)]
const escapeHtml = (value = '') => String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]))

async function json(url, options) {
  const response = await fetch(url, options)
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body.error || body)
    } catch {}
    throw new Error(message)
  }
  if (response.status === 204) return null
  return response.json()
}

async function ensureSession() {
  if (sessionNumber) return sessionNumber
  const session = await json('/api/sessions', { method: 'POST' })
  sessionNumber = session.session_number
  localStorage.setItem(SESSION_KEY, String(sessionNumber))
  el('#sessionBadge').textContent = `SESSION ${sessionNumber}`
  return sessionNumber
}

async function checkHealth() {
  const badge = el('#health')
  clearTimeout(healthTimer)
  try {
    const health = await json('/api/health')
    const warmState = health.warmup?.state
    const stateLabel = warmState === 'warming' ? 'WARMING'
      : warmState === 'ready' ? 'READY'
      : health.worker
    badge.textContent = `${health.transformer_precision?.toUpperCase() || 'GPU'} · ${stateLabel}`
    badge.dataset.state = 'online'
    if (warmState === 'warming') healthTimer = setTimeout(checkHealth, 1500)
  } catch (error) {
    badge.textContent = `离线 · ${error.message}`
    badge.dataset.state = 'offline'
  }
}

function setMode(mode) {
  el('#mode').value = mode
  all('.mode').forEach(button => button.classList.toggle('active', button.dataset.mode === mode))

  const usesFirst = ['i2v', 'flf2v', 'refine_image', 'ref2i'].includes(mode)
  el('#visualInputs').hidden = !usesFirst
  el('#firstAsset').hidden = !usesFirst
  el('#lastAsset').hidden = mode !== 'flf2v'
  el('#audioInputs').hidden = mode !== 'a2v'
  el('#sourceInputs').hidden = !['condition', 'iclora', 'retake', 'extend'].includes(mode)

  const optionMap = {
    retakeStartField: mode === 'retake',
    retakeEndField: mode === 'retake',
    extendDirectionField: mode === 'extend',
    extendSecondsField: mode === 'extend',
    extendContextField: mode === 'extend',
    strengthField: mode === 'refine_image',
    framePositionField: mode === 'ref2i',
  }
  Object.entries(optionMap).forEach(([id, visible]) => { el(`#${id}`).hidden = !visible })
  el('#modeOptions').hidden = !Object.values(optionMap).some(Boolean)

  const still = ['t2i', 'refine_image', 'ref2i'].includes(mode)
  const upscale = document.querySelector('input[name="upscale"]')
  const decoder = document.querySelector('select[name="decoder"]')
  const sourceInput = el('#sourceAsset input')
  sourceInput.accept = ['retake', 'extend'].includes(mode) ? 'video/*' : 'image/*,video/*'
  if (mode === 'ref2i' || ['iclora', 'retake', 'extend'].includes(mode)) {
    upscale.checked = false
    upscale.disabled = true
    decoder.value = 'vae'
    decoder.disabled = true
  } else {
    upscale.disabled = false
    decoder.disabled = false
    if (still) upscale.checked = true
  }
}

all('.mode').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)))

async function uploadAsset(slot, file, label) {
  if (!file) return
  label.classList.add('uploading')
  const strong = label.querySelector('strong')
  const previous = strong.textContent
  strong.textContent = '上传中…'
  try {
    const form = new FormData()
    form.append('file', file)
    const asset = await json('/api/assets', { method: 'POST', body: form })
    assets[slot] = asset
    label.classList.add('ready')
    strong.textContent = file.name
    label.querySelector('small').textContent = `${asset.kind} · ${(asset.size / 1024 / 1024).toFixed(1)} MB`
  } catch (error) {
    assets[slot] = null
    strong.textContent = previous
    el('#submitMessage').textContent = `上传失败：${error.message}`
  } finally {
    label.classList.remove('uploading')
  }
}

[['#firstAsset','first'], ['#lastAsset','last'], ['#sourceAsset','source'], ['#audioAsset','audio']].forEach(([selector, slot]) => {
  const label = el(selector)
  label.querySelector('input').addEventListener('change', event => uploadAsset(slot, event.target.files?.[0], label))
})

async function loadLoras() {
  try {
    loras = await json('/api/loras')
    el('#loraSelect').innerHTML = '<option value="">不使用</option>' + loras.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name || item.id)}</option>`).join('')
  } catch {
    loras = []
  }
}

function statusLabel(job) {
  if (job.status === 'queued') return '等待'
  if (job.status === 'running') return `${Math.round((job.progress || 0) * 100)}%`
  if (job.status === 'completed') return '完成'
  if (job.status === 'interrupted') return '已取消'
  if (job.status === 'failed') return '失败'
  return job.status
}

function modeLabel(mode = 't2av') { return mode.toUpperCase() }

function renderJobs(jobs) {
  const root = el('#queue')
  const active = jobs.filter(job => ['queued', 'running'].includes(job.status))
  const queuedOldestFirst = jobs.filter(job => job.status === 'queued').slice().reverse()
  const queuePosition = new Map(queuedOldestFirst.map((job, index) => [job.id, index + 1]))
  el('#queueSummary').textContent = `${active.length} active · ${jobs.length} shown`
  if (!jobs.length) {
    root.innerHTML = '<div class="empty">还没有任务。</div>'
    return
  }

  // Keep the queue visually stable: at most one full card. Prefer the running
  // job, then the oldest queued job, then the newest historical result.
  const featured = jobs.find(job => job.status === 'running')
    || queuedOldestFirst[0]
    || jobs[0]
  const rest = jobs.filter(job => job.id !== featured.id)

  const actions = job => {
    const activeAction = ['queued','running'].includes(job.status)
      ? `<button class="job-action danger" data-action="interrupt" data-id="${job.id}">取消</button>` : ''
    const deleteAction = ['completed','failed','interrupted'].includes(job.status)
      ? `<button class="job-action" data-action="delete" data-id="${job.id}">删除</button>` : ''
    const download = job.video_url || job.image_url
      ? `<a class="job-action" href="${job.video_url || job.image_url}" download>下载</a>` : ''
    return `${activeAction}${download}${deleteAction}`
  }

  const media = featured.video_url
    ? `<video controls preload="none" src="${featured.video_url}"></video>`
    : featured.image_url ? `<img src="${featured.image_url}" alt="result" loading="lazy" />` : ''
  const metric = featured.generation_seconds == null ? '' : `<span>${featured.generation_seconds.toFixed(1)}s GPU</span>`
  const vram = featured.peak_vram_gb == null ? '' : `<span>${featured.peak_vram_gb.toFixed(1)} GiB</span>`
  const waiting = featured.status === 'queued' ? `<span>队列 #${queuePosition.get(featured.id)}</span>` : ''
  const featuredHtml = `
    <article class="job job-featured job-${featured.status}">
      <div class="job-head">
        <div><strong>${statusLabel(featured)}</strong><span class="mode-chip">${modeLabel(featured.request?.mode)}</span><small>${featured.id.slice(0, 8)}</small></div>
        <div class="job-meta">${waiting}${metric}${vram}</div>
      </div>
      <p>${escapeHtml(featured.request?.prompt || '')}</p>
      ${featured.error ? `<pre>${escapeHtml(featured.error)}</pre>` : ''}
      ${media}
      <div class="job-actions">${actions(featured)}</div>
    </article>`

  const rowsHtml = rest.map(job => {
    const queue = job.status === 'queued' ? `#${queuePosition.get(job.id)}` : statusLabel(job)
    const metric = job.generation_seconds == null ? '' : `${job.generation_seconds.toFixed(1)}s`
    return `
      <div class="queue-row queue-row-${job.status}">
        <div class="queue-row-main">
          <strong>${queue}</strong>
          <span class="mode-chip">${modeLabel(job.request?.mode)}</span>
          <span class="queue-prompt">${escapeHtml(job.request?.prompt || '')}</span>
        </div>
        <div class="queue-row-side"><small>${metric || job.id.slice(0, 8)}</small>${actions(job)}</div>
      </div>`
  }).join('')

  root.innerHTML = featuredHtml + (rowsHtml ? `<div class="queue-list">${rowsHtml}</div>` : '')
}

async function loadJobs() {
  if (!sessionNumber) return renderJobs([])
  try {
    const jobs = await json(`/api/jobs?session_number=${sessionNumber}&limit=50`)
    renderJobs(jobs)
    const active = jobs.some(job => ['queued', 'running'].includes(job.status))
    clearTimeout(pollTimer)
    pollTimer = setTimeout(loadJobs, active ? 1800 : 6000)
  } catch {
    clearTimeout(pollTimer)
    pollTimer = setTimeout(loadJobs, 5000)
  }
}

el('#queue').addEventListener('click', async event => {
  const button = event.target.closest('[data-action]')
  if (!button) return
  button.disabled = true
  try {
    if (button.dataset.action === 'interrupt') {
      await json('/api/interrupt', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_id: button.dataset.id })
      })
    } else if (button.dataset.action === 'delete') {
      await json(`/api/jobs/${button.dataset.id}`, { method: 'DELETE' })
    }
    await loadJobs()
  } catch (error) {
    el('#submitMessage').textContent = `操作失败：${error.message}`
  } finally {
    button.disabled = false
  }
})

function optionalNumber(form, name) {
  const raw = String(form.get(name) ?? '').trim()
  return raw === '' ? null : Number(raw)
}

el('#generate').addEventListener('submit', async event => {
  event.preventDefault()
  const button = el('#submit')
  const message = el('#submitMessage')
  button.disabled = true
  message.textContent = '正在提交…'
  try {
    const form = new FormData(event.currentTarget)
    const mode = form.get('mode')
    if (mode === 'i2v' && !assets.first) throw new Error('I2V 需要先上传首帧图片')
    if (mode === 'flf2v' && (!assets.first || !assets.last)) throw new Error('FLF2V 需要首帧和末帧图片')
    if (['refine_image', 'ref2i'].includes(mode) && (!assets.first || assets.first.kind !== 'image')) throw new Error(`${modeLabel(mode)} 需要参考图片`)
    if (['condition', 'iclora', 'retake', 'extend'].includes(mode) && !assets.source) throw new Error(`${modeLabel(mode)} 需要参考素材`)
    if (['retake', 'extend'].includes(mode) && assets.source?.kind !== 'video') throw new Error(`${modeLabel(mode)} 需要源视频`)
    if (mode === 'a2v' && !assets.audio) throw new Error('A2V 需要先上传音频')
    const loraId = el('#loraSelect').value
    if (mode === 'iclora' && !loraId) throw new Error('IC-LORA 需要选择一个 IC-LoRA')
    if (mode === 'iclora' && !loras.find(item => item.id === loraId)?.generic_iclora_compatible) {
      throw new Error('所选 LoRA 不是兼容的通用 IC-LoRA')
    }

    const [width, height] = form.get('size').split('x').map(Number)
    const session = await ensureSession()
    const conditions = []
    if (['i2v','flf2v','refine_image'].includes(mode)) conditions.push({ asset_id: assets.first.id, kind: 'image', index: 0, strength: 1 })
    if (mode === 'ref2i') conditions.push({ asset_id: assets.first.id, kind: 'image', index: 0, strength: 1 })
    if (mode === 'flf2v') conditions.push({ asset_id: assets.last.id, kind: 'image', index: -1, strength: 1 })
    if (mode === 'condition') conditions.push({ asset_id: assets.source.id, kind: assets.source.kind, index: 0, strength: 1 })
    if (mode === 'iclora') conditions.push({ asset_id: assets.source.id, kind: assets.source.kind, index: 1, strength: 1 })
    if (['retake','extend'].includes(mode)) conditions.push({ asset_id: assets.source.id, kind: 'video', index: 0, strength: 1 })
    const body = {
      session_number: session,
      mode,
      prompt: form.get('prompt').trim(),
      negative_prompt: form.get('negative_prompt').trim(),
      width, height,
      num_frames: mode === 'ref2i' ? 49 : Number(form.get('num_frames')),
      fps: Number(form.get('fps')),
      steps: Number(form.get('steps')),
      guidance_scale: Number(form.get('guidance_scale')),
      seed: Number(form.get('seed')),
      upscale: form.get('upscale') === 'on',
      upscale_method: 'latent',
      temporal_upscale: false,
      decoder: form.get('decoder'),
      conditions,
      audio_asset_id: mode === 'a2v' ? assets.audio.id : null,
      modality_scale: optionalNumber(form, 'modality_scale'),
      audio_guidance_scale: optionalNumber(form, 'audio_guidance_scale'),
      retake_start: mode === 'retake' ? Number(form.get('retake_start')) : null,
      retake_end: mode === 'retake' ? Number(form.get('retake_end')) : null,
      regenerate_video: true,
      regenerate_audio: true,
      extend_direction: mode === 'extend' ? form.get('extend_direction') : 'end',
      extend_seconds: mode === 'extend' ? Number(form.get('extend_seconds')) : 5,
      extend_context_seconds: mode === 'extend' ? Number(form.get('extend_context_seconds')) : 3,
      strength: mode === 'refine_image' ? Number(form.get('strength')) : 1,
      frame_position: mode === 'ref2i' ? form.get('frame_position') : 'last',
      loras: loraId ? [{ id: loraId, strength: Number(el('#loraStrength').value || 1) }] : [],
    }
    const batch = Math.max(1, Math.min(20, Number(form.get('batch') || 1)))
    const jobs = []
    for (let index = 0; index < batch; index += 1) {
      const request = { ...body, seed: body.seed + index }
      const job = await json('/api/jobs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request)
      })
      jobs.push(job)
      message.textContent = batch === 1
        ? `已加入队列 · ${modeLabel(mode)} · ${job.id.slice(0, 8)}`
        : `批量提交中 ${index + 1}/${batch} · ${job.id.slice(0, 8)}`
    }
    message.textContent = batch === 1
      ? `已加入队列 · ${modeLabel(mode)} · ${jobs[0].id.slice(0, 8)}`
      : `已加入 ${jobs.length} 个任务 · Seed ${body.seed}–${body.seed + jobs.length - 1}`
    await loadJobs()
  } catch (error) {
    message.textContent = `提交失败：${error.message}`
  } finally {
    button.disabled = false
  }
})

el('#refresh').addEventListener('click', loadJobs)
el('#health').addEventListener('click', checkHealth)
window.addEventListener('online', () => { checkHealth(); loadJobs() })

await checkHealth()
await ensureSession()
el('#sessionBadge').textContent = `SESSION ${sessionNumber}`
await Promise.all([loadLoras(), loadJobs()])

