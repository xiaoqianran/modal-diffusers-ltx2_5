import './style.css'

const app = document.querySelector('#app')
const SESSION_KEY = 'ltx25.session'
let sessionNumber = Number(localStorage.getItem(SESSION_KEY)) || null
let pollTimer = null
let loras = []
const assets = { first: null, last: null, audio: null }

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
  try {
    const health = await json('/api/health')
    badge.textContent = `${health.transformer_precision?.toUpperCase() || 'GPU'} · ${health.worker}`
    badge.dataset.state = 'online'
  } catch (error) {
    badge.textContent = `离线 · ${error.message}`
    badge.dataset.state = 'offline'
  }
}

function setMode(mode) {
  el('#mode').value = mode
  all('.mode').forEach(button => button.classList.toggle('active', button.dataset.mode === mode))
  el('#visualInputs').hidden = !['i2v', 'flf2v'].includes(mode)
  el('#firstAsset').hidden = !['i2v', 'flf2v'].includes(mode)
  el('#lastAsset').hidden = mode !== 'flf2v'
  el('#audioInputs').hidden = mode !== 'a2v'
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

[['#firstAsset','first'], ['#lastAsset','last'], ['#audioAsset','audio']].forEach(([selector, slot]) => {
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
  root.innerHTML = jobs.map(job => {
    const media = job.video_url
      ? `<video controls preload="none" src="${job.video_url}"></video>`
      : job.image_url ? `<img src="${job.image_url}" alt="result" loading="lazy" />` : ''
    const metric = job.generation_seconds == null ? '' : `<span>${job.generation_seconds.toFixed(1)}s GPU</span>`
    const vram = job.peak_vram_gb == null ? '' : `<span>${job.peak_vram_gb.toFixed(1)} GiB</span>`
    const waiting = job.status === 'queued' ? `<span>队列 #${queuePosition.get(job.id)}</span>` : ''
    const activeActions = ['queued','running'].includes(job.status)
      ? `<button class="job-action danger" data-action="interrupt" data-id="${job.id}">取消</button>` : ''
    const deleteAction = ['completed','failed','interrupted'].includes(job.status)
      ? `<button class="job-action" data-action="delete" data-id="${job.id}">删除</button>` : ''
    const download = job.video_url || job.image_url
      ? `<a class="job-action" href="${job.video_url || job.image_url}" download>下载</a>` : ''
    return `
      <article class="job job-${job.status}">
        <div class="job-head">
          <div><strong>${statusLabel(job)}</strong><span class="mode-chip">${modeLabel(job.request?.mode)}</span><small>${job.id.slice(0, 8)}</small></div>
          <div class="job-meta">${waiting}${metric}${vram}</div>
        </div>
        <p>${escapeHtml(job.request?.prompt || '')}</p>
        ${job.error ? `<pre>${escapeHtml(job.error)}</pre>` : ''}
        ${media}
        <div class="job-actions">${activeActions}${download}${deleteAction}</div>
      </article>`
  }).join('')
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
    if (mode === 'a2v' && !assets.audio) throw new Error('A2V 需要先上传音频')
    const [width, height] = form.get('size').split('x').map(Number)
    const session = await ensureSession()
    const conditions = []
    if (['i2v','flf2v'].includes(mode)) conditions.push({ asset_id: assets.first.id, kind: 'image', index: 0, strength: 1 })
    if (mode === 'flf2v') conditions.push({ asset_id: assets.last.id, kind: 'image', index: -1, strength: 1 })
    const loraId = el('#loraSelect').value
    const body = {
      session_number: session,
      mode,
      prompt: form.get('prompt').trim(),
      negative_prompt: form.get('negative_prompt').trim(),
      width, height,
      num_frames: Number(form.get('num_frames')),
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

