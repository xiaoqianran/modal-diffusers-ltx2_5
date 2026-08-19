const form = document.querySelector('#form');
const tr = (ja, en) => window.LTX_I18N?.language === 'en' ? en : ja;
const submit = document.querySelector('#submit');
const mode = document.querySelector('#mode');
const upscale = document.querySelector('#upscale');
const temporalUpscale = document.querySelector('#temporalUpscale');
const decoder = document.querySelector('#decoder');
const size = document.querySelector('#size');
const simple = document.querySelector('#simpleConditions');
const advanced = document.querySelector('#advancedConditions');
const icLoraEditor = document.querySelector('#icLoraEditor');
const retakeEditor = document.querySelector('#retakeEditor');
const extendEditor = document.querySelector('#extendEditor');
const audioVideoEditor = document.querySelector('#audioVideoEditor');
const rows = document.querySelector('#conditionRows');
const promptInput = document.querySelector('#prompt');
const multishot = document.querySelector('#multishot');
const shotRows = document.querySelector('#shotRows');
const autoDuration = document.querySelector('#autoDuration');
const loraRows = document.querySelector('#loraRows');
let availableLoras = [];
let sessionNumber = null;
let lastJobMetrics = null;
let selectedHistoryIds = [];
const views = ['empty', 'working', 'result', 'error'];
const modeHelp = {
  t2av: 'テキストから同期した映像と音声を生成します。',
  i2v: '先頭画像を保ちながら映像と音声を生成します。',
  flf2v: '先頭画像と末尾画像の間を映像と音声で補間します。',
  condition: '複数の参照画像・動画を、生成映像内の配置位置と反映強度で指定します。',
  iclora: 'IC-LoRAの参照latentを生成latentへ追加し、人物・小道具・場所の一貫性を保って新しい映像を生成します。',
  retake: '元動画の指定時間範囲だけを再生成し、範囲外を維持します。',
  extend: '元動画の動きと音を参照し、先頭または末尾へ新しい区間を追加します。',
  a2v: '入力音声を固定し、発話・音楽・効果音のタイミングに同期する映像を生成します。',
};

function show(id) {
  views.forEach((view) => { document.querySelector(`#${view}`).hidden = view !== id; });
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function ensureSession() {
  const stored = sessionStorage.getItem('ltx25SessionNumber');
  if (stored) {
    sessionNumber = +stored;
  } else {
    const response = await fetch('/api/sessions', { method: 'POST' });
    if (!response.ok) throw new Error(tr('セッションを開始できません', 'Could not start a session'));
    sessionNumber = (await response.json()).session_number;
    sessionStorage.setItem('ltx25SessionNumber', sessionNumber);
  }
  document.querySelector('#sessionBadge').textContent = `SESSION ${String(sessionNumber).padStart(4, '0')}`;
  await loadHistory();
}

function addShot(value = '') {
  const number = shotRows.children.length + 1;
  const row = document.createElement('div');
  row.className = 'shot-row';
  row.innerHTML = `<label><b>SHOT ${number}</b><textarea placeholder="構図、アクション、カメラ、セリフ…"></textarea></label><button type="button" class="remove" aria-label="削除">×</button>`;
  row.querySelector('textarea').value = value;
  row.querySelector('.remove').onclick = () => { row.remove(); renumberShots(); updateConditionSummary(); };
  shotRows.append(row);
  updateConditionSummary();
}

function renumberShots() {
  shotRows.querySelectorAll('.shot-row b').forEach((label, index) => { label.textContent = `SHOT ${index + 1}`; });
}

function getShots() {
  return [...shotRows.querySelectorAll('textarea')].map((item) => item.value.trim()).filter(Boolean);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function addLoraRow(selected = '', strength = 1) {
  if (loraRows.children.length >= 4) return;
  const row = document.createElement('div');
  row.className = 'lora-row';
  const options = availableLoras.map((item) =>
    `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}${item.kind === 'iclora' ? ` · IC ×${item.reference_downscale_factor ?? '?'}` : ''} · ${formatBytes(item.size)}</option>`
  ).join('');
  row.innerHTML = `<label>LoRA<select><option value="">選択してください</option>${options}</select></label><label>強度<input type="number" min="-2" max="2" step="0.05" value="${strength}"></label><button type="button" class="remove" aria-label="削除">×</button>`;
  row.querySelector('select').value = selected;
  row.querySelector('.remove').onclick = () => { row.remove(); updateConditionSummary(); };
  row.querySelectorAll('select,input').forEach((input) => input.addEventListener('input', updateConditionSummary));
  loraRows.append(row);
  updateConditionSummary();
}

function getLoras() {
  const selected = [...loraRows.querySelectorAll('.lora-row')].map((row) => ({
    id: row.querySelector('select').value,
    strength: +row.querySelector('input').value,
  })).filter((item) => item.id);
  if (new Set(selected.map((item) => item.id)).size !== selected.length) {
    throw new Error(tr('同じLoRAを複数回選択することはできません', 'The same LoRA cannot be selected more than once'));
  }
  if (mode.value === 'iclora' && selected.some((item) => !availableLoras.find((entry) => entry.id === item.id)?.generic_iclora_compatible)) {
    throw new Error(tr('この汎用IC-LoRAモードでは参照縮小率1のIC-LoRAだけを使用できます', 'This generic IC-LoRA mode accepts only IC-LoRAs with reference downscale factor 1'));
  }
  return selected;
}

async function loadLoras() {
  const status = document.querySelector('#loraStatus');
  try {
    const response = await fetch('/api/loras');
    if (!response.ok) throw new Error(response.statusText);
    availableLoras = await response.json();
    loraRows.replaceChildren();
    status.textContent = availableLoras.length
      ? `${availableLoras.length}件を検出 · IC-LoRAは参照縮小率も表示します`
      : 'loras/ にLTX-2/2.5互換の .safetensors を配置してください';
  } catch (error) {
    status.textContent = `LoRA一覧を取得できません: ${error.message}`;
  }
  updateConditionSummary();
}

function buildPrompt() {
  const base = promptInput.value.trim();
  if (!multishot.checked) return base;
  const shots = getShots();
  if (!shots.length) throw new Error(tr('Native Multishotのショットを1つ以上入力してください', 'Enter at least one Native Multishot shot'));
  const continuity = 'Keep character identity, wardrobe, environment, lighting, voice, and visual style consistent across every shot.';
  return [base, continuity, ...shots.map((shot, index) => `${index ? 'Hard cut to' : 'Shot'} ${index + 1}: ${shot}`)].join('\n\n');
}

function updateDurationUI() {
  form.elements.num_frames.disabled = autoDuration.checked;
  form.elements.duration_seconds.disabled = autoDuration.checked;
  document.querySelector('#durationRange').hidden = !autoDuration.checked;
  const frames = +form.elements.num_frames.value;
  const fps = +form.elements.fps.value || 24;
  const secondsInput = form.elements.duration_seconds;
  const minSeconds = 8 / fps;
  const maxSeconds = 480 / fps;
  secondsInput.min = minSeconds.toFixed(3);
  secondsInput.max = maxSeconds.toFixed(3);
  document.querySelector('#secondsHelp').textContent =
    `${minSeconds.toFixed(2)}〜${maxSeconds.toFixed(2)}秒・8n+1フレームへ自動調整`;
  document.querySelector('#durationEstimate').textContent = `約${((frames - 1) / fps).toFixed(1)}秒`;
}

function syncSecondsFromFrames(normalize = false) {
  const fps = +form.elements.fps.value || 24;
  let frames = +form.elements.num_frames.value || 9;
  if (normalize) {
    frames = Math.min(481, Math.max(9, Math.round((frames - 1) / 8) * 8 + 1));
    form.elements.num_frames.value = frames;
  }
  form.elements.duration_seconds.value = ((frames - 1) / fps).toFixed(2);
}

function syncFramesFromSeconds() {
  const fps = +form.elements.fps.value || 24;
  const seconds = Math.min(480 / fps, Math.max(8 / fps, +form.elements.duration_seconds.value || 0));
  const intervals = Math.min(480, Math.max(8, Math.round((seconds * fps) / 8) * 8));
  form.elements.num_frames.value = intervals + 1;
  form.elements.duration_seconds.value = (intervals / fps).toFixed(2);
  updateDurationUI();
  updateConditionSummary();
}

function updateConditionSummary() {
  const summary = document.querySelector('#conditionSummary');
  const [width, height] = size.value.split('x').map(Number);
  const finalWidth = upscale.checked ? width * 2 : width;
  const finalHeight = upscale.checked ? height * 2 : height;
  const finalFps = +form.elements.fps.value * (temporalUpscale.checked ? 2 : 1);
  const baseFrames = +form.elements.num_frames.value;
  const finalFrames = temporalUpscale.checked ? (baseFrames - 1) * 2 + 1 : baseFrames;
  const duration = autoDuration.checked
    ? `自動 ${form.elements.min_seconds.value}〜${form.elements.max_seconds.value}秒`
    : `${form.elements.num_frames.value}フレーム（約${((+form.elements.num_frames.value - 1) / +form.elements.fps.value).toFixed(1)}秒）`;
  const referenceCount = mode.value === 'a2v'
    ? (document.querySelector('#audioFirstFrame input[type="file"]')?.files.length || 0)
    : mode.value === 'iclora' ? (document.querySelector('#icLoraReference input[type="file"]')?.files.length || 0)
    : ['retake', 'extend'].includes(mode.value) ? 1 : mode.value === 'condition'
    ? rows.querySelectorAll('.condition-row').length
    : mode.value === 'flf2v' ? 2 : mode.value === 'i2v' ? 1 : 0;
  const values = [
    ['生成方式', mode.options[mode.selectedIndex].textContent],
    ['参照入力', referenceCount ? `${referenceCount}件` : 'なし'],
    ['尺', duration],
    ['FPS', temporalUpscale.checked ? `${form.elements.fps.value} → ${finalFps}` : form.elements.fps.value],
    ['最終フレーム数', autoDuration.checked ? '自動' : `${finalFrames}`],
    ['基準解像度', `${width} × ${height}`],
    ['最終解像度', `${finalWidth} × ${finalHeight}`],
    ['2倍高解像度化', upscale.checked ? 'ON' : 'OFF'],
    ['2倍フレームレート化', temporalUpscale.checked ? 'ON' : 'OFF'],
    ['デコーダー', (upscale.checked || temporalUpscale.checked) ? decoder.options[decoder.selectedIndex].textContent : 'VAE（高速）'],
    ['Multishot', multishot.checked ? `${Math.max(getShots().length, shotRows.children.length)} shots` : 'OFF'],
    ['LoRA', [...loraRows.querySelectorAll('select')].filter((item) => item.value).map((item) => item.options[item.selectedIndex].textContent.split(' · ')[0]).join(' ＋ ') || 'なし'],
    ['Seed', form.elements.seed.value],
  ];
  if (lastJobMetrics) {
    values.push(
      ['生成時間', `${lastJobMetrics.generation_seconds.toFixed(1)}秒`],
      ['最大VRAM使用量', lastJobMetrics.peak_vram_gb == null ? '取得不可' : `${lastJobMetrics.peak_vram_gb.toFixed(2)} GB`],
    );
  }
  summary.replaceChildren();
  values.forEach(([term, description]) => {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = term;
    dd.textContent = description;
    summary.append(dt, dd);
  });
}

function fileField(label, index) {
  const field = document.createElement('label');
  field.className = 'file-field drop-zone';
  field.dataset.index = index;
  field.innerHTML = `<b>${label}</b><span class="drop-copy">ここへ画像をドロップ、またはクリックして選択</span><input type="file" required accept="image/jpeg,image/png,image/webp"><small class="file-name">JPG / PNG / WebP · strength 1.0</small><span class="drop-preview" hidden><img alt="選択した参照画像"></span>`;
  bindDropZone(field);
  return field;
}

function fileIsAccepted(file, input) {
  const name = file.name.toLowerCase();
  const isImage = fileIsImage(file);
  const isVideo = file.type.startsWith('video/') || /\.(mp4|mov|webm|mkv)$/.test(name);
  return input.accept.includes('video') ? isImage || isVideo : isImage && !name.endsWith('.gif');
}

function fileIsImage(file) {
  return file.type.startsWith('image/') || /\.(jpe?g|png|webp|gif)$/i.test(file.name);
}

function updateThumbnail(zone, file) {
  const preview = zone.querySelector('.drop-preview');
  const image = preview?.querySelector('img');
  if (!preview || !image) return;
  if (image.dataset.objectUrl) URL.revokeObjectURL(image.dataset.objectUrl);
  image.removeAttribute('src');
  delete image.dataset.objectUrl;
  preview.hidden = true;
  if (!file || !fileIsImage(file)) return;
  const url = URL.createObjectURL(file);
  image.dataset.objectUrl = url;
  image.src = url;
  preview.hidden = false;
}

function releaseThumbnails(container) {
  container.querySelectorAll('.drop-preview img[data-object-url]').forEach((image) => {
    URL.revokeObjectURL(image.dataset.objectUrl);
  });
}

function updateRenderHelp() {
  const [width, height] = size.value.split('x').map(Number);
  const outputWidth = upscale.checked ? width * 2 : width;
  const outputHeight = upscale.checked ? height * 2 : height;
  const useRefine = upscale.checked || temporalUpscale.checked;
  decoder.disabled = !useRefine;
  if (!useRefine) decoder.value = 'vae';
  const fps = +form.elements.fps.value || 24;
  const renderSteps = [];
  if (upscale.checked) renderSteps.push('空間Latent ×2');
  if (temporalUpscale.checked) renderSteps.push(`時間Latent ×2（${fps} → ${fps * 2} FPS）`);
  document.querySelector('#renderHelp').textContent = useRefine
    ? `基準${width} × ${height} → ${renderSteps.join(' ＋ ')} ＋ 3-step Refine → 最終${outputWidth} × ${outputHeight}`
    : `8-step単段生成 → 最終${outputWidth} × ${outputHeight}（VAEデコード）`;
}

function matchSizeToImage(file) {
  if (!file || (!file.type.startsWith('image/') && !/\.(jpe?g|png|webp)$/i.test(file.name))) return;
  const image = new Image();
  const url = URL.createObjectURL(file);
  image.onload = () => {
    const ratio = image.naturalWidth / image.naturalHeight;
    size.value = ratio > 1.15 ? '768x512' : ratio < 0.87 ? '512x768' : '512x512';
    updateRenderHelp();
    updateConditionSummary();
    URL.revokeObjectURL(url);
  };
  image.onerror = () => URL.revokeObjectURL(url);
  image.src = url;
}

function bindDropZone(zone) {
  const input = zone.querySelector('input[type="file"]');
  const fileName = zone.querySelector('.file-name');
  const initialText = fileName.textContent;

  function displaySelection() {
    zone.classList.remove('invalid');
    zone.classList.toggle('has-file', input.files.length > 0);
    fileName.textContent = input.files[0]?.name || initialText;
    updateThumbnail(zone, input.files[0]);
    if (zone.classList.contains('file-field')) matchSizeToImage(input.files[0]);
  }

  input.addEventListener('change', displaySelection);
  ['dragenter', 'dragover'].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
      zone.classList.add('drag-over');
    });
  });
  ['dragleave', 'dragend'].forEach((eventName) => {
    zone.addEventListener(eventName, () => zone.classList.remove('drag-over'));
  });
  zone.addEventListener('drop', async (event) => {
    event.preventDefault();
    event.stopPropagation();
    zone.classList.remove('drag-over');
    let file = event.dataTransfer?.files[0];
    const historyUrl = event.dataTransfer?.getData('application/x-ltx-history-video');
    if (!file && historyUrl) {
      try {
        fileName.textContent = '履歴動画を読み込み中…';
        const response = await fetch(historyUrl);
        if (!response.ok) throw new Error(response.statusText);
        const blob = await response.blob();
        file = new File([blob], historyUrl.split('/').pop() || 'history.mp4', {
          type: blob.type || 'video/mp4',
        });
      } catch (error) {
        zone.classList.add('invalid');
        fileName.textContent = `履歴動画を取得できません: ${error.message}`;
        return;
      }
    }
    if (!file) return;
    if (!fileIsAccepted(file, input)) {
      input.value = '';
      zone.classList.remove('has-file');
      zone.classList.add('invalid');
      fileName.textContent = 'この形式は使用できません';
      updateThumbnail(zone);
      return;
    }
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    displaySelection();
  });
}

function addCondition() {
  const row = document.createElement('div');
  row.className = 'condition-row';
  row.innerHTML = '<label class="drop-zone compact"><b>画像または動画</b><span class="drop-copy">ここへドロップ、またはクリック</span><input class="condition-file" type="file" required accept="image/*,video/*,.gif"><small class="file-name">未選択</small><span class="drop-preview" hidden><img alt="選択した参照画像"></span></label><label>配置位置（%）<input class="condition-position" type="number" min="0" max="100" step="1" value="0"><small>0=先頭 / 100=末尾</small></label><label>反映強度<input class="condition-strength" type="number" min="0" max="1" step="0.05" value="1"></label><button type="button" class="remove" aria-label="削除">×</button>';
  row.querySelector('.remove').onclick = () => {
    releaseThumbnails(row);
    row.remove();
    updateConditionSummary();
  };
  bindDropZone(row.querySelector('.drop-zone'));
  rows.append(row);
  updateConditionSummary();
}

function renderMode() {
  const selected = mode.value;
  document.querySelector('#modeHelp').textContent = modeHelp[selected];
  releaseThumbnails(simple);
  simple.replaceChildren();
  simple.hidden = !['i2v', 'flf2v'].includes(selected);
  advanced.hidden = selected !== 'condition';
  icLoraEditor.hidden = selected !== 'iclora';
  retakeEditor.hidden = selected !== 'retake';
  extendEditor.hidden = selected !== 'extend';
  audioVideoEditor.hidden = selected !== 'a2v';
  const sourceEdit = ['retake', 'extend', 'iclora'].includes(selected);
  upscale.disabled = sourceEdit;
  temporalUpscale.disabled = sourceEdit;
  if (sourceEdit) {
    upscale.checked = false;
    temporalUpscale.checked = false;
    decoder.value = 'vae';
  }
  if (selected === 'i2v') simple.append(fileField('先頭画像', 0));
  if (selected === 'flf2v') simple.append(fileField('先頭画像', 0), fileField('末尾画像', -1));
  if (selected === 'condition' && !rows.children.length) addCondition();
  if (selected === 'iclora') {
    size.value = '768x448';
    autoDuration.checked = false;
    autoDuration.disabled = true;
  } else {
    autoDuration.disabled = false;
  }
  updateRenderHelp();
}

mode.addEventListener('change', renderMode);
upscale.addEventListener('change', updateRenderHelp);
temporalUpscale.addEventListener('change', updateRenderHelp);
size.addEventListener('change', updateRenderHelp);
autoDuration.addEventListener('change', updateDurationUI);
form.elements.num_frames.addEventListener('input', updateDurationUI);
form.elements.num_frames.addEventListener('change', () => { syncSecondsFromFrames(true); updateDurationUI(); });
form.elements.duration_seconds.addEventListener('change', syncFramesFromSeconds);
form.elements.fps.addEventListener('input', () => { syncSecondsFromFrames(); updateDurationUI(); });
form.addEventListener('input', updateConditionSummary);
form.addEventListener('change', updateConditionSummary);
multishot.addEventListener('change', () => {
  document.querySelector('#multishotEditor').hidden = !multishot.checked;
  if (multishot.checked && !shotRows.children.length) {
    addShot();
    addShot();
  }
});
document.querySelector('#addCondition').onclick = addCondition;
document.querySelector('#addShot').onclick = () => addShot();
renderMode();
bindDropZone(document.querySelector('#retakeSource'));
bindDropZone(document.querySelector('#extendSource'));
bindDropZone(document.querySelector('#audioSource'));
bindDropZone(document.querySelector('#audioFirstFrame'));
bindDropZone(document.querySelector('#icLoraReference'));
updateRenderHelp();
syncSecondsFromFrames();
updateDurationUI();
updateConditionSummary();

async function upload(file) {
  const data = new FormData();
  data.append('file', file);
  const response = await fetch('/api/assets', { method: 'POST', body: data });
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.json();
}

async function collectConditions(selectedMode) {
  const conditions = [];
  if (['i2v', 'flf2v'].includes(selectedMode)) {
    for (const field of simple.querySelectorAll('.file-field')) {
      const asset = await upload(field.querySelector('input').files[0]);
      conditions.push({ asset_id: asset.id, kind: 'image', index: +field.dataset.index, strength: 1 });
    }
  } else if (selectedMode === 'condition') {
    const numFrames = autoDuration.checked ? null : +form.elements.num_frames.value;
    if (numFrames === null) {
      const invalid = [...rows.querySelectorAll('.condition-position')]
        .some((input) => ![0, 100].includes(+input.value));
      if (invalid) throw new Error(tr('自動尺では参照条件の配置位置を0%または100%にしてください', 'With automatic duration, place references at 0% or 100%'));
    }
    const latentFrames = numFrames === null ? null : Math.floor((numFrames - 1) / 8) + 1;
    for (const row of rows.querySelectorAll('.condition-row')) {
      const asset = await upload(row.querySelector('.condition-file').files[0]);
      const position = +row.querySelector('.condition-position').value;
      const index = position >= 100 ? -1 : position <= 0 ? 0 : Math.round((position / 100) * (latentFrames - 1));
      conditions.push({
        asset_id: asset.id,
        kind: asset.kind,
        index,
        strength: +row.querySelector('.condition-strength').value,
      });
    }
  } else if (selectedMode === 'iclora') {
    const file = document.querySelector('#icLoraReference input[type="file"]').files[0];
    if (!file) throw new Error(tr('IC-LoRAの参照画像または動画を選択してください', 'Select an IC-LoRA reference image or video'));
    const asset = await upload(file);
    conditions.push({ asset_id: asset.id, kind: asset.kind, index: 1, strength: 1 });
  } else if (selectedMode === 'retake') {
    const file = document.querySelector('#retakeSource input[type="file"]').files[0];
    if (!file) throw new Error(tr('Retakeの元動画を選択してください', 'Select a source video for Retake'));
    const asset = await upload(file);
    conditions.push({ asset_id: asset.id, kind: 'video', index: 0, strength: 1 });
  } else if (selectedMode === 'extend') {
    const file = document.querySelector('#extendSource input[type="file"]').files[0];
    if (!file) throw new Error(tr('Extendの元動画を選択してください', 'Select a source video for Extend'));
    const asset = await upload(file);
    conditions.push({ asset_id: asset.id, kind: 'video', index: 0, strength: 1 });
  } else if (selectedMode === 'a2v') {
    const image = document.querySelector('#audioFirstFrame input[type="file"]').files[0];
    if (image) {
      const asset = await upload(image);
      conditions.push({ asset_id: asset.id, kind: 'image', index: 0, strength: 1 });
    }
  }
  return conditions;
}

async function enhancePrompt() {
  const button = document.querySelector('#enhancePrompt');
  const status = document.querySelector('#enhanceStatus');
  button.disabled = true;
  status.textContent = '外部LLMで変換中…';
  try {
    const response = await fetch('/api/prompts/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: promptInput.value, mode: mode.value, shots: multishot.checked ? getShots() : [] }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const result = await response.json();
    promptInput.value = result.prompt;
    if (multishot.checked) {
      multishot.checked = false;
      document.querySelector('#multishotEditor').hidden = true;
    }
    status.textContent = `${result.model}で変換しました`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value;
  return node.innerHTML;
}

function updateHistorySelection() {
  document.querySelector('#historySelection').textContent = `${selectedHistoryIds.length}件選択`;
  document.querySelector('#deleteHistory').disabled = selectedHistoryIds.length === 0;
  document.querySelector('#concatHistory').disabled = selectedHistoryIds.length < 2;
}

function openVideoModal(url, filename = 'video.mp4') {
  const dialog = document.querySelector('#videoModal');
  const player = document.querySelector('#modalVideo');
  player.src = url;
  document.querySelector('#modalDownload').href = url;
  document.querySelector('#modalDownload').download = filename;
  dialog.showModal();
  player.play().catch(() => {});
}

function closeVideoModal() {
  const dialog = document.querySelector('#videoModal');
  const player = document.querySelector('#modalVideo');
  player.pause();
  player.removeAttribute('src');
  player.load();
  dialog.close();
}

async function loadHistory() {
  if (!sessionNumber) return;
  const list = document.querySelector('#historyList');
  try {
    const response = await fetch(`/api/jobs?session_number=${sessionNumber}&limit=50`);
    if (!response.ok) throw new Error(response.statusText);
    const jobs = await response.json();
    if (!jobs.length) {
      list.innerHTML = '<p class="hint">このセッションの生成履歴はまだありません。</p>';
      return;
    }
    list.innerHTML = jobs.map((job) => {
      const request = job.request;
      const date = new Date(job.created_at).toLocaleString('ja-JP');
      const checked = selectedHistoryIds.includes(job.id) ? ' checked' : '';
      const video = job.video_url ? `<video src="${job.video_url}" controls preload="metadata" draggable="true" data-history-url="${job.video_url}" data-job-id="${job.id}" title="クリックで原寸表示・入力欄へドラッグできます"></video>` : '';
      const controls = job.video_url ? `<div class="history-card-actions"><label class="history-select"><input type="checkbox" data-job-id="${job.id}"${checked}> 選択</label><a href="${job.video_url}" download="${job.id}.mp4">ダウンロード</a></div>` : '';
      const render = [request.upscale ? '2x resolution' : '', request.temporal_upscale ? '2x FPS' : ''].filter(Boolean).join(' + ') || 'base';
      const metrics = job.generation_seconds == null ? '' : ` · ${job.generation_seconds.toFixed(1)}秒 · VRAM ${job.peak_vram_gb?.toFixed(2) ?? '?'} GB`;
      return `<article class="history-item"><div class="history-meta"><b>#${job.session_number} · ${job.status}</b><span>${date}</span></div>${video}${controls}<p>${escapeHtml(request.prompt)}</p><small>${request.mode} · seed ${request.seed} · ${render}${metrics}</small></article>`;
    }).join('');
    list.querySelectorAll('video[data-history-url]').forEach((video) => {
      let dragged = false;
      video.addEventListener('dragstart', (event) => {
        dragged = true;
        event.dataTransfer.effectAllowed = 'copy';
        event.dataTransfer.setData('application/x-ltx-history-video', video.dataset.historyUrl);
        event.dataTransfer.setData('text/uri-list', new URL(video.dataset.historyUrl, location.href).href);
      });
      video.addEventListener('dragend', () => { setTimeout(() => { dragged = false; }, 0); });
      video.addEventListener('click', () => {
        if (!dragged) openVideoModal(video.dataset.historyUrl, `${video.dataset.jobId}.mp4`);
      });
    });
    list.querySelectorAll('.history-select input').forEach((checkbox) => {
      checkbox.addEventListener('change', () => {
        const id = checkbox.dataset.jobId;
        if (checkbox.checked && !selectedHistoryIds.includes(id)) selectedHistoryIds.push(id);
        if (!checkbox.checked) selectedHistoryIds = selectedHistoryIds.filter((item) => item !== id);
        updateHistorySelection();
      });
    });
    updateHistorySelection();
  } catch (error) {
    list.innerHTML = `<p class="hint">履歴を読み込めません: ${escapeHtml(error.message)}</p>`;
  }
}

document.querySelector('#enhancePrompt').onclick = enhancePrompt;
document.querySelector('#refreshHistory').onclick = loadHistory;
document.querySelector('#deleteHistory').onclick = async () => {
  if (!selectedHistoryIds.length || !confirm(tr(
    `選択した${selectedHistoryIds.length}件の履歴と動画ファイルを削除しますか？`,
    `Delete ${selectedHistoryIds.length} selected history item(s) and video file(s)?`,
  ))) return;
  const button = document.querySelector('#deleteHistory');
  button.disabled = true;
  try {
    for (const id of selectedHistoryIds) {
      const response = await fetch(`/api/jobs/${id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    }
    selectedHistoryIds = [];
    await loadHistory();
  } catch (error) {
    alert(`${tr('削除できません', 'Could not delete')}: ${error.message}`);
    updateHistorySelection();
  }
};
document.querySelector('#concatHistory').onclick = async () => {
  const button = document.querySelector('#concatHistory');
  button.disabled = true;
  button.textContent = '結合中…';
  try {
    const response = await fetch('/api/jobs/concat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_ids: selectedHistoryIds }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const result = await response.json();
    openVideoModal(result.video_url, result.filename);
  } catch (error) {
    alert(`${tr('結合できません', 'Could not merge')}: ${error.message}`);
  } finally {
    button.textContent = '選択順に結合';
    updateHistorySelection();
  }
};
document.querySelector('#closeVideoModal').onclick = closeVideoModal;
document.querySelector('#videoModal').addEventListener('click', (event) => {
  if (event.target === event.currentTarget) closeVideoModal();
});
document.querySelector('#addLora').onclick = () => addLoraRow();
document.querySelector('#refreshLoras').onclick = loadLoras;
loadLoras();
ensureSession().catch((error) => { document.querySelector('#sessionBadge').textContent = error.message; });

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!autoDuration.checked) syncFramesFromSeconds();
  lastJobMetrics = null;
  updateConditionSummary();
  submit.disabled = true;
  show('working');
  const data = new FormData(form);
  const [width, height] = data.get('size').split('x').map(Number);
  try {
    if (!sessionNumber) await ensureSession();
    document.querySelector('#state').textContent = '入力ファイルを準備中';
    const conditions = await collectConditions(data.get('mode'));
    let audioAssetId = null;
    if (data.get('mode') === 'a2v') {
      const audioFile = document.querySelector('#audioSource input[type="file"]').files[0];
      if (!audioFile) throw new Error(tr('Audio → Videoの入力音声を選択してください', 'Select input audio for Audio → Video'));
      audioAssetId = (await upload(audioFile)).id;
    }
    const body = {
      session_number: sessionNumber,
      mode: data.get('mode'),
      upscale: data.get('upscale') === 'on',
      temporal_upscale: data.get('temporal_upscale') === 'on',
      decoder: data.get('decoder'),
      prompt: buildPrompt(),
      negative_prompt: data.get('negative_prompt'),
      width,
      height,
      num_frames: autoDuration.checked ? null : +data.get('num_frames'),
      min_seconds: autoDuration.checked ? +data.get('min_seconds') : 1,
      max_seconds: autoDuration.checked ? +data.get('max_seconds') : 8,
      fps: +data.get('fps'),
      seed: +data.get('seed'),
      conditions,
      retake_start: data.get('mode') === 'retake' ? +data.get('retake_start') : null,
      retake_end: data.get('mode') === 'retake' ? +data.get('retake_end') : null,
      regenerate_video: data.get('regenerate_video') === 'on',
      regenerate_audio: data.get('regenerate_audio') === 'on',
      extend_direction: data.get('extend_direction') || 'end',
      extend_seconds: +(data.get('extend_seconds') || 5),
      extend_context_seconds: +(data.get('extend_context_seconds') || 3),
      audio_asset_id: audioAssetId,
      audio_start: +(data.get('audio_start') || 0),
      audio_duration: data.get('audio_duration') ? +data.get('audio_duration') : null,
      loras: getLoras(),
    };
    let response = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    let job = await response.json();
    while (['queued', 'running'].includes(job.status)) {
      document.querySelector('#state').textContent = job.status === 'queued' ? 'キューで待機中' : '映像と音声を生成中';
      const percent = Math.round(job.progress * 100);
      document.querySelector('#bar').style.width = `${percent}%`;
      document.querySelector('#percent').textContent = `${percent}%`;
      await sleep(1500);
      response = await fetch(`/api/jobs/${job.id}`);
      job = await response.json();
    }
    if (job.status === 'failed') throw new Error(job.error);
    const video = document.querySelector('#video');
    video.src = job.video_url;
    document.querySelector('#download').href = job.video_url;
    lastJobMetrics = {
      generation_seconds: job.generation_seconds,
      peak_vram_gb: job.peak_vram_gb,
    };
    updateConditionSummary();
    show('result');
    await loadHistory();
  } catch (error) {
    document.querySelector('#errorText').textContent = error.message;
    show('error');
  } finally {
    submit.disabled = false;
    loadHistory();
  }
});
