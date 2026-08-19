(() => {
  const language = localStorage.getItem('ltx25Language') === 'en' ? 'en' : 'ja';
  const en = new Map(Object.entries({
    '生成モデルとレンダリング処理を個別に設定できます。': 'Configure the generation model and rendering pipeline independently.',
    '生成方式': 'Generation mode', '生成モード': 'Mode',
    'Retake（指定区間を再生成）': 'Retake (regenerate a selected range)',
    'Extend（動画を前後へ延長）': 'Extend (add video before or after)',
    '参照条件': 'Reference conditions', '画像・動画を生成映像のどの位置に反映するか指定します。': 'Choose where each image or video affects the generated video.',
    '＋ 条件を追加': '+ Add condition', 'IC-LoRA参照シート／動画': 'IC-LoRA reference sheet / video',
    '参照画像または動画をドロップ、またはクリックして選択': 'Drop a reference image or video here, or click to select',
    '画像は全尺の静止参照動画へ変換されます': 'Images are converted into full-length static reference videos',
    '通常LoRAではなく、参照conditioning対応のIC-LoRAを1本選択してください。HDR・DubItなど専用embeddingを要求するLoRAは対象外です。': 'Select one reference-conditioning IC-LoRA, not a standard LoRA. Specialized adapters requiring embeddings, such as HDR or DubIt, are not supported.',
    '元動画': 'Source video', 'ここへ動画をドロップ、またはクリックして選択': 'Drop a video here, or click to select',
    '開始位置（秒）': 'Start (seconds)', '終了位置（秒）': 'End (seconds)',
    '映像を再生成': 'Regenerate video', '音声を再生成': 'Regenerate audio', '範囲外の映像は維持': 'Preserve video outside the range', '範囲外の音声は維持': 'Preserve audio outside the range',
    '延長方向': 'Extend direction', '末尾へ追加': 'Add to end', '先頭へ追加': 'Add to beginning', '延長する長さ（秒）': 'Extension length (seconds)', '参照範囲（秒）': 'Reference range (seconds)',
    '入力音声': 'Input audio', 'ここへ音声をドロップ、またはクリックして選択': 'Drop audio here, or click to select',
    '先頭画像（任意）': 'First frame (optional)', '映像の外観を固定する場合に選択': 'Select to anchor the video appearance',
    '音声開始位置（秒）': 'Audio start (seconds)', '使用時間（秒・空欄は自動）': 'Duration (seconds; blank = automatic)',
    '演出・ショット': 'Direction and shots', 'プロンプト': 'Prompt', '映画的なシーン、動き、カメラ、照明、音声を記述…': 'Describe the cinematic scene, motion, camera, lighting, and audio…',
    'AIでLTX-2.5向けに変換': 'Optimize for LTX-2.5 with AI', '外部LLMの設定が必要です': 'External LLM configuration is required',
    '複数ショットを1回の生成でつなぐ': 'Connect multiple shots in one generation', '＋ ショットを追加': '+ Add shot', 'ネガティブプロンプト': 'Negative prompt',
    '追加学習': 'Adapters', '再読込': 'Reload', 'loras/ を確認中…': 'Checking loras/…',
    'レンダリング': 'Rendering', '基準解像度': 'Base resolution', '横 768 × 512': 'Landscape 768 × 512', 'IC-LoRA推奨 768 × 448': 'IC-LoRA recommended 768 × 448', '縦 512 × 768': 'Portrait 512 × 768', '正方形 512 × 512': 'Square 512 × 512',
    'デコーダー': 'Decoder', 'Diffusion（高精細）': 'Diffusion (high detail)', 'VAE（高速）': 'VAE (fast)',
    'フレーム数': 'Frames', '動画尺（秒）': 'Video duration (seconds)', '8n+1フレームへ自動調整': 'Automatically align to 8n+1 frames',
    '動画の長さを自動決定': 'Determine video duration automatically', 'プロンプトの動作に合わせてduration headが決定': 'The duration head selects a length based on the prompt action',
    '最短（秒）': 'Minimum (seconds)', '最長（秒）': 'Maximum (seconds)', '2倍高解像度化': '2× resolution upscale', '2倍フレームレート化': '2× frame-rate upscale', '尺は維持': 'duration preserved',
    '生成結果': 'Generated video', '設定を選び、生成を開始してください。': 'Choose settings and start generation.', '準備中': 'Preparing', 'MP4をダウンロード': 'Download MP4', '生成に失敗しました': 'Generation failed', '生成する': 'Generate', '生成条件': 'Generation settings',
    'このセッションの生成履歴': 'Generation history for this session', '選択順に結合': 'Merge in selection order', '選択削除': 'Delete selected', '更新': 'Refresh', '履歴を読み込み中…': 'Loading history…', '閉じる': 'Close',
    'テキストから同期した映像と音声を生成します。': 'Generate synchronized video and audio from text.',
    '先頭画像を保ちながら映像と音声を生成します。': 'Generate video and audio while preserving the first image.',
    '先頭画像と末尾画像の間を映像と音声で補間します。': 'Interpolate video and audio between the first and last images.',
    '複数の参照画像・動画を、生成映像内の配置位置と反映強度で指定します。': 'Place multiple reference images or videos at selected positions and strengths.',
    'IC-LoRAの参照latentを生成latentへ追加し、人物・小道具・場所の一貫性を保って新しい映像を生成します。': 'Append IC-LoRA reference latents to preserve characters, props, and locations in a new video.',
    '元動画の指定時間範囲だけを再生成し、範囲外を維持します。': 'Regenerate only a selected source-video range and preserve everything outside it.',
    '元動画の動きと音を参照し、先頭または末尾へ新しい区間を追加します。': 'Use source motion and audio to add a new section at the beginning or end.',
    '入力音声を固定し、発話・音楽・効果音のタイミングに同期する映像を生成します。': 'Keep input audio fixed and generate video synchronized to speech, music, and effects.',
    '選択してください': 'Select an adapter', '強度': 'Strength', '削除': 'Remove', 'なし': 'None', '自動': 'Automatic',
    '生成方式': 'Generation mode', '参照入力': 'Reference inputs', '尺': 'Duration', '最終フレーム数': 'Final frames', '最終解像度': 'Final resolution', '生成時間': 'Generation time', '最大VRAM使用量': 'Peak VRAM', '取得不可': 'Unavailable',
    '画像または動画': 'Image or video', 'ここへドロップ、またはクリック': 'Drop here, or click to select', '未選択': 'Not selected', '配置位置（%）': 'Position (%)', '0=先頭 / 100=末尾': '0 = beginning / 100 = end', '反映強度': 'Condition strength', '先頭画像': 'First image', '末尾画像': 'Last image',
    '履歴動画を読み込み中…': 'Loading history video…', 'この形式は使用できません': 'This file type is not supported',
    '外部LLMで変換中…': 'Rewriting with the external LLM…', 'このセッションの生成履歴はまだありません。': 'No generations in this session yet.',
    'クリックで原寸表示・入力欄へドラッグできます': 'Click for full-size playback or drag into an input', '選択': 'Select', 'ダウンロード': 'Download',
    '結合中…': 'Merging…', '入力ファイルを準備中': 'Preparing input files', 'キューで待機中': 'Waiting in queue', '映像と音声を生成中': 'Generating video and audio',
    '構図、アクション、カメラ、セリフ…': 'Composition, action, camera, dialogue…', '選択した参照画像': 'Selected reference image'
  }));

  function translateText(value) {
    if (language !== 'en') return value;
    const trimmed = value.trim();
    let translated = en.get(trimmed);
    if (!translated) {
      translated = trimmed
        .replace(/^(\d+)件選択$/, '$1 selected')
        .replace(/^(\d+)件$/, '$1 item(s)')
        .replace(/^約([\d.]+)秒$/, 'About $1 s')
        .replace(/^([\d.]+)秒$/, '$1 s')
        .replace(/^(\d+)フレーム（約([\d.]+)秒）$/, '$1 frames (about $2 s)')
        .replace(/^自動 ([\d.]+)〜([\d.]+)秒$/, 'Automatic $1–$2 s')
        .replace(/^([\d.]+)〜([\d.]+)秒・8n\+1フレームへ自動調整$/, '$1–$2 s · automatically aligned to 8n+1 frames')
        .replace(/^基準(.+) → (.+) ＋ 3-step Refine → 最終(.+)$/, 'Base $1 → $2 + 3-step refine → final $3')
        .replace(/^基準(.+) → 最終(.+)$/, 'Base $1 → final $2')
        .replace(/^8-step単段生成 → 最終(.+)（VAEデコード）$/, 'Single 8-step pass → final $1 (VAE decode)')
        .replace(/^空間Latent ×2$/, 'Spatial latent ×2')
        .replace(/^時間Latent ×2（(.+)）$/, 'Temporal latent ×2 ($1)')
        .replace(/^(.+)で変換しました$/, 'Rewritten with $1')
        .replace(/^(\d+)件を検出 · IC-LoRAは参照縮小率も表示します$/, '$1 adapter(s) found · IC-LoRA reference scale is shown')
        .replace(/^LoRA一覧を取得できません: /, 'Could not load LoRA list: ')
        .replace(/^履歴を読み込めません: /, 'Could not load history: ')
        .replace(/^履歴動画を取得できません: /, 'Could not retrieve history video: ')
        .replaceAll('空間Latent', 'Spatial latent')
        .replaceAll('時間Latent', 'Temporal latent')
        .replaceAll('最終', 'final');
    }
    if (translated === trimmed) return value;
    return value.replace(trimmed, translated);
  }

  function apply(root = document.body) {
    if (!root || language !== 'en') return;
    if (root.nodeType === Node.TEXT_NODE) {
      const translated = translateText(root.nodeValue);
      if (translated !== root.nodeValue) root.nodeValue = translated;
      return;
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const translated = translateText(walker.currentNode.nodeValue);
      if (translated !== walker.currentNode.nodeValue) walker.currentNode.nodeValue = translated;
    }
    const elements = root.nodeType === Node.ELEMENT_NODE ? [root, ...root.querySelectorAll('*')] : [];
    elements.forEach((element) => {
      ['placeholder', 'title', 'aria-label'].forEach((attribute) => {
        if (element.hasAttribute?.(attribute)) element.setAttribute(attribute, translateText(element.getAttribute(attribute)));
      });
    });
  }

  document.documentElement.lang = language;
  document.querySelectorAll('[data-language]').forEach((button) => {
    button.classList.toggle('active', button.dataset.language === language);
    button.addEventListener('click', () => {
      localStorage.setItem('ltx25Language', button.dataset.language);
      location.reload();
    });
  });
  apply(document.body);
  new MutationObserver((mutations) => mutations.forEach((mutation) => {
    if (mutation.type === 'characterData') apply(mutation.target);
    mutation.addedNodes.forEach(apply);
  })).observe(document.body, { childList: true, characterData: true, subtree: true });

  window.LTX_I18N = { language, translate: translateText };
})();
