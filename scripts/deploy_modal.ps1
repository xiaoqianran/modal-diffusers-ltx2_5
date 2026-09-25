param(
    [switch]$CheckOnly,
    [switch]$SkipPrepare
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repo '.env'
$python = Join-Path $repo '.venv\Scripts\python.exe'

if (-not (Test-Path $envFile)) {
    throw "Missing .env: $envFile"
}
if (-not (Test-Path $python)) {
    throw "Missing project Python: $python"
}

# Deployment-time media topology is safe to pass as normal environment.
# Credentials deliberately stay out of this script and must live only in the
# Modal Secret named by LTX25_MODAL_MEDIA_SECRET (default: media-storage).
$allowed = @(
    'LTX25_MODAL_APP',
    'LTX25_MODAL_ENVIRONMENT',
    'LTX25_MODAL_MODEL_VOLUME',
    'LTX25_MODAL_STATE_VOLUME',
    'LTX25_MODAL_KERNEL_VOLUME',
    'QWEN_IMAGE21_CACHE_VOLUME',
    'LTX25_MODAL_JOB_DICT',
    'LTX25_MODAL_WORKER_CLASS',
    'LTX25_MODAL_NATIVE_FUNCTION',
    'LTX25_MODAL_HF_SECRET',
    'DIRECTOR_QWEN_ENABLED',
    'LTX25_MODAL_GPU_IDLE_SECONDS',
    'LTX25_PARALLEL_COLD_LOAD',
    'LTX25_CUDA_GRAPH',
    'LTX25_CUDA_GRAPH_MAX_CAPTURES',
    'LTX25_MEDIA_BACKEND',
    'LTX25_MEDIA_PRIMARY_ID',
    'LTX25_MEDIA_PRIMARY_BACKEND',
    'LTX25_MEDIA_PRIMARY_S3_BUCKET',
    'LTX25_MEDIA_PRIMARY_S3_ENDPOINT_URL',
    'LTX25_MEDIA_PRIMARY_S3_REGION',
    'LTX25_MEDIA_FALLBACK_ID',
    'LTX25_MEDIA_FALLBACK_BACKEND',
    'LTX25_MEDIA_FALLBACK_S3_BUCKET',
    'LTX25_MEDIA_FALLBACK_S3_ENDPOINT_URL',
    'LTX25_MEDIA_FALLBACK_S3_REGION',
    'LTX25_MEDIA_BREAKER_FAILURES',
    'LTX25_MEDIA_BREAKER_COOLDOWN_SECONDS',
    'LTX25_MEDIA_PRESIGN_SECONDS',
    'LTX25_MEDIA_MULTIPART_THRESHOLD_MB',
    'LTX25_MEDIA_PART_SIZE_MB',
    'LTX25_MODAL_MEDIA_SECRET'
)

$config = @{}
$rawEnv = @{}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith('#') -or -not $line.Contains('=')) { return }
    $key, $value = $line.Split('=', 2)
    $rawEnv[$key] = $value
    if ($allowed -contains $key) {
        $config[$key] = $value
        [Environment]::SetEnvironmentVariable($key, $value, 'Process')
    }
}

$primaryId = if ($config.ContainsKey('LTX25_MEDIA_PRIMARY_ID')) { $config['LTX25_MEDIA_PRIMARY_ID'] } else { '' }
$primaryBackend = if ($config.ContainsKey('LTX25_MEDIA_PRIMARY_BACKEND')) { $config['LTX25_MEDIA_PRIMARY_BACKEND'] } else { '' }

if ($primaryId -ne 'r2') {
    throw 'R2 must be the configured primary media store before deployment.'
}
if ($primaryBackend -ne 's3') {
    throw 'R2 primary must use the s3 backend.'
}
if (-not $config['LTX25_MEDIA_PRIMARY_S3_BUCKET']) {
    throw 'R2 bucket is missing from .env.'
}

$secretName = $config['LTX25_MODAL_MEDIA_SECRET']
if (-not $secretName) { $secretName = 'media-storage' }

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$modalEnvironment = if ($rawEnv['LTX25_MODAL_ENVIRONMENT']) {
    $rawEnv['LTX25_MODAL_ENVIRONMENT']
} else {
    'main'
}
$env:LTX25_MODAL_ENVIRONMENT = $modalEnvironment
$env:MODAL_ENVIRONMENT = $modalEnvironment
$profile = & $python -m modal profile current 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    Write-Host 'No active Modal profile. Starting Modal setup...'
    & $python -m modal setup
    if ($LASTEXITCODE -ne 0) { throw 'Modal setup failed.' }
}

function Write-SecretJson([hashtable]$Values, [string]$Path) {
    $json = $Values | ConvertTo-Json -Compress
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $json, $utf8)
}

function Sync-ModalSecret([string]$Name, [hashtable]$Values) {
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        Write-SecretJson $Values $tmp
        & $python -m modal secret create $Name --env $modalEnvironment --from-json $tmp --force
        if ($LASTEXITCODE -ne 0) { throw "Failed to create/update Modal Secret '$Name'." }
    }
    finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

$r2Access = $rawEnv['LTX25_MEDIA_PRIMARY_S3_ACCESS_KEY_ID']
$r2Secret = $rawEnv['LTX25_MEDIA_PRIMARY_S3_SECRET_ACCESS_KEY']
if (-not $r2Access -or -not $r2Secret) {
    throw 'R2 credentials are missing from .env.'
}

$mediaSecret = @{
    'AWS_ACCESS_KEY_ID' = $r2Access
    'AWS_SECRET_ACCESS_KEY' = $r2Secret
    'LTX25_MEDIA_PRIMARY_S3_ACCESS_KEY_ID' = $r2Access
    'LTX25_MEDIA_PRIMARY_S3_SECRET_ACCESS_KEY' = $r2Secret
}

$fallbackId = $config['LTX25_MEDIA_FALLBACK_ID']
if ($fallbackId) {
    $fallbackAccess = $rawEnv['LTX25_MEDIA_FALLBACK_S3_ACCESS_KEY_ID']
    $fallbackSecret = $rawEnv['LTX25_MEDIA_FALLBACK_S3_SECRET_ACCESS_KEY']
    if (-not $fallbackAccess -or -not $fallbackSecret) {
        throw 'MinIO fallback credentials are missing from .env.'
    }
    $mediaSecret['LTX25_MEDIA_FALLBACK_S3_ACCESS_KEY_ID'] = $fallbackAccess
    $mediaSecret['LTX25_MEDIA_FALLBACK_S3_SECRET_ACCESS_KEY'] = $fallbackSecret
}

$secretList = & $python -m modal secret list --env $modalEnvironment 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) { throw 'Unable to query Modal secrets.' }

if (-not $CheckOnly) {
    Write-Host "Syncing Modal Secret '$secretName' from local .env (values hidden)..."
    Sync-ModalSecret $secretName $mediaSecret
}

$hfSecretName = if ($rawEnv['LTX25_MODAL_HF_SECRET']) { $rawEnv['LTX25_MODAL_HF_SECRET'] } else { 'huggingface' }
if ($secretList -notmatch [regex]::Escape($hfSecretName)) {
    $hfToken = $rawEnv['HF_TOKEN']
    if (-not $hfToken) { $hfToken = $rawEnv['HUGGING_FACE_HUB_TOKEN'] }
    if (-not $hfToken) { $hfToken = $rawEnv['HUGGINGFACE_TOKEN'] }
    if (-not $hfToken) {
        throw "Modal Secret '$hfSecretName' is missing. Add HF_TOKEN to .env once, then rerun this script."
    }
    if (-not $CheckOnly) {
        Write-Host "Creating Modal Secret '$hfSecretName' from local HF token (value hidden)..."
        Sync-ModalSecret $hfSecretName @{
            'HF_TOKEN' = $hfToken
            'HUGGING_FACE_HUB_TOKEN' = $hfToken
        }
    }
}

Write-Host "Media primary : $($config['LTX25_MEDIA_PRIMARY_ID'])"
Write-Host "Media fallback: $($config['LTX25_MEDIA_FALLBACK_ID'])"
Write-Host "Media secret  : $secretName"
Write-Host "Modal env     : $modalEnvironment"

if (-not $CheckOnly) {
    $jobDictName = if ($rawEnv['LTX25_MODAL_JOB_DICT']) { $rawEnv['LTX25_MODAL_JOB_DICT'] } else { 'ltx25-jobs' }
    Write-Host "Ensuring Modal Dict '$jobDictName' exists..."
    & $python -m modal dict create $jobDictName --env $modalEnvironment 2>$null
}

if ($CheckOnly) {
    Write-Host 'Local configuration check passed.'
    exit 0
}

Push-Location $repo
try {
    if (-not $SkipPrepare) {
        Write-Host 'Preparing model/cache Volumes (existing files are reused)...'
        $previousMinContainers = $env:LTX25_MODAL_GPU_MIN_CONTAINERS
        $env:LTX25_MODAL_GPU_MIN_CONTAINERS = '0'
        & $python -m modal run --env $modalEnvironment modal_app.py::prepare
        $prepareRc = $LASTEXITCODE
        if ($null -eq $previousMinContainers) {
            Remove-Item Env:LTX25_MODAL_GPU_MIN_CONTAINERS -ErrorAction SilentlyContinue
        }
        else {
            $env:LTX25_MODAL_GPU_MIN_CONTAINERS = $previousMinContainers
        }
        if ($prepareRc -ne 0) { exit $prepareRc }
    }
    # The deployed service is intentionally persistent: exactly one resident
    # RTX PRO 6000 until delete-modal.bat stops the App.
    $env:LTX25_MODAL_GPU_MIN_CONTAINERS = '1'
    Write-Host 'Deploying Modal app with recreate strategy...'
    # max_containers=1 + development redeploys favor certainty over zero
    # downtime: once deploy returns, no old container may continue receiving
    # inputs while the replacement Director is warming.
    & $python -m modal deploy --env $modalEnvironment --strategy recreate modal_app.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host 'Waiting for DirectorWorker to finish model startup...'
    $readyScript = Join-Path ([System.IO.Path]::GetTempPath()) ("ltx25-ready-" + [guid]::NewGuid().ToString("N") + ".py")
    $readyRc = 1
    try {
        @'
import json
import os
import modal

app_name = os.environ.get("LTX25_MODAL_APP", "ltx25-nvfp4")
worker_name = os.environ.get("LTX25_MODAL_WORKER_CLASS", "DirectorWorker")
environment_name = os.environ.get("LTX25_MODAL_ENVIRONMENT", "main")
worker = modal.Cls.from_name(
    app_name,
    worker_name,
    environment_name=environment_name,
)()
result = worker.ready.remote()
print("[READY] " + json.dumps(result, default=str))
'@ | Set-Content -LiteralPath $readyScript -Encoding UTF8
        & $python $readyScript
        $readyRc = $LASTEXITCODE
    }
    finally {
        Remove-Item $readyScript -Force -ErrorAction SilentlyContinue
    }
    if ($readyRc -ne 0) {
        Write-Error 'Modal deploy completed, but DirectorWorker did not become ready.'
        exit $readyRc
    }
    Write-Host 'DirectorWorker is ready.'
}
finally {
    Pop-Location
}
