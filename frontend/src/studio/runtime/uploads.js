/**
 * Asset upload helpers.
 *
 * Named `uploads` rather than `assets` to avoid colliding with the `assets`
 * state object. Contains no Vue and no business rules: it turns a File into an
 * uploaded asset.
 *
 * The API client is passed in rather than imported so the runtime owns the
 * dependency and tests can substitute it, matching the rest of the runtime.
 */

import { api as defaultApi } from './api.js'

const IMAGE = /\.(png|jpe?g|webp|bmp|gif|exr)$/i
const VIDEO = /\.(mp4|mov|webm|mkv|m4v)$/i
const AUDIO = /\.(wav|mp3|m4a|flac|ogg|aac|opus)$/i

/** Client-side guess used only to pick an accept filter and preview icon. */
export function guessKind(file) {
  const name = file?.name || ''
  if (IMAGE.test(name)) return 'image'
  if (VIDEO.test(name)) return 'video'
  if (AUDIO.test(name)) return 'audio'
  return 'unknown'
}

const MAX_BYTES = 1024 * 1024 * 1024

/** Validate before spending bandwidth; the server enforces the same limits. */
export function validateFile(file, { kind = null } = {}) {
  if (!file) return '没有选择文件'
  if (file.size === 0) return '文件为空'
  if (file.size > MAX_BYTES) return '文件超过 1 GiB'
  const detected = guessKind(file)
  if (detected === 'unknown') return '不支持的文件类型'
  if (kind === 'image' && detected !== 'image') return '需要图片文件'
  if (kind === 'video' && detected !== 'video') return '需要视频文件'
  if (kind === 'audio' && detected !== 'audio') return '需要音频文件'
  return null
}

/**
 * Upload one file and return the server's asset record.
 * @param {File} file
 * @param {{kind?: string, client?: object}} [options]
 */
export async function uploadFile(file, options = {}) {
  const problem = validateFile(file, options)
  if (problem) throw new Error(problem)
  const client = options.client || defaultApi
  const asset = await client.uploadAsset(file)
  return { ...asset, kind: asset.kind || guessKind(file) }
}
