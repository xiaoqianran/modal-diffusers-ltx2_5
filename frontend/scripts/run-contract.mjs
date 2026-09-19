import { existsSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(here, '..')
const repoRoot = resolve(frontendRoot, '..')
const candidates = process.platform === 'win32'
  ? [resolve(repoRoot, '.venv', 'Scripts', 'python.exe'), 'python']
  : [resolve(repoRoot, '.venv', 'bin', 'python'), 'python3', 'python']

const python = candidates.find(candidate => (
  candidate === 'python' || candidate === 'python3' || existsSync(candidate)
))
if (!python) {
  console.error('Python was not found; cannot run the frontend/backend contract test.')
  process.exit(1)
}
const contract = resolve(frontendRoot, 'src', 'studio', 'model', 'schema.conformance.py')
const result = spawnSync(python, [contract], { cwd: frontendRoot, stdio: 'inherit' })

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}
process.exit(result.status ?? 1)
