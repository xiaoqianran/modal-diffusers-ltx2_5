"""Architecture dependency guards for the frozen LTX25 layout."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ltx25"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _python_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]


def test_runtime_is_cloud_and_http_agnostic():
    imports = _imports(PKG / "runtime.py")
    assert "fastapi" not in imports
    assert "modal" not in imports


def test_director_is_cloud_and_http_agnostic():
    imports = _imports(PKG / "director.py")
    assert "fastapi" not in imports
    assert "modal" not in imports



def test_runtime_job_input_boundary_reaches_both_generation_paths():
    tree = ast.parse((PKG / "runtime.py").read_text(encoding="utf-8"))
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in {"generate", "_generate_impl", "_generate_still_impl"}:
        params = {arg.arg for arg in methods[name].args.args + methods[name].args.kwonlyargs}
        assert "input_dir" in params, name

def test_models_is_cloud_and_http_agnostic():
    imports = _imports(PKG / "models.py")
    assert "fastapi" not in imports
    assert "modal" not in imports
    assert "runtime" not in imports


def test_acceleration_has_no_serving_dependencies():
    for path in _python_files(PKG / "acceleration"):
        imports = _imports(path)
        assert "fastapi" not in imports, path
        assert "modal" not in imports, path


def test_modal_client_does_not_know_runtime():
    imports = _imports(PKG / "modal_client.py")
    assert "fastapi" not in imports
    assert "runtime" not in imports
    assert "models" not in imports


def test_media_store_has_no_serving_or_generation_dependencies():
    imports = _imports(PKG / "media_store.py")
    for forbidden in {"fastapi", "modal", "runtime", "models", "torch", "diffusers"}:
        assert forbidden not in imports


def test_media_storage_owns_routing_without_generation_dependencies():
    imports = _imports(PKG / "media_storage.py")
    for forbidden in {"fastapi", "modal", "runtime", "models", "torch", "diffusers"}:
        assert forbidden not in imports


def test_runtime_does_not_depend_on_media_storage():
    imports = _imports(PKG / "runtime.py")
    assert "ltx25.media_storage" not in imports
    assert "media_storage" not in imports


def test_media_store_does_not_depend_on_routing_layer():
    imports = _imports(PKG / "media_store.py")
    assert "ltx25.media_storage" not in imports
    assert "media_storage" not in imports


def test_api_does_not_import_modal_sdk_or_runtime():
    imports = _imports(PKG / "api.py")
    assert "modal" not in imports
    assert "runtime" not in imports
    assert "models" not in imports


def test_modal_app_is_deployment_shell_only():
    imports = _imports(ROOT / "modal_app.py")
    assert "fastapi" not in imports
    assert "ltx25.api" not in imports
    assert "ltx25.modal_client" not in imports


def test_modal_input_staging_preserves_asset_identity():
    source = (ROOT / "modal_app.py").read_text(encoding="utf-8")
    assert 'target_input = input_dir / f"{asset_id}{Path(ref.key).suffix}"' in source


def test_schemas_are_framework_agnostic():
    imports = _imports(PKG / "schemas.py")
    for forbidden in {"fastapi", "modal", "torch", "diffusers"}:
        assert forbidden not in imports


def test_encoding_has_no_serving_dependency():
    imports = _imports(PKG / "encoding.py")
    assert "fastapi" not in imports
    assert "modal" not in imports


def test_video_encoder_writes_faststart_mp4():
    source = (PKG / "encoding.py").read_text(encoding="utf-8")
    assert '"movflags": "+faststart"' in source


def test_single_primary_package_and_modal_entrypoint():
    assert (ROOT / "modal_app.py").is_file()
    assert not (ROOT / "backend").exists()
    assert not (ROOT / "deploy").exists()
    assert not (PKG / "compat").exists()


def test_reloadable_state_and_loaded_kernel_cache_are_separate():
    source = (ROOT / "modal_app.py").read_text(encoding="utf-8")
    assert 'STATE_VOLUME_NAME = os.environ.get("LTX25_MODAL_STATE_VOLUME", "ltx25-state")' in source
    assert 'KERNEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_KERNEL_VOLUME", "ltx25-kernels")' in source
    assert '"HF_HOME": "/kernel-cache/hf"' in source
    assert '"/data": state_volume' in source
    assert '"/kernel-cache": kernel_volume' in source
    assert '"HF_HUB_DISABLE_TELEMETRY"' not in source


def test_frontend_exposes_upscale_method_contract():
    source = (ROOT / "frontend" / "src" / "studio" / "model" / "generationRequest.js").read_text(encoding="utf-8")
    assert "upscale_method:" in source
    assert "draft.upscaleMethod" in source


def test_frontend_production_entry_uses_vue_runtime_boundary():
    main = (ROOT / "frontend" / "src" / "main.js").read_text(encoding="utf-8")
    app = (ROOT / "frontend" / "src" / "App.vue").read_text(encoding="utf-8")
    vite = (ROOT / "frontend" / "vite.config.js").read_text(encoding="utf-8")

    assert "createApp(App).mount('#app')" in main
    assert "useStudioRuntime" in app
    assert "fetch(" not in app
    assert "plugins: [vue()]" in vite


def test_frontend_views_do_not_own_network_or_polling():
    components = ROOT / "frontend" / "src" / "studio" / "components"
    for path in components.glob("*.vue"):
        source = path.read_text(encoding="utf-8")
        assert "fetch(" not in source, path
        assert "setTimeout(" not in source, path
