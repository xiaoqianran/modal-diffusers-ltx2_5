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


def test_schemas_are_framework_agnostic():
    imports = _imports(PKG / "schemas.py")
    for forbidden in {"fastapi", "modal", "torch", "diffusers"}:
        assert forbidden not in imports


def test_encoding_has_no_serving_dependency():
    imports = _imports(PKG / "encoding.py")
    assert "fastapi" not in imports
    assert "modal" not in imports


def test_single_primary_package_and_modal_entrypoint():
    assert (ROOT / "modal_app.py").is_file()
    assert not (ROOT / "backend").exists()
    assert not (ROOT / "deploy").exists()
    assert not (PKG / "compat").exists()
