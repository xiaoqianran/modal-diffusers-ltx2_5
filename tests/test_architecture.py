"""Architecture dependency checks.

These tests protect the one-way dependency graph documented in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"


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


def test_runtime_has_no_http_or_modal_dependency():
    for path in _python_files(BACKEND / "runtime"):
        imports = _imports(path)
        assert "fastapi" not in imports, path
        assert "modal" not in imports, path


def test_control_has_no_fastapi_or_api_dependency():
    for path in _python_files(BACKEND / "control"):
        imports = _imports(path)
        assert "fastapi" not in imports, path
        assert not any(name.startswith("backend.api") for name in imports), path


def test_primary_api_does_not_depend_on_compat():
    imports = _imports(BACKEND / "api" / "local.py")
    assert not any("compat" in name for name in imports)


def test_primary_api_does_not_import_modal_sdk():
    imports = _imports(BACKEND / "api" / "local.py")
    assert "modal" not in imports
