"""Repository rules, checked on every run: package layers, ASCII source, docstrings, no print.

Cheap invariants that catch whole classes of problems: an upward import (the start of an
import cycle), a stray print in library code, an undocumented function.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src/acquirer_engine"

# docs/hierarchy.md: a module may import only from lower layers
LAYERS = {
    0: {"paths", "errors", "thresholds", "rings", "wording", "report_types", "lbo", "onepager"},
    1: {"env", "config", "data", "profiles", "signals", "valuation", "metrics"},
    2: {"llm", "model", "feedback"},
    3: {"ranking", "review", "factpack"},
    4: {"engine"},
    5: {"graph", "listbuild", "render", "summary", "charts"},
    6: {"execsummary", "figures"},
    7: {"compare", "report"},
    8: {"cli"},
    9: {"__main__"},
}
LAYER_OF = {m: n for n, mods in LAYERS.items() for m in mods}
# Sent to the LLM as JSON Schema: pydantic copies a class docstring into the prompt
SCHEMA_CLASSES = {
    "Section",
    "Risk",
    "Conviction",
    "OnePager",
    "OnePagers",
    "Plan",
    "Critique",
    "ExecSummary",
    "ReviewerMove",
    "ReviewerMoves",
    "AgentPick",
    "AgentAnswer",
}


def _modules() -> dict[str, Path]:
    return {p.stem: p for p in PKG.glob("*.py") if p.stem != "__init__"}


def _runtime_imports(path: Path) -> set[str]:
    """Return the package modules this file imports at runtime (not under TYPE_CHECKING)."""
    found: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            test = node.test
            if (
                getattr(test, "id", None) == "TYPE_CHECKING"
                or getattr(test, "attr", None) == "TYPE_CHECKING"
            ):
                for child in node.orelse:
                    self.visit(child)
                return
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module and node.module.startswith("acquirer_engine"):
                parts = node.module.split(".")
                found.update([parts[1]] if len(parts) > 1 else [a.name for a in node.names])

    Visitor().visit(ast.parse(path.read_text()))
    return found - {"edgar"}


# --- layers ---


class TestLayers:
    def test_every_module_has_a_layer(self):
        assert set(_modules()) == set(LAYER_OF), "add new modules to LAYERS and docs/hierarchy.md"

    def test_imports_only_go_down(self):
        upward = [
            f"{name} (layer {LAYER_OF[name]}) imports {dep} (layer {LAYER_OF[dep]})"
            for name, path in _modules().items()
            for dep in _runtime_imports(path)
            if dep in LAYER_OF and LAYER_OF[dep] >= LAYER_OF[name]
        ]
        assert not upward, upward

    def test_the_hierarchy_doc_lists_every_module(self):
        doc = (ROOT / "docs/hierarchy.md").read_text()
        assert all(f"`{m}`" in doc for m in LAYER_OF)


# --- source rules ---


class TestSource:
    def test_package_source_is_ascii(self):
        # edgar/ is reused unchanged from an earlier project, so it keeps its own text
        files = [*PKG.glob("*.py"), ROOT / "app.py"]
        bad = [p.name for p in files if not p.read_text().isascii()]
        assert not bad, bad

    def test_no_print_outside_the_cli(self):
        """Library modules log; only the command line prints for the user."""
        printing = [
            f"{name}:{node.lineno}"
            for name, path in _modules().items()
            if name not in ("cli", "__main__")
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print"
        ]
        assert not printing, printing

    def test_every_function_and_class_has_a_docstring(self):
        missing = [
            f"{name}.{node.name}"
            for name, path in _modules().items()
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and not node.name.startswith("__")
            and node.name not in SCHEMA_CLASSES
            and not ast.get_docstring(node)
        ]
        assert not missing, missing
