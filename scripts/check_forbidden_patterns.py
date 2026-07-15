"""AST-based checks for Phase 0 forbidden patterns."""

from __future__ import annotations

import ast
import re
import sys
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SDKS = {
    "openai",
    "faster_whisper",
    "torch",
    "transformers",
    "qwen_asr",
    "nemo",
}
NETWORK_MODULES = {"requests", "httpx", "aiohttp", "openai"}
GOLDEN_NAMES = {"update_golden", "write_golden", "overwrite_golden", "regenerate_golden"}
API_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line: int
    message: str


def check_source(source: str, path: Path) -> list[Violation]:
    """Check one Python source string without searching comments or string contents as code."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Violation(str(path), exc.lineno or 1, "syntax error")]

    violations: list[Violation] = []
    relative = path.as_posix().replace("\\", "/")
    is_core = "/core/" in f"/{relative}/"
    is_domain = "/core/domain/" in f"/{relative}/"
    is_gui = "/gui/" in f"/{relative}/"
    is_test = "/tests/" in f"/{relative}/"
    command_path = "/cli/commands/" in f"/{relative}/"

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            is_bare = node.type is None
            is_exception = isinstance(node.type, ast.Name) and node.type.id == "Exception"
            if (
                (is_bare or is_exception)
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
            ):
                violations.append(
                    Violation(str(path), node.lineno, "exception is silently swallowed")
                )
        elif isinstance(node, ast.Call):
            if is_core and isinstance(node.func, ast.Name) and node.func.id == "print":
                violations.append(Violation(str(path), node.lineno, "core code uses print()"))
            if isinstance(node.func, ast.Name) and node.func.id in GOLDEN_NAMES:
                violations.append(Violation(str(path), node.lineno, "automatic golden-file update"))
            if isinstance(node.func, ast.Attribute) and node.func.attr in GOLDEN_NAMES:
                violations.append(Violation(str(path), node.lineno, "automatic golden-file update"))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imported_modules = _imported_modules(node)
            for module in imported_modules:
                root_module = module.split(".", 1)[0]
                if is_gui and root_module in FORBIDDEN_SDKS:
                    violations.append(Violation(str(path), node.lineno, f"GUI imports {module}"))
                if is_domain and (
                    module.startswith("captioner.gui")
                    or module.startswith("captioner.cli")
                    or module.startswith("captioner.adapters")
                    or root_module in FORBIDDEN_SDKS
                    or root_module == "PySide6"
                ):
                    violations.append(Violation(str(path), node.lineno, f"domain imports {module}"))
                if command_path and module.startswith("captioner.cli.commands"):
                    imported_names = _imported_names(node)
                    if "run" in imported_names:
                        violations.append(
                            Violation(
                                str(path), node.lineno, "CLI command imports another command run()"
                            )
                        )
                if is_test and root_module in NETWORK_MODULES:
                    violations.append(
                        Violation(str(path), node.lineno, f"test imports network module {module}")
                    )

        if (
            is_test
            and isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and API_KEY_PATTERN.search(node.value)
        ):
            violations.append(Violation(str(path), node.lineno, "test contains an API key"))

    violations.extend(_comment_violations(source, path))
    return violations


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.module is None:
        return []
    return [node.module]


def _imported_names(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names}
    return {alias.asname or alias.name for alias in node.names}


def _comment_violations(source: str, path: Path) -> list[Violation]:
    violations: list[Violation] = []
    try:
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            comment = token.string
            if re.search(r"#\s*type:\s*ignore\s*$", comment):
                violations.append(
                    Violation(str(path), token.start[0], "type: ignore needs a rule number")
                )
            if re.search(r"#\s*noqa\s*$", comment):
                violations.append(Violation(str(path), token.start[0], "noqa needs an explanation"))
    except (tokenize.TokenError, IndentationError):
        return violations
    return violations


def iter_python_files() -> list[Path]:
    """Return source and test files while excluding generated environments."""
    roots = (ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "main.py")
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*.py") if ".venv" not in path.parts)
    return sorted(files)


def main() -> int:
    """Check the repository and return a process exit code."""
    violations: list[Violation] = []
    for path in iter_python_files():
        violations.extend(check_source(path.read_text(encoding="utf-8"), path.relative_to(ROOT)))
    if violations:
        for violation in violations:
            print(f"{violation.path}:{violation.line}: {violation.message}", file=sys.stderr)
        return 1
    print("forbidden pattern check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
