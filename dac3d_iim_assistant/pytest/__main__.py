"""A tiny pytest-compatible runner used when external pytest is unavailable."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Callable


def discover_test_files(paths: list[str]) -> list[Path]:
    """Collect test files from the provided paths."""
    requested_paths = [Path(path) for path in paths if not path.startswith("-")]
    if not requested_paths:
        requested_paths = [Path("tests")]

    discovered: list[Path] = []
    for path in requested_paths:
        if path.is_file() and path.name.startswith("test_") and path.suffix == ".py":
            discovered.append(path)
            continue
        if path.is_dir():
            discovered.extend(sorted(path.rglob("test_*.py")))
    return discovered


def load_module(path: Path) -> ModuleType:
    """Import a test module from a file path."""
    module_name = ".".join(path.with_suffix("").parts)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def iter_test_functions(module: ModuleType) -> list[Callable[..., object]]:
    """Return top-level callables that look like pytest tests."""
    functions: list[Callable[..., object]] = []
    for name, value in sorted(module.__dict__.items()):
        if name.startswith("test_") and callable(value):
            functions.append(value)
    return functions


def call_test(function: Callable[..., object]) -> None:
    """Execute a test function with a limited fixture set."""
    parameters = inspect.signature(function).parameters
    kwargs = {}
    if "tmp_path" in parameters:
        with TemporaryDirectory() as temporary_directory:
            kwargs["tmp_path"] = Path(temporary_directory)
            function(**kwargs)
        return
    function(**kwargs)


def main() -> int:
    """Run discovered test functions and report a pytest-like summary."""
    test_files = discover_test_files(sys.argv[1:])
    passed = 0
    failed = 0

    for test_file in test_files:
        module = load_module(test_file)
        for function in iter_test_functions(module):
            test_name = f"{test_file.as_posix()}::{function.__name__}"
            try:
                call_test(function)
                passed += 1
            except Exception:
                failed += 1
                print(f"FAILED {test_name}")
                traceback.print_exc()

    if failed:
        print(f"{failed} failed, {passed} passed")
        return 1

    print(f"{passed} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
