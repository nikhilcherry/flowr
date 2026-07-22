import linecache
import sys
import textwrap
import uuid
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    """Point FLOWR_DIR at a fresh per-test store."""
    d = tmp_path / "flowr-store"
    monkeypatch.setenv("FLOWR_DIR", str(d))
    return d


@pytest.fixture
def make_module(tmp_path):
    """Write source to a real .py file and import it (stages need
    file-backed source). Re-calling with the same name simulates an edit."""
    made = []

    def _make(source, name=None):
        name = name or f"flowrtest_{uuid.uuid4().hex[:10]}"
        path = tmp_path / f"{name}.py"
        path.write_text(textwrap.dedent(source))
        linecache.checkcache(str(path))
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        made.append(name)
        return mod

    yield _make
    for n in made:
        sys.modules.pop(n, None)
