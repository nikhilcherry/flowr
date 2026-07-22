"""Phase 2: object store, npz handler, SQLite index, atomicity."""

import shutil

import pytest

from flowr.store import Store, deserialize_value, serialize_value


@pytest.fixture
def store(store_dir):
    s = Store()
    yield s
    s.close()


def test_roundtrip_various_values(store):
    for v in [None, True, 42, 3.5, "text", b"\x00\xff", [1, [2, 3]],
              {"a": 1, "b": [None, "x"]}, (1, 2)]:
        codec, blob = serialize_value(v)
        rh = store.put_bytes(codec, blob)
        assert store.get_value(rh) == v


def test_content_addressing_dedupes(store):
    codec, blob = serialize_value([1, 2, 3])
    r1 = store.put_bytes(codec, blob)
    r2 = store.put_bytes(codec, blob)
    assert r1 == r2
    files = [f for f in store.objects_dir.iterdir()]
    assert len(files) == 1 and files[0].name == r1


def test_no_temp_files_left_behind(store):
    for i in range(20):
        codec, blob = serialize_value(list(range(i)))
        store.put_bytes(codec, blob)
    leftovers = [f for f in store.objects_dir.iterdir() if f.name.startswith(".tmp-")]
    assert leftovers == []


def test_wal_mode(store):
    mode = store.db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_store_survives_directory_move(tmp_path, monkeypatch):
    root_a = tmp_path / "a" / "store"
    monkeypatch.setenv("FLOWR_DIR", str(root_a))
    s = Store()
    codec, blob = serialize_value({"answer": 42})
    rh = s.put_bytes(codec, blob)
    s.close()
    root_b = tmp_path / "b"
    root_b.mkdir()
    shutil.move(str(root_a), str(root_b / "store"))
    s2 = Store(root_b / "store")
    assert s2.get_value(rh) == {"answer": 42}
    s2.close()


def test_missing_object_treated_as_cache_miss(store):
    codec, blob = serialize_value("payload")
    rh = store.put_bytes(codec, blob)
    store.commit_node("k" * 64, "st", "ch", {}, rh, 0.1, 1, [])
    assert store.get_node("k" * 64) is not None
    (store.objects_dir / rh).unlink()
    assert store.get_node("k" * 64) is None      # miss, not crash
    assert store.get_node_row("k" * 64) is not None


def test_commit_node_records_edges_in_order(store):
    codec, blob = serialize_value(1)
    rh = store.put_bytes(codec, blob)
    parents = ["p1" * 32, "p2" * 32, "p3" * 32]
    store.commit_node("c" * 64, "st", "ch", {}, rh, 0.0, 1, parents)
    assert store.parent_keys("c" * 64) == parents


# ---------------------------------------------------------------- npz codec

def test_ndarray_stored_as_npz(store):
    np = pytest.importorskip("numpy")
    a = np.arange(12, dtype="f8").reshape(3, 4)
    codec, blob = serialize_value(a)
    assert codec == "npz-array"
    rh = store.put_bytes(codec, blob)
    out = store.get_value(rh)
    assert np.array_equal(out, a) and out.dtype == a.dtype
    # inspectable by external tools: it is a real npz
    import io
    loaded = np.load(io.BytesIO((store.objects_dir / rh).read_bytes()))
    assert np.array_equal(loaded["arr"], a)


def test_flat_dict_of_ndarrays_stored_as_npz(store):
    np = pytest.importorskip("numpy")
    d = {"time": np.arange(5.0), "flux": np.ones(5, dtype="f4")}
    codec, blob = serialize_value(d)
    assert codec == "npz-dict"
    out = store.get_value(store.put_bytes(codec, blob))
    assert set(out) == {"time", "flux"}
    assert np.array_equal(out["flux"], d["flux"])


def test_npz_encoding_is_deterministic():
    np = pytest.importorskip("numpy")
    a = np.linspace(0, 1, 1000)
    assert serialize_value(a) == serialize_value(a)
    assert serialize_value({"x": a, "y": a * 2}) == serialize_value({"y": a * 2, "x": a})


def test_npz_fallbacks_to_pickle():
    np = pytest.importorskip("numpy")
    assert serialize_value(np.array([object()]))[0] == "pickle"       # object dtype
    assert serialize_value({"a": np.ones(3), "b": "not-an-array"})[0] == "pickle"
    assert serialize_value({})[0] == "pickle"
    assert serialize_value({1: np.ones(3)})[0] == "pickle"            # non-str key
    assert serialize_value({"a/b": np.ones(3)})[0] == "pickle"        # zip-unsafe key


def test_non_contiguous_array_roundtrip(store):
    np = pytest.importorskip("numpy")
    a = np.arange(20).reshape(4, 5)[:, ::2]
    codec, blob = serialize_value(a)
    assert np.array_equal(deserialize_value(codec, blob), a)


def test_mixed_and_plain_values_pickle():
    assert serialize_value([1, 2])[0] == "pickle"
    assert serialize_value("x")[0] == "pickle"
