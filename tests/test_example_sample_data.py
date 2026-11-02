"""Registry, cache and TAP-helper logic of ``examples/_sample_data.py`` (no network)."""

from __future__ import annotations

import functools
import hashlib
import http.server
import importlib
import sys
import threading
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def sd(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.setenv("TORCHSKY_SAMPLE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TORCHSKY_EXAMPLE_FAST", raising=False)
    for name in ("_sample_data", "_sample_tap"):
        sys.modules.pop(name, None)
    return importlib.import_module("_sample_data")


PAYLOAD = b"SIMPLE  =                    T" + b" " * 200


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    root.mkdir()
    (root / "good.fits").write_bytes(PAYLOAD)
    (root / "html.fits").write_bytes(b"<html>error</html>")
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(root)
    )
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", root
    httpd.shutdown()


def _registry(url, **over):
    spec = {
        "url": url,
        "bytes": len(PAYLOAD),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "magic": (b"SIMPLE  =",),
        "suffix": ".fits",
        "large": False,
    }
    spec.update(over)
    return {"t": spec}


def test_builtin_registry_is_well_formed(sd):
    assert {"horsehead", "sdss_spectrum", "cadc_megapipe_cutout", "manga_logcube"} <= set(sd.SAMPLES)
    for name, spec in sd.SAMPLES.items():
        assert spec["url"].startswith(("http://", "https://")), name
        assert spec["magic"] and spec["description"] and spec["license"], name
        assert sd.sample_path(name).name.startswith(name)


def test_cache_dir_override_and_default(sd, monkeypatch, tmp_path):
    assert sd.cache_dir() == tmp_path / "cache"
    monkeypatch.delenv("TORCHSKY_SAMPLE_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert sd.cache_dir() == tmp_path / "xdg" / "torchsky" / "samples"


def test_download_then_cache_hit(sd, server):
    base, root = server
    reg = _registry(f"{base}/good.fits")
    path = sd.fetch_sample("t", registry=reg)
    assert path.read_bytes() == PAYLOAD
    (root / "good.fits").unlink()  # a second call must not hit the server
    assert sd.fetch_sample("t", registry=reg) == path


def test_file_url_supported(sd, tmp_path):
    src = tmp_path / "src.fits"
    src.write_bytes(PAYLOAD)
    assert sd.fetch_sample("t", registry=_registry(src.as_uri())).read_bytes() == PAYLOAD


def test_corrupt_cache_is_evicted_and_refetched(sd, server):
    base, _ = server
    reg = _registry(f"{base}/good.fits")
    path = sd.sample_path("t", reg)
    path.parent.mkdir(parents=True)
    path.write_bytes(PAYLOAD[:50])  # truncated
    assert sd.fetch_sample("t", registry=reg).read_bytes() == PAYLOAD


def test_bad_bytes_rejected(sd, server):
    base, _ = server
    reg = _registry(f"{base}/html.fits", bytes=None, sha256=None)
    with pytest.raises(sd.SampleUnavailable, match="validation"):
        sd.fetch_sample("t", registry=reg)
    assert not sd.sample_path("t", reg).exists()
    assert not list(sd.cache_dir().glob("*.partial"))


def test_sha256_mismatch_rejected(sd, server):
    base, _ = server
    reg = _registry(f"{base}/good.fits", sha256="0" * 64)
    with pytest.raises(sd.SampleUnavailable, match="sha256"):
        sd.fetch_sample("t", registry=reg)


def test_http_error_is_sample_unavailable(sd, server):
    base, _ = server
    with pytest.raises(sd.SampleUnavailable, match="could not download"):
        sd.fetch_sample("t", registry=_registry(f"{base}/missing.fits"))


def test_fast_mode_blocks_download_but_allows_cache(sd, server, monkeypatch):
    base, _ = server
    reg = _registry(f"{base}/good.fits")
    monkeypatch.setenv("TORCHSKY_EXAMPLE_FAST", "1")
    with pytest.raises(sd.SampleUnavailable, match="FAST"):
        sd.fetch_sample("t", registry=reg)
    monkeypatch.delenv("TORCHSKY_EXAMPLE_FAST")
    sd.fetch_sample("t", registry=reg)
    monkeypatch.setenv("TORCHSKY_EXAMPLE_FAST", "1")
    assert sd.fetch_sample("t", registry=reg).is_file()


def test_skip_if_unavailable_exits_77(sd, monkeypatch):
    monkeypatch.setenv("TORCHSKY_EXAMPLE_FAST", "1")
    with pytest.raises(SystemExit) as info:
        with sd.skip_if_unavailable():
            sd.fetch_sample("horsehead")
    assert info.value.code == 77


def test_unknown_sample(sd):
    with pytest.raises(KeyError):
        sd.fetch_sample("nope")


def test_tap_cone_count_check_and_manifest(sd, monkeypatch):
    pytest.importorskip("pyarrow")
    tap = importlib.import_module("_sample_tap")
    calls = []

    def fake_post(url, query):
        calls.append(query)
        if "COUNT(*)" in query:
            return b"n\n3\n"
        return b"source_id,ra,dec\n1,10.0,5.0\n2,10.1,5.1\n3,10.2,5.2\n"

    monkeypatch.setattr(tap, "_post_csv", fake_post)
    table = tap.fetch_cone("gaia_dr3", 10.0, 5.0, 0.1, columns=("source_id", "ra", "dec"))
    assert table.num_rows == 3 and len(calls) == 2
    manifest = tap._read_manifest()
    (entry,) = manifest.values()
    assert entry["rows"] == 3 and len(entry["sha256"]) == 64
    again = tap.fetch_cone("gaia_dr3", 10.0, 5.0, 0.1, columns=("source_id", "ra", "dec"))
    assert again.num_rows == 3 and len(calls) == 2  # served from cache
    monkeypatch.setenv("TORCHSKY_EXAMPLE_FAST", "1")
    with pytest.raises(sd.SampleUnavailable):
        tap.fetch_cone("gaia_dr3", 11.0, 5.0, 0.1)


def test_tap_truncation_detected(sd, monkeypatch):
    pytest.importorskip("pyarrow")
    tap = importlib.import_module("_sample_tap")
    monkeypatch.setattr(
        tap,
        "_post_csv",
        lambda url, q: b"n\n5\n" if "COUNT(*)" in q else b"ra,dec\n1,2\n",
    )
    with pytest.raises(RuntimeError, match="truncated"):
        tap.fetch_cone("twomass", 1.0, 2.0, 0.1, columns=("ra", "dec"))
