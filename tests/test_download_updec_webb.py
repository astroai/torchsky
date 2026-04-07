from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from download_updec_webb import UpdecWebbFile, download_spec  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_download_spec_retries_transient_urlerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = UpdecWebbFile("F277W", 0, "test.fits.fz", 123)
    attempts = {"count": 0}

    def _fake_urlopen(url: str, timeout: float):
        del url, timeout
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise URLError(ConnectionResetError(54, "Connection reset by peer"))
        return _FakeResponse(b"abc")

    monkeypatch.setattr("download_updec_webb.urlopen", _fake_urlopen)
    monkeypatch.setattr("download_updec_webb.time.sleep", lambda _: None)

    dest = download_spec(spec, tmp_path, overwrite=False, attempts=4, backoff_s=0.0)

    assert attempts["count"] == 3
    assert dest.read_bytes() == b"abc"


def test_download_spec_raises_after_final_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = UpdecWebbFile("F277W", 0, "test.fits.fz", 123)

    monkeypatch.setattr(
        "download_updec_webb.urlopen",
        lambda url, timeout: (_ for _ in ()).throw(
            URLError(ConnectionResetError(54, "Connection reset by peer"))
        ),
    )
    monkeypatch.setattr("download_updec_webb.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="failed to download"):
        download_spec(spec, tmp_path, overwrite=False, attempts=2, backoff_s=0.0)

    assert not (tmp_path / "test.fits.fz").exists()
    assert not (tmp_path / "test.fits.fz.part").exists()
