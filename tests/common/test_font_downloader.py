import tempfile
import urllib.error

from generator.components import FontDownloader


def test_required_scripts_detection():
    assert FontDownloader.required_scripts("hello") == {"latin"}
    assert FontDownloader.required_scripts("Привет") == {"cyrillic"}
    assert FontDownloader.required_scripts("Ελληνικά") == {"greek"}
    assert FontDownloader.required_scripts("مرحبا") == {"arabic"}
    # digits / whitespace / punctuation are ignored (covered by every font)
    assert FontDownloader.required_scripts("12:34 .,!") == set()
    # mixed scripts
    assert FontDownloader.required_scripts("abcПри") == {"latin", "cyrillic"}


def test_disabled_downloader_is_noop():
    fd = FontDownloader(cache_dir=tempfile.mkdtemp(), enabled=False)
    assert fd.resolve("hello") is None


def test_covers_with_real_font(tiny_font):
    # tiny_font covers ASCII letters/digits but not Cyrillic.
    assert FontDownloader._covers(tiny_font, {"a", "B", "1"})
    assert not FontDownloader._covers(tiny_font, {"П"})


def test_resolve_uses_downloaded_font(monkeypatch, tiny_font):
    fd = FontDownloader(cache_dir=tempfile.mkdtemp(), enabled=True)
    # Avoid network: any download request resolves to our offline tiny font.
    monkeypatch.setattr(fd, "_download", lambda family, filename: tiny_font)
    path = fd.resolve("hello")
    assert path == tiny_font
    # Single-script result is cached.
    assert fd._resolved.get("latin") == tiny_font


def test_resolve_returns_none_when_no_coverage(monkeypatch, tiny_font):
    fd = FontDownloader(cache_dir=tempfile.mkdtemp(), enabled=True)
    monkeypatch.setattr(fd, "_download", lambda family, filename: tiny_font)
    # tiny_font cannot render Cyrillic, so resolution fails gracefully.
    assert fd.resolve("Привет") is None


def test_resolve_empty_text():
    fd = FontDownloader(cache_dir=tempfile.mkdtemp(), enabled=True)
    assert fd.resolve("   ") is None


def test_failed_download_is_cached_and_not_retried(monkeypatch):
    # A missing file (404) must be attempted - and logged - only once, instead
    # of re-requesting it for every word that needs it.
    fd = FontDownloader(cache_dir=tempfile.mkdtemp(), enabled=True)
    attempts = {"n": 0}

    def boom(req, *a, **k):
        attempts["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    first = fd._download("notosans", "NotoSans[wght].ttf")
    second = fd._download("notosans", "NotoSans[wght].ttf")
    assert first is None and second is None
    assert attempts["n"] == 1  # second call short-circuits via the negative cache
