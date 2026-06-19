import os
import tempfile

from generator.components import BackgroundDownloader


def test_disabled_is_noop():
    bd = BackgroundDownloader(cache_dir=tempfile.mkdtemp(), enabled=False)
    assert bd.download_all() == []


def test_default_entries_resolve_to_image_urls():
    bd = BackgroundDownloader(cache_dir=tempfile.mkdtemp())
    urls = bd._resolve_entries()
    assert len(urls) > 0
    assert all(u.startswith("https://") for u in urls)
    assert all(u.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")) for u in urls)


def test_download_all_offline(monkeypatch):
    cache = tempfile.mkdtemp()
    bd = BackgroundDownloader(cache_dir=cache)

    def fake_download_one(url):
        name = url.rsplit("/", 1)[-1]
        path = os.path.join(cache, name)
        with open(path, "wb") as f:
            f.write(b"fake-image-bytes")
        return path

    monkeypatch.setattr(bd, "_download_one", fake_download_one)
    paths = bd.download_all()
    assert len(paths) == len(bd._resolve_entries())
    assert all(os.path.exists(p) for p in paths)


def test_reuses_existing_cache():
    cache = tempfile.mkdtemp()
    with open(os.path.join(cache, "already.png"), "wb") as f:
        f.write(b"x")
    bd = BackgroundDownloader(cache_dir=cache)
    paths = bd.download_all()  # should return cached without any network call
    assert any(p.endswith("already.png") for p in paths)


def test_absolute_url_entries_pass_through(monkeypatch):
    bd = BackgroundDownloader(cache_dir=tempfile.mkdtemp(), manifest_url="http://example/manifest")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no network")),
    )
    # Manifest fetch fails -> falls back to the default file list (resolved to repo URLs).
    urls = bd._resolve_entries()
    assert len(urls) > 0
