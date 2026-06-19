# Copyright (C) 2021-2026, Felix Dittrich.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
import tempfile
import threading
import urllib.request

__all__ = ["BackgroundDownloader"]

_REPO_RAW = "https://raw.githubusercontent.com/felixdittrich92/docTR-Synth-Generator/main/resources/background_images"

# The curated set bundled in the repository. Kept as an explicit default so the
# downloader works without hitting the (rate-limited) GitHub contents API.
_DEFAULT_FILES: list[str] = [
    "air.png",
    "coffee_122.jpg",
    "color.png",
    "crumbled_1.jpg",
    "crumbled_2.png",
    "crumbled_3.png",
    "crumbled_4.png",
    "dark.png",
    "line_3.png",
    "line_4.png",
    "line_paper.jpg",
    "line_paper_2.jpg",
    "magazin_white.png",
    "noise.jpg",
    "noisy_1.png",
    "old_paper.jpg",
    "old_paper_2.jpg",
    "paper_3.jpg",
    "paper_5.png",
    "paper_white.jpg",
    "tiny_shadow.png",
    "white_1.png",
    "white_paper_2.jpg",
    "wood.png",
    "yellow.png",
]

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_DOWNLOAD_LOCK = threading.Lock()


class BackgroundDownloader:
    """Download and cache a set of background images.

    Args:
        cache_dir (str): Directory to store the downloaded backgrounds.
        source_base_url (str): Base URL files are resolved against.
        manifest_url (str | None): Optional URL of a newline-separated list of
            filenames (resolved against ``source_base_url``) or absolute URLs. If
            ``None`` the built-in default file list is used.
        timeout (int): Per-request timeout in seconds.
        enabled (bool): If ``False`` :meth:`download_all` is a no-op returning ``[]``.
    """

    def __init__(
        self,
        cache_dir: str,
        source_base_url: str = _REPO_RAW,
        manifest_url: str | None = None,
        timeout: int = 30,
        enabled: bool = True,
    ):
        self.cache_dir = cache_dir
        self.source_base_url = source_base_url.rstrip("/")
        self.manifest_url = manifest_url
        self.timeout = timeout
        self.enabled = enabled
        os.makedirs(self.cache_dir, exist_ok=True)

    def _resolve_entries(self) -> list[str]:
        """Return the list of image URLs to download."""
        files = _DEFAULT_FILES
        if self.manifest_url:
            try:
                req = urllib.request.Request(self.manifest_url, headers={"User-Agent": "docTR-Synth-Generator"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    text = resp.read().decode("utf-8", "replace")
                listed = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
                if listed:
                    files = listed
            except Exception as e:  # pragma: no cover - network dependent
                print(f"BackgroundDownloader: could not read manifest {self.manifest_url}: {e}")

        urls = []
        for entry in files:
            urls.append(entry if entry.startswith(("http://", "https://")) else f"{self.source_base_url}/{entry}")
        return urls

    def _download_one(self, url: str) -> str | None:
        name = url.rsplit("/", 1)[-1]
        if not name.lower().endswith(_IMAGE_EXTS):
            return None
        local_path = os.path.join(self.cache_dir, name)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "docTR-Synth-Generator"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
            if not data:
                return None
            fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".part")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, local_path)
            return local_path
        except Exception as e:  # pragma: no cover - network dependent
            print(f"BackgroundDownloader: failed to fetch {url}: {e}")
            return None

    def download_all(self) -> list[str]:
        """Download all background images, returning the local paths obtained."""
        if not self.enabled:
            return []
        # Reuse anything already cached without re-downloading.
        cached = [
            os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.lower().endswith(_IMAGE_EXTS)
        ]
        if cached:
            return sorted(cached)

        paths = []
        with _DOWNLOAD_LOCK:
            for url in self._resolve_entries():
                path = self._download_one(url)
                if path:
                    paths.append(path)
        if not paths:
            print("BackgroundDownloader: no background images could be downloaded.")
        return sorted(paths)
