from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, Optional
from urllib.parse import urlparse

import requests
import yt_dlp
from bs4 import BeautifulSoup


REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_ARTICLE_CHARS = int(os.getenv("MAX_ARTICLE_CHARS", "50000"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "80000"))


class ExtractionError(Exception):
    """Raised when content extraction fails in a controlled way."""


@dataclass(slots=True)
class ExtractedContent:
    title: str
    source: str
    length: str
    content_type: str  # article | media
    text: str
    url: str


class WhisperTranscriber:
    """Lazy-loaded Whisper singleton to avoid slow bot startup."""

    _model: Any = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    try:
                        from faster_whisper import WhisperModel
                    except ImportError as exc:
                        raise ExtractionError(
                            "Whisper runtime is not installed. Run `pip install -r requirements.txt` first."
                        ) from exc

                    cls._model = WhisperModel(
                        WHISPER_MODEL_NAME,
                        device=WHISPER_DEVICE,
                        compute_type=WHISPER_COMPUTE_TYPE,
                    )
        return cls._model

    @classmethod
    def transcribe(cls, file_path: str) -> dict:
        model = cls.get_model()
        segments, _info = model.transcribe(file_path)
        text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
        return {"text": text}


SUPPORTED_MEDIA_PLATFORMS = {
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
}

UNSUPPORTED_SOCIAL_PLATFORMS = {
    "tiktok.com": "TikTok",
    "instagram.com": "Instagram",
    "twitter.com": "Twitter/X",
    "x.com": "Twitter/X",
    "vk.com": "VK",
    "vkvideo.ru": "VK",
}


def extract_content(url: str) -> ExtractedContent:
    """Route URL to the correct extractor."""
    normalized_url = normalize_url(url)
    source = detect_source(normalized_url)

    if source == "YouTube":
        return extract_media_content(normalized_url, source)

    if source in {"TikTok", "Instagram", "Twitter/X", "VK"}:
        raise ExtractionError(
            f"{source} links are not supported in the current hosted version. "
            "Use YouTube links or normal article pages."
        )

    return extract_article_content(normalized_url)


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ExtractionError("Empty URL received.")

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = f"https://{url}"

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ExtractionError("Invalid URL. Please send a full http(s) link.")

    return url


def detect_source(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    domain = domain[4:] if domain.startswith("www.") else domain

    for known_domain, source_name in SUPPORTED_MEDIA_PLATFORMS.items():
        if domain == known_domain or domain.endswith(f".{known_domain}"):
            return source_name

    for known_domain, source_name in UNSUPPORTED_SOCIAL_PLATFORMS.items():
        if domain == known_domain or domain.endswith(f".{known_domain}"):
            return source_name

    return "Article"


def extract_article_content(url: str) -> ExtractedContent:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ExtractionError(f"Failed to fetch the article: {exc}") from exc

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "noscript", "iframe", "svg", "footer", "header", "form"]):
        tag.decompose()

    title = extract_article_title(soup, url)
    text = extract_main_text(soup)

    if len(text.split()) < 80:
        raise ExtractionError(
            "Could not extract enough article text. The page may be unsupported, paywalled, or mostly JavaScript-rendered."
        )

    text = text[:MAX_ARTICLE_CHARS].strip()
    word_count = len(text.split())
    source = urlparse(url).netloc.replace("www.", "")

    return ExtractedContent(
        title=title,
        source=source,
        length=f"{word_count} words",
        content_type="article",
        text=text,
        url=url,
    )


def extract_article_title(soup: BeautifulSoup, url: str) -> str:
    candidates = [
        soup.find("meta", property="og:title"),
        soup.find("meta", attrs={"name": "twitter:title"}),
        soup.find("title"),
        soup.find("h1"),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        if candidate.name == "meta":
            content = candidate.get("content", "").strip()
            if content:
                return content
        else:
            text = candidate.get_text(" ", strip=True)
            if text:
                return text

    return url


def extract_main_text(soup: BeautifulSoup) -> str:
    article_node = soup.find("article")
    if article_node:
        paragraphs = article_node.find_all(["p", "li"])
    else:
        paragraphs = soup.find_all("p")

    cleaned_parts: list[str] = []
    for p in paragraphs:
        text = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(text) >= 40:
            cleaned_parts.append(text)

    if not cleaned_parts:
        body_text = soup.get_text(" ", strip=True)
        body_text = re.sub(r"\s+", " ", body_text)
        return body_text

    text = "\n".join(cleaned_parts)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_media_content(url: str, source: str) -> ExtractedContent:
    with TemporaryDirectory(prefix="tg_summary_") as tmpdir:
        media_path, info = download_media(url, tmpdir)

        if not media_path or not os.path.exists(media_path):
            raise ExtractionError("Media download succeeded but no local file was found for transcription.")

        transcript = WhisperTranscriber.transcribe(media_path)
        transcript_text = (transcript.get("text") or "").strip()

        if len(transcript_text.split()) < 20:
            raise ExtractionError(
                "The media was downloaded, but transcription returned too little text to summarize."
            )

        title = info.get("title") or "Untitled"
        duration_seconds = info.get("duration")
        duration_display = format_duration(duration_seconds) if duration_seconds else f"{len(transcript_text.split())} transcript words"

        return ExtractedContent(
            title=title,
            source=source,
            length=duration_display,
            content_type="media",
            text=transcript_text[:MAX_TRANSCRIPT_CHARS],
            url=url,
        )


def download_media(url: str, tmpdir: str) -> tuple[Optional[str], dict]:
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "paths": {"home": tmpdir},
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "restrictfilenames": True,
        "socket_timeout": REQUEST_TIMEOUT,
        "extract_flat": False,
        "skip_download": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise ExtractionError("yt-dlp could not extract metadata from this URL.")

            if "entries" in info:
                entries = info.get("entries") or []
                info = next((entry for entry in entries if entry), None)
                if not info:
                    raise ExtractionError("Playlist-style URL returned no downloadable entry.")

            media_path = ydl.prepare_filename(info)
            if not os.path.exists(media_path):
                files = [
                    os.path.join(tmpdir, name)
                    for name in os.listdir(tmpdir)
                    if os.path.isfile(os.path.join(tmpdir, name))
                ]
                if files:
                    media_path = max(files, key=os.path.getsize)

            return media_path, info
    except yt_dlp.utils.DownloadError as exc:
        message = str(exc)
        lower_message = message.lower()
        if "private" in lower_message:
            raise ExtractionError("This media appears to be private and cannot be accessed.") from exc
        if "unsupported" in lower_message:
            raise ExtractionError("This URL is not supported by yt-dlp in the current environment.") from exc
        if "login" in lower_message or "sign in" in lower_message:
            raise ExtractionError("This media requires authentication and cannot be processed.") from exc
        raise ExtractionError(f"Failed to download media: {message}") from exc
    except Exception as exc:
        raise ExtractionError(f"Unexpected media extraction error: {exc}") from exc


def format_duration(seconds: int | float | None) -> str:
    if not seconds:
        return "Unknown"

    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
