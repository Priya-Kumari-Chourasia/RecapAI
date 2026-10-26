import os
import platform
import sys
import tempfile

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from pydub import AudioSegment

try:
    from utils.ffmpeg_path import find_ffmpeg_dir
except ImportError:
    from ffmpeg_path import find_ffmpeg_dir

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 10-minute chunks exported as 16 kHz mono 48 kbps MP3 are ~3.6 MB each,
# well under Groq's 25 MB per-request limit on the free tier.
CHUNK_MINUTES = 10

_FFMPEG_DIR = find_ffmpeg_dir()
if _FFMPEG_DIR:
    _EXT = ".exe" if platform.system() == "Windows" else ""
    AudioSegment.converter = os.path.join(_FFMPEG_DIR, f"ffmpeg{_EXT}")
    AudioSegment.ffprobe = os.path.join(_FFMPEG_DIR, f"ffprobe{_EXT}")


class YouTubeBlockedError(RuntimeError):
    """YouTube refused the download (bot check / blocked IP), as opposed to a bad link."""


_BLOCK_MARKERS = (
    "sign in to confirm",
    "not a bot",
    "http error 403",
    "http error 429",
    "use --cookies",
    "confirm your age",
    "this content isn't available",
)

YOUTUBE_BLOCKED_MESSAGE = (
    "YouTube blocked this download. This almost always happens on hosted servers: "
    "YouTube blocks most cloud/datacenter IP addresses (Streamlit Cloud included), "
    "even though the same link works on your own computer.\n\n"
    "**What you can do:** download the video/audio and use **Upload file**, use "
    "**Record** for a live meeting, or run the app locally where YouTube links work."
)


def is_youtube_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _cookie_file() -> str | None:
    """Optional: YouTube cookies (Netscape cookies.txt) from YTDLP_COOKIES_FILE or YTDLP_COOKIES.

    YTDLP_COOKIES holds the file *contents*, which is how you pass them through
    Streamlit secrets. Use a throwaway Google account — accounts used this way can be banned.
    """
    path = os.getenv("YTDLP_COOKIES_FILE")
    if path and os.path.exists(path):
        return path
    contents = os.getenv("YTDLP_COOKIES")
    if contents:
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        handle.write(contents.replace("\\n", "\n"))
        handle.close()
        return handle.name
    return None


def download_youtube_audio(url: str) -> str:
    import yt_dlp

    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "force_ipv4": True,
        "noplaylist": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "64"}],
        "quiet": True,
    }
    if _FFMPEG_DIR:
        ydl_opts["ffmpeg_location"] = _FFMPEG_DIR
    cookie_file = _cookie_file()
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
    if os.getenv("YTDLP_PROXY"):
        ydl_opts["proxy"] = os.getenv("YTDLP_PROXY")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            base, _ = os.path.splitext(ydl.prepare_filename(info))
            return base + ".mp3"
    except yt_dlp.utils.DownloadError as err:
        if any(marker in str(err).lower() for marker in _BLOCK_MARKERS):
            raise YouTubeBlockedError(YOUTUBE_BLOCKED_MESSAGE) from err
        raise RuntimeError(f"Couldn't download that YouTube link: {err}") from err


def load_audio(path: str) -> AudioSegment:
    """Any audio/video file -> 16 kHz mono, the format speech models expect."""
    return AudioSegment.from_file(path).set_channels(1).set_frame_rate(16000)


def chunk_audio(audio: AudioSegment, base_path: str, chunk_minutes: float = CHUNK_MINUTES) -> list[tuple[str, float]]:
    """Split audio into MP3 chunks. Returns (chunk_path, start_offset_seconds) pairs,
    so timestamps from each chunk can be shifted back onto the full recording."""
    chunk_ms = int(chunk_minutes * 60 * 1000)
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk_path = f"{base_path}_chunk_{i}.mp3"
        audio[start : start + chunk_ms].export(chunk_path, format="mp3", bitrate="48k")
        chunks.append((chunk_path, start / 1000))
    return chunks


def process_input(source: str) -> tuple[list[tuple[str, float]], float]:
    """Return (chunks, duration_seconds) for a YouTube URL or a local file path."""
    if is_youtube_url(source):
        print("Detected YouTube URL. Downloading audio...")
        path = download_youtube_audio(source)
    else:
        path = source
    audio = load_audio(path)
    chunks = chunk_audio(audio, os.path.splitext(path)[0])
    print(f"Audio ready — {len(chunks)} chunk(s), {len(audio) / 1000:.0f}s.")
    return chunks, len(audio) / 1000
