import glob
import os
import shutil


def find_ffmpeg_dir() -> str | None:
    """Locate ffmpeg's bin dir even when it's not yet resolvable via PATH
    (e.g. installed by winget after this process was already started)."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return os.path.dirname(ffmpeg_path)
    winget_glob = os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin",
    )
    matches = glob.glob(winget_glob)
    return matches[0] if matches else None


def ensure_ffmpeg_on_path() -> str | None:
    """Prepend ffmpeg's dir to PATH so subprocess calls made by third-party
    libraries (e.g. openai-whisper) that always invoke it by bare name can
    find it, regardless of this process's inherited PATH. Returns the
    ffmpeg dir used, if any."""
    ffmpeg_dir = find_ffmpeg_dir()
    if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg_dir
