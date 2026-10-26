"""Export of the finished video: copy a stitched video to a place the user chose.

The Video Builder's Export button stitches every scene into ``EXPORT_VIDEO.mp4`` in the project folder (the
stitch route does that), asks for a file name with the ``save_video`` native dialog (``builder/paths.py``) and
then calls ``_export_final_video`` to put the video there. The HTTP route lives in ``builder/routes.py``.
"""

import os
import shutil
import tempfile
from typing import Any, Dict

# The stitch the Export button makes is named with this prefix (``EXPORT_VIDEO.mp4``, ``EXPORT_VIDEO_2.mp4`` ...).
# Keep it equal to EXPORT_STITCH_PREFIX in web/music_video_builder/export_video.mjs.
EXPORT_STITCH_PREFIX = "EXPORT_VIDEO"
EXPORT_SUFFIX = ".mp4"
_SOURCE_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
_NAME_FORBIDDEN = '<>:"/\\|?*'


def _export_file_name(name: Any) -> str:
    """Return a file name for the save dialog to start with: safe on every system and ending in ``.mp4``."""
    text = "".join("_" if ch in _NAME_FORBIDDEN or ord(ch) < 32 else ch for ch in str(name or "")).strip().rstrip(". ")
    stem = text[: -len(EXPORT_SUFFIX)] if text.lower().endswith(EXPORT_SUFFIX) else text
    return f"{stem.strip().rstrip('. ') or 'video'}{EXPORT_SUFFIX}"


def _clean_path(raw: Any) -> str:
    return os.path.abspath(str(raw or "").strip().strip('"'))


def _same_place(first: str, second: str) -> bool:
    return os.path.normcase(os.path.normpath(first)) == os.path.normcase(os.path.normpath(second))


def _copy_atomically(source: str, destination: str) -> None:
    """Copy ``source`` to ``destination`` so the destination is never left half written.

    The copy goes to a hidden file beside the destination and replaces it in one step, so a full disk or a
    dropped network share leaves any video that was already at ``destination`` as it was.
    """
    handle, partial = tempfile.mkstemp(prefix=".export_", suffix=".partial", dir=os.path.dirname(destination))
    os.close(handle)
    try:
        shutil.copyfile(source, partial)
        os.replace(partial, destination)
    except BaseException:
        try:
            os.remove(partial)
        except OSError:
            pass
        raise


def _export_final_video(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Copy the stitched video ``source_path`` to ``destination_path`` and report where it went.

    ``destination_path`` is always an ``.mp4`` file in a folder that exists (a name without an extension gets
    ``.mp4``). With ``remove_source`` the stitch is deleted after the copy, but only a file that the Export
    button made: directly in ``project_folder`` and named with ``EXPORT_STITCH_PREFIX``. Any other source stays.
    """
    payload = payload if isinstance(payload, dict) else {}
    source = _clean_path(payload.get("source_path"))
    if not str(payload.get("source_path") or "").strip():
        raise ValueError("The stitched video path is empty.")
    if not os.path.isfile(source):
        raise FileNotFoundError(f"The stitched video was not found: {source}")
    if os.path.splitext(source)[1].lower() not in _SOURCE_SUFFIXES:
        raise ValueError(f"The stitched file is not a supported video file: {source}")

    raw_destination = str(payload.get("destination_path") or "").strip().strip('"')
    if not raw_destination:
        raise ValueError("Choose where to save the video.")
    destination = os.path.abspath(raw_destination)
    suffix = os.path.splitext(destination)[1].lower()
    if not suffix:
        destination += EXPORT_SUFFIX
    elif suffix != EXPORT_SUFFIX:
        raise ValueError(f"The exported video is an {EXPORT_SUFFIX} file. Use a file name that ends in {EXPORT_SUFFIX}.")
    folder = os.path.dirname(destination)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"The folder to save in does not exist: {folder}")
    if _same_place(source, destination):
        raise ValueError("Choose a place other than the stitched video itself.")

    _copy_atomically(source, destination)

    removed = False
    project = _clean_path(payload.get("project_folder")) if str(payload.get("project_folder") or "").strip() else ""
    if (
        payload.get("remove_source")
        and project
        and _same_place(os.path.dirname(source), project)
        and os.path.basename(source).startswith(EXPORT_STITCH_PREFIX)
    ):
        try:
            os.remove(source)
            removed = True
        except OSError:
            pass  # The video is saved. A stitch left in the project folder costs nothing but space.
    return {
        "destination_path": destination,
        "size_bytes": os.path.getsize(destination),
        "removed_source": removed,
    }
