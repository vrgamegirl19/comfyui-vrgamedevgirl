from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "standalone-builder"
ARCHIVE_ROOT = "comfyui-vrgamedevgirl"
MANIFEST_NAME = "svb-builder-manifest.json"
EXCLUDED_PREFIXES = (".agents/", ".codex/", ".git/", ".github/", "tests/")
EXCLUDED_FILES = {"standalone_release.json", "scripts/build_standalone_release.py"}
ALREADY_COMPRESSED = {".7z", ".avi", ".gif", ".gz", ".jpeg", ".jpg", ".mkv", ".mov", ".mp3", ".mp4", ".png", ".webm", ".webp", ".zip"}


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def tracked_files() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    files = []
    for value in raw.split(b"\0"):
        if not value:
            continue
        relative = PurePosixPath(os.fsdecode(value).replace("\\", "/"))
        text = relative.as_posix()
        if text in EXCLUDED_FILES or text.startswith(EXCLUDED_PREFIXES):
            continue
        path = ROOT.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Tracked release file is missing or redirected: {text}")
        files.append(path)
    if not files:
        raise RuntimeError("No tracked files were selected for the Builder release.")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix().casefold())


def validate_files(files: list[Path]) -> None:
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (SyntaxError, UnicodeError) as error:
                raise RuntimeError(f"Invalid Python in {relative}: {error}") from error
        if path.suffix.lower() == ".json" and (relative == "update_notes.json" or relative.startswith("Workflows/UsedForUIDoNotTouch/")):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as error:
                raise RuntimeError(f"Invalid JSON in {relative}: {error}") from error


def write_archive(files: list[Path], destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in ALREADY_COMPRESSED else zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def latest_notes() -> dict:
    document = json.loads((ROOT / "update_notes.json").read_text(encoding="utf-8"))
    releases = document.get("releases", [])
    return releases[0] if releases and isinstance(releases[0], dict) else {}


def write_notes(path: Path, version: str, notes: dict) -> None:
    lines = [f"# AI Video Builder {version}", "", str(notes.get("title") or "Standalone Builder update"), ""]
    for section in notes.get("sections", []):
        if not isinstance(section, dict):
            continue
        lines.extend((f"## {section.get('title') or 'Changes'}", ""))
        for item in section.get("items", []):
            lines.append(f"- {item}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def github_output(name: str, value: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def main() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        version = str(tomllib.load(handle)["project"]["version"])
    config = json.loads((ROOT / "standalone_release.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("product") != "svb-builder":
        raise RuntimeError("standalone_release.json has an unsupported format.")

    files = tracked_files()
    validate_files(files)
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    archive_name = f"comfyui-vrgamedevgirl-{version}.zip"
    archive_path = DIST / archive_name
    write_archive(files, archive_path)
    notes = latest_notes()
    manifest = {
        "schema_version": 1,
        "product": "svb-builder",
        "repository": config["repository"],
        "version": version,
        "tag": f"builder-v{version}",
        "commit": command("git", "rev-parse", "HEAD"),
        "archive_name": archive_name,
        "archive_size": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "requirements_sha256": sha256(ROOT / "requirements.txt"),
        "minimum_standalone_version": config["minimum_standalone_version"],
        "engine_api": config["engine_api"],
        "required_custom_nodes": config["required_custom_nodes"],
        "requires_full_app_update": bool(config.get("requires_full_app_update")),
        "full_app_update_reason": str(config.get("full_app_update_reason", "")),
        "release_notes": notes,
    }
    (DIST / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_notes(DIST / "release-notes.md", version, notes)

    github_output("version", version)
    github_output("tag", manifest["tag"])
    github_output("archive", archive_path.as_posix())
    github_output("manifest", (DIST / MANIFEST_NAME).as_posix())
    github_output("notes", (DIST / "release-notes.md").as_posix())
    print(f"Built {archive_name}: {archive_path.stat().st_size} bytes, {manifest['archive_sha256']}")


if __name__ == "__main__":
    main()
