import asyncio
import json
import os
import subprocess
import sys

from aiohttp import web
from server import PromptServer


UPDATE_BRANCH = "main"
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_RELEASE_NOTES_FILE = "update_notes.json"
_ROUTES_REGISTERED = False


def _run_git(*args, timeout=300):
    try:
        result = subprocess.run(
            ["git", *args], cwd=_NODE_DIR, capture_output=True, text=True,
            errors="replace", timeout=timeout, check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git was not found. Install Git, then try again.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Git timed out while updating. Check your internet connection and try again.") from exc

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        command = "git " + " ".join(args)
        raise RuntimeError(f"{command} failed:\n{output or 'Git returned an unknown error.'}")
    return output


def _install_requirements(timeout=900):
    requirements_path = os.path.join(_NODE_DIR, "requirements.txt")
    if not os.path.isfile(requirements_path):
        raise RuntimeError(f"Updated requirements file was not found: {requirements_path}")

    command = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
    try:
        result = subprocess.run(
            command, cwd=_NODE_DIR, capture_output=True, text=True,
            errors="replace", timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Installing updated Python requirements timed out. "
            f"Run this command manually:\n{subprocess.list2cmdline(command)}"
        ) from exc

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(
            "The code update completed, but installing updated Python requirements failed.\n"
            f"Run this command manually:\n{subprocess.list2cmdline(command)}\n\n"
            f"{output or 'pip returned an unknown error.'}"
        )
    return {
        "command": subprocess.list2cmdline(command),
        "output": output,
    }


def _load_release_notes(ref=""):
    """Load notes from a fetched Git ref, falling back to the installed checkout."""
    raw = ""
    source = "local"
    if ref:
        try:
            raw = _run_git("show", f"{ref}:{_RELEASE_NOTES_FILE}", timeout=20)
            source = ref
        except Exception:
            raw = ""

    if not raw:
        notes_path = os.path.join(_NODE_DIR, _RELEASE_NOTES_FILE)
        if not os.path.isfile(notes_path):
            return {"schema_version": 1, "product": "LTX 2.3 Video Builder", "releases": []}, "none"
        with open(notes_path, "r", encoding="utf-8") as handle:
            raw = handle.read()

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{_RELEASE_NOTES_FILE} is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise RuntimeError(f"{_RELEASE_NOTES_FILE} must contain a JSON object.")
    releases = document.get("releases")
    if not isinstance(releases, list):
        document["releases"] = []
    return document, source


def _git_is_ancestor(commit, ref):
    commit = str(commit or "").strip()
    ref = str(ref or "").strip()
    if not commit or not ref:
        return False
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, ref],
            cwd=_NODE_DIR, capture_output=True, text=True,
            errors="replace", timeout=20, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _git_commit_tree(commit):
    """Return the tree ID for a commit, or an empty string if it is unavailable."""
    commit = str(commit or "").strip()
    if not commit:
        return ""
    try:
        return _run_git("rev-parse", f"{commit}^{{tree}}", timeout=20).strip()
    except Exception:
        return ""


def _git_history_contains_tree(commit, ref):
    """Recognize equivalent content when a PR was squash-merged to another commit ID."""
    target_tree = _git_commit_tree(commit)
    ref = str(ref or "").strip()
    if not target_tree or not ref:
        return False
    try:
        history_trees = _run_git("log", "--format=%T", ref, timeout=30).splitlines()
    except Exception:
        return False
    return target_tree in {tree.strip() for tree in history_trees if tree.strip()}


def _git_contains_release(commit, ref):
    return _git_is_ancestor(commit, ref) or _git_history_contains_tree(commit, ref)


def _release_note_status(document, local_commit, latest_commit):
    available_release_ids = []
    current_release_id = ""
    for release in document.get("releases", []):
        if not isinstance(release, dict):
            continue
        release_id = str(release.get("id") or "").strip()
        commit = str(release.get("commit") or "").strip()
        if not release_id or not commit:
            continue
        installed = _git_contains_release(commit, local_commit)
        published = _git_contains_release(commit, latest_commit)
        if not current_release_id and installed:
            current_release_id = release_id
        if published and not installed:
            available_release_ids.append(release_id)
    return {
        "available_release_ids": available_release_ids,
        "current_release_id": current_release_id,
    }


def _update_to_main():
    if not os.path.isdir(os.path.join(_NODE_DIR, ".git")):
        raise RuntimeError("This installation is not a Git checkout, so the normal Git update commands cannot run.")

    logs = []
    before_commit = _run_git("rev-parse", "HEAD").strip()
    for args in (
        ("fetch", "origin", UPDATE_BRANCH),
        ("switch", UPDATE_BRANCH),
        ("pull", "--ff-only", "origin", UPDATE_BRANCH),
    ):
        output = _run_git(*args)
        logs.append({"command": "git " + " ".join(args), "output": output})

    branch = _run_git("branch", "--show-current").strip()
    if branch != UPDATE_BRANCH:
        raise RuntimeError(f"Git finished on '{branch or '(detached HEAD)'}' instead of '{UPDATE_BRANCH}'.")

    after_commit = _run_git("rev-parse", "HEAD").strip()
    changed_paths = _run_git(
        "diff", "--name-only", before_commit, after_commit, "--", "requirements.txt"
    ).splitlines()
    requirements_changed = "requirements.txt" in {path.strip() for path in changed_paths}
    requirements_installed = False
    requirements_error = ""
    requirements_command = ""

    if requirements_changed:
        try:
            requirements_result = _install_requirements()
            requirements_installed = True
            requirements_command = requirements_result["command"]
            logs.append({
                "command": requirements_result["command"],
                "output": requirements_result["output"],
            })
        except Exception as exc:
            requirements_error = str(exc)

    release_notes, release_notes_source = _load_release_notes()
    return {
        "branch": branch,
        "directory": _NODE_DIR,
        "before_commit": before_commit,
        "after_commit": after_commit,
        "requirements_changed": requirements_changed,
        "requirements_installed": requirements_installed,
        "requirements_error": requirements_error,
        "requirements_command": requirements_command,
        "restart_required": True,
        "release_notes": release_notes,
        "release_notes_source": release_notes_source,
        "logs": logs,
    }


def _main_status():
    """Compare the installed checkout with the production main branch without changing files."""
    if not os.path.isdir(os.path.join(_NODE_DIR, ".git")):
        raise RuntimeError("This installation is not a Git checkout, so its update status cannot be checked.")

    _run_git("fetch", "origin", UPDATE_BRANCH, timeout=20)
    local_commit = _run_git("rev-parse", "HEAD").strip()
    remote_ref = f"origin/{UPDATE_BRANCH}"
    latest_commit = _run_git("rev-parse", remote_ref).strip()
    branch = _run_git("branch", "--show-current").strip()
    behind = int(_run_git("rev-list", "--count", f"HEAD..{remote_ref}").strip() or "0")
    ahead = int(_run_git("rev-list", "--count", f"{remote_ref}..HEAD").strip() or "0")
    tracked_changes = bool(_run_git("status", "--porcelain", "--untracked-files=no").strip())
    release_notes, release_notes_source = _load_release_notes(remote_ref)
    release_status = _release_note_status(release_notes, local_commit, latest_commit)
    latest_content_installed = _git_contains_release(latest_commit, local_commit)

    return {
        "branch": branch,
        "expected_branch": UPDATE_BRANCH,
        "installed_commit": local_commit,
        "latest_commit": latest_commit,
        "behind": behind,
        "ahead": ahead,
        "outdated": behind > 0 and not latest_content_installed,
        "latest_content_installed": latest_content_installed,
        "tracked_changes": tracked_changes,
        "release_notes": release_notes,
        "release_notes_source": release_notes_source,
        **release_status,
    }


def _register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return

    @PromptServer.instance.routes.get("/vrgdg/update/v10/status")
    async def vrgdg_update_v10_status(request):
        try:
            result = await asyncio.to_thread(_main_status)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)})
        return web.json_response({"ok": True, **result})

    @PromptServer.instance.routes.post("/vrgdg/update/v10")
    async def vrgdg_update_v10(request):
        try:
            result = await asyncio.to_thread(_update_to_main)
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        return web.json_response({"ok": True, **result})

    _ROUTES_REGISTERED = True


_register_routes()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
