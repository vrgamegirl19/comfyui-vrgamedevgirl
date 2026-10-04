"""Find what this skill needs that the user's ComfyUI is missing, and (only with --yes) install it.

  python install_missing.py --comfy-root "<ComfyUI folder>"                    # report only: nothing is changed
  python install_missing.py --comfy-root "<ComfyUI folder>" --yes [--groups h3,zimage,music3] [--nodes] [--whisper]

  --yes      download the missing model files (h3 and zimage by default; add music3 if wanted) into ComfyUI/models/...
  --nodes    also git-clone missing custom-node packs into ComfyUI/custom_nodes and pip-install their requirements
             into ComfyUI's Python (restart ComfyUI afterwards)
  --whisper  also pip-install openai-whisper into ComfyUI's Python (for the dialogue checks)

Claude must only run it with --yes after the user has seen the report and agreed (downloads are tens of GB).
It never overwrites, updates or deletes anything: existing files and node folders are left exactly as they are.
Models are found through the running ComfyUI, so models in extra_model_paths.yaml or a custom root count as installed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SKILL_DIR, Comfy  # noqa: E402


def comfy_python(root):
    for cand in (os.path.join(root, "..", "python_embeded", "python.exe"), os.path.join(root, "python_embeded", "python.exe"),
                 os.path.join(root, "venv", "Scripts", "python.exe"), os.path.join(root, "venv", "bin", "python"),
                 os.path.join(root, ".venv", "bin", "python")):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def download(url, dst):
    """Resumable download to dst (via dst.part), with progress every 5%."""
    part = dst + ".part"
    done = os.path.getsize(part) if os.path.isfile(part) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0)) + done
        mode, last = ("ab" if done and resp.status == 206 else "wb"), -5
        if mode == "wb":
            done = 0
        with open(part, mode) as fh:
            while True:
                chunk = resp.read(8 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                pct = int(done * 100 / total) if total else 0
                if pct >= last + 5:
                    print(f"    {os.path.basename(dst)} {pct}% ({done / 1e9:.2f} / {total / 1e9:.2f} GB)", flush=True)
                    last = pct
    os.replace(part, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-root", required=True, help="the ComfyUI folder (the one containing main.py)")
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--groups", default="h3,zimage")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--nodes", action="store_true")
    ap.add_argument("--whisper", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.comfy_root)
    if os.path.isdir(os.path.join(root, "ComfyUI")) and not os.path.isfile(os.path.join(root, "main.py")):
        root = os.path.join(root, "ComfyUI")   # portable install: accept the outer folder too
    if not os.path.isfile(os.path.join(root, "main.py")):
        raise SystemExit(f"{root} does not look like a ComfyUI folder (no main.py)")
    req = json.load(open(os.path.join(SKILL_DIR, "requirements.json"), encoding="utf-8"))
    comfy = Comfy(a.url)
    try:
        comfy.get("/system_stats")
        online = True
        comfy.get("/vrgdg/workflow_runner/lora_list", timeout=60)   # registers the Builder's custom model root, if any
    except Exception:
        online = False
        print("note: ComfyUI is not running, so only the default ComfyUI/models folders are checked")

    missing_nodes = []
    for node in req["custom_nodes"]:
        present = os.path.isdir(os.path.join(root, "custom_nodes", node["name"]))
        if not present and node["probe"] and online:
            present = bool(comfy.get("/object_info/" + urllib.parse.quote(node["probe"]), timeout=60))
        if not present:
            missing_nodes.append(node)

    listings, missing_models = {}, []
    for m in req["models"]:
        if m["folder"] not in listings:
            listings[m["folder"]] = (comfy.get(f"/models/{m['folder']}") if online else []) or []
        names = [os.path.basename(f).lower() for f in listings[m["folder"]]]
        on_disk = os.path.isfile(os.path.join(root, "models", m["folder"], m["file"]))
        similar = any(all(k in n for k in m.get("match", [])) for n in names) if m.get("match") else False
        if not on_disk and m["file"].lower() not in names and not similar:
            missing_models.append(m)

    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    print("\n== custom nodes")
    for n in req["custom_nodes"]:
        print(f"  {'MISSING ' if n in missing_nodes else 'ok      '} {n['name']:38s} {n['url']}")
    print("\n== models")
    for m in req["models"]:
        flag = "MISSING " if m in missing_models else "ok      "
        print(f"  {flag} [{m['group']}] {m['label']:42s} {m['gb']:6.2f} GB  -> models/{m['folder']}/")
    todo = [m for m in missing_models if m["group"] in groups]
    print(f"\nselected groups {groups}: {len(todo)} model files to download, {sum(m['gb'] for m in todo):.1f} GB; "
          f"{len(missing_nodes)} custom-node packs missing")
    for g, text in req["groups"].items():
        print(f"  {g}: {text}")
    if not a.yes:
        print("\nReport only. Nothing was changed. Re-run with --yes (and --nodes / --whisper) after the user agrees.")
        return

    for m in todo:
        dst_dir = os.path.join(root, "models", m["folder"])
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, m["file"])
        if os.path.exists(dst):
            continue
        print(f"downloading {m['label']} ({m['gb']} GB)")
        download(m["url"], dst)

    py = comfy_python(root)
    if a.nodes and missing_nodes:
        if not shutil.which("git"):
            print("git is not installed: install these packs with ComfyUI-Manager or by hand:")
            for n in missing_nodes:
                print("  ", n["url"])
        else:
            for n in missing_nodes:
                dst = os.path.join(root, "custom_nodes", n["name"])
                if os.path.exists(dst):
                    continue
                print(f"cloning {n['name']}")
                subprocess.run(["git", "clone", "--depth", "1", n["url"], dst], check=True)
                reqs = os.path.join(dst, "requirements.txt")
                if os.path.isfile(reqs) and py:
                    subprocess.run([py, "-m", "pip", "install", "-r", reqs], check=True)
            print("RESTART ComfyUI so it loads the new custom nodes.")
    if a.whisper and py:
        subprocess.run([py, "-m", "pip", "install", "openai-whisper"], check=True)
    print("done. Run check_env.py again to confirm everything is found.")


if __name__ == "__main__":
    main()
