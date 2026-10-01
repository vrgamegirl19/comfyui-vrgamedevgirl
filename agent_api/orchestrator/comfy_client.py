"""ComfyUI loopback connector and fake test harness (Section 5.4, 5.5, C4)."""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

from ..errors import (
    ComfyExecutionError,
    ComfyQueueError,
    ComfyTimeoutError,
    JobCancelledError,
)

logger = logging.getLogger("vrgdg.agent_api.comfy_client")


def extract_prompt_error_from_history(history: Dict[str, Any], prompt_id: str) -> Optional[str]:
    """Inspect history payload for execution exceptions or node errors."""
    root = history.get(prompt_id, history)
    if not isinstance(root, dict):
        return None

    messages: List[str] = []
    status = root.get("status", {})
    if isinstance(status, dict):
        status_str = str(status.get("status_str", "")).lower()
        if status_str and status_str not in {"success", "completed"}:
            messages.append(f"status: {status_str}")

        for msg in status.get("messages", []):
            if isinstance(msg, (str, dict)):
                messages.append(str(msg))

        if status.get("exception_message"):
            messages.append(str(status["exception_message"]))

    if root.get("error"):
        messages.append(str(root["error"]))

    if messages:
        return "\n".join(messages)
    return None


def prompt_history_finished(history: Dict[str, Any], prompt_id: str) -> bool:
    """Check if prompt execution has concluded (success or failure)."""
    root = history.get(prompt_id, history)
    if not isinstance(root, dict) or not root:
        return False
    status = root.get("status")
    if isinstance(status, dict):
        s_str = str(status.get("status_str", "")).lower()
        if s_str in {"success", "completed", "error", "failed"}:
            return True
    return bool(root.get("outputs")) or bool(root.get("error"))


def extract_images_from_history(history: Dict[str, Any], prompt_id: str) -> List[Dict[str, Any]]:
    """Extract list of generated image descriptors from history."""
    root = history.get(prompt_id, history)
    outputs = root.get("outputs", {}) if isinstance(root, dict) else {}
    images: List[Dict[str, Any]] = []
    for out in outputs.values():
        if isinstance(out, dict) and isinstance(out.get("images"), list):
            images.extend(out["images"])
    return images


def extract_videos_from_history(history: Dict[str, Any], prompt_id: str) -> List[Dict[str, Any]]:
    """Extract list of generated video descriptors from history."""
    root = history.get(prompt_id, history)
    outputs = root.get("outputs", {}) if isinstance(root, dict) else {}
    videos: List[Dict[str, Any]] = []
    for out in outputs.values():
        if isinstance(out, dict):
            for key in ("videos", "gifs", "animated"):
                if isinstance(out.get(key), list):
                    videos.extend(out[key])
    return videos


def extract_text_from_history(history: Dict[str, Any], prompt_id: str) -> List[str]:
    """Extract string output from text generation nodes."""
    root = history.get(prompt_id, history)
    outputs = root.get("outputs", {}) if isinstance(root, dict) else {}
    texts: List[str] = []
    for out in outputs.values():
        if isinstance(out, dict):
            t = out.get("text") or (out.get("ui", {}).get("text") if isinstance(out.get("ui"), dict) else None)
            if isinstance(t, list):
                for item in t:
                    if item:
                        texts.append(str(item))
            elif isinstance(t, str) and t.strip():
                texts.append(t)
    return texts


class ComfyClient:
    """Loopback HTTP client interacting with the host ComfyUI instance."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url.rstrip("/")

    def queue_prompt(self, prompt: Dict[str, Any], client_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit a prompt graph to ComfyUI /prompt."""
        payload = {
            "prompt": prompt,
            "client_id": client_id or "vrgdg_agent_api",
        }
        url = f"{self.base_url}/prompt"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise ComfyQueueError(f"ComfyUI prompt submission failed ({e.code}): {body}")
        except Exception as e:
            raise ComfyQueueError(f"Failed to connect to ComfyUI at {url}: {e}")

        if result.get("error"):
            raise ComfyQueueError(str(result["error"]))

        if result.get("node_errors"):
            raise ComfyQueueError("Workflow node validation failed", node_errors=result["node_errors"])

        return result

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Fetch prompt execution history from /history/{prompt_id}."""
        url = f"{self.base_url}/history/{urllib.parse.quote(prompt_id)}"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.debug(f"History query error for prompt {prompt_id}: {e}")
            return {}

    def get_queue(self) -> Dict[str, Any]:
        """Fetch queue state from /queue."""
        url = f"{self.base_url}/queue"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return {"queue_running": [], "queue_pending": []}

    def interrupt(self) -> bool:
        """Interrupt the currently executing prompt via POST /interrupt."""
        url = f"{self.base_url}/interrupt"
        try:
            req = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Interrupt call failed: {e}")
            return False

    def delete_queue_items(self, prompt_ids: List[str]) -> bool:
        """Delete specific pending prompts from ComfyUI queue (Section 5.4)."""
        if not prompt_ids:
            return True
        url = f"{self.base_url}/queue"
        payload = json.dumps({"delete": prompt_ids}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Delete queue items failed: {e}")
            return False

    async def wait_for_prompt(
        self,
        prompt_id: str,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 1.0,
        on_status: Optional[Callable[[str], None]] = None,
        check_cancel: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Poll /history/{prompt_id} until completed, checking for cancellation."""
        started = time.time()
        while time.time() - started < timeout_seconds:
            if check_cancel and check_cancel():
                self.interrupt()
                raise JobCancelledError(message="Prompt cancelled by request.")

            history = await asyncio.to_thread(self.get_history, prompt_id)
            if history:
                err = extract_prompt_error_from_history(history, prompt_id)
                if err:
                    raise ComfyExecutionError(f"Workflow execution failed:\n{err}", prompt_id=prompt_id)

                if prompt_history_finished(history, prompt_id):
                    return history

            if on_status:
                on_status("Waiting for ComfyUI prompt to complete...")

            await asyncio.sleep(poll_interval)

        raise ComfyTimeoutError(prompt_id, timeout_seconds)


class FakeComfyClient(ComfyClient):
    """Deterministic mock harness for testing orchestrator & job manager (Section 12 A4)."""

    def __init__(self):
        super().__init__("http://fake.test")
        self.submitted_prompts: List[Dict[str, Any]] = []
        self.interrupted_count: int = 0
        self.deleted_prompt_ids: List[str] = []
        self.mock_history_by_prompt: Dict[str, Dict[str, Any]] = {}
        self.mock_node_errors: Optional[Dict[str, Any]] = None
        self.simulated_latency: float = 0.0
        self._prompt_counter = 0

    def set_mock_output(
        self,
        prompt_id: str,
        images: Optional[List[Dict[str, Any]]] = None,
        videos: Optional[List[Dict[str, Any]]] = None,
        text: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Register pre-canned output for a specific prompt ID."""
        outputs: Dict[str, Any] = {}
        if images:
            outputs["mock_images"] = {"images": images}
        if videos:
            outputs["mock_videos"] = {"videos": videos}
        if text:
            outputs["mock_text"] = {"text": text}

        entry: Dict[str, Any] = {
            "status": {
                "status_str": "error" if error else "success",
                "completed": not bool(error),
            },
            "outputs": outputs,
        }
        if error:
            entry["error"] = error

        self.mock_history_by_prompt[prompt_id] = entry

    def queue_prompt(self, prompt: Dict[str, Any], client_id: Optional[str] = None) -> Dict[str, Any]:
        if self.mock_node_errors:
            raise ComfyQueueError("Workflow node validation failed", node_errors=self.mock_node_errors)

        self._prompt_counter += 1
        prompt_id = f"fake_prompt_{self._prompt_counter}"
        self.submitted_prompts.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "client_id": client_id,
        })

        if prompt_id not in self.mock_history_by_prompt:
            self.set_mock_output(
                prompt_id,
                images=[{"filename": f"img_{self._prompt_counter}.png", "subfolder": "", "type": "output"}],
                videos=[{"filename": f"vid_{self._prompt_counter}.mp4", "subfolder": "", "type": "output"}],
            )

        return {"prompt_id": prompt_id, "number": self._prompt_counter, "node_errors": {}}

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        entry = self.mock_history_by_prompt.get(prompt_id)
        if entry:
            return {prompt_id: entry}
        return {}

    def get_queue(self) -> Dict[str, Any]:
        return {
            "queue_running": [{"prompt_id": p["prompt_id"]} for p in self.submitted_prompts[-1:]],
            "queue_pending": [],
        }

    def interrupt(self) -> bool:
        self.interrupted_count += 1
        return True

    def delete_queue_items(self, prompt_ids: List[str]) -> bool:
        self.deleted_prompt_ids.extend(prompt_ids)
        return True


_GLOBAL_COMFY_CLIENT: Optional[ComfyClient] = None


def get_comfy_client() -> ComfyClient:
    """Return the global ComfyClient instance."""
    global _GLOBAL_COMFY_CLIENT
    if _GLOBAL_COMFY_CLIENT is None:
        _GLOBAL_COMFY_CLIENT = ComfyClient()
    return _GLOBAL_COMFY_CLIENT


def set_comfy_client(client: Optional[ComfyClient]) -> None:
    """Set or override the active ComfyClient instance (e.g. for testing)."""
    global _GLOBAL_COMFY_CLIENT
    _GLOBAL_COMFY_CLIENT = client
