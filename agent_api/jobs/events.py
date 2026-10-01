"""Server-Sent Events (SSE) broadcaster for jobs and updates (Section 5.2, 5.5)."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger("vrgdg.agent_api.events")


class EventBroadcaster:
    """Manages active SSE subscribers and dispatches events matching project filters."""

    def __init__(self):
        # Maps queue to (filter_project_id, Optional[loop])
        self._subscribers: Dict[asyncio.Queue, Tuple[Optional[str], Optional[asyncio.AbstractEventLoop]]] = {}
        self._lock = asyncio.Lock()

    def subscribe(self, project_id: Optional[str] = None) -> asyncio.Queue:
        """Register a new subscriber queue."""
        loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers[queue] = (project_id, loop)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue upon disconnect."""
        self._subscribers.pop(queue, None)

    def emit(self, event_name: str, data: Any, project_id: Optional[str] = None) -> None:
        """Broadcast an event to all interested subscribers. Thread-safe."""
        payload = {
            "event": event_name,
            "data": data,
            "project_id": project_id,
        }

        dead_queues = []
        for queue, (filter_pid, loop) in list(self._subscribers.items()):
            # Filter matches if no filter set, or event has no project_id, or pids match
            if filter_pid is not None and project_id is not None and filter_pid != project_id:
                continue

            try:
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(self._safe_put, queue, payload)
                else:
                    self._safe_put(queue, payload)
            except Exception:
                dead_queues.append(queue)

        for dq in dead_queues:
            self._subscribers.pop(dq, None)

    @staticmethod
    def _safe_put(queue: asyncio.Queue, payload: Dict[str, Any]) -> None:
        try:
            if queue.full():
                # Drop oldest event to avoid blocking
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(payload)
        except Exception:
            pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


_EVENT_BROADCASTER: Optional[EventBroadcaster] = None


def get_event_broadcaster() -> EventBroadcaster:
    """Get the global singleton EventBroadcaster instance."""
    global _EVENT_BROADCASTER
    if _EVENT_BROADCASTER is None:
        _EVENT_BROADCASTER = EventBroadcaster()
    return _EVENT_BROADCASTER
