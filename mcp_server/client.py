"""HTTP API Client for VRGDG Agent API (Section 8, Section 25.1, D12).

Works against both ComfyUI host (/vrgdg/api/v1) and standalone app host (/api/v1).
Uses Python standard library (urllib.request / http.client) with zero external dependencies.
"""

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("vrgdg.mcp.client")

DEFAULT_API_URL = "http://127.0.0.1:8188/vrgdg/api/v1"


class ApiClientError(Exception):
    """Exception raised when an API call fails."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
        next_steps: Optional[List[str]] = None,
    ):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.retryable = retryable
        self.next_steps = next_steps or []


class VrgdgApiClient:
    """HTTP client communicating with the Agent API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        transport: Optional[Callable[[str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]], Tuple[int, Dict[str, Any]]]] = None,
    ):
        raw_url = base_url or os.environ.get("VRGDG_API_URL") or DEFAULT_API_URL
        self.base_url = raw_url.rstrip("/")
        self.api_token = api_token or os.environ.get("VRGDG_API_TOKEN") or ""
        self._transport = transport

    def set_transport(self, transport: Optional[Callable[..., Tuple[int, Dict[str, Any]]]]) -> None:
        """Allow injecting a direct transport for unit tests."""
        self._transport = transport

    def _build_headers(self, if_match_revision: Optional[int] = None) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "vrgdg-mcp-server/1.0",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if if_match_revision is not None:
            headers["If-Match"] = str(if_match_revision)
        return headers

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        if_match_revision: Optional[int] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Send an HTTP request to the Agent API and unwrap the standard JSON envelope."""
        clean_path = path if path.startswith("/") else f"/{path}"
        # Project names can contain spaces; encode the path, leaving separators and existing %XX alone.
        url = f"{self.base_url}{urllib.parse.quote(clean_path, safe='/%')}"

        if params:
            query_str = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            if query_str:
                url = f"{url}?{query_str}"

        # If a custom transport is plugged in (for tests)
        if self._transport is not None:
            status, res_dict = self._transport(method.upper(), url, params, json_data)
            if not res_dict.get("ok", True):
                err = res_dict.get("error") or {}
                raise ApiClientError(
                    code=str(err.get("code") or "REQUEST_FAILED"),
                    message=str(err.get("message") or "Request failed"),
                    status_code=status,
                    details=err.get("details"),
                    retryable=bool(err.get("retryable", False)),
                    next_steps=err.get("next_steps", []),
                )
            return res_dict.get("data") if "data" in res_dict else res_dict

        headers = self._build_headers(if_match_revision=if_match_revision)
        body_bytes = json.dumps(json_data).encode("utf-8") if json_data is not None else None

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method.upper())

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8")
                res_json = json.loads(raw_body) if raw_body else {}

                if not res_json.get("ok", True):
                    err = res_json.get("error") or {}
                    raise ApiClientError(
                        code=str(err.get("code") or "API_ERROR"),
                        message=str(err.get("message") or "Unknown error"),
                        status_code=status_code,
                        details=err.get("details"),
                        retryable=bool(err.get("retryable", False)),
                    )

                return res_json.get("data") if "data" in res_json else res_json

        except urllib.error.HTTPError as he:
            raw_err = he.read().decode("utf-8") if he.fp else ""
            err_json = {}
            try:
                err_json = json.loads(raw_err)
            except Exception:
                pass

            err_data = err_json.get("error") or {}
            code = str(err_data.get("code") or f"HTTP_{he.code}")
            message = str(err_data.get("message") or raw_err or he.reason)
            details = err_data.get("details")
            retryable = bool(err_data.get("retryable", False))

            raise ApiClientError(
                code=code,
                message=message,
                status_code=he.code,
                details=details,
                retryable=retryable,
            ) from he

        except urllib.error.URLError as ue:
            raise ApiClientError(
                code="CONNECTION_REFUSED",
                message=f"Could not connect to VRGDG Agent API at {self.base_url}: {ue.reason}",
                status_code=503,
                retryable=True,
            ) from ue

    def get(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
        return self.request("GET", path, params=params, timeout=timeout)

    def post(self, path: str, json_data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, if_match_revision: Optional[int] = None, timeout: float = 60.0) -> Dict[str, Any]:
        return self.request("POST", path, params=params, json_data=json_data, if_match_revision=if_match_revision, timeout=timeout)

    def put(self, path: str, json_data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, if_match_revision: Optional[int] = None, timeout: float = 60.0) -> Dict[str, Any]:
        return self.request("PUT", path, params=params, json_data=json_data, if_match_revision=if_match_revision, timeout=timeout)

    def patch(self, path: str, json_data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, if_match_revision: Optional[int] = None, timeout: float = 60.0) -> Dict[str, Any]:
        return self.request("PATCH", path, params=params, json_data=json_data, if_match_revision=if_match_revision, timeout=timeout)

    def delete(self, path: str, params: Optional[Dict[str, Any]] = None, json_data: Optional[Dict[str, Any]] = None, if_match_revision: Optional[int] = None, timeout: float = 30.0) -> Dict[str, Any]:
        return self.request("DELETE", path, params=params, json_data=json_data, if_match_revision=if_match_revision, timeout=timeout)

    def job_wait(self, job_id: str, timeout_seconds: float = 60.0, poll_interval: float = 1.0) -> Dict[str, Any]:
        """Poll job status until finished or timeout expires (Design Rule 2)."""
        deadline = time.time() + max(1.0, timeout_seconds)
        last_job: Dict[str, Any] = {}

        while time.time() < deadline:
            res = self.get(f"/jobs/{job_id}")
            last_job = res.get("job") or res
            status = str(last_job.get("status") or "").lower()

            if status in ("succeeded", "completed", "failed", "cancelled", "interrupted"):
                return last_job

            time.sleep(poll_interval)

        return {
            "timeout": True,
            "message": f"Job {job_id} is still running after {timeout_seconds}s.",
            **last_job,
        }
