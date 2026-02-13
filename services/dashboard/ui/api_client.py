from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ApiResult:
    """
    Standardised result object for all API calls.

    Why this exists:
    - The UI should not crash when an API call fails.
    - The UI should always receive the same "shape" of result, so handling is easy.

    Fields:
        ok:
            True if the request succeeded (no exception + expected status code).
        data:
            Parsed response payload (usually JSON) when ok=True.
            Can also be plain text for some health endpoints.
        error:
            Human-readable error message when ok=False.
        status_code:
            HTTP status code if available (e.g., 400/500). None for network errors.
    """
    ok: bool
    data: Any | None = None
    error: str | None = None
    status_code: int | None = None


class ApiClient:
    """
    Small wrapper around requests.Session to make API calls cleaner and consistent.

    What it helps with:
    - Reuses TCP connections (faster than calling requests.get/post every time)
    - Centralises timeouts (prevents the dashboard from hanging forever)
    - Centralises error handling (always returns ApiResult)
    """

    def __init__(self, timeout: int = 30) -> None:
        # requests.Session keeps connections open for reuse (performance)
        self.session = requests.Session()

        # Default timeout (seconds) for API calls; prevents infinite waiting
        self.timeout = timeout

    def post_json(self, url: str, payload: dict) -> ApiResult:
        """
        POST JSON to an endpoint (Content-Type: application/json).

        Used for:
        - Single prediction endpoint (inference /predict)
        - Any endpoints that expect a JSON body

        Returns:
            ApiResult(ok=True, data=<parsed_json>) on success
            ApiResult(ok=False, error=...) on failure
        """
        try:
            # json= automatically serialises payload and sets the correct header.
            r = self.session.post(url, json=payload, timeout=self.timeout)

            # Convert HTTP error codes (4xx/5xx) into exceptions
            r.raise_for_status()

            # Parse and return JSON response
            return ApiResult(ok=True, data=r.json(), status_code=r.status_code)

        except Exception as e:
            # Some exceptions (HTTPError) include a response with status_code; others (timeout) don't.
            status = getattr(getattr(e, "response", None), "status_code", None)
            return ApiResult(ok=False, error=str(e), status_code=status)

    def post_file(self, url: str, file_bytes: bytes, filename: str) -> ApiResult:
        """
        POST a file using multipart/form-data.

        Used for:
        - Batch prediction upload (batch-predict /predict-file)
        - Training upload (trainer /train)

        Notes:
        - We label the upload as "text/csv" because we expect CSV files.
        """
        try:
            # 'files' triggers multipart upload (like a browser form upload)
            files = {"file": (filename, file_bytes, "text/csv")}
            r = self.session.post(url, files=files, timeout=self.timeout)
            r.raise_for_status()

            # Many training endpoints return JSON status
            return ApiResult(ok=True, data=r.json(), status_code=r.status_code)

        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            return ApiResult(ok=False, error=str(e), status_code=status)

    def get_health(self, url: str) -> ApiResult:
        """
        Quick GET health check.

        Why short timeout:
        - Health endpoints should respond quickly.
        - If a service is down, we want to show status fast.

        Behavior:
        - If status_code == 200: try JSON, else fall back to text.
        - If status_code != 200: treat as down and return response text as error.
        """
        try:
            r = self.session.get(url, timeout=8)

            if r.status_code == 200:
                # Some services return JSON; some return plain text
                try:
                    return ApiResult(ok=True, data=r.json(), status_code=r.status_code)
                except Exception:
                    return ApiResult(ok=True, data=r.text, status_code=r.status_code)

            # Non-200: include response text for easier debugging
            return ApiResult(ok=False, error=r.text, status_code=r.status_code)

        except Exception as e:
            # Network errors (DNS failure, refused connection, timeout, etc.)
            return ApiResult(ok=False, error=str(e))
