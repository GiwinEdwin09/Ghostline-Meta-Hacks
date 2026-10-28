"""Deleting uploads from the Meta Files API (build plan Parts 22 and 28).

`DELETE {META_BASE_URL}/files/{file_id}` (dev.meta.ai/docs/file-handling). Files also expire
after 7 days on their own, so a failure here is logged by the caller and never blocks a delete.
"""

from typing import Protocol

import httpx

from ghostline.config import Settings


class FileDeleter(Protocol):
    def delete(self, file_id: str) -> bool:
        """True if the file is gone (including when it was already gone)."""
        ...


class HttpFileDeleter:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self._base = settings.meta_base_url.rstrip("/")
        self._key = settings.model_api_key
        self._client = client or httpx.Client(timeout=10.0)

    def delete(self, file_id: str) -> bool:
        if not self._key:
            return False
        try:
            response = self._client.delete(
                f"{self._base}/files/{file_id}", headers={"Authorization": f"Bearer {self._key}"}
            )
        except httpx.HTTPError:
            return False
        return response.is_success or response.status_code == 404


class NoFileDeleter:
    """Used when no API key is configured: nothing to delete remotely."""

    def delete(self, file_id: str) -> bool:
        return False
