"""Google Cloud Storage & Local File Manager for Shelf, Catalog, and Planogram Buckets."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, List, Optional, Tuple

from google.genai import types

from shelf_benchmark.auth import create_storage_client
from shelf_benchmark.config import BucketConfig


class StorageManager:
    """Manages GCS buckets and transparent loading of GCS URIs (`gs://...`) or local files."""

    def __init__(self, project_id: str, bucket_config: BucketConfig):
        self.project_id = project_id
        self.bucket_config = bucket_config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = create_storage_client(self.project_id)
        return self._client

    @staticmethod
    def parse_gcs_uri(uri: str) -> Tuple[str, str]:
        """Split gs://bucket/path/to/blob into (bucket, blob_name)."""
        if not uri.startswith("gs://"):
            raise ValueError(f"Not a GCS URI: {uri}")
        without_scheme = uri[len("gs://") :]
        parts = without_scheme.split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        return bucket_name, blob_name

    @staticmethod
    def guess_mime_type(uri_or_path: str) -> str:
        mime, _ = mimetypes.guess_type(uri_or_path)
        if mime:
            return mime
        if uri_or_path.lower().endswith(".png"):
            return "image/png"
        if uri_or_path.lower().endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        if uri_or_path.lower().endswith(".webp"):
            return "image/webp"
        return "image/png"

    def to_genai_part(self, image_uri_or_path: str) -> types.Part:
        """Convert a GCS URI (`gs://...`) or local image path into a `google.genai.types.Part`."""
        mime_type = self.guess_mime_type(image_uri_or_path)
        if image_uri_or_path.startswith("gs://"):
            return types.Part.from_uri(file_uri=image_uri_or_path, mime_type=mime_type)
        p = Path(image_uri_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {image_uri_or_path}")
        return types.Part.from_bytes(data=p.read_bytes(), mime_type=mime_type)

    def upload_file(self, local_path: str | Path, destination_gcs_uri: str) -> str:
        """Upload a local file to a GCS URI, creating the bucket if needed."""
        bucket_name, blob_name = self.parse_gcs_uri(destination_gcs_uri)
        bucket = self.client.bucket(bucket_name)
        if not bucket.exists():
            bucket = self.client.create_bucket(bucket_name, location="us-central1")
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        return destination_gcs_uri

    def read_text(self, uri_or_path: str) -> str:
        """Read text content from either a `gs://` URI or a local path."""
        if uri_or_path.startswith("gs://"):
            bucket_name, blob_name = self.parse_gcs_uri(uri_or_path)
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.download_as_text(encoding="utf-8")
        return Path(uri_or_path).read_text(encoding="utf-8")

    def read_json(self, uri_or_path: Optional[str]) -> Optional[Any]:
        """Read JSON from either a `gs://` URI or local path; falls back to local `configs/` if GCS object missing."""
        if not uri_or_path:
            return None
        try:
            return json.loads(self.read_text(uri_or_path))
        except Exception:
            if uri_or_path.startswith("gs://"):
                _, blob_name = self.parse_gcs_uri(uri_or_path)
                fallback = Path("configs") / Path(blob_name).name
                if fallback.exists():
                    return json.loads(fallback.read_text(encoding="utf-8"))
            return None

    def list_gcs_images(self, bucket_uri: str, prefix: str = "") -> List[str]:
        """List image URIs in a GCS bucket."""
        bucket_name, base_prefix = self.parse_gcs_uri(bucket_uri)
        full_prefix = f"{base_prefix.rstrip('/')}/{prefix}".strip("/") if (base_prefix or prefix) else ""
        blobs = self.client.list_blobs(bucket_name, prefix=full_prefix or None)
        uris: List[str] = []
        for b in blobs:
            if b.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                uris.append(f"gs://{bucket_name}/{b.name}")
        return uris
