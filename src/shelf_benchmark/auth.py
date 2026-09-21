"""GCP Authentication & Vertex AI / Cloud Storage / BigQuery Client Factory.

Supports both Application Default Credentials (ADC) and seamless fallback to
`gcloud auth print-access-token` with quota project configuration so the suite
runs out-of-the-box in Cloudtop, CI/CD, or standard developer workstations.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import google.auth
from google import genai
from google.cloud import bigquery, storage
from google.oauth2 import credentials


def ensure_gcp_env() -> None:
    """Disable client certificate mTLS override if not configured for API endpoints."""
    if os.environ.get("GOOGLE_API_USE_CLIENT_CERTIFICATE", "").lower() != "false":
        os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"


def get_gcp_credentials(project_id: str = "unilever-shelf-understanding") -> google.auth.credentials.Credentials:
    """Obtain valid GCP credentials with quota_project_id set."""
    ensure_gcp_env()
    try:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode().strip()
        if token:
            return credentials.Credentials(token=token, quota_project_id=project_id)
    except Exception:
        pass

    creds, _ = google.auth.default(quota_project_id=project_id)
    return creds


def create_genai_client(
    project_id: str = "unilever-shelf-understanding",
    location: str = "global",
    creds: Optional[google.auth.credentials.Credentials] = None,
) -> genai.Client:
    """Create a Vertex AI google-genai Client."""
    ensure_gcp_env()
    active_creds = creds or get_gcp_credentials(project_id)
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=active_creds,
    )


def create_storage_client(
    project_id: str = "unilever-shelf-understanding",
    creds: Optional[google.auth.credentials.Credentials] = None,
) -> storage.Client:
    """Create a Google Cloud Storage client."""
    ensure_gcp_env()
    active_creds = creds or get_gcp_credentials(project_id)
    return storage.Client(project=project_id, credentials=active_creds)


def create_bigquery_client(
    project_id: str = "unilever-shelf-understanding",
    creds: Optional[google.auth.credentials.Credentials] = None,
) -> bigquery.Client:
    """Create a Google Cloud BigQuery client."""
    ensure_gcp_env()
    active_creds = creds or get_gcp_credentials(project_id)
    return bigquery.Client(project=project_id, credentials=active_creds)
