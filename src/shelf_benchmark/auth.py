"""GCP Authentication & Vertex AI / Cloud Storage / BigQuery Client Factory.

Supports both Application Default Credentials (ADC) and seamless fallback to
`gcloud auth print-access-token` with quota project configuration so the suite
runs out-of-the-box in Cloudtop, CI/CD, or standard developer workstations.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

# Must be set BEFORE importing google.auth / google.genai to avoid mTLS client-cert errors on Cloudtop
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

import google.auth
import google.auth.transport.requests
import google.cloud.storage as storage
from google import genai
from google.cloud import bigquery
from google.oauth2 import credentials


def ensure_gcp_env() -> None:
    """Disable client certificate mTLS override if not configured for API endpoints."""
    os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"


def get_gcp_credentials(project_id: str = "unilever-shelf-understanding") -> google.auth.credentials.Credentials:
    """Obtain valid GCP credentials with quota_project_id set (prioritizing Application Default Credentials)."""
    ensure_gcp_env()
    try:
        creds, _ = google.auth.default(quota_project_id=project_id)
        if hasattr(creds, "with_quota_project"):
            creds = creds.with_quota_project(project_id)
        if not getattr(creds, "token", None) or getattr(creds, "expired", False):
            creds.refresh(google.auth.transport.requests.Request())
        if getattr(creds, "token", None):
            return creds
    except Exception:
        pass

    try:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
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
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
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
