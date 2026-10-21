#!/usr/bin/env python3
"""Automated deployment script for MaxViT Tiny on Vertex AI Model Registry & Endpoints.

Usage:
    python3 scripts/deploy_maxvit_endpoint.py --project ul-fs-s-sandbx-16-prj --region us-central1
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import google.auth
from google.auth.transport.requests import AuthorizedSession

MODEL_DISPLAY_NAME = "marketshare-maxvit-tiny-tf-224"
ENDPOINT_DISPLAY_NAME = "marketshare-maxvit-vertex-endpoint"
ARTIFACT_URI = "gs://marketshare-datasets/models/maxvit_variants"
DEFAULT_CONTAINER = (
    "us-central1-docker.pkg.dev/ul-fs-s-sandbx-16-prj/cloud-run-source-deploy/shelf-bench:latest"
)
MACHINE_TYPE = "n1-standard-4"


def get_authorized_session() -> tuple[AuthorizedSession, str]:
    creds, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return AuthorizedSession(creds), project or "ul-fs-s-sandbx-16-prj"


def register_model_if_missing(
    session: AuthorizedSession,
    project: str,
    region: str,
    display_name: str = MODEL_DISPLAY_NAME,
    artifact_uri: str = ARTIFACT_URI,
    container_uri: str = DEFAULT_CONTAINER,
) -> str:
    """Register MaxViT in Vertex AI Model Registry if not already registered."""
    base_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}"
    models_url = f"{base_url}/models"

    print(f"Checking existing models for '{display_name}' in {region}...")
    resp = session.get(models_url)
    resp.raise_for_status()
    models_list = resp.json().get("models", [])

    for m in models_list:
        if m.get("displayName") == display_name:
            model_resource = m["name"]
            print(f"✔ Found existing registered model: {model_resource}")
            return model_resource

    print(f"Uploading model '{display_name}' to Vertex AI Model Registry...")
    payload: dict[str, Any] = {
        "model": {
            "displayName": display_name,
            "description": (
                "[market_share/classification] Google MaxViT Tiny (maxvit_tiny_tf_224) "
                "1,922-Class Variant Classifier on hul_variants."
            ),
            "artifactUri": artifact_uri,
            "containerSpec": {"imageUri": container_uri},
            "labels": {
                "accuracy": "0-871",
                "classes": "1922",
                "framework": "pytorch",
                "model-family": "maxvit",
                "app": "shelf-bench",
            },
        }
    }

    upload_resp = session.post(f"{models_url}:upload", json=payload)
    upload_resp.raise_for_status()
    op_name = upload_resp.json().get("name")
    print(f"Upload operation submitted: {op_name}")

    # Poll operation until done
    while True:
        time.sleep(5)
        op_res = session.get(
            f"https://{region}-aiplatform.googleapis.com/v1/{op_name}"
        ).json()
        if op_res.get("done"):
            if "error" in op_res:
                raise RuntimeError(f"Model upload failed: {op_res['error']}")
            model_res = op_res.get("response", {}).get("model")
            print(f"✔ Successfully registered model: {model_res}")
            return model_res
        print("  ...uploading model artifacts to registry...")


def get_or_create_endpoint(
    session: AuthorizedSession,
    project: str,
    region: str,
    endpoint_name: str = ENDPOINT_DISPLAY_NAME,
) -> str:
    """Fetch existing endpoint or create a new one."""
    base_url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}"
    endpoints_url = f"{base_url}/endpoints"

    print(f"Checking existing endpoints for '{endpoint_name}' in {region}...")
    resp = session.get(endpoints_url)
    resp.raise_for_status()
    endpoints = resp.json().get("endpoints", [])

    for ep in endpoints:
        if ep.get("displayName") == endpoint_name:
            ep_resource = ep["name"]
            print(f"✔ Found existing endpoint: {ep_resource}")
            return ep_resource

    print(f"Creating new endpoint '{endpoint_name}'...")
    create_resp = session.post(
        endpoints_url, json={"displayName": endpoint_name}
    )
    create_resp.raise_for_status()
    op_name = create_resp.json().get("name")
    print(f"Create endpoint operation submitted: {op_name}")

    while True:
        time.sleep(4)
        op_res = session.get(
            f"https://{region}-aiplatform.googleapis.com/v1/{op_name}"
        ).json()
        if op_res.get("done"):
            if "error" in op_res:
                raise RuntimeError(
                    f"Endpoint creation failed: {op_res['error']}"
                )
            ep_res = op_res.get("response", {}).get("name")
            print(f"✔ Successfully created endpoint: {ep_res}")
            return ep_res
        print("  ...creating endpoint...")


def deploy_model_to_endpoint(
    session: AuthorizedSession,
    region: str,
    endpoint_resource: str,
    model_resource: str,
    deployed_display_name: str = MODEL_DISPLAY_NAME,
    machine_type: str = MACHINE_TYPE,
) -> None:
    """Deploy the registered model to the Vertex AI endpoint."""
    # Check if already deployed
    ep_info = session.get(
        f"https://{region}-aiplatform.googleapis.com/v1/{endpoint_resource}"
    ).json()
    deployed_models = ep_info.get("deployedModels", [])
    for dm in deployed_models:
        if dm.get("model") == model_resource or dm.get("displayName") == deployed_display_name:
            print(f"✔ Model already deployed on endpoint (ID: {dm.get('id')})")
            return

    print(f"Deploying model {model_resource} to endpoint {endpoint_resource}...")
    payload = {
        "deployedModel": {
            "model": model_resource,
            "displayName": deployed_display_name,
            "dedicatedResources": {
                "machineSpec": {"machineType": machine_type},
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
            },
        },
        "trafficSplit": {"0": 100},
    }

    deploy_resp = session.post(
        f"https://{region}-aiplatform.googleapis.com/v1/{endpoint_resource}:deployModel",
        json=payload,
    )
    deploy_resp.raise_for_status()
    op_name = deploy_resp.json().get("name")
    print(f"Deploy operation initiated: {op_name}")
    print(f"Note: Vertex AI cluster deployment takes ~15-25 minutes.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy MaxViT Tiny to Vertex AI")
    parser.add_argument("--project", default="ul-fs-s-sandbx-16-prj", help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--artifact-uri", default=ARTIFACT_URI, help="GCS URI to model artifacts")
    args = parser.parse_args()

    session, project = get_authorized_session()
    project = args.project or project
    region = args.region

    print("==================================================")
    print(" MaxViT Tiny Vertex AI Model & Endpoint Deployment")
    print(f" Project: {project} | Region: {region}")
    print("==================================================")

    model_res = register_model_if_missing(
        session, project, region, artifact_uri=args.artifact_uri
    )
    endpoint_res = get_or_create_endpoint(session, project, region)
    deploy_model_to_endpoint(session, region, endpoint_res, model_res)

    print("\n✔ Deployment pipeline submitted successfully.")
    print(f"Model: {model_res}")
    print(f"Endpoint: {endpoint_res}")


if __name__ == "__main__":
    main()
