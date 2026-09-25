#!/usr/bin/env python3
"""Upload Doc 1 and Doc 2 to Google Drive as native Google Docs (`application/vnd.google-apps.document`).

Usage:
    # 1. Ensure ADC includes the `drive.file` scope:
    OAUTHLIB_RELAX_TOKEN_SCOPE=1 gcloud auth application-default login \
      --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/drive.file,https://www.googleapis.com/auth/userinfo.email,openid \
      --no-launch-browser

    # 2. Run the uploader:
    PYTHONPATH=src:. python3 scripts/upload_docs_to_google_drive.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


DOCS_TO_UPLOAD = [
    (
        Path("docs/DOC1_PRINCIPAL_FDE_ARCHITECTURE_AND_WALKTHROUGH.md"),
        "Unilever Shelf Intelligence — Doc 1: Principal FDE Architecture, Business Meaning & E2E Walkthrough",
    ),
    (
        Path("docs/DOC2_DEEP_DIVE_MODEL_PERFORMANCE_KPIS_AND_LIMITATIONS.md"),
        "Unilever Shelf Intelligence — Doc 2: Multi-Level Model Performance, Causal Ablation & Failure Modes",
    ),
]


def _markdown_to_simple_html(md_text: str, title: str) -> str:
    """Convert Markdown tables, headers, bullets, and code blocks to clean HTML for Google Docs import."""
    lines = md_text.splitlines()
    out = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Arial,sans-serif;line-height:1.5;color:#111;} "
        "table{border-collapse:collapse;width:100%;margin:12px 0;} "
        "th,td{border:1px solid #ccc;padding:6px 8px;font-size:10pt;text-align:left;vertical-align:top;} "
        "th{background:#f1f5f9;font-weight:bold;} "
        "pre{background:#f8fafc;padding:10px;border:1px solid #e2e8f0;font-family:monospace;font-size:9.5pt;} "
        "code{font-family:monospace;background:#f1f5f9;padding:1px 3px;}</style></head><body>",
    ]
    in_code = False
    in_table = False
    in_list = False

    def fmt_inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return s

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            if in_list:
                out.append("</ul>")
                in_list = False
            if not in_code:
                out.append("<pre>")
                in_code = True
            else:
                out.append("</pre>")
                in_code = False
            continue
        if in_code:
            out.append(html.escape(line))
            continue

        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set(":-") for c in cells):
                continue
            if not in_table:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append("<table><thead><tr>" + "".join(f"<th>{fmt_inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{fmt_inline(c)}</td>" for c in cells) + "</tr>")
            continue
        elif in_table:
            out.append("</tbody></table>")
            in_table = False

        if line.startswith("# "):
            out.append(f"<h1>{fmt_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{fmt_inline(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{fmt_inline(line[4:])}</h3>")
        elif line.startswith(("* ", "- ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{fmt_inline(line[2:])}</li>")
        elif line.strip():
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<p>{fmt_inline(line)}</p>")

    if in_table:
        out.append("</tbody></table>")
    if in_list:
        out.append("</ul>")
    out.append("</body></html>")
    return "\n".join(out)


def upload_docs_to_drive() -> list[dict]:
    adc = json.loads(Path("~/.config/gcloud/application_default_credentials.json").expanduser().read_text())
    token_req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=urllib.parse.urlencode({
            "client_id": adc["client_id"],
            "client_secret": adc["client_secret"],
            "refresh_token": adc["refresh_token"],
            "grant_type": "refresh_token",
        }).encode(),
    )
    token = json.loads(urllib.request.urlopen(token_req, timeout=10).read().decode())["access_token"]
    candidate_projects = []
    for p in ("subs-proto-com-sandbox-7-9aefe", adc.get("quota_project_id"), "loas-jjuneja", "jjuneja-fde-sandbox"):
        if p and p not in candidate_projects:
            candidate_projects.append(p)

    # Ensure drive.googleapis.com is enabled on the primary quota project
    try:
        enable_req = urllib.request.Request(
            f"https://serviceusage.googleapis.com/v1/projects/{candidate_projects[0]}/services/drive.googleapis.com:enable",
            data=b"{}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(enable_req, timeout=15).read()
    except Exception:
        pass

    results = []
    boundary = "===shelf_bench_drive_upload_boundary==="
    for md_path, doc_title in DOCS_TO_UPLOAD:
        html_content = _markdown_to_simple_html(md_path.read_text(), doc_title)
        metadata = json.dumps({
            "name": doc_title,
            "mimeType": "application/vnd.google-apps.document",
        })
        body = (
            f"--{boundary}\r\n"
            "Content-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{metadata}\r\n"
            f"--{boundary}\r\n"
            "Content-Type: text/html; charset=UTF-8\r\n\r\n"
            f"{html_content}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        header_candidates = [
            {
                "Authorization": f"Bearer {token}",
                "x-goog-user-project": proj,
                "Content-Type": f"multipart/related; boundary={boundary}",
            }
            for proj in candidate_projects
        ]
        last_err = None
        for hdrs in header_candidates:
            try:
                req = urllib.request.Request(
                    "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,webViewLink",
                    data=body,
                    headers=hdrs,
                    method="POST",
                )
                resp = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
                print(f"Created Google Doc: {resp['name']}\n  URL: {resp['webViewLink']}")
                results.append(resp)
                last_err = None
                break
            except urllib.error.HTTPError as e:
                last_err = (e.code, e.read().decode())
        if last_err is not None:
            raise RuntimeError(f"Drive upload failed across {candidate_projects}: {last_err}")
    return results


if __name__ == "__main__":
    upload_docs_to_drive()
