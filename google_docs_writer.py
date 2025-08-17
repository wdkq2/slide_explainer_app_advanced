"""Create and populate Google Docs with generated slide summaries.

This module handles authentication with Google APIs and writes
summarised lecture content to a new Google Document. When a service
account key JSON (``type": "service_account"``) is supplied it is used
directly, so no browser-based OAuth flow is triggered. If another JSON
credentials file is supplied (e.g. OAuth client secrets) the user will
be prompted to grant access in a browser.

For instructions on creating a service account and downloading the
JSON key file, refer to the README.md in the project root.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import json

from google.oauth2.service_account import Credentials as SACredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def _load_credentials(credentials_path: Optional[str], scopes: list[str]):
    """Load Google API credentials.

    If ``credentials_path`` points to a service account JSON it will be used
    directly. Otherwise an OAuth installed-app flow is initiated. Supplying a
    service account JSON therefore allows non-interactive use on headless
    environments like GitHub Codespaces.
    """
    if not credentials_path:
        raise ValueError("A credentials JSON path must be provided")

    with open(credentials_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    if info.get("type") == "service_account":
        return SACredentials.from_service_account_info(info, scopes=scopes)

    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes=scopes)
    return flow.run_local_server(port=0)


def create_document_from_summaries(
    credentials_path: Optional[str],
    title: str,
    summaries: Dict[int, str],
    *,
    share_email: Optional[str] = None,
) -> str:
    """Create a Google Doc and populate it with slide summaries."""

    scopes = [
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = _load_credentials(credentials_path, scopes)
    docs_service = build("docs", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)

    # Create an empty document
    doc = docs_service.documents().create(body={"title": title}).execute()
    doc_id = doc.get("documentId")

    # Compose the text content. Sort pages to ensure order.
    content_lines = []
    for page_idx in sorted(summaries.keys()):
        summary = summaries[page_idx]
        content_lines.append(f"페이지 {page_idx + 1}\n{summary}\n\n")
    combined_text = "".join(content_lines)

    # Insert the content at the beginning of the document
    requests = [
        {
            "insertText": {
                "location": {"index": 1},
                "text": combined_text,
            }
        }
    ]
    docs_service.documents().batchUpdate(
        documentId=doc_id, body={"requests": requests}
    ).execute()

    # If using a service account, optionally share the document with a user
    if share_email and isinstance(creds, SACredentials):
        try:
            drive_service.permissions().create(
                fileId=doc_id,
                body={"type": "user", "role": "writer", "emailAddress": share_email},
                fields="id",
            ).execute()
        except Exception as exc:
            logging.warning(
                "Failed to share document %s with %s: %s", doc_id, share_email, exc
            )

    return doc_id

