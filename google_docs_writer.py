"""Create and populate Google Docs with generated slide summaries.

This module handles authentication with Google APIs and writes
summarised lecture content to a new Google Document. It supports
service account credentials by default and falls back to user OAuth
when a service account key file is not provided or cannot be used.

For instructions on creating a service account and downloading the
JSON key file, refer to the README.md in the project root.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.service_account import Credentials as SACredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


def _load_credentials(
    credentials_path: Optional[str], scopes: list[str]
):
    """Load Google API credentials.

    Attempts to create service account credentials from a JSON key
    file. If that fails or if ``credentials_path`` is None, initiates
    an OAuth flow for installed applications. The latter requires
    user interaction during the first run.

    Parameters
    ----------
    credentials_path : Optional[str]
        Path to a service account JSON key file. If None or invalid,
        OAuth flow will be used.
    scopes : list[str]
        List of OAuth scopes required for Google Docs access.

    Returns
    -------
    google.auth.credentials.Credentials
        Authorised credentials for Google APIs.
    """
    creds = None
    if credentials_path:
        try:
            creds = SACredentials.from_service_account_file(
                credentials_path, scopes=scopes
            )
        except Exception as exc:
            logging.warning(
                "Failed to load service account credentials from %s: %s",
                credentials_path,
                exc,
            )
    if not creds:
        # Fallback to user OAuth flow. This will open a browser on first run.
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes=scopes)
        creds = flow.run_local_server(port=0)
    return creds


def create_document_from_summaries(
    credentials_path: Optional[str],
    title: str,
    summaries: Dict[int, str],
    *,
    share_email: Optional[str] = None,
) -> str:
    """Create a Google Doc and populate it with slide summaries.

    Parameters
    ----------
    credentials_path : Optional[str]
        Path to a Google service account JSON key file or OAuth
        client secrets. If None, the default credentials will be used.
    title : str
        Title of the document to create.
    summaries : Dict[int, str]
        Mapping from page indices to their summaries.
    share_email : Optional[str], optional
        If provided and a service account is used, this email will be
        granted edit access to the document. Ignored for user OAuth.

    Returns
    -------
    str
        The document ID of the created Google Doc.
    """
    scopes = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]
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