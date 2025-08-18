from __future__ import annotations

"""Save slide summaries to Google Drive using Google Colab."""

import os
from typing import Dict

try:  # Colab-specific import
    from google.colab import drive  # type: ignore
except Exception:  # pragma: no cover
    drive = None  # type: ignore


def save_document_to_drive(
    title: str,
    summaries: Dict[int, str],
    drive_dir: str = "/content/drive/MyDrive",
) -> str:
    """Write summaries to a text file in Google Drive.

    Parameters
    ----------
    title:
        Name of the output document (without extension).
    summaries:
        Mapping of page index to summary text.
    drive_dir:
        Target directory in the mounted Google Drive.

    Returns
    -------
    str
        Full path to the saved file.
    """

    if drive is None:
        raise RuntimeError("Google Colab environment is required to use Drive.")

    # Ensure Drive is mounted
    drive.mount("/content/drive", force_remount=False)

    os.makedirs(drive_dir, exist_ok=True)
    file_path = os.path.join(drive_dir, f"{title}.txt")

    lines: list[str] = []
    for page_idx in sorted(summaries.keys()):
        summary = summaries[page_idx]
        lines.append(f"페이지 {page_idx + 1}\n{summary}\n\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return file_path
