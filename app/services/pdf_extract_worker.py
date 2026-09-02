"""Isolated PDF text extractor.

Run as a subprocess so a malformed PDF cannot crash the FastAPI process.
"""

import json
import sys

import pymupdf as fitz


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    pages = []
    document = fitz.open(sys.argv[1])
    for page_num, page in enumerate(document, start=1):
        pages.append({"page": page_num, "text": page.get_text("text") or ""})
    document.close()
    sys.stdout.write(json.dumps(pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
