from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from src.utils.io import ensure_dir
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def download_file(url: str, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or stage SCREEN cCRE source files")
    parser.add_argument(
        "--screen-url",
        default=None,
        help="Direct URL to the SCREEN cCRE table or BED file",
    )
    parser.add_argument(
        "--output",
        default="data/raw/screen/screen_ccre.tsv.gz",
        help="Destination path for the downloaded SCREEN file",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        LOGGER.info("SCREEN file already exists at %s", output_path)
        return

    if not args.screen_url:
        raise SystemExit(
            "No --screen-url provided. Either place a SCREEN export under "
            "data/raw/screen/ or rerun with an explicit download URL."
        )

    LOGGER.info("Downloading SCREEN file from %s", args.screen_url)
    download_file(args.screen_url, output_path)
    LOGGER.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
