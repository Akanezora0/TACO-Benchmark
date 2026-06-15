#!/usr/bin/env python3
"""Download and extract the TACO-Benchmark dataset from Google Drive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taco.core.dataset import (  # noqa: E402
    GOOGLE_DRIVE_VIEW_URL,
    download_and_extract,
    download_archive,
    extract_archive,
    verify_dataset,
)
from taco.core.paths import BENCHMARK_DATA_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract the TACO-Benchmark dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_DATA_DIR,
        help="Target data directory (default: benchmark/data)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for the downloaded archive cache",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the archive; do not extract",
    )
    parser.add_argument(
        "--extract-only",
        type=Path,
        metavar="ARCHIVE",
        help="Extract an existing local archive instead of downloading",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset layout and exit (no download)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download / re-extract even if data already exists",
    )
    args = parser.parse_args(argv)

    if args.verify:
        missing = verify_dataset(args.output)
        if missing:
            print("Dataset verification FAILED. Missing paths:")
            for rel in missing:
                print(f"  - {args.output / rel}")
            print(f"\nDownload: {GOOGLE_DRIVE_VIEW_URL}")
            print("Or run: python scripts/download_dataset.py")
            return 1
        print(f"Dataset OK at {args.output.resolve()}")
        return 0

    if args.extract_only:
        data_dir = extract_archive(args.extract_only, data_dir=args.output, force=args.force)
    elif args.download_only:
        archive = download_archive(cache_dir=args.cache_dir, force=args.force)
        print(f"Archive saved to {archive.resolve()}")
        return 0
    else:
        data_dir = download_and_extract(
            data_dir=args.output,
            cache_dir=args.cache_dir,
            force=args.force,
        )

    missing = verify_dataset(data_dir)
    if missing:
        print("Warning: extraction finished but some expected paths are missing:")
        for rel in missing:
            print(f"  - {data_dir / rel}")
        return 1

    print(f"Dataset ready at {data_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
