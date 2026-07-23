from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


def compute_hashes(path: Path, *, part_bytes: int) -> dict[str, object]:
    if part_bytes <= 0:
        raise ValueError("part_bytes must be positive")
    overall_md5 = hashlib.md5()
    overall_sha256 = hashlib.sha256()
    part_digests: list[bytes] = []
    byte_size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(part_bytes)
            if not chunk:
                break
            byte_size += len(chunk)
            overall_md5.update(chunk)
            overall_sha256.update(chunk)
            part_digests.append(hashlib.md5(chunk).digest())
    if not part_digests:
        raise ValueError("input file is empty")
    multipart = hashlib.md5(b"".join(part_digests)).hexdigest()
    return {
        "schema_version": 1,
        "path": str(path.resolve()),
        "byte_size": byte_size,
        "part_bytes": part_bytes,
        "part_count": len(part_digests),
        "md5": overall_md5.hexdigest(),
        "sha256": overall_sha256.hexdigest(),
        "s3_multipart_etag": f"{multipart}-{len(part_digests)}",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--part-bytes", type=int, default=1_073_741_824)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compute_hashes(args.path, part_bytes=args.part_bytes)
    rendered = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="ascii")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
