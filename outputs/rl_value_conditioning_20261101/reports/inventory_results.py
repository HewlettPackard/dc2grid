"""Hash the preserved research outputs without committing bulk replay or trace files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# isort: off
from ctx import ROOT, PRIOR, PROJECT, cap_cpu, file_hash, write_json
# isort: on


def main():
    cap_cpu(23)
    destination = ROOT / "reports/result_inventory.json"
    if destination.exists():
        raise FileExistsError("Result inventories are immutable")
    rows = []
    for directory in (PRIOR, ROOT):
        for path in sorted(directory.rglob("*")):
            if path == destination or any(
                part in ("cache", "__pycache__", ".pytest_cache")
                for part in path.relative_to(directory).parts
            ):
                continue
            if path.is_symlink():
                rows.append(
                    {
                        "path": str(path.relative_to(PROJECT)),
                        "type": "symlink",
                        "target": os.readlink(path),
                    }
                )
            elif path.is_file():
                rows.append(
                    {
                        "path": str(path.relative_to(PROJECT)),
                        "type": "file",
                        "bytes": path.stat().st_size,
                        "sha256": file_hash(path),
                    }
                )
    write_json(
        destination,
        {
            "scope": [str(PRIOR), str(ROOT)],
            "file_or_link_count": len(rows),
            "total_file_bytes": sum(row.get("bytes", 0) for row in rows),
            "bulk_data_remain_on_disk_with_hashes": True,
            "excludes": ["this manifest", "cache", "__pycache__", ".pytest_cache"],
            "files": rows,
        },
    )
    print(len(rows), "paths;", sum(row.get("bytes", 0) for row in rows), "bytes")


if __name__ == "__main__":
    main()
