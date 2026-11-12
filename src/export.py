"""JSONL export (one record per athlete per frame) plus run_meta.json."""
import json
from pathlib import Path


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w")

    def write(self, record: dict):
        self._file.write(json.dumps(record) + "\n")

    def close(self):
        self._file.close()


def write_run_meta(path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str))
