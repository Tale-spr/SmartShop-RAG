from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def current_date_tag() -> str:
    return datetime.now().strftime("%Y_%m_%d")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        file_path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with file_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    normalized[key] = json.dumps(value, ensure_ascii=False)
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
