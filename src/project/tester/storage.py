"""Zapis i odczyt wynikow: runs.csv, histories.jsonl, summary.csv, meta.json."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

RUN_COLUMNS = [
    "algo", "set_id", "params", "instance", "N", "I", "run", "seed", "max_nfe", "nfe", "iterations",
    "best_f", "feasible", "set_mismatch", "time_s", "best_solution",
]
INT_COLUMNS = {"N", "I", "run", "seed", "nfe", "iterations"}
FLOAT_COLUMNS = {"best_f", "time_s"}


def save_runs(out_dir: Path, records: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    # Krzywe zbieznosci f_best(NFE): tylko zdarzenia poprawy, wiec plik jest maly.
    with open(out_dir / "histories.jsonl", "w") as fh:
        for r in records:
            key = {k: r[k] for k in ("algo", "set_id", "instance", "N", "I", "run")}
            fh.write(json.dumps({**key, "history": r["history"]}) + "\n")


def load_runs(out_dir: Path, with_history: bool = False) -> list[dict]:
    records = []
    with open(out_dir / "runs.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            for c in INT_COLUMNS:
                row[c] = int(row[c])
            for c in FLOAT_COLUMNS:
                row[c] = float(row[c])
            row["feasible"] = row["feasible"] == "True"
            records.append(row)
    if with_history:
        with open(out_dir / "histories.jsonl") as fh:
            for row, line in zip(records, fh):
                row["history"] = json.loads(line)["history"]
    return records


def write_table(path: Path, rows: list[dict]) -> None:
    """CSV z unia kolumn (parametry roznych algorytmow maja rozne nazwy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for r in rows:
        columns.extend(c for c in r if c not in columns)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_meta(out_dir: Path, **info) -> None:
    """Metadane reprodukowalnosci: wersje, seedy, siatki, opis instancji."""
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "project_version": metadata.version("project"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        **info,
    }
    with open(out_dir / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
