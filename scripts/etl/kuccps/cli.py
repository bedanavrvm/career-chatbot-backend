"""
KUCCPS ETL CLI utilities (ad-hoc tools)

Command: eligibility-check
  Evaluate eligibility and weighted cluster points for a specific program code
  using candidate grades provided as a JSON string.

Example (from backend/ in venv):
  python scripts/etl/kuccps/cli.py eligibility-check \
    --program-code 1263101 \
    --grades '{"ENG":"B+","MAT":"A-","PHY":"B","CHE":"B-"}' \
    --config scripts/etl/kuccps/config.yaml

Notes
- If programs_deduped.csv exists in processed/, it is used; otherwise programs.csv is used.
- You can pass --processed-dir instead of --config to point directly to a folder.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any

# Local imports
# These modules are in the same folder when run as: python scripts/etl/kuccps/cli.py ...
from eligibility import evaluate_eligibility  # type: ignore

# Optional: use existing Config helper if --config is provided
try:
    from etl import Config  # type: ignore
except Exception:  # pragma: no cover
    Config = None  # type: ignore


def _detect_reader(f) -> csv.DictReader:
    """Robust TSV/CSV reader (mirrors normalize/load logic)."""
    head = f.readline()
    rest = f.read(8192)
    f.seek(0)
    try:
        if "\t" in head:
            return csv.DictReader(f, delimiter="\t")
        sample = head + rest
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
        return csv.DictReader(f, dialect=dialect)
    except Exception:
        f.seek(0)
        return csv.DictReader(f, delimiter="\t")


def _load_program_row(processed_dir: Path, program_code: str) -> Dict[str, Any]:
    base = Path(processed_dir)
    cand = base / "programs_deduped.csv"
    if not cand.exists():
        cand = base / "programs.csv"
    if not cand.exists():
        raise FileNotFoundError(f"Programs file not found in {processed_dir}")
    with open(cand, encoding="utf-8") as f:
        reader = _detect_reader(f)
        for row in reader:
            code = (row.get("program_code") or row.get("code") or "").strip()
            if code == program_code:
                return row
    raise ValueError(f"Program code {program_code} not found in {cand}")


def cmd_eligibility_check(args: argparse.Namespace) -> int:
    # Determine processed_dir
    processed_dir = None
    if getattr(args, "processed_dir", None):
        processed_dir = Path(args.processed_dir).resolve()
    elif getattr(args, "config", None) and Config is not None:
        cfg = Config.from_yaml(Path(args.config))
        processed_dir = cfg.processed_dir
    else:
        # Default to local processed folder next to this script
        processed_dir = Path(__file__).resolve().parent / "processed"

    pc = str(args.program_code).strip()
    try:
        prog = _load_program_row(processed_dir, pc)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 2

    try:
        grades: Dict[str, str] = json.loads(args.grades or "{}")
    except Exception as e:
        print(json.dumps({"error": f"Invalid --grades JSON: {e}"}))
        return 2

    res = evaluate_eligibility(prog, grades)
    # Enrich output with program/institution identifiers for readability
    out = {
        "institution_code": (prog.get("institution_code") or "").strip(),
        "institution_name": (prog.get("institution_name") or "").strip(),
        "program_code": (prog.get("program_code") or prog.get("code") or "").strip(),
        "program_name": (prog.get("name") or prog.get("normalized_name") or "").strip(),
        "normalized_name": (prog.get("normalized_name") or "").strip(),
        "result": res,
    }
    print(json.dumps(out, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KUCCPS ETL ad-hoc CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("eligibility-check", help="Compute eligibility and cluster points for a program code")
    s.add_argument("--program-code", required=True, help="Program code, e.g., 1263101")
    s.add_argument("--grades", required=True, help="JSON map of subject grades e.g. '{\"ENG\":\"B+\",\"MAT\":\"A-\"}'")
    s.add_argument("--config", help="Path to config.yaml (to resolve processed_dir)")
    s.add_argument("--processed-dir", help="Override processed directory path")

    return p.parse_args()


def main() -> None:
    ns = parse_args()
    if ns.cmd == "eligibility-check":
        exit(cmd_eligibility_check(ns))


if __name__ == "__main__":
    main()
