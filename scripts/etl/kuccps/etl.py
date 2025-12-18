"""
KUCCPS ETL CLI (skeleton)

This script provides a minimal, well-structured ETL pipeline for KUCCPS data.
- extract: copies/optionally parses source PDFs into raw tabular artifacts
- transform: normalizes and prepares CSVs matching our canonical schema
- load: loads processed CSVs into Django models (catalog app)

Usage examples (run from project root; activate venv first):
  source venv/bin/activate
  python backend/scripts/etl/kuccps/etl.py extract --config backend/scripts/etl/kuccps/config.yaml
  python backend/scripts/etl/kuccps/etl.py transform --config backend/scripts/etl/kuccps/config.yaml
  python backend/scripts/etl/kuccps/etl.py load --config backend/scripts/etl/kuccps/config.yaml
  python backend/scripts/etl/kuccps/etl.py all --config backend/scripts/etl/kuccps/config.yaml

Notes
- This is a skeleton: PDF parsing is stubbed. You may replace extract/transform
  internals with pdfplumber/tabula logic or curated spreadsheets.
- The transform step creates empty CSVs with headers if none exist, to guide curation.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import re
import json

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    yaml = None

# ---------------------------
# Paths & Logging
# ---------------------------
HERE = Path(__file__).resolve()
KUCCPS_DIR = HERE.parent
ETL_DIR = KUCCPS_DIR.parent
SCRIPTS_DIR = ETL_DIR.parent
BACKEND_DIR = SCRIPTS_DIR.parent
LOGS_DIR = KUCCPS_DIR / "logs"
RAW_DIR = KUCCPS_DIR / "raw"
PROCESSED_DIR = KUCCPS_DIR / "processed"
MAPPINGS_DIR = KUCCPS_DIR / "mappings"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("kuccps_etl")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOGS_DIR / f"etl_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


@dataclass
class Config:
    """Lightweight config object loaded from YAML with sensible defaults."""
    dataset_year: int
    dataset_root: Path
    inputs: Dict[str, Any]
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR

    @staticmethod
    def from_yaml(path: Path) -> "Config":
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Add PyYAML to backend/requirements.txt and pip install.")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        dataset_year = int(data.get("dataset_year", datetime.utcnow().year))
        dataset_root = Path(data.get("dataset_root", BACKEND_DIR.parent / "Student Career Datasets")).resolve()
        inputs = data.get("inputs", {})
        raw_dir_val = data.get("raw_dir")
        processed_dir_val = data.get("processed_dir")
        raw_dir = Path(raw_dir_val).resolve() if raw_dir_val and str(raw_dir_val).strip() else RAW_DIR
        processed_dir = Path(processed_dir_val).resolve() if processed_dir_val and str(processed_dir_val).strip() else PROCESSED_DIR
        return Config(dataset_year=dataset_year, dataset_root=dataset_root, inputs=inputs, raw_dir=raw_dir, processed_dir=processed_dir)


# ---------------------------
# Utilities
# ---------------------------

def ensure_dirs(cfg: Config) -> None:
    """Ensure raw/processed directories exist."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)


def copy_inputs(cfg: Config) -> None:
    """Copy referenced input files into raw_dir for tracking (no parsing yet)."""
    copied = 0
    for key, rel in cfg.inputs.items():
        src = (cfg.dataset_root / rel).resolve()
        if not src.exists():
            logger.warning("Input missing: %s -> %s", key, src)
            continue
        dst = cfg.raw_dir / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
            copied += 1
            logger.info("Copied %s -> %s", src, dst)
    logger.info("extract: %d inputs copied into %s", copied, cfg.raw_dir)


def extract_programs(cfg: Config) -> None:
    """Focused extractor for the programs PDF only (DEGREE_PROGRAMMES_2025.pdf).

    Actions:
    - copy programs_pdf to raw/
    - export page text into raw/programs_text/
    - attempt naive table extraction per page into raw/programs_tables/
    """
    ensure_dirs(cfg)
    key = "programs_pdf"
    rel = cfg.inputs.get(key)
    if not rel:
        logger.error("No '%s' configured in inputs", key)
        return
    src = (cfg.dataset_root / rel).resolve()
    if not src.exists():
        logger.error("Programs PDF not found: %s", src)
        return
    dst_pdf = cfg.raw_dir / src.name
    if not dst_pdf.exists():
        dst_pdf.write_bytes(src.read_bytes())
        logger.info("Copied programs PDF -> %s", dst_pdf)

    text_dir = cfg.raw_dir / "programs_text"
    tables_dir = cfg.raw_dir / "programs_tables"
    text_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pdfplumber  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.error("pdfplumber not installed: %s. pip install pdfplumber", e)
        return

    pages_cnt = 0
    tables_cnt = 0
    with pdfplumber.open(str(src)) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            pages_cnt += 1
            # dump text
            try:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                (text_dir / f"page_{pi:04d}.txt").write_text(txt, encoding="utf-8")
            except Exception as e:
                logger.warning("page %d text extract failed: %s", pi, e)
            # naive table extraction
            try:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                }) or []
                for ti, table in enumerate(tables, start=1):
                    tables_cnt += 1
                    out = tables_dir / f"page_{pi:04d}_table_{ti:02d}.csv"
                    with open(out, "w", newline="", encoding="utf-8") as f:
                        # Write as TSV to avoid quoting around commas in programme names
                        writer = csv.writer(f, delimiter="\t")
                        for row in table:
                            cleaned_row: List[str] = []
                            for c in row:
                                s = (c or "").replace("\r", " ").replace("\n", " ")
                                s = re.sub(r"\s+", " ", s).strip()
                                cleaned_row.append(s)
                            writer.writerow(cleaned_row)
            except Exception as e:
                logger.warning("page %d table extract failed: %s", pi, e)

    logger.info("extract-programs: pages=%d tables=%d output in %s and %s", pages_cnt, tables_cnt, text_dir, tables_dir)


# ---------------------------
# Transform (skeleton)
# ---------------------------
EXPECTED = {
    "institutions.csv": ["code", "name", "alias", "region", "county", "country", "website"],
    "fields.csv": ["name", "parent", "description"],
    "subjects.csv": ["code", "name", "group", "alt_codes"],
    "programs.csv": [
        "source_index", "institution_code", "institution_name", "field_name", "program_code", "course_suffix", "name", "normalized_name", "level", "campus", "region", "duration_years", "award", "mode", "subject_requirements_json",
    ],
    "yearly_cutoffs.csv": ["program_normalized_name", "institution_code", "program_code", "year", "cutoff", "capacity", "notes"],
    "normalization_rules.csv": ["type", "source_value", "normalized_value"],
}


def bootstrap_csvs(cfg: Config) -> None:
    """Create empty CSVs with headers if they don't exist to guide curation or later parsing."""
    for fname, headers in EXPECTED.items():
        fpath = cfg.processed_dir / fname
        if not fpath.exists():
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            logger.info("Created template CSV: %s", fpath)
    logger.info("transform: ensured template CSVs in %s", cfg.processed_dir)


# ---------------------------
# Transform: programs tables -> processed CSVs
# ---------------------------

NUM_RE = re.compile(r"^\d+(?:\.|)$")


def _is_header_row(cells: List[str]) -> bool:
    joined = " ".join(cells).upper()
    return "PROG" in joined and "CODE" in joined and "INSTITUTION" in joined


def _is_program_block_header(cells: List[str]) -> bool:
    if not cells:
        return False
    first = (cells[0] or "").strip()
    if not first:
        return False
    if NUM_RE.match(first):
        return False
    # treat as header if all other cells are empty
    others_empty = all(((c or "").strip() == "") for c in cells[1:])
    return others_empty


def _infer_level(text: str) -> str:
    t = (text or "").strip().upper()
    if t.startswith("BACHELOR"):
        return "bachelor"
    if t.startswith("DIPLOMA"):
        return "diploma"
    if t.startswith("CERTIFICATE"):
        return "certificate"
    return "bachelor"


SUBJ_RE = re.compile(r"(?P<subject>[A-Za-z.&\s]+?)(?:\s*A)?\((?P<code>\d[\s]?\d[\s]?\d)\)")


def _parse_requirements(subject_cells: List[str]) -> Dict[str, Any]:
    text = ",".join([(c or "").replace("\n", " ").strip() for c in subject_cells if c]).strip(", ")
    if not text:
        return {}
    required: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    for token in [t.strip() for t in text.split(",") if t.strip()]:
        if ":" not in token:
            continue
        left, grade = token.split(":", 1)
        grade = grade.strip()
        options = [o.strip() for o in left.split("/") if o.strip()]
        opts: List[Dict[str, Any]] = []
        for opt in options:
            m = SUBJ_RE.search(opt)
            if not m:
                continue
            subj = re.sub(r"\s+", " ", m.group("subject").strip())
            code = re.sub(r"\s+", "", m.group("code"))
            opts.append({"subject": subj, "code": code, "min_grade": grade})
        if not opts:
            continue
        if len(opts) == 1:
            required.append(opts[0])
        else:
            groups.append({"pick": 1, "options": opts})
    out: Dict[str, Any] = {}
    if required:
        out["required"] = required
    if groups:
        out["groups"] = groups
    return out


def _clean(cell: Optional[str]) -> str:
    return (cell or "").replace("\n", " ").replace("\r", " ").strip()


def _strip_quotes(text: Optional[str]) -> str:
    s = (text or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return re.sub(r"\s+", " ", s).strip()


def transform_programs(cfg: Config) -> None:
    """Parse extracted programs tables into processed/programs.csv and yearly_cutoffs.csv."""
    ensure_dirs(cfg)
    tables_candidates = [cfg.raw_dir / "programs_tables", BACKEND_DIR / "programs_tables"]
    tables_dir = None
    for c in tables_candidates:
        if c.exists():
            tables_dir = c
            break
    if not tables_dir:
        logger.error("programs_tables directory not found in %s", tables_candidates)
        return

    programs_out = cfg.processed_dir / "programs.csv"
    cutoffs_out = cfg.processed_dir / "yearly_cutoffs.csv"

    programs_rows: List[List[str]] = []
    cutoffs_rows: List[List[str]] = []
    code_corrections: List[List[str]] = []  # [source_index, institution_name, original_code, corrected_code, reason, program_header]

    current_program: Optional[str] = None
    current_level: str = "bachelor"
    block_rows: List[Dict[str, Any]] = []  # buffer rows until next header to allow suffix-mode correction
    inst_prefix_counts: Dict[str, Dict[str, int]] = {}
    inst_modal_prefix_map: Dict[str, str] = {}

    def _digits(s: str) -> str:
        return "".join(ch for ch in (s or "") if ch.isdigit())

    def _is_valid_code(code: str) -> bool:
        """Accepts numeric codes (>=7 digits) and alphanumeric codes of form ddddLdd (e.g., 1080B61)."""
        c = (code or "").strip().upper()
        if not c:
            return False
        if re.fullmatch(r"\d{7,}", _digits(c)):
            return True
        if re.fullmatch(r"\d{4}[A-Z]\d{2,3}", c):
            return True
        return False

    def finalize_block() -> None:
        # Compute majority 3-digit suffix across numeric codes within the block and correct outliers.
        nonlocal block_rows, programs_rows, cutoffs_rows, code_corrections, current_program, current_level
        if not block_rows:
            return
        suffix_counts: Dict[str, int] = {}
        for br in block_rows:
            d = _digits(br["prog_code"])  # digits-only
            if len(d) >= 7:
                suf = d[-3:]
                suffix_counts[suf] = suffix_counts.get(suf, 0) + 1
        # pick mode if it has at least 2 occurrences
        suffix_mode = None
        if suffix_counts:
            suffix_mode = max(suffix_counts.items(), key=lambda x: x[1])[0]
            if suffix_counts[suffix_mode] < 2:
                suffix_mode = None
        # Correct outliers and emit rows
        for br in block_rows:
            original_code = br["prog_code"]
            corrected_code = original_code
            d = _digits(original_code)
            # Strict validation (allow alphanumeric forms like 1080B61)
            if not _is_valid_code(original_code):
                code_corrections.append([br["serial"], br["inst_name"], original_code, original_code, "invalid_format", current_program or ""]) 
            if suffix_mode and len(d) >= 7 and d[-3:] != suffix_mode and d[:4].isdigit():
                # Heuristic: enforce common suffix for numeric codes in the block
                corrected_code = f"{d[:4]}{suffix_mode}"
                # Record audit only if changed
                if corrected_code != original_code:
                    code_corrections.append([
                        br["serial"], br["inst_name"], original_code, corrected_code, "block_suffix_mode", current_program or ""
                    ])
            # Derive course suffix from digits
            d2 = _digits(corrected_code)
            course_suffix = d2[-3:] if len(d2) >= 3 else ""
            # Institution-prefix modal correction: if this row's prefix deviates but suffix matches block suffix, rewrite to <modal_prefix><block_suffix>
            inst_name = br["inst_name"].strip()
            modal_prefix = inst_modal_prefix_map.get(inst_name)
            if modal_prefix and suffix_mode and len(d2) >= 7:
                cur_prefix = d2[:4]
                cur_suffix = d2[-3:]
                if cur_prefix != modal_prefix and cur_suffix == suffix_mode:
                    new_code = f"{modal_prefix}{suffix_mode}"
                    if new_code != corrected_code:
                        code_corrections.append([
                            br["serial"], inst_name, corrected_code, new_code, "inst_prefix_modal", current_program or ""
                        ])
                        corrected_code = new_code
                        d2 = _digits(corrected_code)
                        course_suffix = d2[-3:] if len(d2) >= 3 else ""
            # programs.csv row (align with EXPECTED order)
            clean_prog_name = _strip_quotes(br["prog_name"])  # remove stray quotes
            clean_norm_name = _strip_quotes(current_program or br["prog_name"])  # header text cleaned
            programs_rows.append([
                br["serial"],
                "",  # institution_code (derived later in normalize)
                _strip_quotes(br["inst_name"]),
                "",  # field_name
                corrected_code,
                course_suffix,
                clean_prog_name,
                clean_norm_name,
                current_level,
                "",  # campus
                "",  # region
                "",  # duration_years
                "",  # award
                "",  # mode
                br["req_json"],
            ])
            # yearly_cutoffs.csv rows
            if br["cutoff_2023"] and br["cutoff_2023"] != "-":
                cutoffs_rows.append([clean_norm_name, "", corrected_code, 2023, br["cutoff_2023"], "", ""])
            if br["cutoff_2022"] and br["cutoff_2022"] != "-":
                cutoffs_rows.append([clean_norm_name, "", corrected_code, 2022, br["cutoff_2022"], "", ""])
        block_rows = []

    files = sorted([p for p in tables_dir.glob("*.csv")])
    # Prepass: learn modal 4-digit prefixes per institution across entire dataset
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            # Auto-detect delimiter: support TSV (extract) and legacy CSV
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
            except Exception:
                dialect = csv.excel_tab if "\t" in sample else csv.excel
            reader = csv.reader(f, dialect=dialect)
            for row in reader:
                cells = [_clean(c) for c in row]
                if not cells or not any(cells):
                    continue
                first = cells[0]
                if not NUM_RE.match(first or ""):
                    continue
                while len(cells) < 4:
                    cells.append("")
                prog_code = cells[1]
                inst_name = cells[2]
                d = _digits(prog_code)
                if len(d) >= 4 and inst_name:
                    pref = d[:4]
                    inst_prefix_counts.setdefault(inst_name, {})
                    inst_prefix_counts[inst_name][pref] = inst_prefix_counts[inst_name].get(pref, 0) + 1
    for inst, counts in inst_prefix_counts.items():
        # choose the prefix with max frequency; require at least 2 occurrences to avoid noise
        pref, cnt = max(counts.items(), key=lambda x: x[1])
        if cnt >= 2:
            inst_modal_prefix_map[inst] = pref

    # Main pass: parse and correct block-by-block
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            # Auto-detect delimiter: support TSV (extract) and legacy CSV
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
            except Exception:
                dialect = csv.excel_tab if "\t" in sample else csv.excel
            reader = csv.reader(f, dialect=dialect)
            for row in reader:
                cells = [_clean(c) for c in row]
                if not any(cells):
                    continue
                if _is_header_row(cells):
                    continue
                # Program block header
                if _is_program_block_header(cells):
                    # Finish previous block before starting a new one
                    finalize_block()
                    current_program = cells[0]
                    current_level = _infer_level(current_program)
                    continue
                # Data rows
                if not cells:
                    continue
                first = cells[0]
                if not NUM_RE.match(first or ""):
                    continue
                # pad to at least 10 columns
                while len(cells) < 10:
                    cells.append("")
                serial, prog_code, inst_name, prog_name = cells[0], cells[1], cells[2], cells[3]
                cutoff_2023, cutoff_2022 = cells[4], cells[5]
                subj_cells = cells[6:10]
                req = _parse_requirements(subj_cells)
                req_json = json.dumps(req, ensure_ascii=False) if req else ""
                # Buffer row for block-level correction and emission
                block_rows.append({
                    "serial": serial,
                    "prog_code": prog_code,
                    "inst_name": inst_name,
                    "prog_name": prog_name,
                    "cutoff_2023": cutoff_2023,
                    "cutoff_2022": cutoff_2022,
                    "req_json": req_json,
                })

    # finalize last block at EOF
    finalize_block()

    # Write outputs (overwrite each run)
    with open(programs_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(EXPECTED["programs.csv"])
        if programs_rows:
            writer.writerows(programs_rows)
    with open(cutoffs_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(EXPECTED["yearly_cutoffs.csv"])
        if cutoffs_rows:
            writer.writerows(cutoffs_rows)
    logger.info("transform-programs: wrote %d programs and %d cutoff rows", len(programs_rows), len(cutoffs_rows))

    # Write code corrections audit if any
    if code_corrections:
        corr_out = cfg.processed_dir / "_code_corrections.csv"
        with open(corr_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_index", "institution_name", "original_code", "corrected_code", "reason", "program_header"])
            w.writerows(code_corrections)


# ---------------------------
# Transform: normalization/derivation
# ---------------------------

def _ensure_fields_map() -> Path:
    path = MAPPINGS_DIR / "fields_map.csv"
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pattern", "field_name"])  # substring or regex-like token
            w.writerow(["BIOCHEMISTRY|BIOLOGY", "Life Sciences"])
            w.writerow(["GEOLOGY|GEOSCIENCE", "Earth Sciences"])
            w.writerow(["ARCHITECTURE", "Architecture"])
            w.writerow(["ARTS", "Arts & Humanities"])
            w.writerow(["PSYCHOLOGY|SOCIOLOGY|POLITICAL", "Social Sciences"])
    return path


def _ensure_inst_meta() -> Path:
    path = MAPPINGS_DIR / "institutions_meta.csv"
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Extended header supports both code and name-based overrides. Backward compatible loader below.
            w.writerow(["institution_code", "name", "name_contains", "alias", "website"])  # overrides
    return path


def _load_inst_meta() -> List[Dict[str, str]]:
    p = _ensure_inst_meta()
    rows: List[Dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            code = (row.get("institution_code") or "").strip()
            # Support legacy 3-col files where only name_contains/alias/website exist
            name_contains = (row.get("name_contains") or row.get("name") or "").strip()
            alias = (row.get("alias") or "").strip()
            website = (row.get("website") or "").strip()
            name = (row.get("name") or "").strip()
            item = {
                "institution_code": code,
                "name_contains": name_contains.upper(),
                "alias": alias,
                "website": website,
                "name": name,
            }
            if code or name_contains:
                rows.append(item)
    return rows


def _alias_from_name(name: str) -> str:
    """Generate a simple alias from institution name (e.g., JKUAT, UON, MKU)."""
    if not name:
        return ""
    n = re.sub(r"[^A-Za-z&\s]", " ", name.upper())
    parts = [p for p in n.split() if p and p not in {"OF", "AND", "THE"}]
    alias = "".join(p[0] for p in parts if p)
    # Special cases common in Kenya
    specials = {
        "JOMO KENYATTA UNIVERSITY OF AGRICULTURE AND TECHNOLOGY": "JKUAT",
        "UNIVERSITY OF NAIROBI": "UON",
        "MOUNT KENYA UNIVERSITY": "MKU",
        "KENYATTA UNIVERSITY": "KU",
        "TECHNICAL UNIVERSITY OF KENYA": "TUK",
        "TECHNICAL UNIVERSITY OF MOMBASA": "TUM",
    }
    return specials.get(name.upper(), alias)

def _ensure_inst_geo() -> Path:
    path = MAPPINGS_DIR / "institutions_geo.csv"
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Allow an optional name column for readability.
            w.writerow(["institution_code", "name", "region", "county"])  # fill as needed
    return path


def _load_fields_map() -> List[Dict[str, str]]:
    p = _ensure_fields_map()
    out: List[Dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pat = (row.get("pattern") or "").strip()
            fld = (row.get("field_name") or "").strip()
            if pat and fld:
                out.append({"pattern": pat, "field_name": fld})
    return out


def _load_inst_geo() -> Dict[str, Dict[str, str]]:
    p = _ensure_inst_geo()
    out: Dict[str, Dict[str, str]] = {}
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            code = (row.get("institution_code") or "").strip()
            if code:
                out[code] = {
                    "region": (row.get("region") or "").strip(),
                    "county": (row.get("county") or "").strip(),
                }
    return out


def _ensure_inst_campuses() -> Path:
    path = MAPPINGS_DIR / "institutions_campuses.csv"
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # Non-main branches: town/county/region for specific campuses. Main is implied by institutions_geo.csv
            w.writerow(["institution_code", "campus", "town", "county", "region"])  # list only non-main branches
    return path


def _load_inst_campuses() -> Dict[str, Dict[str, Dict[str, str]]]:
    p = _ensure_inst_campuses()
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            code = (row.get("institution_code") or "").strip()
            campus = (row.get("campus") or "").strip().upper()
            if not code or not campus:
                continue
            out.setdefault(code, {})[campus] = {
                "town": (row.get("town") or "").strip(),
                "county": (row.get("county") or "").strip(),
                "region": (row.get("region") or "").strip(),
            }
    return out


def _derive_institution_code(program_code: str) -> str:
    s = (program_code or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:4] if len(digits) >= 4 else ""


def transform_normalize(cfg: Config) -> None:
    """Normalize/derive missing fields in processed/programs.csv and emit institutions.csv.

    - Fill institution_code from program code prefix
    - Classify field_name using mappings/mappings/fields_map.csv
    - Join region/county from mappings/institutions_geo.csv
    - Write processed/institutions.csv and overwrite processed/programs.csv
    - Write processed/_unclassified_programs.csv for manual mapping
    """
    ensure_dirs(cfg)
    fields_map = _load_fields_map()
    inst_geo = _load_inst_geo()
    inst_meta = _load_inst_meta()
    inst_campuses = _load_inst_campuses()

    programs_in = cfg.processed_dir / "programs.csv"
    if not programs_in.exists():
        logger.error("programs.csv not found in %s. Run transform-programs first.", cfg.processed_dir)
        return

    # Read all rows and normalize header keys (strip spaces)
    with open(programs_in, encoding="utf-8") as f:
        # Robust TSV handling:
        # - transform-programs writes TSV; subject_requirements_json cells contain commas
        # - csv.Sniffer can mis-detect delimiter as ',' when sampling JSON-heavy rows
        # - Prefer explicit TSV if we see tabs in the header; otherwise try Sniffer; fallback to TSV
        header_line = f.readline()
        rest = f.read(8192)
        f.seek(0)
        try:
            if "\t" in header_line:
                reader = csv.DictReader(f, delimiter="\t")
            else:
                sample = header_line + rest
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                reader = csv.DictReader(f, dialect=dialect)
        except Exception:
            # Default to TSV as a safe preference
            f.seek(0)
            reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        if reader.fieldnames:
            normalized_fields = [fn.strip() for fn in reader.fieldnames]
            if normalized_fields != reader.fieldnames:
                # remap keys per row using stripped field names
                remapped: List[Dict[str, Any]] = []
                for r in rows:
                    nr: Dict[str, Any] = {}
                    for old, val in r.items():
                        nr[(old or "").strip()] = val
                    remapped.append(nr)
                rows = remapped

    # Prepare outputs
    unclassified: List[List[str]] = []
    inst_map: Dict[str, Dict[str, str]] = {}

    def classify_field(normalized_name: str) -> str:
        name_u = (normalized_name or "").upper()
        for m in fields_map:
            pats = [p.strip() for p in m["pattern"].split("|") if p.strip()]
            if any(pat in name_u for pat in pats):
                return m["field_name"]
        return ""

    # Update rows
    rows_out: List[Dict[str, Any]] = []
    for r in rows:
        # Skip empty rows with no identifying data
        if not any([(r.get("program_code") or r.get("code") or "").strip(),
                    (r.get("name") or "").strip(),
                    (r.get("normalized_name") or "").strip(),
                    (r.get("institution_name") or "").strip()]):
            continue

        prog_code = (r.get("program_code") or r.get("code") or "").strip()
        inst_code = _derive_institution_code(prog_code)
        r["institution_code"] = inst_code
        # ensure course_suffix present
        if not (r.get("course_suffix") or "").strip():
            digits = "".join(ch for ch in prog_code if ch.isdigit())
            if digits:
                r["course_suffix"] = digits[-3:] if len(digits) >= 3 else ""

        # field classification
        if not (r.get("field_name") or "").strip():
            base_name = (r.get("normalized_name") or r.get("name") or "").strip()
            if base_name:
                fld = classify_field(base_name)
                r["field_name"] = fld
                if not fld:
                    unclassified.append([base_name, prog_code, "no_field_match"])

        # region join: campus-level region only if provided, otherwise fall back to institution-level region
        if inst_code and (not r.get("region") or not r.get("region").strip()):
            campus = (r.get("campus") or "").strip().upper()
            # Try campus-level mapping first
            campus_geo = None
            if campus and inst_code in inst_campuses:
                campus_geo = inst_campuses[inst_code].get(campus)
            campus_region = (campus_geo.get("region", "").strip() if campus_geo else "")
            if campus_region:
                r["region"] = campus_region
            else:
                # Fallback to code-level geo
                code_geo = inst_geo.get(inst_code)
                if code_geo:
                    r["region"] = code_geo.get("region", "")
            # county remains institutions-only

        # collect institutions
        name = (r.get("institution_name") or "").strip()
        if name.startswith('"') and name.endswith('"'):
            name = name[1:-1]
        name = re.sub(r"\s+", " ", name).strip()
        if inst_code and name:
            if inst_code not in inst_map:
                # apply optional meta overrides by substring match
                alias = _alias_from_name(name)
                website = ""
                upper_name = name.upper()
                # Prefer direct code match from meta; fallback to substring
                matched = False
                for m in inst_meta:
                    if m.get("institution_code") and m.get("institution_code") == inst_code:
                        if m.get("alias"):
                            alias = m["alias"]
                        if m.get("website"):
                            website = m["website"]
                        matched = True
                        break
                if not matched:
                    for m in inst_meta:
                        if m.get("name_contains") and m["name_contains"] in upper_name:
                            if m.get("alias"):
                                alias = m["alias"]
                            if m.get("website"):
                                website = m["website"]
                            break
                inst_map[inst_code] = {
                    "code": inst_code,
                    "name": name,
                    "alias": alias,
                    "region": inst_geo.get(inst_code, {}).get("region", ""),
                    "county": inst_geo.get(inst_code, {}).get("county", ""),
                    "country": "Kenya",
                    "website": website,
                }

        # keep non-empty normalized rows only
        rows_out.append(r)

    # Drop rows that are entirely empty (no code, no names)
    rows = [
        rr for rr in rows_out
        if any([
            (rr.get("program_code") or rr.get("code") or "").strip(),
            (rr.get("name") or "").strip(),
            (rr.get("normalized_name") or "").strip(),
            (rr.get("institution_name") or "").strip(),
        ])
    ]

    # Write institutions.csv
    inst_out = cfg.processed_dir / "institutions.csv"
    with open(inst_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EXPECTED["institutions.csv"])
        for code in sorted(inst_map.keys()):
            d = inst_map[code]
            w.writerow([d["code"], d["name"], d["alias"], d["region"], d["county"], d.get("country", "Kenya"), d["website"]])

    # Overwrite programs.csv
    with open(programs_in, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(EXPECTED["programs.csv"])
        for r in rows:
            inst_name_clean = _strip_quotes(r.get("institution_name", ""))
            name_clean = _strip_quotes(r.get("name", ""))
            norm_clean = _strip_quotes(r.get("normalized_name", ""))
            w.writerow([
                r.get("source_index", ""),
                r.get("institution_code", ""),
                inst_name_clean,
                r.get("field_name", ""),
                r.get("program_code", r.get("code", "")),
                r.get("course_suffix", ""),
                name_clean,
                norm_clean,
                r.get("level", ""),
                r.get("campus", ""),
                r.get("region", ""),
                r.get("duration_years", ""),
                r.get("award", ""),
                r.get("mode", ""),
                r.get("subject_requirements_json", ""),
            ])

    # Write program_offerings summary (unique institutions per normalized course)
    offer_map = defaultdict(set)
    for r in rows:
        key_name = _strip_quotes((r.get("normalized_name") or r.get("name") or "").strip())
        key_suffix = (r.get("course_suffix") or "").strip()
        inst_code = (r.get("institution_code") or "").strip()
        if key_name and inst_code:
            offer_map[(key_name, key_suffix)].add(inst_code)
    if offer_map:
        with open(cfg.processed_dir / "program_offerings.csv", "w", newline="", encoding="utf-8") as f:
            # Write as TSV to minimize quoting
            w = csv.writer(f, delimiter="\t")
            w.writerow(["program_normalized_name", "course_suffix", "offerings_count"])
            for (pn, cs), insts in sorted(offer_map.items()):
                w.writerow([pn, cs, len(insts)])

    # Broad offerings: aggregate variants under base course names
    def _coarse_program_name(name: str) -> str:
        u = name.upper().strip()
        # Group common degree families under a base label
        families = [
            "BACHELOR OF ARTS",
            "BACHELOR OF SCIENCE",
            "BACHELOR OF EDUCATION",
            "BACHELOR OF TECHNOLOGY",
            "BACHELOR OF COMMERCE",
            "BACHELOR OF ENGINEERING",
            "BACHELOR OF LAWS",
            "BACHELOR OF MEDICINE",
        ]
        for fam in families:
            if u.startswith(fam):
                return fam
        return name

    broad_offer_map = defaultdict(set)
    for r in rows:
        raw_name = _strip_quotes((r.get("normalized_name") or r.get("name") or "").strip())
        key_name = _coarse_program_name(raw_name)
        inst_code = (r.get("institution_code") or "").strip()
        if key_name and inst_code:
            broad_offer_map[key_name].add(inst_code)
    if broad_offer_map:
        with open(cfg.processed_dir / "program_offerings_broad.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["program_normalized_name", "offerings_count"])
            for pn, insts in sorted(broad_offer_map.items()):
                w.writerow([pn, len(insts)])

    # Write unclassified
    if unclassified:
        with open(cfg.processed_dir / "_unclassified_programs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["normalized_name", "program_code", "reason"])
            w.writerows(unclassified)
    # Dedup candidates: identify duplicate program rows within the same institution/name/level/campus
    # This prepares review files without mutating data. A later step can use this to drive DedupMatch records.
    def _key_for_dedup(rr: Dict[str, Any]) -> Tuple[str, str, str, str]:
        inst = (rr.get("institution_code") or "").strip()
        name = _strip_quotes((rr.get("normalized_name") or rr.get("name") or "").strip())
        level = (rr.get("level") or "").strip().lower() or "bachelor"
        campus = (rr.get("campus") or "").strip()
        return inst, name, level, campus

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rr in rows:
        groups[_key_for_dedup(rr)].append(rr)

    dups = {k: v for k, v in groups.items() if len(v) > 1}

    def _code_key(rc: Dict[str, Any]) -> Tuple[int, str]:
        code = (rc.get("program_code") or rc.get("code") or "").strip()
        # Prefer pure numeric codes over alphanumeric; then lexicographic
        has_alpha = int(any(ch.isalpha() for ch in code))
        return (has_alpha, code)

    if dups:
        cand_path = cfg.processed_dir / "dedup_candidates.csv"
        with open(cand_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "institution_code",
                "institution_name",
                "normalized_name",
                "level",
                "campus",
                "rows_count",
                "program_codes",
                "name_variants",
                "suggested_master_program_code",
            ])
            for (inst, name, level, campus), items in sorted(dups.items()):
                codes = sorted(((it.get("program_code") or it.get("code") or "").strip() for it in items))
                names = sorted(set(_strip_quotes((it.get("name") or "").strip()) for it in items))
                master = sorted(items, key=_code_key)[0]
                master_code = (master.get("program_code") or master.get("code") or "").strip()
                inst_name = _strip_quotes((items[0].get("institution_name") or "").strip())
                w.writerow([inst, inst_name, name, level, campus, len(items), "|".join(codes), "|".join(names), master_code])

        # Summary by institution: number of duplicate groups and total duplicate rows
        summary_path = cfg.processed_dir / "dedup_summary.csv"
        by_inst: Dict[str, Tuple[int, int, str]] = defaultdict(lambda: (0, 0, ""))
        for (inst, name, level, campus), items in dups.items():
            groups_count, rows_count, inst_name = by_inst[inst]
            by_inst[inst] = (groups_count + 1, rows_count + len(items), _strip_quotes((items[0].get("institution_name") or "").strip()))
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["institution_code", "institution_name", "duplicate_groups", "duplicate_rows"])
            for inst in sorted(by_inst.keys()):
                g, r, nm = by_inst[inst]
                w.writerow([inst, nm, g, r])

    logger.info("transform-normalize: institutions=%d, programs=%d, unclassified=%d", len(inst_map), len(rows), len(unclassified))


# ---------------------------
# Load (Django ORM)
# ---------------------------

def setup_django() -> None:
    """Initialize Django so we can import and use ORM in a standalone script."""
    sys.path.append(str(BACKEND_DIR))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
    import django  # type: ignore
    django.setup()


def load_csvs(cfg: Config, dry_run: bool = False) -> None:
    """Load processed CSVs into catalog models. Uses upserts (get_or_create) for idempotency.

    Expected files are created by transform(). Users can replace with curated data before load().
    """
    setup_django()
    from django.db import transaction
    from catalog.models import Institution, Field, Subject, Program, YearlyCutoff, NormalizationRule, ProgramLevel

    @transaction.atomic
    def _load():
        # Institutions
        inst_path = cfg.processed_dir / "institutions.csv"
        if inst_path.exists():
            with open(inst_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    Institution.objects.get_or_create(
                        code=row.get("code", "").strip(),
                        defaults={
                            "name": row.get("name", "").strip(),
                            "alias": row.get("alias", "").strip(),
                            "region": row.get("region", "").strip(),
                            "county": row.get("county", "").strip(),
                            "website": row.get("website", "").strip(),
                        },
                    )
        # Fields (simple by-name hierarchy)
        fields_cache = {}
        fld_path = cfg.processed_dir / "fields.csv"
        if fld_path.exists():
            with open(fld_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # First pass create parents
                for row in reader:
                    name = row.get("name", "").strip()
                    if not name:
                        continue
                    obj, _ = Field.objects.get_or_create(name=name)
                    fields_cache[name] = obj
                f.seek(0)
                next(f)  # skip header
                for row in csv.DictReader(f):
                    name = row.get("name", "").strip()
                    parent_name = row.get("parent", "").strip()
                    if name and parent_name and parent_name in fields_cache:
                        fld = fields_cache[name]
                        if fld.parent_id is None:
                            fld.parent = fields_cache[parent_name]
                            fld.save(update_fields=["parent"])
        # Subjects
        subj_path = cfg.processed_dir / "subjects.csv"
        if subj_path.exists():
            with open(subj_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    Subject.objects.get_or_create(
                        code=row.get("code", "").strip(),
                        defaults={
                            "name": row.get("name", "").strip(),
                            "group": row.get("group", "").strip(),
                            "alt_codes": [v.strip() for v in (row.get("alt_codes", "").split("|") if row.get("alt_codes") else [])],
                        },
                    )
        # Programs: prefer deduplicated file if present (ensures master selection is honored)
        prog_path = cfg.processed_dir / "programs_deduped.csv"
        if not prog_path.exists():
            prog_path = cfg.processed_dir / "programs.csv"
        if prog_path.exists():
            with open(prog_path, encoding="utf-8") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                except Exception:
                    dialect = csv.excel_tab if "\t" in sample else csv.excel
                reader = csv.DictReader(f, dialect=dialect)
                for row in reader:
                    inst_code = row.get("institution_code", "").strip()
                    field_name = row.get("field_name", "").strip()
                    inst = Institution.objects.filter(code=inst_code).first()
                    fld = Field.objects.filter(name=field_name).first()
                    level = row.get("level", "").strip().lower() or ProgramLevel.BACHELOR
                    # Basic sanitize for normalized_name fallback
                    normalized_name = row.get("normalized_name") or row.get("name") or ""
                    normalized_name = normalized_name.strip()
                    Program.objects.get_or_create(
                        institution=inst,
                        normalized_name=normalized_name,
                        level=level,
                        campus=row.get("campus", "").strip(),
                        defaults={
                            "field": fld,
                            "code": (row.get("program_code") or row.get("code") or "").strip(),
                            "name": row.get("name", "").strip(),
                            "region": row.get("region", "").strip(),
                            "duration_years": (row.get("duration_years") or None),
                            "award": row.get("award", "").strip(),
                            "mode": row.get("mode", "").strip(),
                            "subject_requirements": _safe_json(row.get("subject_requirements_json", "{}")),
                        },
                    )
        # Cutoffs
        cut_path = cfg.processed_dir / "yearly_cutoffs.csv"
        if cut_path.exists():
            with open(cut_path, encoding="utf-8") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                except Exception:
                    dialect = csv.excel_tab if "\t" in sample else csv.excel
                reader = csv.DictReader(f, dialect=dialect)
                for row in reader:
                    prog = None
                    # Preferred: program_code match
                    prog_code = (row.get("program_code") or "").strip()
                    if prog_code:
                        prog = Program.objects.filter(code=prog_code).first()
                    # Fallback: institution_code + normalized_name
                    if not prog:
                        inst_code = (row.get("institution_code") or "").strip()
                        prog_name = (row.get("program_normalized_name") or "").strip()
                        if inst_code and prog_name:
                            inst = Institution.objects.filter(code=inst_code).first()
                            if inst:
                                prog = Program.objects.filter(institution=inst, normalized_name=prog_name).first()
                    if not prog:
                        logger.warning("Cutoff row skipped (program not found): %s", row)
                        continue
                    YearlyCutoff.objects.get_or_create(
                        program=prog,
                        year=int(row.get("year") or 0),
                        defaults={
                            "cutoff": row.get("cutoff") or 0,
                            "capacity": (int(row.get("capacity") or 0) or None),
                            "notes": (row.get("notes") or "").strip(),
                        },
                    )
        # Normalization rules
        norm_path = cfg.processed_dir / "normalization_rules.csv"
        if norm_path.exists():
            with open(norm_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    NormalizationRule.objects.get_or_create(
                        type=row.get("type", "").strip(),
                        source_value=row.get("source_value", "").strip(),
                        defaults={"normalized_value": row.get("normalized_value", "").strip()},
                    )

    if dry_run:
        logger.info("load: DRY RUN - skipping DB writes")
        return
    _load()
    logger.info("load: finished writing to database")


def dedup_programs(cfg: Config, inplace: bool = False) -> None:
    """Produce processed/programs_deduped.csv keeping 1 row per (inst, name, level, campus).
    Selection rule: prefer purely numeric program_code; then lexicographic among codes.
    Also emit processed/dedup_suppressed.csv listing suppressed rows with their master.
    """
    ensure_dirs(cfg)
    src = cfg.processed_dir / "programs.csv"
    if not src.exists():
        logger.error("programs.csv not found in %s. Run transform-normalize first.", cfg.processed_dir)
        return
    # Read TSV robustly (same approach as normalize)
    with open(src, encoding="utf-8") as f:
        header_line = f.readline()
        rest = f.read(8192)
        f.seek(0)
        try:
            if "\t" in header_line:
                reader = csv.DictReader(f, delimiter="\t")
            else:
                sample = header_line + rest
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                reader = csv.DictReader(f, dialect=dialect)
        except Exception:
            f.seek(0)
            reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    def _key(rr: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (
            (rr.get("institution_code") or "").strip(),
            _strip_quotes((rr.get("normalized_name") or rr.get("name") or "").strip()),
            (rr.get("level") or "").strip().lower() or "bachelor",
            (rr.get("campus") or "").strip(),
        )

    def _code_key(rc: Dict[str, Any]) -> Tuple[int, str]:
        code = (rc.get("program_code") or rc.get("code") or "").strip()
        has_alpha = int(any(ch.isalpha() for ch in code))
        return (has_alpha, code)

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        # skip empty rows
        if not any([(r.get("program_code") or r.get("code") or "").strip(), (r.get("institution_name") or "").strip()]):
            continue
        groups[_key(r)].append(r)

    masters: List[Dict[str, Any]] = []
    suppressed: List[List[str]] = []
    for k, items in groups.items():
        if not items:
            continue
        # Choose master by code heuristic
        master = sorted(items, key=_code_key)[0]
        masters.append(master)
        if len(items) > 1:
            master_code = (master.get("program_code") or master.get("code") or "").strip()
            for it in items:
                if it is master:
                    continue
                suppressed.append([
                    (it.get("institution_code") or "").strip(),
                    _strip_quotes((it.get("institution_name") or "").strip()),
                    _strip_quotes((it.get("normalized_name") or it.get("name") or "").strip()),
                    (it.get("level") or "").strip().lower() or "bachelor",
                    (it.get("campus") or "").strip(),
                    (it.get("program_code") or it.get("code") or "").strip(),
                    master_code,
                ])

    # Write outputs
    out_path = cfg.processed_dir / "programs_deduped.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(EXPECTED["programs.csv"])  # same schema
        for m in masters:
            w.writerow([
                m.get("source_index", ""),
                m.get("institution_code", ""),
                _strip_quotes(m.get("institution_name", "")),
                m.get("field_name", ""),
                m.get("program_code", m.get("code", "")),
                m.get("course_suffix", ""),
                _strip_quotes(m.get("name", "")),
                _strip_quotes(m.get("normalized_name", "")),
                m.get("level", ""),
                m.get("campus", ""),
                m.get("region", ""),
                m.get("duration_years", ""),
                m.get("award", ""),
                m.get("mode", ""),
                m.get("subject_requirements_json", ""),
            ])

    sup_path = cfg.processed_dir / "dedup_suppressed.csv"
    if suppressed:
        with open(sup_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "institution_code",
                "institution_name",
                "normalized_name",
                "level",
                "campus",
                "suppressed_program_code",
                "master_program_code",
            ])
            w.writerows(suppressed)
    # Optionally overwrite programs.csv with deduped masters
    if inplace:
        try:
            data = out_path.read_text(encoding="utf-8")
            (cfg.processed_dir / "programs.csv").write_text(data, encoding="utf-8")
            logger.info("dedup-programs: inplace write to programs.csv successful")
        except Exception as e:
            logger.error("dedup-programs: inplace write failed: %s", e)
    logger.info("dedup-programs: wrote %d masters and %d suppressed rows", len(masters), len(suppressed))


def dq_report(cfg: Config) -> None:
    """Generate processed/dq_report.csv with key KPIs.
    - Programs missing region (program-level)
    - Programs missing county (via institutions.csv county by institution_code)
    - Unclassified count and top unclassified names (if file exists)
    - Duplicate summary totals (if dedup_summary.csv exists)
    """
    ensure_dirs(cfg)
    prog_path = cfg.processed_dir / "programs_deduped.csv"
    if not prog_path.exists():
        prog_path = cfg.processed_dir / "programs.csv"
    programs: List[Dict[str, Any]] = []
    if prog_path.exists():
        with open(prog_path, encoding="utf-8") as f:
            header_line = f.readline()
            rest = f.read(8192)
            f.seek(0)
            try:
                if "\t" in header_line:
                    reader = csv.DictReader(f, delimiter="\t")
                else:
                    sample = header_line + rest
                    dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                    reader = csv.DictReader(f, dialect=dialect)
            except Exception:
                f.seek(0)
                reader = csv.DictReader(f, delimiter="\t")
            programs = list(reader)

    # Institutions county map
    inst_county: Dict[str, str] = {}
    inst_path = cfg.processed_dir / "institutions.csv"
    if inst_path.exists():
        with open(inst_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code = (row.get("code") or "").strip()
                inst_county[code] = (row.get("county") or "").strip()

    total_programs = len(programs)
    missing_region_programs = sum(1 for r in programs if not (r.get("region") or "").strip())
    missing_county_programs = 0
    for r in programs:
        inst = (r.get("institution_code") or "").strip()
        if not inst_county.get(inst, ""):
            missing_county_programs += 1

    # Unclassified
    unclassified_path = cfg.processed_dir / "_unclassified_programs.csv"
    unclassified_rows: List[Dict[str, Any]] = []
    if unclassified_path.exists():
        with open(unclassified_path, encoding="utf-8") as f:
            unclassified_rows = list(csv.DictReader(f))
    unclassified_count = len(unclassified_rows)
    top_unclassified = []
    if unclassified_rows:
        cnt = Counter((r.get("normalized_name") or "").strip() for r in unclassified_rows)
        top_unclassified = cnt.most_common(10)

    # Duplicates summary
    dup_groups = 0
    dup_rows = 0
    dup_sum_path = cfg.processed_dir / "dedup_summary.csv"
    if dup_sum_path.exists():
        with open(dup_sum_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    dup_groups += int(row.get("duplicate_groups") or 0)
                    dup_rows += int(row.get("duplicate_rows") or 0)
                except Exception:
                    pass

    # Write report
    out = cfg.processed_dir / "dq_report.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "notes"])
        w.writerow(["programs_total", total_programs, "base=" + prog_path.name])
        w.writerow(["programs_missing_region", missing_region_programs, "program-level region cell is empty"])
        w.writerow(["programs_missing_county", missing_county_programs, "derived via institutions.csv county"])
        w.writerow(["duplicates_groups", dup_groups, "sum of dedup_summary.csv duplicate_groups"])
        w.writerow(["duplicates_rows", dup_rows, "sum of dedup_summary.csv duplicate_rows"])
        w.writerow(["unclassified_count", unclassified_count, "rows in _unclassified_programs.csv if present"])
        for name, count in top_unclassified:
            w.writerow(["unclassified_top_name", count, name])
    logger.info("dq-report: wrote %s", out)


def _safe_json(val: str) -> Dict[str, Any]:
    """Small helper: parse lightweight JSON from CSV cell if provided; else empty dict.
    We avoid importing json to keep deps minimal and because content is optional.
    """
    import json
    try:
        return json.loads(val or "{}")
    except Exception:
        return {}


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KUCCPS ETL (skeleton)")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("extract", "transform", "load", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True, help="Path to config.yaml")
        if name == "load":
            sp.add_argument("--dry-run", action="store_true")

    sp_prog = sub.add_parser("extract-programs")
    sp_prog.add_argument("--config", required=True, help="Path to config.yaml")
    sp_tprog = sub.add_parser("transform-programs")
    sp_tprog.add_argument("--config", required=True, help="Path to config.yaml")
    sp_tnorm = sub.add_parser("transform-normalize")
    sp_tnorm.add_argument("--config", required=True, help="Path to config.yaml")
    sp_dedup = sub.add_parser("dedup-programs")
    sp_dedup.add_argument("--config", required=True, help="Path to config.yaml")
    sp_dedup.add_argument("--inplace", action="store_true")
    sp_dq = sub.add_parser("dq-report")
    sp_dq.add_argument("--config", required=True, help="Path to config.yaml")
    return p.parse_args()


def main() -> None:
    ns = parse_args()
    cfg = Config.from_yaml(Path(ns.config))
    ensure_dirs(cfg)

    if ns.cmd == "extract":
        copy_inputs(cfg)
    elif ns.cmd == "extract-programs":
        extract_programs(cfg)
    elif ns.cmd == "transform-programs":
        transform_programs(cfg)
    elif ns.cmd == "transform-normalize":
        transform_normalize(cfg)
    elif ns.cmd == "dedup-programs":
        dedup_programs(cfg, inplace=getattr(ns, "inplace", False))
    elif ns.cmd == "dq-report":
        dq_report(cfg)
    elif ns.cmd == "transform":
        bootstrap_csvs(cfg)
    elif ns.cmd == "load":
        load_csvs(cfg, dry_run=getattr(ns, "dry_run", False))
    elif ns.cmd == "all":
        copy_inputs(cfg)
        bootstrap_csvs(cfg)
        load_csvs(cfg)


if __name__ == "__main__":
    main()
