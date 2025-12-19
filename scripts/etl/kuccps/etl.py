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
from decimal import Decimal, InvalidOperation

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
    # Support absolute or dataset_root-relative paths
    rel_path = Path(str(rel))
    src = rel_path if rel_path.is_absolute() else (cfg.dataset_root / rel_path)
    src = src.resolve()
    if not src.exists():
        logger.error("Programs PDF not found: %s", src)
        return
    # Namespace under raw/<source_id>/ to avoid collisions across multiple sources
    source_id = str(cfg.inputs.get("source_id") or "default")
    ns_dir = (cfg.raw_dir / source_id)
    ns_dir.mkdir(parents=True, exist_ok=True)
    dst_pdf = ns_dir / src.name
    if not dst_pdf.exists():
        dst_pdf.write_bytes(src.read_bytes())
        logger.info("Copied programs PDF -> %s", dst_pdf)

    text_dir = ns_dir / "programs_text"
    tables_dir = ns_dir / "programs_tables"
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
        "source_index", "institution_code", "institution_name", "field_name", "program_code", "course_suffix", "name", "normalized_name", "level", "campus", "region", "duration_years", "award", "mode", "subject_requirements_json", "source_id",
    ],
    "yearly_cutoffs.csv": ["program_normalized_name", "institution_code", "program_code", "year", "cutoff", "capacity", "notes", "source_id"],
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


def _looks_like_text(s: Optional[str]) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    return any(ch.isalpha() for ch in t) and len(t) >= 3


def transform_programs(cfg: Config) -> None:
    """Parse extracted programs tables into processed/programs.csv and yearly_cutoffs.csv.
    Now supports multiple sources under raw/<source_id>/programs_tables and optional format adapters.
    """
    ensure_dirs(cfg)
    # Discover all table directories: unnamespaced and namespaced
    tables_dirs: List[Tuple[str, Path]] = []
    # Legacy unnamespaced
    legacy = cfg.raw_dir / "programs_tables"
    if legacy.exists():
        tables_dirs.append(("default", legacy))
    # Namespaced
    for p in sorted(cfg.raw_dir.glob("*/programs_tables")):
        if p.is_dir():
            tables_dirs.append((p.parent.name, p))
    if not tables_dirs and (BACKEND_DIR / "programs_tables").exists():
        tables_dirs.append(("default", BACKEND_DIR / "programs_tables"))
    if not tables_dirs:
        logger.error("programs_tables directory not found under raw/* or legacy paths")
        return

    # Load optional format adapters YAML from inputs[format_adapter]
    adapter_map: Dict[str, Any] = {}
    adapter_path = cfg.inputs.get("format_adapter")
    if adapter_path:
        try:
            ap = Path(str(adapter_path))
            if not ap.is_absolute():
                ap = (cfg.dataset_root / ap).resolve()
            if yaml is None:
                raise RuntimeError("PyYAML not installed for format adapter support")
            if ap.exists():
                adapter_map = yaml.safe_load(ap.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("format_adapter load failed: %s", e)

    def _adapter_for(src: str, year: int) -> Dict[str, Any]:
        # Structure: { format_adapters: { <source_id>: { <year>: {...} | "default": {...} } } }
        root = adapter_map.get("format_adapters") or {}
        by_src = root.get(src) or {}
        return by_src.get(year) or by_src.get("default") or {}

    programs_out = cfg.processed_dir / "programs.csv"
    cutoffs_out = cfg.processed_dir / "yearly_cutoffs.csv"

    programs_rows: List[List[str]] = []
    cutoffs_rows: List[List[str]] = []
    costs_rows: List[List[str]] = []  # [program_code, institution_name, program_name, cost, source_id]
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
        suffix_name_counts: Dict[str, Counter] = defaultdict(Counter)
        for br in block_rows:
            d = _digits(br["prog_code"])  # digits-only
            if len(d) >= 7:
                suf = d[-3:]
                suffix_counts[suf] = suffix_counts.get(suf, 0) + 1
                # collect candidate program names for this suffix when textual
                nm = _strip_quotes(br.get("prog_name", ""))
                if _looks_like_text(nm):
                    suffix_name_counts[suf][nm] += 1
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
            # If name looks numeric or blank, try recover from suffix majority; fallback to normalized header
            if not _looks_like_text(clean_prog_name):
                if course_suffix and suffix_name_counts.get(course_suffix):
                    clean_prog_name = suffix_name_counts[course_suffix].most_common(1)[0][0]
                else:
                    clean_prog_name = clean_norm_name
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
                br.get("source_id", "default"),
            ])
            # yearly_cutoffs.csv rows for all detected years
            for y, val in sorted((br.get("cutoffs") or {}).items()):
                if not val or val == "-":
                    continue
                try:
                    y_int = int(str(y))
                except Exception:
                    continue
                cutoffs_rows.append([clean_norm_name, "", corrected_code, y_int, val, "", "", br.get("source_id", "default")])
            # program_costs.csv rows (optional per row)
            cost_raw = (br.get("cost") or "").strip()
            if cost_raw:
                costs_rows.append([corrected_code, _strip_quotes(br["inst_name"]), clean_prog_name, cost_raw, br.get("source_id", "default")])
        block_rows = []

    # Collect files from all sources and learn prefix modes across all
    all_files: List[Tuple[str, Path]] = []
    for src_id, td in tables_dirs:
        for p in sorted(td.glob("*.csv")):
            all_files.append((src_id, p))
    # Prepass: learn modal 4-digit prefixes per institution across entire merged dataset
    for _src_id, fp in all_files:
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
    for src_id, fp in all_files:
        with open(fp, encoding="utf-8") as f:
            # Auto-detect delimiter: support TSV (extract) and legacy CSV
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
            except Exception:
                dialect = csv.excel_tab if "\t" in sample else csv.excel
            reader = csv.reader(f, dialect=dialect)
            adapter = _adapter_for(src_id, cfg.dataset_year)
            serial_idx = adapter.get("serial_idx", 0)
            prog_code_idx = adapter.get("prog_code_idx", 1)
            inst_name_idx = adapter.get("inst_name_idx", 2)
            prog_name_idx = adapter.get("prog_name_idx", 3)
            cost_idx = adapter.get("program_cost_idx", -1)
            cutoff_cols: Dict[str, int] = adapter.get("cutoff_year_columns", {"2023": 4, "2022": 5})
            subj_indices: List[int] = adapter.get("subject_columns", [6, 7, 8, 9])
            YEAR_HDR_RE = re.compile(r"^(?:(\d{4})\s*(?:CUTOFF)?)$", re.IGNORECASE)
            YEAR_TOKEN_RE = re.compile(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)")
            for row in reader:
                cells = [_clean(c) for c in row]
                if not any(cells):
                    continue
                if _is_header_row(cells):
                    # Auto-detect year columns from header cells if adapter isn't explicit
                    detected: Dict[str, int] = {}
                    for idx, cell in enumerate(cells):
                        m = YEAR_HDR_RE.match(cell.replace(" ", "")) if cell else None
                        if m:
                            detected[m.group(1)] = idx
                        else:
                            # Fallback: find any 4-digit year token anywhere in the header cell
                            toks = YEAR_TOKEN_RE.findall((cell or ""))
                            if toks:
                                # Prefer the first token as the column's year
                                detected.setdefault(toks[0], idx)
                    if detected:
                        cutoff_cols = detected
                    # Also auto-detect main columns by header names
                    ucells = [(cell or "").upper() for cell in cells]
                    def _find(*need: str) -> int:
                        for i, u in enumerate(ucells):
                            if all(n in u for n in need):
                                return i
                        return -1
                    p_idx = _find("PROG", "CODE")
                    if p_idx != -1:
                        prog_code_idx = p_idx
                    i_idx = _find("INSTITUTION", "NAME")
                    if i_idx != -1:
                        inst_name_idx = i_idx
                    n_idx = _find("PROGRAMME", "NAME")
                    if n_idx != -1:
                        prog_name_idx = n_idx
                    c_idx = _find("PROGRAMME", "COST")
                    if c_idx == -1:
                        c_idx = _find("PROGRAM", "COST")
                    if c_idx == -1:
                        c_idx = _find("COST")
                    if c_idx != -1:
                        cost_idx = c_idx
                    # Serial column commonly labeled '#' or 'NO'
                    s_idx = _find("#")
                    if s_idx == -1:
                        s_idx = _find("NO")
                    if s_idx != -1:
                        serial_idx = s_idx
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
                first = cells[serial_idx] if serial_idx < len(cells) else ""
                if not NUM_RE.match(first or ""):
                    # Fallback: accept row if program_code cell looks valid
                    cand_code = cells[prog_code_idx] if prog_code_idx < len(cells) else ""
                    if not _is_valid_code(cand_code):
                        continue
                # pad to max referenced index
                min_len = max([serial_idx, prog_code_idx, inst_name_idx, prog_name_idx, cost_idx] + list(cutoff_cols.values()) + subj_indices) + 1
                while len(cells) < min_len:
                    cells.append("")
                serial = cells[serial_idx]
                prog_code = cells[prog_code_idx]
                # Recover from mis-detected column indices (missing header rows)
                if not _is_valid_code(prog_code):
                    candidates: List[Tuple[int, str]] = []
                    scan_upto = min(len(cells), 10)
                    for i in range(scan_upto):
                        val = cells[i]
                        if _is_valid_code(val):
                            candidates.append((i, val))
                    if candidates:
                        # Prefer first candidate different from serial_idx
                        for i, val in candidates:
                            if i != serial_idx:
                                prog_code = val
                                break
                        else:
                            prog_code = candidates[0][1]
                inst_name = cells[inst_name_idx]
                prog_name = cells[prog_name_idx]
                # Gather all year cutoffs present
                cutoffs_map: Dict[str, str] = {}
                for y, idx in cutoff_cols.items():
                    if idx >= 0 and idx < len(cells):
                        cutoffs_map[y] = cells[idx]
                prog_cost = cells[cost_idx] if (cost_idx is not None and cost_idx >= 0 and cost_idx < len(cells)) else ""
                subj_cells = [cells[i] for i in subj_indices if i < len(cells)]
                req = _parse_requirements(subj_cells)
                req_json = json.dumps(req, ensure_ascii=False) if req else ""
                # Buffer row for block-level correction and emission
                block_rows.append({
                    "serial": serial,
                    "prog_code": prog_code,
                    "inst_name": inst_name,
                    "prog_name": prog_name,
                    "cutoffs": cutoffs_map,
                    "req_json": req_json,
                    "source_id": src_id,
                    "cost": prog_cost,
                })

    # finalize last block at EOF
    finalize_block()

    # Write outputs (overwrite each run, merged across sources)
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
    # Optional program costs
    if costs_rows:
        cost_out = cfg.processed_dir / "program_costs.csv"
        with open(cost_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["program_code", "institution_name", "program_name", "cost", "source_id"])
            writer.writerows(costs_rows)
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

def _ensure_suffix_map_overrides() -> Path:
    """Ensure mappings/course_suffix_map_overrides.csv exists with header.
    Admins can place curated overrides that bind a course_suffix to a
    canonical normalized_name and/or field_name. Empty cells are ignored.
    """
    path = MAPPINGS_DIR / "course_suffix_map_overrides.csv"
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["course_suffix", "normalized_name", "field_name"])  # optional overrides per suffix
    return path


def _load_suffix_overrides() -> Dict[str, Dict[str, str]]:
    """Load optional course suffix overrides from mappings/ CSV.
    Returns { suffix: {"normalized_name": str, "field_name": str} }.
    """
    p = _ensure_suffix_map_overrides()
    out: Dict[str, Dict[str, str]] = {}
    try:
        with open(p, encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                sfx = (row.get("course_suffix") or "").strip()
                if not sfx:
                    continue
                out[sfx] = {
                    "normalized_name": (row.get("normalized_name") or "").strip(),
                    "field_name": (row.get("field_name") or "").strip(),
                }
    except Exception:
        pass
    return out


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


def _is_valid_prog_code(code: str) -> bool:
    """Heuristic validity for KUCCPS program codes.
    Accepts numeric codes with >=7 digits, or alphanumeric ddddLdd(+).
    """
    c = (code or "").strip().upper()
    if not c:
        return False
    digits = "".join(ch for ch in c if ch.isdigit())
    if len(digits) >= 7:
        return True
    return bool(re.fullmatch(r"\d{4}[A-Z]\d{2,3}", c))


def _inst_code_from_meta(name: str, inst_meta: List[Dict[str, str]]) -> str:
    """Try resolve institution_code by substring matching against institutions_meta.csv."""
    nm = (name or "").upper().strip()
    if not nm:
        return ""
    # Prefer exact 'name' matches with institution_code when present
    for m in inst_meta:
        if (m.get("name") or "").upper().strip() == nm and (m.get("institution_code") or "").strip():
            return (m.get("institution_code") or "").strip()
    # Fallback: name_contains substring rule
    for m in inst_meta:
        needle = (m.get("name_contains") or "").upper().strip()
        code = (m.get("institution_code") or "").strip()
        if needle and code and needle in nm:
            return code
    return ""

def _is_code_like_name(text: Optional[str]) -> bool:
    """Heuristic: treat a program name cell as code-like (invalid) when it
    matches program code patterns or contains no alphabetic characters.
    Examples: "1063B76", "1246155", "1112233".
    """
    s = (text or "").strip()
    if not s:
        return True
    u = s.upper()
    # Program code patterns (numeric 7+ digits OR ddddLdd(+))
    if _is_valid_prog_code(u):
        return True
    # No alphabetic characters -> not a descriptive name
    if not any(ch.isalpha() for ch in u):
        return True
    return False

def transform_normalize(cfg: Config) -> None:
    """Normalize/derive missing fields in processed/programs.csv and emit institutions.csv.

    - Fill institution_code from program code prefix
    - Classify field_name using mappings/mappings/fields_map.csv
    - Join county/region using campus-level overrides from mappings/institutions_campuses.csv
      (when campus is set and matches a non-main campus); fall back to mappings/institutions_geo.csv
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
    load_errors: List[List[str]] = []  # [reason, institution_name, program_name, program_code, source_id, extra]

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
        # Validate program code
        if not _is_valid_prog_code(prog_code):
            load_errors.append([
                "invalid_program_code",
                (r.get("institution_name") or "").strip(),
                (r.get("name") or r.get("normalized_name") or "").strip(),
                prog_code,
                (r.get("source_id") or "").strip(),
                (r.get("source_index") or "").strip(),
            ])
            continue
        inst_code = _derive_institution_code(prog_code)
        if not inst_code:
            # Try resolve from institutions_meta name mappings
            guess = _inst_code_from_meta((r.get("institution_name") or "").strip(), inst_meta)
            if guess:
                inst_code = guess
            else:
                load_errors.append([
                    "missing_institution_code",
                    (r.get("institution_name") or "").strip(),
                    (r.get("name") or r.get("normalized_name") or "").strip(),
                    prog_code,
                    (r.get("source_id") or "").strip(),
                    (r.get("source_index") or "").strip(),
                ])
                continue
        r["institution_code"] = inst_code
        # ensure course_suffix present
        if not (r.get("course_suffix") or "").strip():
            digits = "".join(ch for ch in prog_code if ch.isdigit())
            if digits:
                r["course_suffix"] = digits[-3:] if len(digits) >= 3 else ""

        # Validate program name columns and enforce Bachelor naming conventions.
        level_val = (r.get("level") or "").strip().lower() or "bachelor"
        name_val = (r.get("name") or "").strip()
        norm_val = (r.get("normalized_name") or "").strip()
        # Drop if both name and normalized_name are code-like (non-descriptive)
        if _is_code_like_name(name_val) and _is_code_like_name(norm_val):
            load_errors.append([
                "invalid_name_code",
                (r.get("institution_name") or "").strip(),
                (r.get("name") or r.get("normalized_name") or "").strip(),
                prog_code,
                (r.get("source_id") or "").strip(),
                (r.get("source_index") or "").strip(),
            ])
            continue
        # Enforce: bachelor programs must start with 'BACHELOR OF '
        label = (norm_val or name_val)
        if level_val == "bachelor" and label and not label.upper().startswith("BACHELOR OF"):
            load_errors.append([
                "invalid_bachelor_name",
                (r.get("institution_name") or "").strip(),
                label,
                prog_code,
                (r.get("source_id") or "").strip(),
                (r.get("source_index") or "").strip(),
            ])
            continue

        # field classification
        if not (r.get("field_name") or "").strip():
            base_name = (r.get("normalized_name") or r.get("name") or "").strip()
            if base_name:
                fld = classify_field(base_name)
                r["field_name"] = fld
                if not fld:
                    unclassified.append([base_name, prog_code, "no_field_match"])

        # county/region join with campus-level override (non-main campuses) then fallback to institution-level
        if inst_code:
            campus = (r.get("campus") or "").strip().upper()
            campus_geo = None
            if campus and inst_code in inst_campuses:
                campus_geo = inst_campuses[inst_code].get(campus)
            # County
            if not (r.get("county") or "").strip():
                if campus_geo and (campus_geo.get("county") or "").strip():
                    r["county"] = campus_geo.get("county", "")
                else:
                    code_geo = inst_geo.get(inst_code)
                    if code_geo:
                        r["county"] = code_geo.get("county", "")
            # Region
            if not (r.get("region") or "").strip():
                if campus_geo and (campus_geo.get("region") or "").strip():
                    r["region"] = campus_geo.get("region", "")
                else:
                    code_geo = inst_geo.get(inst_code)
                    if code_geo:
                        r["region"] = code_geo.get("region", "")

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

    # Build lookups for backfilling names
    inst_name_lookup: Dict[str, str] = {}
    # Prefer explicit meta mapping by code->name
    for m in inst_meta:
        code = (m.get("institution_code") or "").strip()
        name = (m.get("name") or "").strip()
        if code and name:
            inst_name_lookup.setdefault(code, name)
    # Learn from rows we already saw
    for code, meta in inst_map.items():
        if (meta.get("name") or "").strip():
            inst_name_lookup.setdefault(code, meta["name"])

    # Majority program name by course_suffix (prefer normalized_name)
    suffix_name_majority: Dict[str, str] = {}
    tmp_counts: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        suf = (r.get("course_suffix") or "").strip()
        if not suf:
            continue
        cand = _strip_quotes((r.get("normalized_name") or r.get("name") or "").strip())
        if _looks_like_text(cand):
            tmp_counts[suf][cand] += 1
    for suf, cnt in tmp_counts.items():
        if cnt:
            suffix_name_majority[suf] = cnt.most_common(1)[0][0]

    # Second pass: backfill institution_name and repair numeric program names
    for r in rows:
        code = (r.get("institution_code") or "").strip()
        if not _looks_like_text(r.get("institution_name")) and code and inst_name_lookup.get(code):
            r["institution_name"] = inst_name_lookup[code]
        nm = (r.get("name") or "").strip()
        if not _looks_like_text(nm):
            suf = (r.get("course_suffix") or "").strip()
            fallback = suffix_name_majority.get(suf) or (r.get("normalized_name") or "").strip()
            if fallback:
                r["name"] = fallback
        # Ensure normalized_name is textual too
        if not _looks_like_text(r.get("normalized_name")):
            r["normalized_name"] = (r.get("name") or r.get("normalized_name") or "").strip()

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
                r.get("source_id", "default"),
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

    # Write load errors CSV for QA
    if load_errors:
        with open(cfg.processed_dir / "_load_errors.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["reason", "institution_name", "program_name", "program_code", "source_id", "source_index"])
            w.writerows(load_errors)

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


def load_csvs(cfg: Config, dry_run: bool = False) -> Dict[str, Any]:
    """Load processed CSVs into catalog models with change-tracking stats.

    Returns a dict with:
    - totals: counters for created/updated/unchanged across Institutions, Subjects, Programs, YearlyCutoffs, Program costs
    - by_source: per-source counters for Programs, YearlyCutoffs, and Program costs (using source_id from processed files)
    Expected files are created by transform(). Users can replace with curated data before load().
    """
    setup_django()
    from django.db import transaction
    from django.db.models import Q
    from catalog.models import (
        Institution,
        Field,
        Subject,
        Program,
        YearlyCutoff,
        NormalizationRule,
        ProgramLevel,
        InstitutionCampus,
        ProgramOfferingAggregate,
        ProgramOfferingBroadAggregate,
        DedupCandidateGroup,
        DedupSummary,
        CodeCorrectionAudit,
        ETLRun,
        DQReportEntry,
        ClusterSubjectRule,
        ProgramRequirementNormalized,
        ProgramRequirementGroup,
        ProgramRequirementOption,
    )

    @transaction.atomic
    def _load() -> Dict[str, Any]:
        changes: Dict[str, int] = defaultdict(int)
        by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        def inc(key: str, sid: Optional[str] = None) -> None:
            changes[key] += 1
            if sid:
                by_source[sid][key] += 1
        # Institutions
        inst_path = cfg.processed_dir / "institutions.csv"
        if inst_path.exists():
            with open(inst_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code_val = row.get("code", "").strip()
                    if not code_val:
                        continue
                    new_vals = {
                        "name": row.get("name", "").strip(),
                        "alias": row.get("alias", "").strip(),
                        "region": row.get("region", "").strip(),
                        "county": row.get("county", "").strip(),
                        "website": row.get("website", "").strip(),
                    }
                    inst = Institution.objects.filter(code=code_val).first()
                    if not inst:
                        Institution.objects.create(code=code_val, **new_vals)
                        inc("institutions_created")
                    else:
                        changed = False
                        for k, v in new_vals.items():
                            if getattr(inst, k) != v:
                                setattr(inst, k, v)
                                changed = True
                        if changed:
                            inst.save(update_fields=list(new_vals.keys()))
                            inc("institutions_updated")
                        else:
                            inc("institutions_unchanged")
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
        # Subjects: preload canonical first (mappings/subjects_canonical.csv), then processed/subjects.csv
        canon_subj = MAPPINGS_DIR / "subjects_canonical.csv"
        def _load_subjects_file(fp: Path) -> None:
            with open(fp, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = (row.get("code") or "").strip()
                    name = (row.get("name") or "").strip()
                    group = (row.get("group") or "").strip()
                    alt_codes = [v.strip() for v in ((row.get("alt_codes") or "").split("|") if row.get("alt_codes") else [])]
                    if not code and not name:
                        continue
                    obj, created = Subject.objects.get_or_create(code=code, defaults={"name": name or code, "group": group, "alt_codes": alt_codes})
                    if created:
                        inc("subjects_created")
                    else:
                        changed = False
                        if name and obj.name != name:
                            obj.name = name; changed = True
                        if group and obj.group != group:
                            obj.group = group; changed = True
                        if alt_codes and obj.alt_codes != alt_codes:
                            obj.alt_codes = alt_codes; changed = True
                        if changed:
                            obj.save(update_fields=["name", "group", "alt_codes"])
                            inc("subjects_updated")
                        else:
                            inc("subjects_unchanged")
        if canon_subj.exists():
            _load_subjects_file(canon_subj)
        subj_path = cfg.processed_dir / "subjects.csv"
        if subj_path.exists():
            _load_subjects_file(subj_path)
        # Programs: prefer deduplicated file if present (ensures master selection is honored)
        prog_path = cfg.processed_dir / "programs_deduped.csv"
        if not prog_path.exists():
            prog_path = cfg.processed_dir / "programs.csv"
        if prog_path.exists():
            with open(prog_path, encoding="utf-8") as f:
                # Loader notes:
                # - programs.csv is emitted as TSV by transform_normalize to avoid heavy quoting in JSON columns
                # - csv.Sniffer can mis-detect comma when subject_requirements_json has many commas
                # - Prefer explicit tab delimiter when the header contains tabs; otherwise fall back to Sniffer, then TSV
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
                for row in reader:
                    inst_code = row.get("institution_code", "").strip()
                    field_name = row.get("field_name", "").strip()
                    # Map institution_code -> Institution FK; skip unsafe rows instead of violating NOT NULL
                    if not inst_code:
                        logger.warning("Program row skipped (missing institution_code): %s", row)
                        continue
                    inst = Institution.objects.filter(code=inst_code).first()
                    if not inst:
                        logger.warning("Program row skipped (institution not found for code=%s)", inst_code)
                        continue
                    fld = Field.objects.filter(name=field_name).first() if field_name else None
                    level = row.get("level", "").strip().lower() or ProgramLevel.BACHELOR
                    normalized_name = (row.get("normalized_name") or row.get("name") or "").strip()
                    if not normalized_name:
                        logger.warning("Program row skipped (missing normalized/name): %s", row)
                        continue
                    sid = (row.get("source_id") or "").strip() or "default"
                    # Diff-aware upsert for Program
                    def _as_int(v):
                        try:
                            s = str(v).strip()
                            return int(s) if s else None
                        except Exception:
                            return None
                    key_kwargs = dict(
                        institution=inst,
                        normalized_name=normalized_name,
                        level=level,
                        campus=(row.get("campus") or "").strip(),
                    )
                    new_vals = {
                        "field": fld,
                        "code": (row.get("program_code") or row.get("code") or "").strip(),
                        "name": (row.get("name") or "").strip(),
                        "region": (row.get("region") or "").strip(),
                        "duration_years": _as_int(row.get("duration_years")),
                        "award": (row.get("award") or "").strip(),
                        "mode": (row.get("mode") or "").strip(),
                        "subject_requirements": _safe_json(row.get("subject_requirements_json") or "{}"),
                    }
                    prog = Program.objects.filter(**key_kwargs).first()
                    if not prog:
                        Program.objects.create(**key_kwargs, **new_vals)
                        inc("programs_created", sid)
                    else:
                        changed = False
                        for fname, fval in new_vals.items():
                            current = getattr(prog, fname)
                            if fname == "subject_requirements":
                                if (current or {}) != (fval or {}):
                                    setattr(prog, fname, fval)
                                    changed = True
                            else:
                                if current != fval:
                                    setattr(prog, fname, fval)
                                    changed = True
                        if changed:
                            prog.save()
                            inc("programs_updated", sid)
                        else:
                            inc("programs_unchanged", sid)
        # Program costs
        cost_path = cfg.processed_dir / "program_costs.csv"
        if cost_path.exists():
            with open(cost_path, encoding="utf-8") as f:
                # TSV-aware
                header = f.readline()
                rest = f.read(4096)
                f.seek(0)
                try:
                    if "\t" in header:
                        reader = csv.DictReader(f, delimiter="\t")
                    else:
                        sample = header + rest
                        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                        reader = csv.DictReader(f, dialect=dialect)
                except Exception:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
                latest: Dict[str, Dict[str, str]] = {}
                for r in rows:
                    code = (r.get("program_code") or "").strip()
                    if code:
                        latest[code] = r
                for code, row in latest.items():
                    prog = Program.objects.filter(code=code).first()
                    if not prog:
                        continue
                    raw_cost = (row.get("cost") or "").strip()
                    sid = (row.get("source_id") or "").strip() or "default"
                    # Parse numeric cost if possible (keep raw too)
                    digits = re.sub(r"[^0-9.]+", "", raw_cost)
                    num = None
                    if digits:
                        try:
                            num = float(digits)
                        except Exception:
                            num = None
                    meta = prog.metadata or {}
                    prev_num = meta.get("cost") if isinstance(meta, dict) else None
                    prev_raw = meta.get("cost_raw") if isinstance(meta, dict) else None
                    new_meta = {
                        "cost": num,
                        "cost_raw": raw_cost,
                        "cost_source_id": (row.get("source_id") or "").strip(),
                    }
                    if prev_num == new_meta["cost"] and str(prev_raw or "").strip() == new_meta["cost_raw"]:
                        inc("program_costs_unchanged", sid)
                        continue
                    prog.metadata = {**(meta or {}), **new_meta}
                    prog.save(update_fields=["metadata"])
                    if prev_num is None and (prev_raw is None or str(prev_raw).strip() == ""):
                        inc("program_costs_created", sid)
                    else:
                        inc("program_costs_updated", sid)
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
                    try:
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
                        sid = (row.get("source_id") or "").strip() or "default"
                        # Parse year
                        try:
                            year = int(str(row.get("year") or "").strip())
                        except Exception:
                            logger.warning("Cutoff row skipped (invalid year): %s", row)
                            continue
                        # Parse cutoff as Decimal, cleaning stray characters
                        cutoff_raw = str(row.get("cutoff") or "").strip()
                        cutoff_clean = cutoff_raw.replace(",", "")
                        try:
                            cutoff_val = Decimal(cutoff_clean)
                        except InvalidOperation:
                            logger.warning("Cutoff row skipped (invalid cutoff '%s'): %s", cutoff_raw, row)
                            continue
                        # Capacity (optional)
                        try:
                            cap_val = int(str(row.get("capacity") or "").strip()) if row.get("capacity") else None
                        except Exception:
                            cap_val = None
                        notes_val = (row.get("notes") or "").strip()
                        yc = YearlyCutoff.objects.filter(program=prog, year=year).first()
                        if not yc:
                            YearlyCutoff.objects.create(program=prog, year=year, cutoff=cutoff_val, capacity=cap_val, notes=notes_val)
                            inc("yearly_cutoffs_created", sid)
                        else:
                            changed = False
                            if yc.cutoff != cutoff_val:
                                yc.cutoff = cutoff_val; changed = True
                            if yc.capacity != cap_val:
                                yc.capacity = cap_val; changed = True
                            if (yc.notes or "") != notes_val:
                                yc.notes = notes_val; changed = True
                            if changed:
                                yc.save(update_fields=["cutoff", "capacity", "notes"])
                                inc("yearly_cutoffs_updated", sid)
                            else:
                                inc("yearly_cutoffs_unchanged", sid)
                    except Exception as e:
                        logger.warning("Cutoff row skipped (exception: %s): %s", e, row)
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

        # Institution campuses (from mappings/institutions_campuses.csv)
        try:
            campuses_path = MAPPINGS_DIR / "institutions_campuses.csv"
            if campuses_path.exists():
                with open(campuses_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        code = (row.get("institution_code") or "").strip()
                        campus = (row.get("campus") or "").strip()
                        if not code or not campus:
                            continue
                        inst = Institution.objects.filter(code=code).first()
                        if not inst:
                            continue
                        InstitutionCampus.objects.update_or_create(
                            institution=inst,
                            campus=campus,
                            defaults={
                                "town": (row.get("town") or "").strip(),
                                "county": (row.get("county") or "").strip(),
                                "region": (row.get("region") or "").strip(),
                            },
                        )
        except Exception:
            logger.exception("Failed loading InstitutionCampus")

        # Program offerings aggregates (processed/program_offerings.csv)
        po_path = cfg.processed_dir / "program_offerings.csv"
        if po_path.exists():
            with open(po_path, encoding="utf-8") as f:
                # TSV-aware
                header = f.readline()
                rest = f.read(4096)
                f.seek(0)
                try:
                    if "\t" in header:
                        reader = csv.DictReader(f, delimiter="\t")
                    else:
                        sample = header + rest
                        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                        reader = csv.DictReader(f, dialect=dialect)
                except Exception:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    ProgramOfferingAggregate.objects.update_or_create(
                        program_normalized_name=(row.get("program_normalized_name") or "").strip(),
                        course_suffix=(row.get("course_suffix") or "").strip(),
                        defaults={
                            "offerings_count": int(row.get("offerings_count") or 0),
                        },
                    )

        # Program offerings broad aggregates (processed/program_offerings_broad.csv)
        pob_path = cfg.processed_dir / "program_offerings_broad.csv"
        if pob_path.exists():
            with open(pob_path, encoding="utf-8") as f:
                # TSV-aware
                header = f.readline()
                rest = f.read(4096)
                f.seek(0)
                try:
                    if "\t" in header:
                        reader = csv.DictReader(f, delimiter="\t")
                    else:
                        sample = header + rest
                        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                        reader = csv.DictReader(f, dialect=dialect)
                except Exception:
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    ProgramOfferingBroadAggregate.objects.update_or_create(
                        program_normalized_name=(row.get("program_normalized_name") or "").strip(),
                        defaults={
                            "offerings_count": int(row.get("offerings_count") or 0),
                        },
                    )

        # Dedup reports (processed/dedup_candidates.csv and dedup_summary.csv)
        cand_path = cfg.processed_dir / "dedup_candidates.csv"
        if cand_path.exists():
            with open(cand_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = (row.get("institution_code") or "").strip()
                    inst = Institution.objects.filter(code=code).first() if code else None
                    program_codes = [c for c in (row.get("program_codes") or "").split("|") if c]
                    name_variants = [n for n in (row.get("name_variants") or "").split("|") if n]
                    DedupCandidateGroup.objects.update_or_create(
                        institution=inst,
                        normalized_name=(row.get("normalized_name") or "").strip(),
                        level=((row.get("level") or "").strip().lower() or ProgramLevel.BACHELOR),
                        campus=(row.get("campus") or "").strip(),
                        defaults={
                            "institution_code": code,
                            "institution_name": (row.get("institution_name") or "").strip(),
                            "rows_count": int(row.get("rows_count") or 0),
                            "program_codes": program_codes,
                            "name_variants": name_variants,
                            "suggested_master_program_code": (row.get("suggested_master_program_code") or "").strip(),
                        },
                    )
        summ_path = cfg.processed_dir / "dedup_summary.csv"
        if summ_path.exists():
            with open(summ_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = (row.get("institution_code") or "").strip()
                    inst = Institution.objects.filter(code=code).first() if code else None
                    DedupSummary.objects.update_or_create(
                        institution=inst,
                        defaults={
                            "institution_code": code,
                            "institution_name": (row.get("institution_name") or "").strip(),
                            "duplicate_groups": int(row.get("duplicate_groups") or 0),
                            "duplicate_rows": int(row.get("duplicate_rows") or 0),
                        },
                    )

        # Code corrections audit (processed/_code_corrections.csv)
        corr_path = cfg.processed_dir / "_code_corrections.csv"
        if corr_path.exists():
            with open(corr_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    before = (row.get("original_code") or "").strip()
                    after = (row.get("corrected_code") or "").strip()
                    reason = (row.get("reason") or "").strip()
                    inst_code = _derive_institution_code(before) if before else ""
                    CodeCorrectionAudit.objects.update_or_create(
                        program_code_before=before,
                        program_code_after=after,
                        correction_type=reason,
                        defaults={
                            "institution_code": inst_code,
                            "group_key": (row.get("program_header") or "").strip(),
                            "reason": reason,
                            "metadata": {"source_index": (row.get("source_index") or "").strip(), "institution_name": (row.get("institution_name") or "").strip()},
                        },
                    )

        # DQ report (processed/dq_report.csv)
        dq_path = cfg.processed_dir / "dq_report.csv"
        if dq_path.exists():
            metrics = {}
            with open(dq_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for r in rows:
                    metrics[r.get("metric", "")] = r.get("value")
            run = ETLRun.objects.create(action="dq-report", stats=metrics)
            for r in rows:
                DQReportEntry.objects.create(
                    run=run,
                    metric_name=(r.get("metric") or "").strip(),
                    value=(r.get("value") or "").strip(),
                    scope=(r.get("scope") or "").strip(),
                    extra={k: v for k, v in r.items() if k not in {"metric", "value", "scope"}},
                )

        # Cluster subject rules (mappings/cluster_subjects.csv)
        try:
            csr_path = MAPPINGS_DIR / "cluster_subjects.csv"
            if csr_path.exists():
                with open(csr_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        prog_pat = (row.get("program_pattern") or "").strip()
                        if not prog_pat:
                            continue
                        ClusterSubjectRule.objects.update_or_create(
                            program_pattern=prog_pat,
                            defaults={"subjects_grammar": (row.get("subjects") or "").strip()},
                        )
        except Exception:
            logger.exception("Failed loading ClusterSubjectRule")

        # Optional: Normalize program requirements into JSON helper + fully relational groups/options
        for prog in Program.objects.all().only("id", "subject_requirements"):
            try:
                req = prog.subject_requirements or {}
                # Store JSON snapshot
                ProgramRequirementNormalized.objects.update_or_create(
                    program=prog,
                    defaults={
                        "required": req.get("required", []),
                        "groups": req.get("groups", []),
                        "notes": req.get("notes", ""),
                    },
                )
                # Rebuild relational groups/options
                ProgramRequirementGroup.objects.filter(program=prog).delete()

                # Helper: subject resolution or creation
                def _resolve_subject(subj_code: str, subj_name: str):
                    code = (subj_code or "").strip().upper()
                    name = re.sub(r"\s+", " ", (subj_name or "").strip()).title()
                    if not code and not name:
                        return None
                    s = None
                    if code:
                        s = Subject.objects.filter(Q(code__iexact=code) | Q(alt_codes__contains=[code])).first()
                    if not s and name:
                        s = Subject.objects.filter(name__iexact=name).first()
                    if not s and code:
                        s, _ = Subject.objects.get_or_create(code=code, defaults={"name": name or code, "group": ""})
                    return s

                group_order = 1
                # Required list: each becomes its own group with pick=1
                for item in req.get("required", []):
                    grp = ProgramRequirementGroup.objects.create(
                        program=prog,
                        name=f"Subj {group_order}",
                        pick=1,
                        order=group_order,
                        metadata={"type": "required"},
                    )
                    sub_code = (item.get("code") or item.get("subject_code") or "").strip()
                    sub_name = (item.get("subject") or "").strip()
                    subj = _resolve_subject(sub_code, sub_name)
                    ProgramRequirementOption.objects.create(
                        group=grp,
                        subject=subj,
                        subject_code=sub_code or sub_name,
                        min_grade=(item.get("min_grade") or "").strip(),
                        order=1,
                    )
                    group_order += 1

                # Groups with alternatives (pick indicates either/or count)
                for g in req.get("groups", []):
                    pick = int(g.get("pick") or 1)
                    grp = ProgramRequirementGroup.objects.create(
                        program=prog,
                        name=f"Subj {group_order}",
                        pick=pick,
                        order=group_order,
                        metadata={"type": "group"},
                    )
                    opt_order = 1
                    for opt in g.get("options", []):
                        sub_code = (opt.get("code") or opt.get("subject_code") or "").strip()
                        sub_name = (opt.get("subject") or "").strip()
                        subj = _resolve_subject(sub_code, sub_name)
                        ProgramRequirementOption.objects.create(
                            group=grp,
                            subject=subj,
                            subject_code=sub_code or sub_name,
                            min_grade=(opt.get("min_grade") or "").strip(),
                            order=opt_order,
                        )
                        opt_order += 1
                    group_order += 1
            except Exception:
                # Skip malformed JSON rows gracefully
                continue

        # Aggregate change-tracking results
        return {
            "totals": dict(changes),
            "by_source": {k: dict(v) for k, v in by_source.items()},
        }

    if dry_run:
        logger.info("load: DRY RUN - skipping DB writes")
        return {"totals": {}, "by_source": {}}
    stats = _load()
    logger.info("load: finished writing to database")
    return stats


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

    # Prefilter invalid rows based on name/normalized_name rules
    def _invalid_row_local(rr: Dict[str, Any]) -> bool:
        lvl = (rr.get("level") or "").strip().lower() or "bachelor"
        nm = (rr.get("name") or "").strip()
        nn = (rr.get("normalized_name") or "").strip()
        if _is_code_like_name(nm) and _is_code_like_name(nn):
            return True
        label = (nn or nm)
        if lvl == "bachelor" and label and not label.upper().startswith("BACHELOR OF"):
            return True
        return False
    rows = [r for r in rows if not _invalid_row_local(r)]

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

    # First, enforce uniqueness by program_code when present
    coded_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    fallback_rows: List[Dict[str, Any]] = []
    for r in rows:
        # skip empty rows
        code_val = (r.get("program_code") or r.get("code") or "").strip()
        name_val = (r.get("institution_name") or "").strip()
        if not any([code_val, name_val]):
            continue
        if code_val:
            coded_groups[code_val].append(r)
        else:
            fallback_rows.append(r)

    def _digits_only(s: str) -> str:
        return "".join(ch for ch in (s or "") if ch.isdigit())

    def _merge_rows(code: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Prefer values from the item whose institution_code matches the code prefix
        pref = _digits_only(code)[:4]
        preferred_item = None
        for it in items:
            if pref and (it.get("institution_code") or "").strip() == pref:
                preferred_item = it; break
        base = dict(preferred_item or items[0])
        # Ensure program_code and course_suffix are consistent
        base["program_code"] = code
        if not (base.get("course_suffix") or "").strip():
            d = _digits_only(code)
            base["course_suffix"] = d[-3:] if len(d) >= 3 else (base.get("course_suffix") or "")
        # Fields to merge by preferring first non-empty
        fields = [
            "source_index","institution_code","institution_name","field_name","program_code","course_suffix",
            "name","normalized_name","level","campus","region","duration_years","award","mode",
            "subject_requirements_json","source_id",
        ]
        for f in fields:
            if (base.get(f) or "").strip():
                continue
            for it in items:
                val = (it.get(f) or "").strip()
                if val:
                    base[f] = val
                    break
        # Majority choice for key text fields
        def _majority(field: str) -> str:
            c = Counter()
            for it in items:
                v = _strip_quotes(it.get(field, ""))
                if _looks_like_text(v):
                    c[v] += 1
            return c.most_common(1)[0][0] if c else (base.get(field) or "")
        for tf in ("institution_name", "name", "normalized_name"):
            if not _looks_like_text(base.get(tf)):
                maj = _majority(tf)
                if maj:
                    base[tf] = maj
        return base

    masters: List[Dict[str, Any]] = []
    suppressed: List[List[str]] = []
    # Merge duplicates per program_code
    for code, items in coded_groups.items():
        merged = _merge_rows(code, items)
        masters.append(merged)
        if len(items) > 1:
            for it in items:
                if it is merged:
                    continue
                suppressed.append([
                    (it.get("institution_code") or "").strip(),
                    _strip_quotes((it.get("institution_name") or "").strip()),
                    _strip_quotes((it.get("normalized_name") or it.get("name") or "").strip()),
                    (it.get("level") or "").strip().lower() or "bachelor",
                    (it.get("campus") or "").strip(),
                    (it.get("program_code") or it.get("code") or "").strip(),
                    code,
                ])

    # Fallback dedup for rows without program_code: legacy grouping
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in fallback_rows:
        groups[_key(r)].append(r)
    for k, items in groups.items():
        if not items:
            continue
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

    # Build helper maps for final cleanup
    inst_name_by_code: Dict[str, str] = {}
    try:
        inst_fp = cfg.processed_dir / "institutions.csv"
        if inst_fp.exists():
            with open(inst_fp, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    c = (r.get("code") or "").strip(); n = (r.get("name") or "").strip()
                    if c and n:
                        inst_name_by_code[c] = n
    except Exception:
        pass
    # Majority normalized name per course_suffix
    name_by_suffix: Dict[str, str] = {}
    tmp_cnt: Dict[str, Counter] = defaultdict(Counter)
    for m in masters:
        suf = (m.get("course_suffix") or "").strip()
        nm = _strip_quotes((m.get("normalized_name") or m.get("name") or "").strip())
        if suf and _looks_like_text(nm):
            tmp_cnt[suf][nm] += 1
    for suf, cnt in tmp_cnt.items():
        if cnt:
            name_by_suffix[suf] = cnt.most_common(1)[0][0]

    # Canonical mapping from course_suffix -> (normalized_name, field_name) from input rows
    canon_name_by_suffix: Dict[str, str] = {}
    canon_field_by_suffix: Dict[str, str] = {}
    _n_cnt: Dict[str, Counter] = defaultdict(Counter)
    _f_cnt: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        sfx = (r.get("course_suffix") or "").strip()
        if not sfx:
            continue
        rn = _strip_quotes((r.get("normalized_name") or r.get("name") or "").strip())
        if rn and rn.upper().startswith("BACHELOR OF"):
            _n_cnt[sfx][rn] += 1
        rf = (r.get("field_name") or "").strip()
        if rf:
            _f_cnt[sfx][rf] += 1
    for sfx, cnt in _n_cnt.items():
        if cnt:
            canon_name_by_suffix[sfx] = cnt.most_common(1)[0][0]
    for sfx, cnt in _f_cnt.items():
        if cnt:
            canon_field_by_suffix[sfx] = cnt.most_common(1)[0][0]

    # Apply admin overrides (if any)
    try:
        _ov = _load_suffix_overrides()
        if _ov:
            for sfx, vals in _ov.items():
                if (vals.get("normalized_name") or "").strip():
                    canon_name_by_suffix[sfx] = vals["normalized_name"].strip()
                if (vals.get("field_name") or "").strip():
                    canon_field_by_suffix[sfx] = vals["field_name"].strip()
    except Exception:
        pass

    # Persist final canonical mapping for visibility and downstream use
    try:
        map_out = cfg.processed_dir / "course_suffix_map.csv"
        with open(map_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["course_suffix", "normalized_name", "field_name"])  # canonical values after overrides
            keys = sorted(set(list(canon_name_by_suffix.keys()) + list(canon_field_by_suffix.keys())))
            for k in keys:
                w.writerow([k, canon_name_by_suffix.get(k, ""), canon_field_by_suffix.get(k, "")])
    except Exception:
        pass

    # Write outputs
    out_path = cfg.processed_dir / "programs_deduped.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(EXPECTED["programs.csv"])  # same schema
        for m in masters:
            # Repairs: institution_name, course_suffix, name/normalized_name
            inst_code = (m.get("institution_code") or "").strip()
            inst_name = _strip_quotes(m.get("institution_name", ""))
            if (not _looks_like_text(inst_name)) and inst_code and inst_code in inst_name_by_code:
                inst_name = inst_name_by_code[inst_code]
            pc = (m.get("program_code") or m.get("code") or "").strip()
            digits = "".join(ch for ch in pc if ch.isdigit())
            course_suffix = (m.get("course_suffix") or "").strip()
            if not course_suffix and digits:
                course_suffix = digits[-3:] if len(digits) >= 3 else ""
            name_val = _strip_quotes(m.get("name", ""))
            norm_val = _strip_quotes(m.get("normalized_name", ""))
            if not _looks_like_text(name_val):
                if course_suffix and name_by_suffix.get(course_suffix):
                    name_val = name_by_suffix[course_suffix]
                elif _looks_like_text(norm_val):
                    name_val = norm_val
            if not _looks_like_text(norm_val):
                norm_val = name_val or norm_val
            # Enforce canonical suffix mapping where available
            if course_suffix and canon_name_by_suffix.get(course_suffix):
                canon_nm = canon_name_by_suffix[course_suffix]
                name_val = canon_nm
                norm_val = canon_nm
            field_val = m.get("field_name", "")
            if course_suffix and canon_field_by_suffix.get(course_suffix):
                field_val = canon_field_by_suffix[course_suffix]
            w.writerow([
                m.get("source_index", ""),
                m.get("institution_code", ""),
                inst_name,
                field_val,
                m.get("program_code", m.get("code", "")),
                course_suffix,
                name_val,
                norm_val,
                m.get("level", ""),
                m.get("campus", ""),
                m.get("region", ""),
                m.get("duration_years", ""),
                m.get("award", ""),
                m.get("mode", ""),
                m.get("subject_requirements_json", ""),
                m.get("source_id", ""),
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
