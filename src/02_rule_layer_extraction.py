"""Extract rule-based entities and relations from the ingested guideline files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow the script to find shared helpers.
_here = Path(__file__).resolve().parent
if (_here / "_helpers.py").exists():
    sys.path.insert(0, str(_here))
elif (_here.parent / "src" / "_helpers.py").exists():
    sys.path.insert(0, str(_here.parent / "src"))

from _helpers import (
    ensure_dirs, INTERIM_DIR, OUTPUT_DIR,
    read_jsonl, write_jsonl, short, stable_id,
    mgdl_to_mmolL, normalize_comparator,
)

# CLI
def parse_args():
    ap = argparse.ArgumentParser(description="Rule-based KG entity/relation extractor v3.0")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument(
        "--docs", nargs="*", default=None,
        help="Limit to specific doc_keys e.g. --docs ada_26 nice_t1"
    )
    return ap.parse_args()


# Guideline metadata used when creating graph nodes.
GUIDELINE_META: Dict[str, Dict[str, Any]] = {
    "ADA": {
        "display_name": "ADA Standards of Care in Diabetes 2026",
        "geography": "US",
        "year": 2026,
        "glucose_unit": "mg/dL",   # ADA primary unit
    },
    "NICE": {
        "display_name": "NICE Type 1 & Type 2 Diabetes Guidelines",
        "geography": "UK",
        "year": 2024,
        "glucose_unit": "mmol/L",
    },
    "IDF": {
        "display_name": "IDF Global Clinical Practice Recommendations 2025",
        "geography": "Global",
        "year": 2025,
        "glucose_unit": "mmol/L",
    },
    "Diabetes Canada": {
        "display_name": "Diabetes Canada Clinical Practice Guidelines 2018",
        "geography": "Canada",
        "year": 2018,
        "glucose_unit": "mmol/L",
    },
}

# In-memory graph stores.
def make_stores():
    return {
        "entities": {},   # (etype, org_scope, key_tuple) -> entity dict
        "relations": [],
        "rel_seen": set(),
    }


def add_entity(
    stores: Dict,
    etype: str,
    org: str,
    key: Tuple,
    props: Dict[str, Any],
) -> Dict[str, Any]:
    """Add an entity, keeping guideline-specific entities separate."""
    store_key = (etype, org, key)
    if store_key in stores["entities"]:
        ent = stores["entities"][store_key]
        # Merge: fill in missing fields only
        for k, v in props.items():
            if v is not None and ent.get(k) in (None, "", []):
                ent[k] = v
        return ent
    eid = stable_id(etype, org, *[str(x) for x in key],
                    prefix=f"{etype[:3].lower()}_")
    ent = {"type": etype, "id": eid, "organization": org}
    ent.update(props)
    stores["entities"][store_key] = ent
    return ent


def add_relation(
    stores: Dict,
    src_id: str,
    rtype: str,
    dst_id: str,
    **props,
) -> None:
    """Add a relation once and keep any extra provenance found later."""
    key = (src_id, rtype, dst_id)

    if key in stores["rel_seen"]:
        # Preserve additional page provenance on the existing relation.
        for row in stores["relations"]:
            if (row["src"], row["type"], row["dst"]) != key:
                continue

            new_page = props.get("page")
            if new_page is not None:
                pages = row.get("pages")

                if pages is None:
                    pages = []
                    old_page = row.get("page")
                    if old_page is not None:
                        pages.append(old_page)

                if new_page not in pages:
                    pages.append(new_page)

                if len(pages) > 1:
                    row["pages"] = pages

            # Fill properties that were absent on the first occurrence.
            for prop_key, prop_value in props.items():
                if prop_key == "page":
                    continue
                if prop_value is not None and row.get(prop_key) in (None, "", []):
                    row[prop_key] = prop_value
            return

    stores["rel_seen"].add(key)
    row = {"src": src_id, "type": rtype, "dst": dst_id}
    row.update(props)
    stores["relations"].append(row)


# Normalisation helpers.
TEST_ALIASES = {
    "fpg": "FPG",
    "fasting plasma glucose": "FPG",
    "fasting glucose": "FPG",
    "a1c": "HbA1c",
    "hba1c": "HbA1c",
    "glycated hemoglobin": "HbA1c",
    "glycated haemoglobin": "HbA1c",
    "ogtt": "2h-OGTT",
    "2h ogtt": "2h-OGTT",
    "2-h ogtt": "2h-OGTT",
    "75g ogtt": "2h-OGTT",
    "random pg": "Random PG",
    "random plasma glucose": "Random PG",
    "cgm": "CGM",
    "time in range": "CGM-TIR",
    "tir": "CGM-TIR",
    "post-prandial": "Post-prandial Glucose",
    "postprandial": "Post-prandial Glucose",
    "smbg": "SMBG",
}

def norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def norm_test(name: str) -> str:
    return TEST_ALIASES.get(norm(name).lower(), norm(name))

def to_mmol(val: float, unit: str) -> Optional[float]:
    """Normalise glucose values to mmol/L for cross-guideline comparison."""
    u = (unit or "").lower().strip()
    if "mmol" in u:
        return round(val, 2)
    if "mg" in u:
        return mgdl_to_mmolL(val)
    return None


# Patterns used to find clinical targets in text.
# HbA1c in percent.
RE_HBA1C_PCT = re.compile(
    r"(?:HbA1c|A1C|glycated h[ae]moglobin)[^.]{0,60}"
    r"([\u2265\u2264<>]|>=|<=)\s*(\d+\.?\d*)\s*%",
    re.I,
)
# HbA1c in mmol/mol.
RE_HBA1C_MMOL = re.compile(
    r"(?:HbA1c|A1C)[^.]{0,40}"
    r"([\u2265\u2264<>]|>=|<=)\s*(\d{2,3})\s*mmol/mol",
    re.I,
)
# Fasting plasma glucose in mmol/L.
RE_FPG_MMOL = re.compile(
    r"(?:fasting|FPG|fasting plasma glucose)[^.]{0,40}"
    r"([\u2265\u2264<>]|>=|<=)\s*(\d+\.?\d*)\s*mmol/[Ll]",
    re.I,
)
# Fasting plasma glucose in mg/dL.
RE_FPG_MGDL = re.compile(
    r"(?:fasting|FPG|fasting plasma glucose)[^.]{0,40}"
    r"([\u2265\u2264<>]|>=|<=)\s*(\d{2,3})\s*mg/d[Ll]",
    re.I,
)
# Two-hour OGTT in mmol/L.
RE_OGTT_MMOL = re.compile(
    r"(?:2.?h|2-hour|OGTT)[^.]{0,40}"
    r"([\u2265\u2264<>]|>=|<=)\s*(\d+\.?\d*)\s*mmol/[Ll]",
    re.I,
)
# Two-hour OGTT in mg/dL.
RE_OGTT_MGDL = re.compile(
    r"(?:2.?h|2-hour|OGTT)[^.]{0,40}"
    r"([\u2265\u2264<>]|>=|<=)\s*(\d{2,3})\s*mg/d[Ll]",
    re.I,
)
# Random plasma glucose.
RE_RPG_MMOL = re.compile(
    r"(?:random|RPG|random plasma glucose)[^.]{0,40}"
    r"([\u2265\u2264<>]|>=|<=)\s*(\d+\.?\d*)\s*mmol/[Ll]",
    re.I,
)
RE_RPG_MGDL = re.compile(
    r"(?:random|RPG)[^.]{0,40}"
    r"([\u2265\u2264<>]|>=|<=)\s*(\d{2,3})\s*mg/d[Ll]",
    re.I,
)
# Continuous glucose monitoring and time in range.
RE_TIR = re.compile(r"time[- ]?in[- ]?range|TIR|\bCGM\b", re.I)
RE_TIR_PCT = re.compile(
    r"time[- ]?in[- ]?range[^.]{0,40}"
    r"([\u2265\u2264<>])\s*(\d+)\s*%",
    re.I,
)

# Table caption classifiers
CAP_DIAGNOSTIC   = re.compile(r"diagn|criteria|classif", re.I)
CAP_TARGET       = re.compile(r"target|goal|glyc[ae]mic|hba1c|a1c", re.I)
CAP_MONITORING   = re.compile(r"monitor|screen|follow.?up|frequen", re.I)
CAP_TREATMENT    = re.compile(r"treat|therap|pharmac|drug|medic|insulin", re.I)
CAP_RISK_FACTOR  = re.compile(r"risk factor|complication|cardiovascular", re.I)

# Find the ingested guideline documents.
def discover_docs(interim_dir: Path, limit_to: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Return list of meta dicts for all valid doc directories."""
    docs_file = interim_dir / "documents.jsonl"
    docs = []
    if docs_file.exists():
        docs = [json.loads(l) for l in docs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        # Fallback: scan directories
        for d in sorted(interim_dir.iterdir()):
            meta_f = d / "meta.json"
            if d.is_dir() and meta_f.exists():
                docs.append(json.loads(meta_f.read_text(encoding="utf-8")))

    if limit_to:
        docs = [d for d in docs if d.get("doc_id") in limit_to]
    return docs


# Create guideline and organisation nodes.
def seed_guideline_nodes(stores, meta: Dict[str, Any]) -> Tuple[Dict, Dict]:
    org_name = meta.get("organization", "Unknown")
    doc_id   = meta.get("doc_id", "")
    gm       = GUIDELINE_META.get(org_name, {})

    org_ent = add_entity(stores, "Organisation", org_name, (org_name,), {
        "name":      org_name,
        "geography": meta.get("geography") or gm.get("geography", ""),
    })

    gl_ent = add_entity(stores, "Guideline", org_name, (doc_id,), {
        "name":        gm.get("display_name", meta.get("source_file", doc_id)),
        "doc_id":      doc_id,
        "source_file": meta.get("source_file", ""),
        "year":        meta.get("year_hint") or gm.get("year"),
        "geography":   meta.get("geography") or gm.get("geography", ""),
        "n_pages":     meta.get("n_pages"),
    })

    add_relation(stores, org_ent["id"], "PUBLISHES", gl_ent["id"])
    return org_ent, gl_ent


# Revision-note filter: ADA Summary of Revisions uses this language (pp 1-18)
_REVISION_RE = re.compile(
    r"\bwas\s+(updated|revised|added|modified|removed|enhanced|clarified|amended|renamed|restructured)\b"
    r"|\bstates?\s+that\b"
    r"|\bwere\s+(updated|revised|added|modified|removed|enhanced|clarified|amended)\b"
    r"|\bnow\s+(includes?|recommends?|reflects?|incorporates?)\b",
    re.I,
)

def _is_revision_note(text: str, page: int, org: str) -> bool:
    """Return True if this is an ADA Summary-of-Revisions entry, not a clinical reco."""
    if org == "ADA" and page <= 18 and _REVISION_RE.search(text):
        return True
    return False


# Create shared diabetes condition nodes.
def seed_conditions(stores) -> Dict[str, Dict]:
    conditions = {}
    for name in ["Type 2 Diabetes", "Type 1 Diabetes", "Prediabetes",
                 "Intermediate Hyperglycaemia", "Gestational Diabetes"]:
        # Conditions are shared across guidelines.
        ent = add_entity(stores, "Condition", "global", (name,), {"name": name})
        conditions[name.lower()] = ent
    return conditions


# Extract entities from recommendation text.
def extract_from_recommendations(
    stores, doc_dir: Path, org: str, gl_ent: Dict,
    conditions: Dict, verbose: bool
) -> int:
    reco_path = doc_dir / "recommendations.jsonl"
    if not reco_path.exists():
        return 0

    count = 0
    for row in read_jsonl(reco_path):
        text  = norm(row.get("text", ""))
        label = norm(row.get("label", ""))
        page  = row.get("page_start", 0)

        if len(text.split()) < 5:
            continue

        # Skip ADA revision notes that are not clinical recommendations.
        if _is_revision_note(text, page, org):
            continue

        # Keep each recommendation linked to its source guideline.
        reco = add_entity(stores, "Recommendation", org, (label, text[:80]), {
            "label":       label,
            "text":        text,
            "page_start":  page,
            "source_file": gl_ent.get("source_file", ""),
        })
        add_relation(stores, gl_ent["id"], "HAS_RECOMMENDATION", reco["id"],
                     label=label, page=page)

        # Extract numeric targets from the recommendation text.
        _extract_numeric_from_text(stores, text, org, page, conditions, reco["id"])
        count += 1

    if verbose:
        print(f"    recommendations: {count}")
    return count


# Extract numeric targets from text chunks.
def extract_from_chunks(
    stores, doc_dir: Path, org: str, gl_ent: Dict,
    conditions: Dict, verbose: bool
) -> int:
    chunks_path = doc_dir / "chunks.jsonl"
    if not chunks_path.exists():
        return 0

    count = 0
    for chunk in read_jsonl(chunks_path):
        text = norm(chunk.get("text", ""))
        page = chunk.get("page_start", chunk.get("page_id", 0))
        if len(text.split()) < 10:
            continue
        extracted = _extract_numeric_from_text(
            stores, text, org, page, conditions, source_id=None
        )
        count += extracted

    if verbose:
        print(f"    chunk numeric extractions: {count}")
    return count



def detect_condition(text: str, conditions: Dict) -> Optional[Dict]:
    """Return Type 1 or Type 2 diabetes only when the text is unambiguous."""
    t1 = bool(re.search(
        r"\btype\s*1\b|\bT1DM\b|\btype-1\b",
        text,
        re.I,
    ))
    t2 = bool(re.search(
        r"\btype\s*2\b|\bT2DM\b|\btype-2\b",
        text,
        re.I,
    ))

    if t1 and not t2:
        return conditions.get("type 1 diabetes")

    if t2 and not t1:
        return conditions.get("type 2 diabetes")

    return None

def _extract_numeric_from_text(
    stores, text: str, org: str, page: int,
    conditions: Dict, source_id: Optional[str]
) -> int:
    """Extract numeric clinical targets from a block of text."""
    count = 0
    dm = detect_condition(text, conditions)

    # HbA1c in %
    for m in RE_HBA1C_PCT.finditer(text):
        op  = normalize_comparator(m.group(1))
        val = float(m.group(2))
        if not (4.0 <= val <= 15.0):   # sanity range
            continue
        tgt = add_entity(stores, "GlycaemicTarget", org,
                         ("HbA1c", "%", op, val), {
            "metric": "HbA1c", "units": "%", "op": op, "value": val,
            "value_mmol_mol": round((val - 2.152) / 0.09148, 0),  # IFCC conversion
            "page_id": page, "source_quote": short(text, 160),
            "confidence": 0.75, "extract_method": "rule",
        })
        test = add_entity(stores, "Test", "global", ("HbA1c",), {"name": "HbA1c"})
        add_relation(stores, tgt["id"], "MEASURED_BY", test["id"])
        if dm:
            add_relation(stores, dm["id"], "HAS_TARGET", tgt["id"],
                         organization=org, page=page)
        if source_id:
            add_relation(stores, source_id, "SPECIFIES_TARGET", tgt["id"])
        count += 1

    # HbA1c in mmol/mol (NICE / UK)
    for m in RE_HBA1C_MMOL.finditer(text):
        op  = normalize_comparator(m.group(1))
        val = float(m.group(2))
        if not (20 <= val <= 200):
            continue
        tgt = add_entity(stores, "GlycaemicTarget", org,
                         ("HbA1c", "mmol/mol", op, val), {
            "metric": "HbA1c", "units": "mmol/mol", "op": op, "value": val,
            "value_pct": round(0.09148 * val + 2.152, 1),  # IFCC inverse
            "page_id": page, "source_quote": short(text, 160),
            "confidence": 0.75, "extract_method": "rule",
        })
        test = add_entity(stores, "Test", "global", ("HbA1c",), {"name": "HbA1c"})
        add_relation(stores, tgt["id"], "MEASURED_BY", test["id"])
        if dm:
            add_relation(stores, dm["id"], "HAS_TARGET", tgt["id"],
                         organization=org, page=page)
        if source_id:
            add_relation(stores, source_id, "SPECIFIES_TARGET", tgt["id"])
        count += 1

    # FPG mmol/L — clinical range 4.0–10.0 for screening, up to 15 for severe
    for m in RE_FPG_MMOL.finditer(text):
        op  = normalize_comparator(m.group(1))
        val = float(m.group(2))
        if not (4.0 <= val <= 15.0):   # tightened from (2.0, 20.0)
            continue
        _add_glucose_target(stores, org, "FPG", "mmol/L", op, val, val,
                            page, text, conditions, source_id)
        count += 1

    # FPG mg/dL (ADA) — clinical range 72–270
    for m in RE_FPG_MGDL.finditer(text):
        op    = normalize_comparator(m.group(1))
        val   = float(m.group(2))
        mmol  = mgdl_to_mmolL(val)
        if not (72 <= val <= 270):   # tightened from (36, 360)
            continue
        _add_glucose_target(stores, org, "FPG", "mg/dL", op, val, mmol,
                            page, text, conditions, source_id)
        count += 1

    # 2h-OGTT mmol/L — clinical range 7.0–15.0
    for m in RE_OGTT_MMOL.finditer(text):
        op  = normalize_comparator(m.group(1))
        val = float(m.group(2))
        if not (5.5 <= val <= 16.0):   # tightened from (5.0, 25.0)
            continue
        _add_glucose_target(stores, org, "2h-OGTT", "mmol/L", op, val, val,
                            page, text, conditions, source_id)
        count += 1

    # 2h-OGTT mg/dL — clinical range 99–288
    for m in RE_OGTT_MGDL.finditer(text):
        op   = normalize_comparator(m.group(1))
        val  = float(m.group(2))
        mmol = mgdl_to_mmolL(val)
        if not (99 <= val <= 288):   # tightened from (90, 450)
            continue
        _add_glucose_target(stores, org, "2h-OGTT", "mg/dL", op, val, mmol,
                            page, text, conditions, source_id)
        count += 1

    # Random plasma glucose.
    for m in RE_RPG_MMOL.finditer(text):
        op  = normalize_comparator(m.group(1))
        val = float(m.group(2))
        _add_glucose_target(stores, org, "Random PG", "mmol/L", op, val, val,
                            page, text, conditions, source_id)
        count += 1
    for m in RE_RPG_MGDL.finditer(text):
        op   = normalize_comparator(m.group(1))
        val  = float(m.group(2))
        mmol = mgdl_to_mmolL(val)
        _add_glucose_target(stores, org, "Random PG", "mg/dL", op, val, mmol,
                            page, text, conditions, source_id)
        count += 1

    # CGM / TIR — Fix 3: one node per org (not one per page)
    # Key on org only so all CGM mentions update the same node
    if RE_TIR.search(text):
        mp = add_entity(stores, "MonitoringProtocol", org,
                        ("CGM/TIR",), {        # single key per org
            "parameter": "CGM / Time-in-Range",
            "confidence": 0.55, "extract_method": "rule",
        })
        for m in RE_TIR_PCT.finditer(text):
            op  = normalize_comparator(m.group(1))
            val = float(m.group(2))
            # Only overwrite if we have a real value and didn't have one before
            if mp.get("tir_op") is None:
                mp["tir_op"]    = op
                mp["tir_value"] = val
                mp["page_id"]   = page
                mp["source_quote"] = short(text, 120)
        test_cgm = add_entity(stores, "Test", "global", ("CGM",), {"name": "CGM"})
        add_relation(stores, test_cgm["id"], "HAS_PROTOCOL", mp["id"],
                     organization=org)
        count += 1

    return count


def _add_glucose_target(
    stores, org: str, test_name: str, unit: str,
    op: str, val: float, val_mmol: float,
    page: int, text: str,
    conditions: Dict, source_id: Optional[str],
):
    dm = detect_condition(text, conditions)
    tgt = add_entity(stores, "GlycaemicTarget", org,
                     (test_name, unit, op, val), {
        "metric": test_name, "units": unit, "op": op, "value": val,
        "value_mmol_L": val_mmol,
        "page_id": page, "source_quote": short(text, 160),
        "confidence": 0.70, "extract_method": "rule",
    })
    test = add_entity(stores, "Test", "global", (norm_test(test_name),),
                      {"name": norm_test(test_name)})
    add_relation(stores, tgt["id"], "MEASURED_BY", test["id"])
    if dm:
        add_relation(stores, dm["id"], "HAS_TARGET", tgt["id"],
                     organization=org, page=page)
    if source_id:
        add_relation(stores, source_id, "SPECIFIES_TARGET", tgt["id"])


# Extract entities from tables.
def extract_from_tables(
    stores, doc_dir: Path, org: str, gl_ent: Dict,
    conditions: Dict, verbose: bool
) -> int:
    idx_path = doc_dir / "tables" / "index.jsonl"
    if not idx_path.exists():
        return 0

    count = 0
    for entry in read_jsonl(idx_path):
        caption   = norm(entry.get("caption", ""))
        page      = entry.get("page_id", 0)
        csv_rel   = entry.get("csv", "")
        tbl_id    = entry.get("id", "")
        if not csv_rel:
            continue

        csv_path = doc_dir / csv_rel
        if not csv_path.exists():
            continue

        try:
            rows = list(csv.reader(open(csv_path, encoding="utf-8")))
        except Exception:
            continue

        if len(rows) < 2:
            continue

        # Skip revision-note captions (e.g. "Table 14.1 was added to...")
        # These appear in ADA's Summary of Revisions and carry no clinical data
        if _REVISION_RE.search(caption):
            continue

        # Classify table by caption
        is_diag     = bool(CAP_DIAGNOSTIC.search(caption))
        is_target   = bool(CAP_TARGET.search(caption))
        is_monitor  = bool(CAP_MONITORING.search(caption))
        is_treat    = bool(CAP_TREATMENT.search(caption))

        # Create Table node linked to Guideline
        tbl_ent = add_entity(stores, "Table", org, (tbl_id,), {
            "table_id": tbl_id,
            "caption":  caption,
            "page_id":  page,
            "n_rows":   entry.get("n_rows"),
            "n_cols":   entry.get("n_cols"),
            "confidence": entry.get("confidence", 0.0),
        })
        add_relation(stores, gl_ent["id"], "CONTAINS_TABLE", tbl_ent["id"],
                     page=page)

        # Join all cell text for pattern matching
        full_text = " ".join(
            cell for row in rows for cell in row if cell.strip()
        )

        if is_diag or is_target:
            # Run numeric extractors on the full table text.
            # Condition linkage is now determined inside
            # _extract_numeric_from_text via detect_condition(), based on
            # the actual table text rather than a fixed default.
            extracted = _extract_numeric_from_text(
                stores, full_text, org, page, conditions, tbl_ent["id"]
            )
            count += extracted

        if is_monitor:
            mp = add_entity(stores, "MonitoringProtocol", org,
                            ("table", caption[:60]), {
                "parameter": caption,
                "page_id": page,
                "source": "table",
                "table_id": tbl_id,
                "confidence": 0.65,
                "extract_method": "rule",
            })
            add_relation(stores, tbl_ent["id"], "DESCRIBES_PROTOCOL", mp["id"])
            count += 1

        if is_treat:
            # Ignore revision-note captions.
            if _REVISION_RE.search(caption):
                pass
            else:
                tr = add_entity(stores, "TreatmentRecommendation", org,
                                ("table", caption[:60]), {
                    "setting": "table-derived",
                    "caption": caption,
                    "page_id": page,
                    "table_id": tbl_id,
                    "confidence": 0.65,
                    "extract_method": "rule",
                })
                add_relation(stores, gl_ent["id"], "HAS_RECOMMENDATION", tr["id"],
                             page=page, source="table")
                count += 1

    if verbose:
        print(f"    table entities created: {count}")
    return count


# Link matching glucose targets across guidelines.
def add_cross_guideline_equivalences(stores, verbose: bool) -> int:
    targets = [
        e for e in stores["entities"].values()
        if e.get("type") == "GlycaemicTarget" and e.get("value_mmol_L") is not None
    ]

    # Group targets by metric, comparator, and normalised value.
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for t in targets:
        key = (
            t.get("metric", ""),
            t.get("op", ""),
            round(float(t["value_mmol_L"]), 1),
        )
        groups[key].append(t)

    count = 0
    for key, group in groups.items():
        if len(group) < 2:
            continue
        # Link matching targets from different organisations.
        orgs_seen = set()
        for i, a in enumerate(group):
            if a["organization"] in orgs_seen:
                continue
            for b in group[i+1:]:
                if b["organization"] == a["organization"]:
                    continue
                add_relation(
                    stores, a["id"], "EQUIVALENT_TO", b["id"],
                    metric=key[0], op=key[1], value_mmol_L=key[2],
                    note="Same numeric threshold across guidelines",
                )
                count += 1
            orgs_seen.add(a["organization"])

    if verbose:
        print(f"  cross-guideline EQUIVALENT_TO relations: {count}")
    return count


# Main pipeline.
def main():
    args = parse_args()
    ensure_dirs()
    (OUTPUT_DIR / "kg-jsonl").mkdir(parents=True, exist_ok=True)

    interim_dir = INTERIM_DIR
    docs = discover_docs(interim_dir, args.docs)

    if not docs:
        print("[error] No documents found in", interim_dir)
        sys.exit(1)

    print(f"[start] Processing {len(docs)} documents")

    stores = make_stores()
    conditions = seed_conditions(stores)

    for doc_meta in docs:
        doc_id  = doc_meta.get("doc_id", "")
        org     = doc_meta.get("organization", "Unknown")
        doc_dir = interim_dir / doc_id

        if not doc_dir.exists():
            print(f"[skip] {doc_id} — directory not found")
            continue

        print(f"  [{org}] {doc_id}")

        # Create source nodes.
        org_ent, gl_ent = seed_guideline_nodes(stores, doc_meta)

        # Extract from recommendations, chunks, and tables.
        r1 = extract_from_recommendations(
            stores, doc_dir, org, gl_ent, conditions, args.verbose)
        r2 = extract_from_chunks(
            stores, doc_dir, org, gl_ent, conditions, args.verbose)
        r3 = extract_from_tables(
            stores, doc_dir, org, gl_ent, conditions, args.verbose)

        print(f"    recos={r1}  chunk_extractions={r2}  table_entities={r3}")

    # Add cross-guideline target equivalences.
    print("\n[cross-guideline] Computing EQUIVALENT_TO relations...")
    add_cross_guideline_equivalences(stores, args.verbose)

    # Save graph files.
    entities  = list(stores["entities"].values())
    relations = stores["relations"]

    ent_path = OUTPUT_DIR / "kg-jsonl" / "entities.jsonl"
    rel_path = OUTPUT_DIR / "kg-jsonl" / "relations.jsonl"
    write_jsonl(ent_path, entities)
    write_jsonl(rel_path, relations)

    # Print a short summary.
    from collections import Counter
    etype_counts = Counter(e["type"] for e in entities)
    rtype_counts  = Counter(r["type"] for r in relations)

    print(f"\n[done] entities: {len(entities)}  relations: {len(relations)}")
    print("\nEntity types:")
    for etype, cnt in sorted(etype_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<28} {cnt:>5}")
    print("\nRelation types:")
    for rtype, cnt in sorted(rtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {rtype:<28} {cnt:>5}")
    print(f"\n[wrote] {ent_path}")
    print(f"[wrote] {rel_path}")


if __name__ == "__main__":
    main()