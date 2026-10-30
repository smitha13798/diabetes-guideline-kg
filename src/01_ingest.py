#!/usr/bin/env python3
"""Parse diabetes guideline PDFs into structured text, tables, recommendations, and corpus files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import fitz  # PyMuPDF
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyMuPDF is required: pip install pymupdf") from e

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from docling.document_converter import DocumentConverter  # type: ignore
except Exception:
    DocumentConverter = None

ROOT = Path.cwd()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CANDIDATES = [ROOT / "data" / "input", SCRIPT_DIR.parent / "data" / "input"]
DEFAULT_OUTPUT_CANDIDATES = [ROOT / "data" / "interim", SCRIPT_DIR.parent / "data" / "interim"]

# Compiled regular expressions
TITLE_CASE_RE = re.compile(r"^[A-Z][A-Za-z0-9,/()''\%\-]+(?:\s+[A-Z][A-Za-z0-9,/()''\%\-]+){0,14}$")
ALL_CAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s,.:;()%\-/]{2,120}$")
TABLE_CAPTION_RE = re.compile(r"^table\s+([A-Za-z0-9.\-]+)", re.I)
FIGURE_CAPTION_RE = re.compile(r"^figure\s+([A-Za-z0-9.\-]+)", re.I)
NICE_RECO_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)")
YEAR_RE = re.compile(r"(20\d{2})")
DOT_LEADER_RE = re.compile(r"\.{3,}\s*\d+$")
DOI_RE = re.compile(r"\bdoi\b", re.I)
JOURNAL_META_RE = re.compile(r"\b(can j diabetes|vol\.|issue|copyright|author|affiliation|received|accepted)\b", re.I)
ADA_NAME_RE = re.compile(r"(^|[_\-\s])ada([_\-\s]|$)", re.I)
CANADA_NAME_RE = re.compile(r"canada(diabetesguidelines)?|canadiandiabetesguidelines|can_j_diabetes|canjdiabetes", re.I)
NICE_TABLEISH_RE = re.compile(r"(table|algorithm|summary of recommendations|drug|dose|benefits|risks|monitoring)", re.I)
BULLETISH_RE = re.compile(r"^[•\-–]|^[A-Za-z]\)\s")
CONTENTS_CELL_RE = re.compile(r"^(contents|chapter\s+\d+|pg\.?\s*\d+|foreward|introduction|references)$", re.I)
TITLE_PAGE_RE = re.compile(r"(standards of care in diabetes|international diabetes federation|national institute for health and care excellence|diabetes canada)", re.I)

# Numbered recommendations used in Diabetes Canada documents.
CANADA_RECO_RE = re.compile(r"^(\d+[a-z]?)\.\s+(.{20,})", re.DOTALL)

# Recommendation blocks used in ADA documents.
ADA_RECO_BLOCK_RE = re.compile(r"^Recommendations?\s+(\d+\.\d+[a-z]?)\s+(.+)", re.I | re.DOTALL)

# Care-level recommendations used in IDF documents.
IDF_CARE_RE = re.compile(r"^(Optimal Care|Basic Care|Recommended Care|Comprehensive Care)[:\s]+(.{10,})", re.I | re.DOTALL)

# Only the opening pages of journal-style documents are treated as front matter.
JOURNAL_FRONTMATTER_PAGE_LIMIT = 2

# Guideline profiles
GUIDELINE_PROFILES: Dict[str, Dict[str, Any]] = {
    "IDF": {
        "organization": "IDF",
        "geography": "Global",
        "heading_allow": [r"^chapter\s+\d+", r"^(introduction|foreward|references|glossary|appendix)\b"],
        "heading_deny": [r"^(optimal care|basic care|who criteria|ada criteria|definition|rationale)$"],
        "table_caption": TABLE_CAPTION_RE,
        "figure_caption": FIGURE_CAPTION_RE,
        "nice_numbered_sections": False,
        "journal_style": False,
    },
    "ADA": {
        "organization": "ADA",
        "geography": "US",
        "heading_allow": [r"^(summary of revisions|standards of care|classification and diagnosis|prevention or delay|comprehensive medical evaluation|glycemic targets|diabetes technology|obesity and weight management|pharmacologic approaches|cardiovascular disease|chronic kidney disease|older adults|children and adolescents|management of diabetes in pregnancy)\b"],
        "heading_deny": [r"^(professional practice committee|american diabetes association|copyright|doi)$"],
        "table_caption": TABLE_CAPTION_RE,
        "figure_caption": FIGURE_CAPTION_RE,
        "nice_numbered_sections": False,
        "journal_style": False,
    },
    "NICE": {
        "organization": "NICE",
        "geography": "UK",
        "heading_allow": [r"^\d+(?:\.\d+)+\s+", r"^(recommendations|context|terms used|recommendation)\b"],
        "heading_deny": [r"^(your responsibility|contents)$"],
        "table_caption": TABLE_CAPTION_RE,
        "figure_caption": FIGURE_CAPTION_RE,
        "nice_numbered_sections": True,
        "journal_style": False,
    },
    "Diabetes Canada": {
        "organization": "Diabetes Canada",
        "geography": "Canada",
        "heading_allow": [r"^(key messages|recommendations|introduction|screening|treatment|references|acknowledg)\b"],
        "heading_deny": [r"^(canadian journal of diabetes|can j diabetes|doi|author|affiliation)$"],
        "table_caption": TABLE_CAPTION_RE,
        "figure_caption": FIGURE_CAPTION_RE,
        "nice_numbered_sections": False,
        "journal_style": True,
    },
}

CONTENT_RULES = [
    {"organization": "ADA", "pattern": r"American Diabetes Association|Standards of Care in Diabetes", "geography": "US"},
    {"organization": "NICE", "pattern": r"National Institute for Health and Care Excellence|\bNICE\b", "geography": "UK"},
    {"organization": "Diabetes Canada", "pattern": r"Diabetes Canada|Canadian Journal of Diabetes|Can J Diabetes", "geography": "Canada"},
    {"organization": "IDF", "pattern": r"International Diabetes Federation|\bIDF\b", "geography": "Global"},
]

@dataclass
class Args:
    """Command-line options for one ingestion run."""

    input_path: Path
    output_path: Path
    no_figures: bool = False
    no_structured: bool = False
    no_flow: bool = False
    no_docling: bool = False
    tables_lite: bool = False

# Utilities
def stable_id(text: str, prefix: str = "id_") -> str:
    return prefix + hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# Some Diabetes Canada PDFs contain words joined together during text extraction.
# (e.g. "Ageofonset" -> "Age of onset", "Type1diabetes" -> "Type 1 diabetes").
# Docling preserves this faithfully. Two-stage repair:
#   Stage 1 — split digit↔letter boundaries unconditionally (never intentional
#              in clinical text; A1C/HbA1c are safe because no letter runs attach).
#   Stage 2 — split camelCase transitions only when density is high enough
#              to avoid breaking normal abbreviations in English prose.
_CAMEL_FIX_RE   = re.compile(r"([a-z])([A-Z])")
_DIGIT_ATTACH_RE = re.compile(r"([A-Za-z])(\d)")
_DIGIT_LEAD_RE   = re.compile(r"(\d)([A-Za-z])")
_CONCAT_DETECT_RE = re.compile(r"[a-z][A-Z]")

def fix_concatenated_text(text: str) -> str:
    """Add missing spaces to joined text from Diabetes Canada PDFs."""
    if not text or len(text) < 6:
        return text
    # Separate joined letters and numbers.
    text = _DIGIT_ATTACH_RE.sub(r"\1 \2", text)
    text = _DIGIT_LEAD_RE.sub(r"\1 \2", text)
    # Split joined words only when the pattern is common in the text.
    if len(_CONCAT_DETECT_RE.findall(text)) / max(1, len(text)) >= 0.04:
        text = _CAMEL_FIX_RE.sub(r"\1 \2", text)
    return norm(text)

def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip().lower()).strip("_")
    return s or "doc"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# Source / profile inference
def infer_guideline_source(pdf_path: Path, probe_text: str = "") -> Dict[str, Any]:
    name = pdf_path.stem.lower()
    text = probe_text or ""
    result = {"organization": None, "geography": None, "year_hint": None, "inference_method": "filename"}

    filename_org = None
    if CANADA_NAME_RE.search(name) or name.startswith("canada"):
        filename_org = ("Diabetes Canada", "Canada")
    elif "nice" in name:
        filename_org = ("NICE", "UK")
    elif "idf" in name:
        filename_org = ("IDF", "Global")
    elif ADA_NAME_RE.search(name):
        filename_org = ("ADA", "US")

    if filename_org:
        result.update({"organization": filename_org[0], "geography": filename_org[1]})

    y = YEAR_RE.search(pdf_path.name)
    if y:
        result["year_hint"] = int(y.group(1))

    matched_rule = None
    for rule in CONTENT_RULES:
        if text and re.search(rule["pattern"], text, re.I):
            matched_rule = rule
            break

    if matched_rule:
        if filename_org is None:
            result["organization"] = matched_rule["organization"]
            result["geography"] = matched_rule["geography"]
            result["inference_method"] = "content"
        elif matched_rule["organization"] == result["organization"]:
            result["inference_method"] = "mixed"
        elif result["organization"] in {"Diabetes Canada", "NICE", "IDF"}:
            result["inference_method"] = "filename"
        else:
            result["organization"] = matched_rule["organization"]
            result["geography"] = matched_rule["geography"]
            result["inference_method"] = "mixed"

    if not result["organization"]:
        result.update({"organization": "IDF", "geography": "Global"})
    return result

def get_profile(org: str) -> Dict[str, Any]:
    return GUIDELINE_PROFILES.get(org, GUIDELINE_PROFILES["IDF"])

# Docling
def try_docling(pdf_path: Path) -> Dict[str, Any]:
    if DocumentConverter is None:
        return {"engine": "none", "ok": False, "blocks": [], "raw_text": "", "pages": []}
    try:
        conv = DocumentConverter()
        result = conv.convert(str(pdf_path))
        doc = getattr(result, "document", None)
        text = ""
        if doc is not None:
            if hasattr(doc, "export_to_markdown"):
                text = doc.export_to_markdown() or ""
            elif hasattr(doc, "export_to_text"):
                text = doc.export_to_text() or ""
        return {"engine": "docling", "ok": True, "blocks": [], "raw_text": text, "pages": []}
    except Exception:
        return {"engine": "docling", "ok": False, "blocks": [], "raw_text": "", "pages": []}

# Structural PDF parsing
def parse_pdf_structural(pdf_path: Path) -> Dict[str, Any]:
    doc = fitz.open(str(pdf_path))
    pages: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    first_two_text: List[str] = []
    for page_idx, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        w, h = float(page.rect.width), float(page.rect.height)
        page_lines: List[str] = []
        for b_idx, b in enumerate(page_dict.get("blocks", []), start=1):
            if b.get("type") != 0:
                continue
            lines = []
            spans_all = []
            for line in b.get("lines", []):
                parts = []
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if txt:
                        parts.append(txt)
                        spans_all.append(span)
                if parts:
                    lines.append(norm(" ".join(parts)))
            text = norm("\n".join(lines))
            text = fix_concatenated_text(text)  # FIX A: repair Canada concatenated encoding
            if not text:
                continue
            if page_idx <= 2:
                page_lines.append(text)
            x0, y0, x1, y1 = map(float, b.get("bbox", [0, 0, 0, 0]))
            fonts = [float(s.get("size", 0)) for s in spans_all] or [0.0]
            flags = [int(s.get("flags", 0)) for s in spans_all] or [0]
            block = {
                "id": stable_id(f"{pdf_path.name}:{page_idx}:{b_idx}:{text[:80]}", "blk_"),
                "page_id": page_idx,
                "bbox": [x0, y0, x1, y1],
                "text": text,
                "avg_font_size": sum(fonts) / max(1, len(fonts)),
                "max_font_size": max(fonts),
                "font_bold": any(f & 16 for f in flags),
                "page_width": w,
                "page_height": h,
            }
            blocks.append(block)
        raw = page.get_text("text") or ""
        pages.append({
            "id": stable_id(f"{pdf_path.name}:page:{page_idx}", "pg_"),
            "page_id": page_idx,
            "width": w,
            "height": h,
            "text": raw,
        })
        if page_idx <= 2:
            first_two_text.append(raw)
    doc.close()
    return {"pages": pages, "blocks": blocks, "probe_text": "\n".join(first_two_text)}

# Column detection / ordering
def detect_columns(page_blocks: List[Dict[str, Any]], page_width: float) -> int:
    if len(page_blocks) < 8:
        return 1
    centers = [((b["bbox"][0] + b["bbox"][2]) / 2.0) for b in page_blocks if len(norm(b["text"])) > 20]
    if len(centers) < 6:
        return 1
    mid = page_width / 2.0
    left = sum(1 for c in centers if c < mid * 0.9)
    right = sum(1 for c in centers if c > mid * 1.1)
    return 2 if left >= 3 and right >= 3 else 1

def order_blocks(page_blocks: List[Dict[str, Any]], columns: int, page_width: float) -> List[Dict[str, Any]]:
    if columns == 1:
        return sorted(page_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    split_x = page_width / 2.0
    left = [b for b in page_blocks if ((b["bbox"][0] + b["bbox"][2]) / 2.0) <= split_x]
    right = [b for b in page_blocks if ((b["bbox"][0] + b["bbox"][2]) / 2.0) > split_x]
    left = sorted(left, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    right = sorted(right, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return left + right

# Page classification
def table_signal_score(page_blocks: List[Dict[str, Any]]) -> float:
    texts = [norm(b["text"]) for b in page_blocks if norm(b["text"])]
    if not texts:
        return 0.0
    short = sum(1 for t in texts if len(t.split()) <= 6)
    numericish = sum(1 for t in texts if sum(ch.isdigit() for ch in t) >= 2)
    x0s = [round(float(b["bbox"][0]) / 12.0) for b in page_blocks if len(norm(b["text"]).split()) <= 10]
    repeated_x = sum(c for _, c in Counter(x0s).items() if c >= 3)
    total = max(1, len(texts))
    return 0.4 * (short / total) + 0.35 * (numericish / total) + 0.25 * min(1.0, repeated_x / max(3, total / 2.0))

def caption_density(page_blocks: List[Dict[str, Any]]) -> int:
    texts = [norm(b["text"]) for b in page_blocks]
    return sum(1 for t in texts if TABLE_CAPTION_RE.match(t) or FIGURE_CAPTION_RE.match(t))

def classify_page(
    page_blocks: List[Dict[str, Any]],
    profile: Dict[str, Any],
    page_idx: int = 999,  # FIX-1: page index to gate journal_frontmatter classification
) -> str:
    if not page_blocks:
        return "blank"
    texts = [norm(b["text"]) for b in page_blocks]
    joined = "\n".join(texts)
    short = sum(1 for t in texts if len(t.split()) <= 6)
    dotted = sum(1 for t in texts if DOT_LEADER_RE.search(t))
    caps = sum(1 for t in texts if ALL_CAPS_RE.match(t))
    table_caps = sum(1 for t in texts if TABLE_CAPTION_RE.match(t))
    fig_caps = sum(1 for t in texts if FIGURE_CAPTION_RE.match(t))
    long_paras = sum(1 for t in texts if len(t.split()) >= 18)
    bulletish = sum(1 for t in texts if BULLETISH_RE.search(t))
    nums = sum(ch.isdigit() for ch in joined)
    letters = sum(ch.isalpha() for ch in joined) or 1
    digit_ratio = nums / max(1, nums + letters)
    lower = joined.lower()
    signal = table_signal_score(page_blocks)
    if dotted >= 2 or (short >= 8 and ("contents" in lower or "chapter" in lower)):
        return "contents"
    if TITLE_PAGE_RE.search(joined) and long_paras <= 4 and short >= 5:
        return "frontmatter"
    if table_caps >= 1:
        return "table_like"
    if fig_caps >= 1:
        return "figure_like"
    # FIX-1: gate journal_frontmatter to first JOURNAL_FRONTMATTER_PAGE_LIMIT pages only.
    # Previously this fired on every page of journal-style docs (Canada guidelines),
    # causing the entire body to be misclassified and skipped by recover_headings.
    if (
        profile.get("journal_style")
        and page_idx <= JOURNAL_FRONTMATTER_PAGE_LIMIT
        and (DOI_RE.search(joined) or JOURNAL_META_RE.search(joined))
        and long_paras <= 4
    ):
        return "journal_frontmatter"
    if profile["organization"] == "NICE" and (NICE_RECO_RE.search(joined) or "recommendation" in lower):
        return "recommendations"
    if bulletish >= 4 and long_paras <= 4:
        return "recommendations"
    if signal >= 0.56 and short >= 6 and digit_ratio >= 0.08:
        return "table_like"
    if long_paras >= 5:
        return "prose"
    if caps >= 6 and short >= 8:
        return "contents"
    return "prose"

# Heading recovery
def looks_like_heading(block: Dict[str, Any], body_font: float, profile: Dict[str, Any], page_type: str) -> Tuple[bool, float]:
    text = norm(block.get("text", "").split("\n")[0])
    if not text or len(text) > 160:
        return False, 0.0
    lower = text.lower()
    if page_type in {"contents", "journal_frontmatter"}:
        return False, 0.0
    if TABLE_CAPTION_RE.match(text) or FIGURE_CAPTION_RE.match(text):
        return False, 0.0
    # Keep NICE table and algorithm captions from being treated as headings.

    if profile.get("nice_numbered_sections") and NICE_TABLEISH_RE.search(text):
        return False, 0.0
    for pat in profile.get("heading_deny", []):
        if re.search(pat, lower, re.I):
            return False, 0.0
    allow_hit = any(re.search(pat, lower, re.I) for pat in profile.get("heading_allow", []))
    fs = max(float(block.get("avg_font_size") or 0.0), float(block.get("max_font_size") or 0.0))
    bold = bool(block.get("font_bold"))
    title_case = bool(TITLE_CASE_RE.match(text))
    all_caps = bool(ALL_CAPS_RE.match(text)) and len(text.split()) <= 12
    words = len(text.split())
    score = 0.0
    if allow_hit:
        score += 0.55
    if profile.get("nice_numbered_sections") and NICE_RECO_RE.match(text):
        score += 0.45
    if fs >= body_font * 1.22:
        score += 0.25
    if fs >= body_font * 1.35:
        score += 0.15
    if bold:
        score += 0.18
    if title_case or all_caps:
        score += 0.12
    if words <= 12:
        score += 0.08
    if words <= 3 and not allow_hit and not all_caps:
        score -= 0.12
    if ":" in text and words <= 8 and not allow_hit:
        score -= 0.15
    if sum(ch.isdigit() for ch in text) >= 5 and not profile.get("nice_numbered_sections"):
        score -= 0.12
    return score >= 0.52, max(0.0, min(1.0, score))

def recover_headings(
    page_blocks: List[Dict[str, Any]],
    body_font: float,
    profile: Dict[str, Any],
    page_type: str,
) -> List[Dict[str, Any]]:
    out = []
    ordered = page_blocks
    # Skip repeated headings on the same page.

    seen_titles: set = set()
    for idx, b in enumerate(ordered):
        ok, conf = looks_like_heading(b, body_font, profile, page_type)
        if ok:
            title = norm(b["text"])
            # Skip an exact duplicate heading.
            if title in seen_titles:
                continue
            seen_titles.add(title)
            row = dict(b)
            row["id"] = stable_id(f"heading:{b['page_id']}:{title[:80]}", "hd_")
            row["title"] = title
            row["confidence"] = conf
            out.append(row)
            continue
        # ADA recovery: bold short lines with surrounding whitespace.
        if profile["organization"] == "ADA":
            text = norm(b["text"])
            prev_y = ordered[idx - 1]["bbox"][3] if idx > 0 else 0
            gap_above = b["bbox"][1] - prev_y
            if b.get("font_bold") and len(text.split()) <= 10 and gap_above >= 10 and len(text) <= 90:
                if text not in seen_titles:
                    seen_titles.add(text)
                    row = dict(b)
                    row["id"] = stable_id(f"heading:{b['page_id']}:{text[:80]}", "hd_")
                    row["title"] = text
                    row["confidence"] = 0.56
                    out.append(row)
    return out

# Caption detection
def detect_captions(blocks: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    caps: List[Dict[str, Any]] = []
    for b in blocks:
        text = norm(b["text"])
        kind = None
        if profile["table_caption"].match(text):
            kind = "table"
        elif profile["figure_caption"].match(text):
            kind = "figure"
        if kind:
            caps.append({
                "id": stable_id(f"cap:{b['page_id']}:{text[:90]}", "cap_"),
                "page_id": b["page_id"],
                "kind": kind,
                "text": text,
                "bbox": b["bbox"],
                "confidence": 0.9,
            })
    return caps

# Helpers
def page_blocks_map(blocks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for b in blocks:
        out[int(b["page_id"])].append(b)
    return out

def bbox_union(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

# Section building
def build_sections(
    pages: List[Dict[str, Any]],
    page_blocks_by_id: Dict[int, List[Dict[str, Any]]],
    headings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hmap: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for h in sorted(headings, key=lambda x: (x["page_id"], x["bbox"][1], x["bbox"][0])):
        hmap[h["page_id"]].append(h)
    sections: List[Dict[str, Any]] = []
    current = {
        "id": stable_id("preamble", "sec_"),
        "title": "Preamble",
        "page_start": 1,
        "page_end": 1,
        "text_parts": [],
        "confidence": 0.4,
    }
    for page in pages:
        pid = int(page["page_id"])
        pblocks = sorted(page_blocks_by_id.get(pid, []), key=lambda b: (b["bbox"][1], b["bbox"][0]))
        page_heads = hmap.get(pid, [])
        head_i = 0
        for b in pblocks:
            while head_i < len(page_heads) and page_heads[head_i]["bbox"][1] <= b["bbox"][1] + 1:
                h = page_heads[head_i]
                if current["text_parts"] or current["title"] != "Preamble":
                    current["text"] = norm("\n".join(current.pop("text_parts")))
                    sections.append(current)
                current = {
                    "id": stable_id(f"sec:{pid}:{h['title']}", "sec_"),
                    "title": h["title"],
                    "page_start": pid,
                    "page_end": pid,
                    "text_parts": [],
                    "confidence": h.get("confidence", 0.6),
                }
                head_i += 1
            current["text_parts"].append(b["text"])
            current["page_end"] = pid
    current["text"] = norm("\n".join(current.pop("text_parts")))
    sections.append(current)
    return [s for s in sections if norm(s.get("text", ""))]

# Chunking
def make_chunks(sections: List[Dict[str, Any]], doc_key: str, chunk_words: int = 220) -> List[Dict[str, Any]]:
    rows = []
    for sec in sections:
        words = sec.get("text", "").split()
        if not words:
            continue
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i:i + chunk_words])
            rows.append({
                "id": stable_id(f"{doc_key}:{sec['id']}:{i}", "chk_"),
                "section_id": sec["id"],
                "title": sec["title"],
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "text": chunk,
            })
    return rows

# Table extraction helpers
def rows_look_like_prose(rows: List[List[str]]) -> bool:
    flat = [norm(c) for r in rows for c in r if norm(c)]
    if not flat:
        return False
    long_cells = sum(1 for c in flat if len(c.split()) >= 12)
    bullet_cells = sum(1 for c in flat if BULLETISH_RE.search(c))
    avg_words = sum(len(c.split()) for c in flat) / max(1, len(flat))
    return avg_words >= 8 or long_cells >= max(2, len(flat) // 3) or bullet_cells >= max(2, len(flat) // 3)

def rows_look_like_bullets(rows: List[List[str]]) -> bool:
    flat = [norm(c) for r in rows for c in r if norm(c)]
    if not flat:
        return False
    bullet_like = sum(1 for c in flat if BULLETISH_RE.search(c) or c.lower().startswith(("recommend", "consider ", "offer ", "do not ")))
    shortish = sum(1 for c in flat if len(c.split()) <= 10)
    return bullet_like >= max(2, len(flat) // 3) and shortish >= max(2, len(flat) // 3)

def rows_look_like_contents(rows: List[List[str]]) -> bool:
    flat = [norm(c) for r in rows for c in r if norm(c)]
    if not flat:
        return False
    dotted = sum(1 for c in flat if DOT_LEADER_RE.search(c))
    contentsish = sum(1 for c in flat if CONTENTS_CELL_RE.search(c))
    return dotted >= 2 or contentsish >= max(2, len(flat) // 3)

def has_real_tabularity(rows: List[List[str]]) -> bool:
    if len(rows) < 2:
        return False
    widths = [len(r) for r in rows]
    mode_w = Counter(widths).most_common(1)[0][0]
    if mode_w < 2:
        return False
    consistent = sum(1 for w in widths if abs(w - mode_w) <= 1) / max(1, len(widths))
    nonempty_by_row = [sum(1 for c in r if norm(c)) for r in rows]
    numeric_cells = sum(1 for r in rows for c in r if re.search(r"\d", norm(c)))
    return consistent >= 0.6 and sum(1 for x in nonempty_by_row if x >= 2) >= max(2, len(rows) // 2) and numeric_cells >= 2

def looks_like_false_table(rows: List[List[str]], caption: Optional[str], page_type: str) -> bool:
    if not rows:
        return True
    widths = [len(r) for r in rows]
    mode_w = Counter(widths).most_common(1)[0][0]
    if page_type in {"contents", "journal_frontmatter"}:
        return True
    if page_type == "prose" and not caption:
        return True
    if mode_w <= 1:
        return True
    if len(rows) < 2:
        return True
    if rows_look_like_contents(rows):
        return True
    if rows_look_like_bullets(rows) and not caption:
        return True
    if rows_look_like_prose(rows) and not caption and not has_real_tabularity(rows):
        return True
    headerish = sum(1 for c in rows[0] if norm(c))
    if headerish <= 1 and rows_look_like_prose(rows) and not caption:
        return True
    return False

def table_confidence(rows: List[List[str]], caption: Optional[str], page_type: str) -> float:
    if not rows:
        return 0.0
    n_rows = len(rows)
    widths = [len(r) for r in rows]
    mode_w = Counter(widths).most_common(1)[0][0]
    consistent = sum(1 for w in widths if abs(w - mode_w) <= 1) / max(1, n_rows)
    nonempty = sum(1 for r in rows for c in r if norm(c))
    avg_nonempty = nonempty / max(1, n_rows)
    score = 0.0
    if caption:
        score += 0.4
    if n_rows >= 3:
        score += 0.15
    if mode_w >= 2:
        score += 0.2
    if mode_w >= 3:
        score += 0.05
    if consistent > 0.7:
        score += 0.1
    if avg_nonempty >= 2.5:
        score += 0.08
    if page_type == "table_like":
        score += 0.10
    if page_type == "recommendations":
        score -= 0.18
    if rows_look_like_prose(rows):
        score -= 0.22
    if rows_look_like_bullets(rows):
        score -= 0.20
    if rows_look_like_contents(rows):
        score -= 0.40
    if has_real_tabularity(rows):
        score += 0.10
    if mode_w == 1:
        score -= 0.30
    return max(0.0, min(1.0, score))

def extract_tables_region_first(
    pdf_path: Path,
    pages: List[Dict[str, Any]],
    blocks: List[Dict[str, Any]],
    captions: List[Dict[str, Any]],
    page_types: Dict[int, str],
    tables_lite: bool,
    profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if pdfplumber is None:
        return []
    page_blockmap = page_blocks_map(blocks)
    caps_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in captions:
        if c["kind"] == "table":
            caps_by_page[int(c["page_id"])].append(c)
    results: List[Dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p_idx, plumber_page in enumerate(pdf.pages, start=1):
            page_type = page_types.get(p_idx, "prose")
            page_caps = sorted(caps_by_page.get(p_idx, []), key=lambda c: c["bbox"][1])
            candidates: List[Tuple[Tuple[float, float, float, float], Optional[str]]] = []
            page_h = float(plumber_page.height)
            page_w = float(plumber_page.width)
            if page_caps:
                for i, cap in enumerate(page_caps):
                    top = max(0.0, float(cap["bbox"][3]) + 2.0)
                    bottom = float(page_caps[i + 1]["bbox"][1] - 6.0) if i + 1 < len(page_caps) else page_h - 8.0
                    if bottom - top >= 35:
                        candidates.append(((8.0, top, page_w - 8.0, bottom), cap["text"]))
            elif page_type == "table_like":
                page_signal = table_signal_score(page_blockmap.get(p_idx, []))
                page_text = " ".join(norm(b["text"]) for b in page_blockmap.get(p_idx, []))
                if page_signal >= 0.62 and not TITLE_PAGE_RE.search(page_text):
                    candidates.append(((6.0, 6.0, page_w - 6.0, page_h - 6.0), None))
            else:
                continue
            seen_fps = set()
            for bbox, cap_text in candidates:
                crop = plumber_page.crop(bbox)
                settings_list = [{"vertical_strategy": "lines", "horizontal_strategy": "lines"}]
                if not tables_lite:
                    settings_list.append({"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 3, "intersection_tolerance": 3})
                    settings_list.append({"vertical_strategy": "lines", "horizontal_strategy": "text", "snap_tolerance": 4, "intersection_tolerance": 4})
                for settings in settings_list:
                    try:
                        extracted = crop.extract_tables(table_settings=settings) or []
                    except Exception:
                        extracted = []
                    for table in extracted:
                        # Repair joined text in table cells from journal-style PDFs.

                        is_journal = profile and profile.get("journal_style")
                        if is_journal:
                            rows = [[fix_concatenated_text(norm(c or "")) for c in row]
                                    for row in table if row and any(norm(c or "") for c in row)]
                        else:
                            rows = [[norm(c or "") for c in row]
                                    for row in table if row and any(norm(c or "") for c in row)]
                        if not rows or looks_like_false_table(rows, cap_text, page_type):
                            continue
                        conf = table_confidence(rows, cap_text, page_type)
                        threshold = 0.60 if cap_text else 0.68
                        if page_type == "recommendations":
                            threshold += 0.05
                        if conf < threshold:
                            continue
                        fp = hashlib.sha1(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()
                        if fp in seen_fps:
                            continue
                        seen_fps.add(fp)
                        if cap_text is None:
                            if page_type != "table_like":
                                continue
                        results.append({
                            "id": stable_id(f"{pdf_path.name}:{p_idx}:{fp}", "tbl_"),
                            "page_id": p_idx,
                            "bbox": list(bbox),
                            "caption": cap_text,
                            "rows": rows,
                            "confidence": conf,
                            "kind": "semantic_table",
                        })
    return results

# Table / figure saving
def save_tables(out_dir: Path, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tdir = out_dir / "tables"
    ensure_dir(tdir)
    index = []
    for i, t in enumerate(tables, start=1):
        csv_path = tdir / f"table_{t['page_id']}_{i}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(t["rows"])
        index.append({
            "id": t["id"],
            "page_id": t["page_id"],
            "caption": t.get("caption"),
            "bbox": t.get("bbox"),
            "confidence": t.get("confidence", 0.0),
            "csv": csv_path.relative_to(out_dir).as_posix(),
            "csv_abs": str(csv_path.resolve()),
            "n_rows": len(t["rows"]),
            "n_cols": max((len(r) for r in t["rows"]), default=0),
        })
    write_jsonl(tdir / "index.jsonl", index)
    return index

def extract_figures_basic(
    pdf_path: Path,
    captions: List[Dict[str, Any]],
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    caps_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for c in captions:
        if c["kind"] == "figure":
            caps_by_page[int(c["page_id"])].append(c)
    out = []
    for p_idx, page in enumerate(doc, start=1):
        images = page.get_images(full=True)
        if not images:
            continue
        for j, img in enumerate(images, start=1):
            try:
                xref = img[0]
                info = page.get_image_rects(xref)
                if not info:
                    continue
                rect = info[0]
                nearby = None
                best = 1e9
                for cap in caps_by_page.get(p_idx, []):
                    gap = abs(rect.y0 - cap["bbox"][3])
                    if gap < best:
                        best = gap
                        nearby = cap
                if nearby and best < 160:
                    out.append({
                        "id": stable_id(f"fig:{pdf_path.name}:{p_idx}:{xref}", "fig_"),
                        "page_id": p_idx,
                        "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "caption": nearby["text"],
                        "xref": xref,
                        "confidence": 0.7,
                    })
            except Exception:
                continue
    doc.close()
    return out

def save_figures(pdf_path: Path, out_dir: Path, figures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fdir = out_dir / "figures"
    ensure_dir(fdir)
    doc = fitz.open(str(pdf_path))
    index = []
    for i, fig in enumerate(figures, start=1):
        try:
            pix = fitz.Pixmap(doc, int(fig["xref"]))
            if pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            out_path = fdir / f"figure_{fig['page_id']}_{i}.png"
            pix.save(str(out_path))
            index.append({
                "id": fig["id"],
                "page_id": fig["page_id"],
                "bbox": fig["bbox"],
                "caption": fig.get("caption"),
                "confidence": fig.get("confidence", 0.0),
                "image_path": out_path.relative_to(out_dir).as_posix(),
                "image_path_abs": str(out_path.resolve()),
            })
        except Exception:
            continue
    doc.close()
    write_jsonl(fdir / "index.jsonl", index)
    return index

# Structured extraction helpers
def structured_from_tables(profile: Dict[str, Any], table_index: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"recommendations": [], "targets": [], "monitoring": [], "treatment": []}
    for row in table_index:
        cap = norm(row.get("caption") or "")
        low = cap.lower()
        base = {"id": row["id"], "page_id": row["page_id"], "source_table_id": row["id"], "caption": cap, "csv": row["csv"]}
        if any(k in low for k in ["target", "hba1c", "glycemic"]):
            out["targets"].append(base)
        if any(k in low for k in ["monitor", "screen", "follow-up"]):
            out["monitoring"].append(base)
        if any(k in low for k in ["treatment", "therapy", "pharmacologic", "insulin"]):
            out["treatment"].append(base)
        if profile["organization"] == "NICE" and re.match(r"^\d+(?:\.\d+)+", low):
            out["recommendations"].append(base)
    return out

def extract_nice_recommendations(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for sec in sections:
        for line in sec.get("text", "").splitlines():
            line = norm(line)
            m = NICE_RECO_RE.match(line)
            if m:
                key = (m.group(1), line)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "id": stable_id(f"nice-reco:{sec['id']}:{m.group(1)}", "reco_"),
                    "section_id": sec["id"],
                    "label": m.group(1),
                    "text": line,
                    "page_start": sec["page_start"],
                    "source": "NICE",
                })
    return out

def extract_recommendation_blocks(profile: Dict[str, Any], sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract recommendation-like lines from non-NICE sections."""
    out = []
    seen = set()
    for sec in sections:
        title_low = sec.get("title", "").lower()
        text = sec.get("text", "")
        if profile["organization"] == "NICE":
            continue

        if any(k in title_low for k in ["recommendation", "key messages"]):
            lines = [norm(x) for x in text.splitlines() if norm(x)]
            for i, line in enumerate(lines, start=1):
                if len(line.split()) < 4:
                    continue
                # Check common recommendation markers.
                if not (
                    BULLETISH_RE.search(line)
                    or line[:1].isdigit()
                    or line.lower().startswith(("recommend", "use ", "consider ", "offer ", "do not "))
                ):
                    continue
                key = (sec["id"], line)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "id": stable_id(f"reco:{sec['id']}:{i}", "reco_"),
                    "section_id": sec["id"],
                    "text": line,
                    "page_start": sec["page_start"],
                    "source": profile["organization"],
                })

        # Handle numbered Diabetes Canada recommendations.

        if profile["organization"] == "Diabetes Canada":
            lines = [norm(x) for x in text.splitlines() if norm(x)]
            for i, line in enumerate(lines, start=1):
                m = CANADA_RECO_RE.match(line)
                if not m:
                    continue
                if len(line.split()) < 5:
                    continue
                key = (sec["id"], m.group(1), line[:60])
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "id": stable_id(f"canada-reco:{sec['id']}:{m.group(1)}:{i}", "reco_"),
                    "section_id": sec["id"],
                    "label": m.group(1),
                    "text": line,
                    "page_start": sec["page_start"],
                    "source": "Diabetes Canada",
                })
    return out

# ADA recommendation extraction.
# ADA recommendation boxes often arrive as single text blocks.

# Read them before section text is merged.
def extract_ada_recommendation_blocks(
    blocks: List[Dict[str, Any]],
    doc_key: str,
) -> List[Dict[str, Any]]:
    """Extract labelled ADA recommendations from raw PDF blocks."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for b in blocks:
        text = norm(b.get("text", ""))
        m = ADA_RECO_BLOCK_RE.match(text)
        if not m:
            continue
        label = m.group(1)
        if label in seen:
            continue
        seen.add(label)
        out.append({
            "id": stable_id(f"ada-reco:{doc_key}:{label}", "reco_"),
            "section_id": None,
            "label": label,
            "text": text,
            "page_start": b["page_id"],
            "source": "ADA",
        })
    return out

# Read IDF care-level recommendations.
# IDF recommendations use care-level labels instead of numbered lists.

def extract_idf_recommendations(
    blocks: List[Dict[str, Any]],
    doc_key: str,
) -> List[Dict[str, Any]]:
    """Extract IDF care-level recommendations from raw PDF blocks."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for b in blocks:
        text = norm(b.get("text", ""))
        m = IDF_CARE_RE.match(text)
        if not m:
            continue
        label = m.group(1).title()
        # Allow the same care label to appear on different pages.
        key = (b["page_id"], label, text[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "id": stable_id(f"idf-reco:{doc_key}:{b['page_id']}:{label}", "reco_"),
            "section_id": None,
            "label": label,
            "text": text,
            "page_start": b["page_id"],
            "source": "IDF",
        })
    return out

# Diabetes Canada recommendations are read directly from numbered text blocks.

_CANADA_REF_RE = re.compile(
    r"\b(et al\.|doi|diabetes care|can j diabetes|j clin|lancet|N Engl|"
    r"BMJ|JAMA|Ann Intern|Diabetes Metab|Endocr Rev)\b", re.I
)

def extract_canada_recommendation_blocks(
    blocks: List[Dict[str, Any]], doc_key: str
) -> List[Dict[str, Any]]:
    """Extract numbered Diabetes Canada recommendations from raw PDF blocks."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for b in blocks:
        text = norm(b.get("text", ""))
        if not CANADA_RECO_RE.match(text):
            continue
        if _CANADA_REF_RE.search(text):   # Skip reference-list entries.
            continue
        if len(text.split()) < 6:          # Skip short numeric fragments.
            continue
        key = text[:80]
        if key in seen:
            continue
        seen.add(key)
        m = CANADA_RECO_RE.match(text)
        out.append({
            "id": stable_id(f"canada-reco:{doc_key}:{b['page_id']}:{m.group(1)}", "reco_"),
            "section_id": "",
            "label": m.group(1) if m else "",
            "text": text,
            "page_start": int(b.get("page_id", 0)),
            "source": "Diabetes Canada",
        })
    return out

# Flow edge seeding
def seed_flow_edges(
    profile: Dict[str, Any],
    sections: List[Dict[str, Any]],
    tables: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    edges = []
    for sec in sections:
        low = sec["title"].lower()
        if any(k in low for k in ["screen", "diagnos"]):
            edges.append({"id": stable_id(f"edge:{sec['id']}:screening", "edge_"), "src": sec["id"], "dst_type": "concept", "dst": "screening", "relation": "about"})
        if any(k in low for k in ["target", "glycemic"]):
            edges.append({"id": stable_id(f"edge:{sec['id']}:glycemic", "edge_"), "src": sec["id"], "dst_type": "concept", "dst": "glycemic_target", "relation": "about"})
        if any(k in low for k in ["treatment", "therapy", "pharmacologic"]):
            edges.append({"id": stable_id(f"edge:{sec['id']}:treatment", "edge_"), "src": sec["id"], "dst_type": "concept", "dst": "treatment", "relation": "about"})
    return edges

# PDF collection
def collect_pdfs(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.pdf"))
    raise FileNotFoundError(f"No PDFs found at {input_path}")

def default_input_path() -> Path:
    for p in DEFAULT_INPUT_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_INPUT_CANDIDATES[0]

def default_output_path() -> Path:
    for p in DEFAULT_OUTPUT_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_OUTPUT_CANDIDATES[0]

# Per-PDF processing
def process_one_pdf(
    pdf_path: Path,
    out_root: Path,
    args: Args,
) -> Tuple[bool, Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    doc_key = slugify(pdf_path.stem)
    outputs: Dict[str, List[Dict[str, Any]]] = {}
    try:
        parsed = parse_pdf_structural(pdf_path)
        meta = infer_guideline_source(pdf_path, parsed["probe_text"])
        profile = get_profile(meta["organization"])
        print(f"[start] {doc_key} org={meta['organization']} year={meta.get('year_hint')} file={pdf_path.name}")
        docling = {"ok": False, "engine": "none", "raw_text": ""}
        if not args.no_docling:
            docling = try_docling(pdf_path)
        if not docling.get("ok"):
            print(f"[warn] docling fallback: {doc_key}")
        elif docling.get("raw_text"):
            docling_path = out_dir / "docling.md"
            docling_path.write_text(docling["raw_text"], encoding="utf-8")
        pages = parsed["pages"]
        blocks_by_page = page_blocks_map(parsed["blocks"])
        pages_out = []
        ordered_blocks: List[Dict[str, Any]] = []
        page_types: Dict[int, str] = {}
        headings: List[Dict[str, Any]] = []
        body_fonts = [b["avg_font_size"] for b in parsed["blocks"] if b.get("avg_font_size")]
        body_font = Counter(round(f, 1) for f in body_fonts).most_common(1)[0][0] if body_fonts else 10.0
        for p in pages:
            pid = int(p["page_id"])
            pblocks = blocks_by_page.get(pid, [])
            cols = detect_columns(pblocks, float(p["width"]))
            ordered = order_blocks(pblocks, cols, float(p["width"]))
            ptype = classify_page(ordered, profile, page_idx=pid)
            page_types[pid] = ptype
            for b in ordered:
                bb = dict(b)
                bb["column_count"] = cols
                bb["page_type"] = ptype
                ordered_blocks.append(bb)
            pages_out.append({**p, "column_count": cols, "page_type": ptype})
            headings.extend(recover_headings(ordered, body_font, profile, ptype))
        captions = detect_captions(ordered_blocks, profile)
        tables = extract_tables_region_first(pdf_path, pages_out, ordered_blocks, captions, page_types, args.tables_lite, profile)
        figures = [] if args.no_figures else extract_figures_basic(pdf_path, captions, pages_out)
        sections = build_sections(pages_out, page_blocks_map(ordered_blocks), headings)
        chunks = make_chunks(sections, doc_key)

        # Extract recommendations using the rules for each publisher.
        nice_recos: List[Dict[str, Any]] = []
        ada_recos: List[Dict[str, Any]] = []
        idf_recos: List[Dict[str, Any]] = []
        canada_recos: List[Dict[str, Any]] = []
        generic_recos: List[Dict[str, Any]] = []

        if profile["organization"] == "NICE":
            nice_recos = extract_nice_recommendations(sections)
        elif profile["organization"] == "ADA":
            # Read ADA recommendations before section text is merged.
            ada_recos = extract_ada_recommendation_blocks(ordered_blocks, doc_key)
        elif profile["organization"] == "IDF":
            # Read IDF care-level recommendations.
            idf_recos = extract_idf_recommendations(ordered_blocks, doc_key)
        elif profile["organization"] == "Diabetes Canada":
            # Read numbered Diabetes Canada recommendations from raw blocks.
            canada_recos = extract_canada_recommendation_blocks(ordered_blocks, doc_key)

        # Use the generic extractor when no publisher-specific result is available.

        generic_recos = extract_recommendation_blocks(profile, sections)

        # Prefer publisher-specific recommendations over the generic result.
        reco_rows: List[Dict[str, Any]] = nice_recos or ada_recos or idf_recos or canada_recos or generic_recos
        # -------------------------------------------------------------------

        out_dir = out_root / doc_key
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ensure_dir(out_dir)
        write_jsonl(out_dir / "pages.jsonl", pages_out)
        write_jsonl(out_dir / "blocks.jsonl", ordered_blocks)
        write_jsonl(out_dir / "captions.jsonl", captions)
        write_jsonl(out_dir / "headings.jsonl", headings)
        write_jsonl(out_dir / "sections.jsonl", sections)
        write_jsonl(out_dir / "chunks.jsonl", chunks)
        if reco_rows:
            write_jsonl(out_dir / "recommendations.jsonl", reco_rows)
        table_index = save_tables(out_dir, tables)
        figure_index = [] if args.no_figures else save_figures(pdf_path, out_dir, figures)
        if args.no_figures:
            print(f"[warn] figures disabled: {doc_key}")
        if not captions:
            print(f"[warn] no captions: {doc_key}")
        if not table_index:
            print(f"[warn] no tables: {doc_key}")
        if not figure_index and not args.no_figures:
            print(f"[warn] no figures: {doc_key}")
        structured = {} if args.no_structured else structured_from_tables(profile, table_index)
        if not args.no_structured:
            for key, rows in structured.items():
                write_jsonl(out_dir / f"{key}.jsonl", rows)
            if reco_rows:
                write_jsonl(out_dir / "recommendations.jsonl", reco_rows)
        flow_edges = [] if args.no_flow else seed_flow_edges(profile, sections, table_index, figure_index)
        if not args.no_flow:
            write_jsonl(out_dir / "flow_edges.jsonl", flow_edges)
        doc_record = {
            "doc_id": doc_key,
            "source_file": pdf_path.name,
            "source_path": str(pdf_path.resolve()),
            "organization": meta["organization"],
            "geography": meta["geography"],
            "year_hint": meta.get("year_hint"),
            "inference_method": meta.get("inference_method"),
            "profile": profile["organization"],
            "docling_ok": docling.get("ok", False),
            "n_pages": len(pages_out),
            "n_blocks": len(ordered_blocks),
            "n_headings": len(headings),
            "n_sections": len(sections),
            "n_chunks": len(chunks),
            "n_captions": len(captions),
            "n_tables": len(table_index),
            "n_figures": len(figure_index),
            "n_recommendations": len(reco_rows),
            "page_type_counts": dict(Counter(p["page_type"] for p in pages_out)),
            "notes": [],
            "status": "ok",
        }
        if not captions:
            doc_record["notes"].append("No captions found")
        if not table_index:
            doc_record["notes"].append("No tables found")
        if not figure_index:
            doc_record["notes"].append("No figures found")
        if not reco_rows:
            doc_record["notes"].append("No recommendations found")
        if not docling.get("ok"):
            doc_record["notes"].append("Docling failed; fell back to PyMuPDF/pdfplumber")
        write_json(out_dir / "meta.json", doc_record)
        outputs = {
            "pages": pages_out,
            "blocks": ordered_blocks,
            "sections": sections,
            "chunks": chunks,
            "tables": table_index,
            "figures": figure_index,
            "headings": headings,
            "captions": captions,
            "recommendations": reco_rows,
        }
        print(f"[ok] {doc_key} pages={len(pages_out)} tables={len(table_index)} recos={len(reco_rows)} figures={len(figure_index)}")
        return True, doc_record, outputs
    except Exception as e:
        print(f"[error] {doc_key} {e}")
        traceback.print_exc()
        doc_record = {
            "doc_id": doc_key,
            "source_file": pdf_path.name,
            "source_path": str(pdf_path.resolve()),
            "status": "failed",
            "error": str(e),
            "notes": ["Processing failed"],
        }
        return False, doc_record, outputs

# Corpus assembly
def write_corpus(
    out_root: Path,
    records: List[Dict[str, Any]],
    all_outputs: Dict[str, List[Dict[str, Any]]],
) -> None:
    write_jsonl(out_root / "documents.jsonl", records)
    for key, rows in all_outputs.items():
        write_jsonl(out_root / f"corpus_{key}.jsonl", rows)
    org_counts = Counter(r.get("organization") for r in records if r.get("status") == "ok")
    corpus_meta = {
        "n_documents": len(records),
        "n_succeeded": sum(1 for r in records if r.get("status") == "ok"),
        "n_failed": sum(1 for r in records if r.get("status") != "ok"),
        "docs_by_organization": dict(org_counts),
        "totals": {
            "pages": sum(int(r.get("n_pages", 0) or 0) for r in records),
            "tables": sum(int(r.get("n_tables", 0) or 0) for r in records),
            "figures": sum(int(r.get("n_figures", 0) or 0) for r in records),
            "sections": sum(int(r.get("n_sections", 0) or 0) for r in records),
            "recommendations": sum(int(r.get("n_recommendations", 0) or 0) for r in records),
        },
        "documents": [
            {
                "doc_id": r.get("doc_id"),
                "source_file": r.get("source_file"),
                "organization": r.get("organization"),
                "profile": r.get("profile"),
                "n_pages": r.get("n_pages", 0),
                "n_tables": r.get("n_tables", 0),
                "n_figures": r.get("n_figures", 0),
                "n_sections": r.get("n_sections", 0),
                "n_recommendations": r.get("n_recommendations", 0),
                "status": r.get("status"),
            }
            for r in records
        ],
    }
    write_json(out_root / "corpus_meta.json", corpus_meta)

# CLI
def parse_args(argv: Optional[Sequence[str]] = None) -> Args:
    ap = argparse.ArgumentParser(description="Hybrid guideline PDF parser v5 — patched")
    ap.add_argument("input_path", nargs="?", default=str(default_input_path()))
    ap.add_argument("output_path", nargs="?", default=str(default_output_path()))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-structured", action="store_true")
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--tables-lite", action="store_true")
    ap.add_argument("--no-docling", action="store_true")
    ns = ap.parse_args(argv)
    return Args(
        input_path=Path(ns.input_path),
        output_path=Path(ns.output_path),
        no_figures=ns.no_figures,
        no_structured=ns.no_structured,
        no_flow=ns.no_flow,
        tables_lite=ns.tables_lite,
        no_docling=ns.no_docling,
    )

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_dir(args.output_path)
    pdfs = collect_pdfs(args.input_path)
    records: List[Dict[str, Any]] = []
    all_outputs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    succeeded = 0
    failed = 0
    for pdf in pdfs:
        ok, rec, outputs = process_one_pdf(pdf, args.output_path, args)
        records.append(rec)
        if ok:
            succeeded += 1
            for key, rows in outputs.items():
                for row in rows:
                    rr = dict(row)
                    rr["doc_id"] = rec["doc_id"]
                    rr["source_file"] = rec["source_file"]
                    all_outputs[key].append(rr)
        else:
            failed += 1
    write_corpus(args.output_path, records, all_outputs)
    print("\n[done] batch ingest complete")
    print(f"  input: {args.input_path}")
    print(f"  out:   {args.output_path}")
    print(f"  docs:  {succeeded} succeeded")
    print(f"  errs:  {failed} failed")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
