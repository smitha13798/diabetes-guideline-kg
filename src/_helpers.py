import json, re, os, sys, hashlib
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional, Tuple

"""Shared file, ID, text, and unit-conversion helpers used by the pipeline."""

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
OUTPUT_DIR = ROOT / "outputs"


def ensure_dirs():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    (INTERIM_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (INTERIM_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (INTERIM_DIR / "docling").mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "kg-jsonl").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "cypher").mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: Path):
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("PyYAML not installed. Install using: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


#  v3.0: Config paths now point to versioned schema + patterns
def config_paths(version: str = "v3"):
    """
    version can be: 'v2', 'v3'
    """
    if version == "v2":
        return {
            "schema":   CONFIG_DIR / "schema.json",
            "patterns": CONFIG_DIR / "patterns.yml",
            "ontology": CONFIG_DIR / "ontology.yml",
        }
    else:
        return {
            "schema":   CONFIG_DIR / "schema.v3.json",
            "patterns": CONFIG_DIR / "patterns.v3.yml",
            "ontology": CONFIG_DIR / "ontology.v3.yml",
        }

def load_docling_json(path: Path) -> Dict[str, Any]:
    """Load a Docling PDF parse JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_bbox(bbox: List[float]) -> Tuple[float, float, float, float]:
    """Ensure bbox is (x0, y0, x1, y1) float tuple."""
    if not bbox:
        return (0, 0, 0, 0)
    return tuple(float(x) for x in bbox)

def bbox_center(b):
    (x0, y0, x1, y1) = normalize_bbox(b)
    return ((x0 + x1) / 2, (y0 + y1) / 2)

def bbox_distance(b1, b2):
    (cx1, cy1) = bbox_center(b1)
    (cx2, cy2) = bbox_center(b2)
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5

def short(text: str, n: int = 200):
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= n else t[:n-1] + "…"

def find_iter(pattern: str, text: str):
    """v3: Python-compatible groups only using (?P<name>)"""
    return list(re.finditer(pattern, text, flags=re.MULTILINE))


# Stable identifiers
def canonical_str(v: Any) -> str:
    """Normalize floats, lists, enums, units for ID hashing."""
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, list):
        return "_".join(sorted(canonical_str(x) for x in v))
    if v is None:
        return ""
    return str(v).strip().lower()


def stable_id_from_entity(entity_type: str, props: Dict[str, Any]) -> str:
    """
    v3.0: Deterministic ID from entity type and key properties.
    Example:
      GlycaemicTarget(metric='HbA1c', units='%', op='≥', value=7.0)
    """
    parts = [entity_type]
    for k in sorted(props.keys()):
        parts.append(k)
        parts.append(canonical_str(props[k]))

    base = "::".join(parts)
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{entity_type}_{h}"


def stable_id(*parts: str, prefix: str = "") -> str:
    s = "::".join([p or "" for p in parts])
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}{h}"


#  Unit conversions and normalization
def normalize_decimal(s: str) -> str:
    """Replace comma decimal separators with dot."""
    return s.replace(",", ".")

def mmolL_to_mgdl(val: float) -> float:
    return round(val * 18.0, 1)

def mgdl_to_mmolL(val: float) -> float:
    return round(val / 18.0, 2)


#  comparators normalization
COMPARATOR_MAP = {
    "<=": "≤",
    ">=": "≥",
    "=<": "≤",
    "=>": "≥"
}

def normalize_comparator(op: str) -> str:
    if not op:
        return ""
    op = op.strip()
    if op in COMPARATOR_MAP:
        return COMPARATOR_MAP[op]
    return op


#  BBOX Utilities (PyMuPDF-compatible legacy)
def nearest_by_y(bboxes: List[Tuple[float, float, float, float]], ref: Tuple[float, float, float, float]):
    ry0, ry1 = ref[1], ref[3]

    def dist(b):
        y0, y1 = b[1], b[3]
        return min(abs(y0 - ry1), abs(y1 - ry0), abs((y0 + y1)/2 - (ry0 + ry1)/2))

    return min(bboxes, key=dist) if bboxes else None

#  PDF path resolution

def default_pdf_path():
    p = DATA_DIR / "input" / "IDF.pdf"
    if not p.exists():
        p = ROOT / "IDF.pdf"
    return p
