"""Merge, normalise, and deduplicate the extracted graph data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_here = Path(__file__).resolve().parent
if (_here / "_helpers.py").exists():
    sys.path.insert(0, str(_here))

from _helpers import ensure_dirs, OUTPUT_DIR, read_jsonl, write_jsonl, stable_id

# CLI
def parse_args():
    ap = argparse.ArgumentParser(description="Normalise and deduplicate KG entities v1.0")
    ap.add_argument("--min-confidence", type=float, default=0.45,
                    help="Minimum confidence for heuristic relations (default 0.45)")
    ap.add_argument("--keep-heuristic-contraindications", action="store_true", default=False,
                    help="Keep heuristic CONTRAINDICATED_IN relations (noisy — off by default)")
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap.parse_args()


# Entity types shared across guidelines.
GLOBAL_TYPES = {"Drug", "ClinicalCondition", "Intervention", "Test", "Condition"}

# Entity types kept separate for each guideline.
ORG_SCOPED_TYPES = {
    "GlycaemicTarget", "Recommendation", "Table", "Guideline",
    "Organisation", "MonitoringProtocol", "TreatmentRecommendation",
}

# Relation types that are not part of the graph schema.
SPURIOUS_RELATION_TYPES = {"RESEARCHING", "INTERVENTION"}

# Name normalisation.
def norm(s: Optional[str]) -> str:
    import re
    return re.sub(r"\s+", " ", (s or "").strip())

# Common drug-name aliases.
DRUG_ALIASES: Dict[str, str] = {
    # GLP-1
    "glp-1 ra":                         "GLP-1 receptor agonist",
    "glp1 ra":                          "GLP-1 receptor agonist",
    "glp-1 receptor agonists":          "GLP-1 receptor agonist",
    "glucagon-like peptide 1 receptor agonist (glp-1 ra)": "GLP-1 receptor agonist",
    "glucagon-like peptide-1 receptor agonist": "GLP-1 receptor agonist",
    # SGLT2
    "sglt2 inhibitor":                  "SGLT2 inhibitor",
    "sglt-2 inhibitor":                 "SGLT2 inhibitor",
    "sglt-2 inhibitors":                "SGLT2 inhibitor",
    "sglt2 inhibitors":                 "SGLT2 inhibitor",
    "sglt2i":                           "SGLT2 inhibitor",
    "sodium-glucose cotransporter 2 inhibitor": "SGLT2 inhibitor",
    "sodium–glucose cotransporter 2 inhibitor": "SGLT2 inhibitor",
    # DPP-4
    "dpp4 inhibitor":                   "DPP-4 inhibitor",
    "dpp-4 inhibitors":                 "DPP-4 inhibitor",
    # ACE
    "ace inhibitors":                   "ACE inhibitor",
    "acei":                             "ACE inhibitor",
    "angiotensin-converting enzyme inhibitor": "ACE inhibitor",
    "angiotensin converting enzyme inhibitor": "ACE inhibitor",
    # ARB
    "angiotensin receptor blocker":     "ARB",
    "arbs":                             "ARB",
    "arb therapy":                      "ARB",
    # Sulfonylurea
    "sulphonylurea":                    "sulfonylurea",
    "sulfonyl urea":                    "sulfonylurea",
    "sulfonylureas":                    "sulfonylurea",
    # Other
    "thiazolidinedione":                "pioglitazone",
    "insulin therapy":                  "insulin",
    "basal insulin":                    "insulin",
    "statins":                          "statin",
    "statin therapy":                   "statin",
}

# Test names that should not be treated as drugs.
MISCLASSIFIED_AS_DRUG = {
    "a1c", "hba1c", "hba1c level", "blood glucose", "glucose",
    "fasting plasma glucose", "fpg", "cgm", "smbg",
}

CONDITION_ALIASES: Dict[str, str] = {
    "ckd":                              "Chronic Kidney Disease",
    "chronic kidney disease":           "Chronic Kidney Disease",
    "renal disease":                    "Chronic Kidney Disease",
    "heart failure":                    "Heart Failure",
    "cardiac failure":                  "Heart Failure",
    "hf":                              "Heart Failure",
    "ascvd":                            "Atherosclerotic Cardiovascular Disease",
    "cardiovascular disease":           "Atherosclerotic Cardiovascular Disease",
    "cvd":                              "Atherosclerotic Cardiovascular Disease",
    "atherosclerotic cardiovascular disease (ascvd)": "Atherosclerotic Cardiovascular Disease",
    "nafld":                            "Non-alcoholic Fatty Liver Disease",
    "nash":                             "Non-alcoholic Fatty Liver Disease",
    "dka":                              "Diabetic Ketoacidosis",
    "pad":                              "Peripheral Artery Disease",
    "hypoglycemia":                     "Hypoglycaemia",
    "hypoglycaemia":                    "Hypoglycaemia",
    "obesity":                          "Obesity",
    "overweight and obesity":           "Obesity",
}

def canonical_drug_name(name: str) -> str:
    return DRUG_ALIASES.get(name.lower().strip(), norm(name))

def canonical_condition_name(name: str) -> str:
    return CONDITION_ALIASES.get(name.lower().strip(), norm(name))

def canonical_name(etype: str, name: str) -> str:
    if etype == "Drug":
        return canonical_drug_name(name)
    if etype == "ClinicalCondition":
        return canonical_condition_name(name)
    return norm(name)


# Entity and relation cleanup.
def dedup_global_entities(
    all_entities: List[Dict],
    verbose: bool,
) -> Tuple[Dict[str, str], List[Dict]]:
    """Merge shared entities while keeping guideline-specific entities separate."""
    id_map: Dict[str, str] = {}          # old_id -> canonical_id
    global_pool: Dict[str, Dict] = {}    # canonical_key -> entity
    org_scoped_pool: Dict[str, Dict] = {}

    for e in all_entities:
        etype = e.get("type", "")
        eid   = e.get("id", "")
        name  = canonical_name(etype, e.get("name", ""))

        # Drop test names that were labelled as drugs.
        if etype == "Drug" and name.lower() in MISCLASSIFIED_AS_DRUG:
            continue

        if etype in GLOBAL_TYPES:
            # Merge shared entities by type and canonical name.
            canon_key = f"{etype}::{name.lower()}"
            if canon_key not in global_pool:
                global_id = stable_id(etype, name, prefix=f"{etype[:3].lower()}_")
                merged = {
                    "type":          etype,
                    "id":            global_id,
                    "name":          name,
                    "organization":  "global",
                    "mentioned_by":  [],
                }
                # Keep extra fields from the first matching entity.
                for k, v in e.items():
                    if k not in merged and v is not None:
                        merged[k] = v
                global_pool[canon_key] = merged

            global_ent = global_pool[canon_key]
            # Record which guidelines mention the entity.
            org = e.get("organization", "")
            if org and org != "global" and org not in global_ent["mentioned_by"]:
                global_ent["mentioned_by"].append(org)

            id_map[eid] = global_ent["id"]

        else:
            # Keep guideline-specific entities unchanged.
            org_scoped_pool[eid] = e
            id_map[eid] = eid

    if verbose:
        print(f"  Global entities after dedup: {len(global_pool)}")
        print(f"  Org-scoped entities kept:    {len(org_scoped_pool)}")

    return id_map, list(global_pool.values()) + list(org_scoped_pool.values())


def rewire_relations(
    all_relations: List[Dict],
    id_map: Dict[str, str],
    min_confidence: float,
    drop_heuristic_contras: bool,
    verbose: bool,
) -> List[Dict]:
    """Rewire relations to canonical IDs and remove duplicates or weak edges."""
    out: List[Dict] = []
    seen: Set[Tuple] = set()
    dropped = 0

    for r in all_relations:
        src = id_map.get(r.get("src", ""), r.get("src", ""))
        dst = id_map.get(r.get("dst", ""), r.get("dst", ""))
        rtype = r.get("type", "")

        if not src or not dst or not rtype:
            dropped += 1
            continue

        # Drop relation types outside the graph schema.
        if rtype in SPURIOUS_RELATION_TYPES:
            dropped += 1
            continue

        # Drop self-referencing relations.
        if src == dst:
            dropped += 1
            continue

        # Drop weak heuristic relations.
        conf = float(r.get("confidence", 1.0))
        method = r.get("extract_method", "")
        if method == "heuristic" and conf < min_confidence:
            dropped += 1
            continue

        # Drop heuristic contraindications unless explicitly requested.
        if drop_heuristic_contras and rtype == "CONTRAINDICATED_IN" and method == "heuristic":
            dropped += 1
            continue

        key = (src, rtype, dst)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)

        rr = dict(r)
        rr["src"] = src
        rr["dst"] = dst
        out.append(rr)

    if verbose:
        print(f"  Relations kept: {len(out)} | dropped: {dropped}")

    return out


def fix_condition_mentioned_by(
    entities: List[Dict],
    relations: List[Dict],
) -> None:
    """Fill guideline mentions for shared condition nodes from their relations."""
    eid_map = {e['id']: e for e in entities}
    for r in relations:
        if r.get('type') not in ('HAS_TARGET', 'HAS_RECOMMENDATION', 'RECOMMENDS'):
            continue
        src_ent = eid_map.get(r.get('src', ''))
        if not src_ent or src_ent.get('type') != 'Condition':
            continue
        org = r.get('organization', '')
        if not org:
            continue
        mb = src_ent.setdefault('mentioned_by', [])
        if org not in mb:
            mb.append(org)


def add_mentioned_in_edges(
    entities: List[Dict],
    guideline_entities: List[Dict],
) -> List[Dict]:
    """Link each shared entity to the guidelines that mention it."""
    gl_by_org: Dict[str, str] = {}
    for gl in guideline_entities:
        if gl.get("type") == "Guideline":
            org = gl.get("organization", "")
            if org not in gl_by_org:
                gl_by_org[org] = gl["id"]

    edges = []
    seen: Set[Tuple] = set()
    for e in entities:
        if e.get("organization") != "global":
            continue
        for org in e.get("mentioned_by", []):
            gl_id = gl_by_org.get(org)
            if not gl_id:
                continue
            key = (e["id"], "MENTIONED_IN", gl_id)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"src": e["id"], "type": "MENTIONED_IN", "dst": gl_id,
                          "organization": org})
    return edges


# Print graph counts.
def print_summary(entities: List[Dict], relations: List[Dict]):
    from collections import Counter
    etype_c = Counter(e.get("type", "?") for e in entities)
    rtype_c = Counter(r.get("type", "?") for r in relations)
    org_c   = Counter(e.get("organization", "?") for e in entities)

    print(f"\nFinal entity count: {len(entities)}")
    print("By type:")
    for k, v in sorted(etype_c.items(), key=lambda x: -x[1]):
        print(f"  {k:<30} {v:>4}")

    print(f"\nFinal relation count: {len(relations)}")
    print("By type:")
    for k, v in sorted(rtype_c.items(), key=lambda x: -x[1]):
        print(f"  {k:<30} {v:>4}")

    print("\nEntities by scope:")
    global_e = sum(1 for e in entities if e.get("organization") == "global")
    org_e    = len(entities) - global_e
    print(f"  Global (shared across guidelines): {global_e}")
    print(f"  Org-scoped (guideline-specific):   {org_e}")


# Main pipeline.
def main():
    args = parse_args()
    ensure_dirs()
    out_dir = OUTPUT_DIR / "kg-jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rule-based and contextual extraction outputs.
    ent04_path = out_dir / "entities.jsonl"
    ent05_path = out_dir / "llm_entities.jsonl"
    rel04_path = out_dir / "relations.jsonl"
    rel05_path = out_dir / "llm_relations.jsonl"

    for p in [ent04_path, rel04_path]:
        if not p.exists():
            print(f"[error] Missing {p} — run script 04 first")
            sys.exit(1)

    ents04 = list(read_jsonl(ent04_path))
    rels04 = list(read_jsonl(rel04_path))
    ents05 = list(read_jsonl(ent05_path)) if ent05_path.exists() else []
    rels05 = list(read_jsonl(rel05_path)) if rel05_path.exists() else []

    print(f"[load] Script 04: {len(ents04)} entities, {len(rels04)} relations")
    print(f"[load] Script 05: {len(ents05)} entities, {len(rels05)} relations")

    all_entities  = ents04 + ents05
    all_relations = rels04 + rels05

    # Merge shared entities.
    print("\n[step 1] Deduplicating global entities (Drug, ClinicalCondition, Intervention)...")
    id_map, merged_entities = dedup_global_entities(all_entities, args.verbose)

    # Rewire relations and remove noise.
    print("[step 2] Rewiring relations to canonical IDs and filtering noise...")
    merged_relations = rewire_relations(
        all_relations, id_map,
        min_confidence=args.min_confidence,
        drop_heuristic_contras=not args.keep_heuristic_contraindications,
        verbose=args.verbose,
    )

    # Fill guideline mentions for seeded condition nodes.
    print("[step 3] Fixing Condition node mentioned_by from relations...")
    fix_condition_mentioned_by(merged_entities, merged_relations)

    # Add links from shared entities to their source guidelines.
    print("[step 4] Adding MENTIONED_IN cross-reference edges...")
    guideline_ents = [e for e in merged_entities if e.get("type") == "Guideline"]
    global_ents    = [e for e in merged_entities if e.get("organization") == "global"]
    mentioned_edges = add_mentioned_in_edges(global_ents, guideline_ents)
    merged_relations.extend(mentioned_edges)
    print(f"  MENTIONED_IN edges added: {len(mentioned_edges)}")

    # Save the final graph.
    final_ent_path = out_dir / "final_entities.jsonl"
    final_rel_path = out_dir / "final_relations.jsonl"
    write_jsonl(final_ent_path, merged_entities)
    write_jsonl(final_rel_path, merged_relations)

    print_summary(merged_entities, merged_relations)

    print(f"\n[wrote] {final_ent_path}")
    print(f"[wrote] {final_rel_path}")


if __name__ == "__main__":
    main()