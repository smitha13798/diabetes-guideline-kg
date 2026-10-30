"""Extract contextual clinical entities and relations from guideline text."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Shared helper setup.
_here = Path(__file__).resolve().parent
if (_here / "_helpers.py").exists():
    sys.path.insert(0, str(_here))
elif (_here.parent / "src" / "_helpers.py").exists():
    sys.path.insert(0, str(_here.parent / "src"))

from _helpers import (
    ensure_dirs, INTERIM_DIR, OUTPUT_DIR,
    read_jsonl, write_jsonl, short, stable_id,
)

# CLI
def parse_args():
    ap = argparse.ArgumentParser(description="LLM entity/relation extractor v3.0")
    ap.add_argument("--provider", choices=["openai", "ollama", "none"],
                    default=os.getenv("LLM_PROVIDER", "none"))
    ap.add_argument("--model",      default=os.getenv("LLM_MODEL", ""))
    ap.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--docs",       nargs="*", default=None,
                    help="Limit to specific doc_keys e.g. --docs ada_26 nice_t2")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Max chunks to process per document (useful for testing)")
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap.parse_args()


# Entity types extracted from clinical context.
LLM_ENTITY_TYPES = {
    "Drug": "A pharmaceutical agent, drug class, or medication "
            "(e.g. metformin, SGLT2 inhibitor, GLP-1 RA, insulin glargine)",
    "Population": "A patient group defined by age, condition, or clinical context "
                  "(e.g. adults with T2DM, elderly patients, people with CKD stage 3)",
    "ClinicalCondition": "A comorbidity, complication, or clinical state "
                         "(e.g. heart failure, CKD, retinopathy, hypoglycaemia)",
    "Intervention": "A non-drug clinical action or programme "
                    "(e.g. lifestyle modification, bariatric surgery, diabetes education)",
}

LLM_RELATION_TYPES = {
    "RECOMMENDED_FOR":    "Drug or Intervention recommended for a Population or ClinicalCondition",
    "CONTRAINDICATED_IN": "Drug contraindicated in a Population or ClinicalCondition",
    "FIRST_LINE_FOR":     "Drug is first-line treatment for a Population or ClinicalCondition",
    "SECOND_LINE_FOR":    "Drug is second-line treatment for a Population or ClinicalCondition",
    "MANAGES":            "Drug or Intervention manages a ClinicalCondition",
    "USED_WITH":          "Drug used in combination with another Drug",
    "REQUIRES_MONITORING":"Drug or condition requires a specific Test or monitoring protocol",
    "ASSOCIATED_WITH":    "ClinicalCondition associated with another ClinicalCondition",
}

# Prompts used for contextual extraction.
SYSTEM_PROMPT = """You are a clinical knowledge graph extractor specialising in diabetes guidelines.

Extract ONLY the following entity types from the TEXT:
{entity_type_descriptions}

Extract ONLY the following relation types:
{relation_type_descriptions}

RULES:
- Return STRICT JSON with keys: "entities" (list) and "relations" (list).
- Each entity needs: "id" (short unique string within this response), "type" (from the list above), "name" (canonical name), "organization" (which guideline this came from).
- Each relation needs: "type" (from list above), "from" (entity id), "to" (entity id).
- Drug names: use the most specific name mentioned. If a class is mentioned (e.g. SGLT2 inhibitor), create a class entity. If a specific drug is mentioned (e.g. empagliflozin), create both.
- Do NOT extract GlycaemicTarget, Recommendation, or numeric thresholds — those are handled separately.
- Do NOT invent relations not explicitly stated in the text.
- If nothing relevant is found, return: {{"entities": [], "relations": []}}
- Return ONLY JSON. No markdown fences. No commentary."""

USER_PROMPT = """GUIDELINE: {org} ({year})
PAGE: {page}

KNOWN FACTS ON THIS PAGE (already extracted — do not re-extract these):
{known_facts}

TEXT:
{chunk_text}

Return ONLY JSON."""


def build_system_prompt() -> str:
    ent_desc = "\n".join(
        f"  {name}: {desc}" for name, desc in LLM_ENTITY_TYPES.items()
    )
    rel_desc = "\n".join(
        f"  {name}: {desc}" for name, desc in LLM_RELATION_TYPES.items()
    )
    return SYSTEM_PROMPT.format(
        entity_type_descriptions=ent_desc,
        relation_type_descriptions=rel_desc,
    )


# Model backends.
def call_openai(model: str, system: str, user: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


def call_ollama(model: str, system: str, user: str, base_url: str) -> str:
    import requests
    base = re.sub(r"/api(?:/.*)?$", "", base_url.rstrip("/"), flags=re.I)
    payload = {
        "model":    model or "llama3",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "options":    {"temperature": 0.0, "num_predict": 1024},
        "keep_alive": "10m",
        "stream":     False,
    }
    try:
        r = requests.post(base + "/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "") or json.dumps(data)
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama error: {e}") from e


def parse_llm_response(raw: str) -> Tuple[List[Dict], List[Dict]]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S)
    m1, m2 = raw.find("{"), raw.rfind("}")
    if m1 != -1 and m2 != -1:
        raw = raw[m1:m2+1]
    try:
        data = json.loads(raw)
    except Exception:
        return [], []
    if not isinstance(data, dict):
        return [], []
    return (
        data.get("entities", []) if isinstance(data.get("entities"), list) else [],
        data.get("relations", []) if isinstance(data.get("relations"), list) else [],
    )


# Pattern-based fallback used when no model is selected.

# Common drug names and classes.
_DRUG_CLASSES = [
    ("SGLT2 inhibitor",       re.compile(r"\bSGLT.?2\s+inhibitor|\bSGLT.?2i\b|sodium.glucose\s+cotransporter", re.I)),
    ("GLP-1 receptor agonist",re.compile(r"\bGLP.1\s+(?:receptor\s+)?agonist|\bGLP.1\s*RA\b|glucagon.like\s+peptide.1", re.I)),
    ("DPP-4 inhibitor",       re.compile(r"\bDPP.4\s+inhibitor|\bgliptin\b", re.I)),
    ("metformin",             re.compile(r"\bmetformin\b", re.I)),
    ("insulin",               re.compile(r"\binsulin\b(?!\s+resistance|\s+sensitiv|\s+secret)", re.I)),
    ("sulfonylurea",          re.compile(r"\bsulfonylurea|\bglibenclamide|\bglipizide|\bglimepiride\b", re.I)),
    ("empagliflozin",         re.compile(r"\bempagliflozin\b", re.I)),
    ("dapagliflozin",         re.compile(r"\bdapagliflozin\b", re.I)),
    ("canagliflozin",         re.compile(r"\bcanagliflozin\b", re.I)),
    ("semaglutide",           re.compile(r"\bsemaglutide\b", re.I)),
    ("liraglutide",           re.compile(r"\bliraglutide\b", re.I)),
    ("dulaglutide",           re.compile(r"\bdulaglutide\b", re.I)),
    ("pioglitazone",          re.compile(r"\bpioglitazone|\bthiazolidinedione\b", re.I)),
    ("statin",                re.compile(r"\bstatin\b|\batorvastatin|\brosuvastatin\b", re.I)),
    ("ACE inhibitor",         re.compile(r"\bACE\s+inhibitor|\bACEi\b|\blisinopril|\benalapril\b", re.I)),
    ("ARB",                   re.compile(r"\bARB\b|angiotensin\s+receptor\s+blocker|\blosartan|\birbesartan\b", re.I)),
]

# Common clinical conditions.
_CONDITION_PATTERNS = [
    ("Heart Failure",         re.compile(r"\bheart\s+failure\b", re.I)),
    ("Chronic Kidney Disease",re.compile(r"\bCKD\b|\bchronic\s+kidney\s+disease\b", re.I)),
    ("Retinopathy",           re.compile(r"\bretinopathy\b", re.I)),
    ("Neuropathy",            re.compile(r"\bneuropathy\b", re.I)),
    ("Peripheral Artery Disease", re.compile(r"\bPAD\b|\bperipheral\s+artery\s+disease\b", re.I)),
    ("Hypertension",          re.compile(r"\bhypertension\b|\bhigh\s+blood\s+pressure\b", re.I)),
    ("Obesity",               re.compile(r"\bobesity\b|\bobese\b", re.I)),
    ("Hypoglycaemia",         re.compile(r"\bhypoglycae?mia\b|\blow\s+blood\s+(?:glucose|sugar)\b", re.I)),
    ("Atherosclerotic Cardiovascular Disease", re.compile(r"\bASCVD\b|\batherosclerotic\s+cardiovascular\b", re.I)),
    ("Non-alcoholic Fatty Liver Disease", re.compile(r"\bNAFLD\b|\bNASH\b|\bmetabolic.associated\s+(?:steatotic|fatty)\b", re.I)),
    ("Diabetic Ketoacidosis", re.compile(r"\bDKA\b|\bdiabetic\s+ketoacidosis\b", re.I)),
]

# Common non-drug interventions.
_INTERVENTION_PATTERNS = [
    ("Lifestyle Modification",re.compile(r"\blifestyle\s+(?:modification|intervention|change)\b", re.I)),
    ("Bariatric Surgery",     re.compile(r"\bbariatric\s+surgery|\bmetabolic\s+surgery\b", re.I)),
    ("Medical Nutrition Therapy", re.compile(r"\bmedical\s+nutrition\s+therapy|\bMNT\b", re.I)),
    ("Diabetes Self-Management Education", re.compile(r"\bDSME\b|\bdiabetes\s+(?:self.management\s+)?education\b", re.I)),
    ("Physical Activity",     re.compile(r"\bphysical\s+activity|\baerobic\s+exercise|\bresistance\s+training\b", re.I)),
]

# Words that suggest common clinical relations.
_FIRST_LINE = re.compile(r"\bfirst.line\b|\binitially\s+(?:recommend|prescrib|use)\b|\bpreferred\b", re.I)
_SECOND_LINE = re.compile(r"\bsecond.line\b|\bif\s+(?:not\s+tolerat|inadequate|insufficien)", re.I)
_CONTRAINDICATED = re.compile(r"\bcontraindicated?\b|\bshould\s+not\s+be\s+used\b|\bavoid\b", re.I)
_RECOMMENDED = re.compile(r"\brecommend|\bshould\s+(?:be\s+)?(?:used?|considered?|offered?)\b|\badvised?\b", re.I)
_MANAGES = re.compile(r"\bmanage[sd]?\b|\btreat[sed]?\b|\bimprove[sd]?\b|\breduce[sd]?\b", re.I)


def extract_heuristic(
    text: str, org: str, page: int
) -> Tuple[List[Dict], List[Dict]]:
    """Extract common entities and relations with simple text patterns."""
    ents: List[Dict] = []
    rels: List[Dict] = []
    eid_map: Dict[str, str] = {}

    def get_or_add(name: str, etype: str) -> str:
        key = f"{etype}::{name.lower()}"
        if key not in eid_map:
            eid = f"h_{abs(hash(key)) % 10**9}"
            eid_map[key] = eid
            ents.append({
                "id": eid, "type": etype, "name": name,
                "organization": org, "source_page": page,
                "extract_method": "heuristic", "confidence": 0.5,
            })
        return eid_map[key]

    # Find matching entities.
    drugs = []
    for name, pat in _DRUG_CLASSES:
        if pat.search(text):
            drugs.append(get_or_add(name, "Drug"))

    conditions = []
    for name, pat in _CONDITION_PATTERNS:
        if pat.search(text):
            conditions.append(get_or_add(name, "ClinicalCondition"))

    for name, pat in _INTERVENTION_PATTERNS:
        if pat.search(text):
            get_or_add(name, "Intervention")

    # Build relations from co-occurring entities and trigger words.
    for drug_id in drugs:
        drug_name = next((e["name"] for e in ents if e["id"] == drug_id), "")
        for cond_id in conditions:
            if _CONTRAINDICATED.search(text):
                rels.append({"type": "CONTRAINDICATED_IN", "from": drug_id,
                             "to": cond_id, "source_page": page,
                             "source_quote": short(text, 120),
                             "confidence": 0.45, "extract_method": "heuristic"})
            elif _FIRST_LINE.search(text):
                rels.append({"type": "FIRST_LINE_FOR", "from": drug_id,
                             "to": cond_id, "source_page": page,
                             "source_quote": short(text, 120),
                             "confidence": 0.5, "extract_method": "heuristic"})
            elif _SECOND_LINE.search(text):
                rels.append({"type": "SECOND_LINE_FOR", "from": drug_id,
                             "to": cond_id, "source_page": page,
                             "source_quote": short(text, 120),
                             "confidence": 0.45, "extract_method": "heuristic"})
            elif _RECOMMENDED.search(text):
                rels.append({"type": "RECOMMENDED_FOR", "from": drug_id,
                             "to": cond_id, "source_page": page,
                             "source_quote": short(text, 120),
                             "confidence": 0.45, "extract_method": "heuristic"})
            elif _MANAGES.search(text):
                rels.append({"type": "MANAGES", "from": drug_id,
                             "to": cond_id, "source_page": page,
                             "source_quote": short(text, 120),
                             "confidence": 0.4, "extract_method": "heuristic"})

    return ents, rels


# Match newly extracted entities to the existing graph.
def load_canonical(path: Path) -> Dict[str, Any]:
    """Load existing graph entities for name matching."""
    idx: Dict[str, Dict] = {}
    name_type_idx: Dict[Tuple[str, str], str] = {}  # (name_lower, type) -> id
    if not path.exists():
        return {"by_id": idx, "by_name_type": name_type_idx}
    for e in read_jsonl(path):
        idx[e["id"]] = e
        key = (e.get("name", "").lower().strip(), e.get("type", ""))
        if key[0]:
            name_type_idx[key] = e["id"]
    return {"by_id": idx, "by_name_type": name_type_idx}


def resolve_entity(e: Dict, canonical: Dict) -> Optional[str]:
    """Match a new entity to an existing entity by name and type."""
    name = e.get("name", "").lower().strip()
    etype = e.get("type", "")
    return canonical["by_name_type"].get((name, etype))


def merge_outputs(
    all_ents: List[Dict],
    all_rels: List[Dict],
    canonical: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    """Resolve extracted entities to graph IDs and remove duplicate relations."""
    local_to_global: Dict[str, str] = {}
    out_entities: Dict[str, Dict] = {}

    for e in all_ents:
        local_id = e.get("id", "")
        canonical_id = resolve_entity(e, canonical)
        if canonical_id:
            local_to_global[local_id] = canonical_id
            # Don't add to out_entities — it already exists in script 04 output
        else:
            global_id = stable_id(
                e.get("type", ""), e.get("organization", ""),
                e.get("name", ""),
                prefix="llm_",
            )
            local_to_global[local_id] = global_id
            if global_id not in out_entities:
                ee = dict(e)
                ee["id"] = global_id
                out_entities[global_id] = ee

    # Update relation endpoints to resolved entity IDs.
    out_relations: List[Dict] = []
    seen: set = set()
    for r in all_rels:
        src_local = r.get("from") or r.get("src")
        dst_local = r.get("to") or r.get("dst")
        rtype = r.get("type")
        if not src_local or not dst_local or not rtype:
            continue
        src_g = local_to_global.get(src_local)
        dst_g = local_to_global.get(dst_local)
        if not src_g or not dst_g:
            continue
        key = (src_g, rtype, dst_g)
        if key in seen:
            continue
        seen.add(key)
        rr = {"src": src_g, "type": rtype, "dst": dst_g}
        for k, v in r.items():
            if k not in ("from", "to", "src", "dst", "type"):
                rr[k] = v
        out_relations.append(rr)

    return list(out_entities.values()), out_relations


# Skip chunks that are unlikely to contain relevant clinical content.
_CLINICALLY_RICH = re.compile(
    r"\b(metformin|insulin|SGLT|GLP.1|DPP.4|sulfonylurea|empagliflozin|"
    r"semaglutide|dapagliflozin|liraglutide|pioglitazone|statin|"
    r"heart\s+failure|CKD|kidney\s+disease|retinopathy|neuropathy|"
    r"contraindicated|first.line|second.line|lifestyle|bariatric|"
    r"recommend[s]?\s+(?:use|initiat|start|prescrib|consider))\b",
    re.I,
)

def is_worth_processing(text: str) -> bool:
    """Return True when a chunk contains useful clinical signals."""
    return bool(_CLINICALLY_RICH.search(text)) and len(text.split()) >= 40


# Add rule-extracted facts from the same page to the prompt.
def build_known_facts(page: int, canonical: Dict) -> str:
    """Summarise rule-extracted facts from the current page."""
    lines = []
    for e in canonical["by_id"].values():
        if e.get("type") == "GlycaemicTarget" and e.get("page_id") == page:
            lines.append(
                f"TARGET: {e.get('metric')} {e.get('op','')} "
                f"{e.get('value','')} {e.get('units','')}"
            )
        if e.get("type") == "Recommendation" and e.get("page_start") == page:
            lines.append(f"RECO [{e.get('label','')}]: {short(e.get('text',''), 100)}")
    return "\n".join(lines[:20]) or "(none)"


# Main pipeline.
def main():
    args = parse_args()
    ensure_dirs()
    (OUTPUT_DIR / "kg-jsonl").mkdir(parents=True, exist_ok=True)

    # Load entities created by the rule-based extractor.
    canon_path = OUTPUT_DIR / "kg-jsonl" / "entities.jsonl"
    canonical = load_canonical(canon_path)
    if args.verbose:
        print(f"[load] canonical entities: {len(canonical['by_id'])}")

    # Load the ingested guideline list.
    docs_file = INTERIM_DIR / "documents.jsonl"
    if not docs_file.exists():
        print(f"[error] {docs_file} not found — run ingestion pipeline first")
        sys.exit(1)

    docs = [json.loads(l) for l in docs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.docs:
        docs = [d for d in docs if d.get("doc_id") in args.docs]

    print(f"[start] Processing {len(docs)} documents  provider={args.provider}")

    system_prompt = build_system_prompt()
    all_ents: List[Dict] = []
    all_rels: List[Dict] = []

    for doc_meta in docs:
        doc_id  = doc_meta.get("doc_id", "")
        org     = doc_meta.get("organization", "Unknown")
        year    = doc_meta.get("year_hint", "")
        doc_dir = INTERIM_DIR / doc_id
        chunks_path = doc_dir / "chunks.jsonl"

        if not chunks_path.exists():
            print(f"  [skip] {doc_id} — no chunks.jsonl")
            continue

        chunks = [c for c in read_jsonl(chunks_path) if is_worth_processing(
            c.get("text", c.get("content", ""))
        )]

        if args.max_chunks:
            chunks = chunks[:args.max_chunks]

        print(f"  [{org}] {doc_id}: {len(chunks)} chunks to process")

        doc_ents: List[Dict] = []
        doc_rels: List[Dict] = []

        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", chunk.get("content", "")).strip()
            page = int(chunk.get("page_start", chunk.get("page_id", 0)))

            if args.provider == "none":
                ents, rels = extract_heuristic(text, org, page)
            else:
                known = build_known_facts(page, canonical)
                user_prompt = USER_PROMPT.format(
                    org=org, year=year or "current",
                    page=page, known_facts=known,
                    chunk_text=text[:1800],
                )
                try:
                    if args.provider == "openai":
                        raw = call_openai(args.model, system_prompt, user_prompt)
                    else:
                        raw = call_ollama(
                            args.model, system_prompt, user_prompt, args.ollama_url
                        )
                    ents, rels = parse_llm_response(raw)

                    # Keep source information on model output.
                    for e in ents:
                        e.setdefault("organization", org)
                        e.setdefault("source_page", page)
                        e.setdefault("extract_method", "llm")
                        e.setdefault("confidence", 0.7)
                    for r in rels:
                        r.setdefault("source_page", page)
                        r.setdefault("extract_method", "llm")

                except RuntimeError as e:
                    print(f"    [error] chunk {i}: {e}")
                    continue

            doc_ents.extend(ents)
            doc_rels.extend(rels)

            if args.verbose and i % 20 == 0:
                print(f"    [{i}/{len(chunks)}] ents={len(doc_ents)} rels={len(doc_rels)}")

        all_ents.extend(doc_ents)
        all_rels.extend(doc_rels)
        print(f"    done: raw ents={len(doc_ents)} rels={len(doc_rels)}")

    # Resolve new entities against the existing graph.
    print(f"\n[merge] resolving {len(all_ents)} entities onto canonical graph...")
    merged_ents, merged_rels = merge_outputs(all_ents, all_rels, canonical)

    # Save extracted entities and relations.
    ent_path = OUTPUT_DIR / "kg-jsonl" / "llm_entities.jsonl"
    rel_path = OUTPUT_DIR / "kg-jsonl" / "llm_relations.jsonl"
    write_jsonl(ent_path, merged_ents)
    write_jsonl(rel_path, merged_rels)

    # Print a short summary.
    from collections import Counter
    etype_counts = Counter(e.get("type", "?") for e in merged_ents)
    rtype_counts = Counter(r.get("type", "?") for r in merged_rels)

    print(f"\n[done] LLM entities: {len(merged_ents)}  relations: {len(merged_rels)}")
    print("\nNew entity types:")
    for etype, cnt in sorted(etype_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<30} {cnt:>4}")
    print("\nNew relation types:")
    for rtype, cnt in sorted(rtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {rtype:<30} {cnt:>4}")
    print(f"\n[wrote] {ent_path}")
    print(f"[wrote] {rel_path}")


if __name__ == "__main__":
    main()
