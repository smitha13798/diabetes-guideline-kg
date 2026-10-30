"""Generate Cypher files and optionally load the final graph into Neo4j."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_here = Path(__file__).resolve().parent
if (_here / "_helpers.py").exists():
    sys.path.insert(0, str(_here))

from _helpers import OUTPUT_DIR, read_jsonl

KG_DIR    = OUTPUT_DIR / "kg-jsonl"
CYPHER_DIR = OUTPUT_DIR / "cypher"
CYPHER_DIR.mkdir(parents=True, exist_ok=True)

ENT_PATH = KG_DIR / "final_entities.jsonl"
REL_PATH = KG_DIR / "final_relations.jsonl"

# Number of records sent to Neo4j at a time.
BATCH_SIZE = 500


# CLI
def parse_args():
    ap = argparse.ArgumentParser(description="Load KG into Neo4j v3.0")
    ap.add_argument("--apply",    action="store_true",
                    help="Actually load into Neo4j (requires connection env vars)")
    ap.add_argument("--uri",      default=os.getenv("NEO4J_URI",      "bolt://localhost:7687"))
    ap.add_argument("--user",     default=os.getenv("NEO4J_USER",     "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--wipe",     action="store_true",
                    help="Delete all existing nodes/relations before loading (use with care)")
    return ap.parse_args()


# Format Python values for Cypher.
_PROP_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def prop_key(k: str) -> str:
    return k if _PROP_KEY_RE.match(k or "") else f"`{k}`"

def cypher_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, list):
        items = ", ".join(cypher_value(x) for x in v)
        return f"[{items}]"
    if isinstance(v, dict):
        return '"' + json.dumps(v, ensure_ascii=False).replace('"', '\\"') + '"'
    s = str(v)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


# Build Cypher statements.
def build_entity_cypher(entities: List[Dict]) -> List[str]:
    """Build Cypher statements for graph entities."""
    from collections import defaultdict
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for e in entities:
        label = e.get("type")
        if not label or not e.get("id"):
            continue
        by_label[label].append(e)

    lines: List[str] = []

    # Add one uniqueness constraint for each node type.
    lines.append("// === CONSTRAINTS ===")
    for label in sorted(by_label.keys()):
        lines.append(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) "
            f"REQUIRE n.id IS UNIQUE;"
        )
    lines.append("")

    # Create or update nodes.
    lines.append("// === NODES ===")
    for label, rows in sorted(by_label.items()):
        lines.append(f"// {label} ({len(rows)} nodes)")
        for e in rows:
            props = []
            for k, v in e.items():
                if k in ("type", "id") or k.startswith("_") or v is None:
                    continue
                props.append(f"n.{prop_key(k)} = {cypher_value(v)}")
            set_clause = (" SET " + ", ".join(props)) if props else ""
            lines.append(
                f'MERGE (n:`{label}` {{id: {cypher_value(e["id"])}}})' 
                f'{set_clause};'
            )
        lines.append("")

    return lines


def build_relation_cypher(relations: List[Dict]) -> List[str]:
    """Build Cypher statements for graph relations."""
    lines: List[str] = []
    lines.append("// === RELATIONS ===")

    for r in relations:
        rtype  = r.get("type", "")
        src_id = r.get("src", "")
        dst_id = r.get("dst", "")
        if not rtype or not src_id or not dst_id:
            continue

        props = []
        for k, v in r.items():
            if k in ("type", "src", "dst", "from", "to") or k.startswith("_") or v is None:
                continue
            props.append(f"rel.{prop_key(k)} = {cypher_value(v)}")
        set_clause = (" SET " + ", ".join(props)) if props else ""

        lines.append(
            f"MATCH (a {{id: {cypher_value(src_id)}}}), "
            f"(b {{id: {cypher_value(dst_id)}}}) "
            f"MERGE (a)-[rel:`{rtype}`]->(b){set_clause};"
        )

    return lines


# Load the graph directly into Neo4j in batches.
def apply_to_neo4j(
    entities: List[Dict],
    relations: List[Dict],
    uri: str, user: str, password: str,
    batch_size: int,
    wipe: bool,
):
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[error] neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        if wipe:
            print("[wipe] Deleting all existing nodes and relations...")
            session.run("MATCH (n) DETACH DELETE n")

        # Create node ID constraints.
        from collections import defaultdict
        by_label: Dict[str, List[Dict]] = defaultdict(list)
        for e in entities:
            label = e.get("type")
            if label and e.get("id"):
                by_label[label].append(e)

        for label in sorted(by_label.keys()):
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) "
                f"REQUIRE n.id IS UNIQUE"
            )
        print(f"  Constraints created for {len(by_label)} labels")

        # Load nodes in batches.
        total_ents = 0
        for label, rows in by_label.items():
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                # Keep node properties separate from the label and ID.
                batch_data = []
                for e in batch:
                    props = {k: v for k, v in e.items()
                             if k not in ("type", "id") and v is not None}
                    batch_data.append({"id": e["id"], "props": props})

                session.run(
                    f"UNWIND $batch AS row "
                    f"MERGE (n:`{label}` {{id: row.id}}) "
                    f"SET n += row.props",
                    batch=batch_data,
                )
                total_ents += len(batch)
        print(f"  Nodes ingested: {total_ents}")

        # Load relations in batches.
        total_rels = 0
        # Group relations by type before loading.
        by_rtype: Dict[str, List[Dict]] = defaultdict(list)
        for r in relations:
            rtype = r.get("type", "")
            src   = r.get("src", "")
            dst   = r.get("dst", "")
            if rtype and src and dst:
                by_rtype[rtype].append(r)

        for rtype, rlist in by_rtype.items():
            for i in range(0, len(rlist), batch_size):
                batch = rlist[i:i + batch_size]
                batch_data = []
                for r in batch:
                    props = {k: v for k, v in r.items()
                             if k not in ("type", "src", "dst", "from", "to")
                             and v is not None}
                    batch_data.append({
                        "src": r["src"], "dst": r["dst"], "props": props
                    })

                session.run(
                    f"UNWIND $batch AS row "
                    f"MATCH (a {{id: row.src}}), (b {{id: row.dst}}) "
                    f"MERGE (a)-[rel:`{rtype}`]->(b) "
                    f"SET rel += row.props",
                    batch=batch_data,
                )
                total_rels += len(batch)
        print(f"  Relations ingested: {total_rels}")

    driver.close()
    print("[ok] Neo4j ingestion complete")


# Main pipeline.
def main():
    args = parse_args()

    if not ENT_PATH.exists():
        print(f"[error] {ENT_PATH} not found — run script 06 first")
        sys.exit(1)

    entities  = list(read_jsonl(ENT_PATH))
    relations = list(read_jsonl(REL_PATH))

    print(f"[load] {len(entities)} entities, {len(relations)} relations")

    # Generate Cypher files for the final graph.
    ent_lines = build_entity_cypher(entities)
    rel_lines = build_relation_cypher(relations)

    ent_cypher_path = CYPHER_DIR / "load_entities.cypher"
    rel_cypher_path = CYPHER_DIR / "load_relations.cypher"

    ent_cypher_path.write_text("\n".join(ent_lines), encoding="utf-8")
    rel_cypher_path.write_text("\n".join(rel_lines), encoding="utf-8")

    print(f"[wrote] {ent_cypher_path}  ({len(ent_lines)} lines)")
    print(f"[wrote] {rel_cypher_path}  ({len(rel_lines)} lines)")

    if args.apply:
        if not args.password:
            print("[error] --password required for --apply (or set NEO4J_PASSWORD env var)")
            sys.exit(1)
        print(f"\n[apply] Connecting to {args.uri} as {args.user}...")
        apply_to_neo4j(
            entities, relations,
            args.uri, args.user, args.password,
            args.batch_size, args.wipe,
        )
    else:
        print("\n[next] To load into Neo4j:")
        print("  Option A — Neo4j Browser: paste load_entities.cypher then load_relations.cypher")
        print("  Option B — cypher-shell:")
        print(f"    cypher-shell -u neo4j -p <password> < {ent_cypher_path}")
        print(f"    cypher-shell -u neo4j -p <password> < {rel_cypher_path}")
        print("  Option C — direct Python:")
        print("    python src/07_load_to_neo4j.py --apply --password <password>")


if __name__ == "__main__":
    main()