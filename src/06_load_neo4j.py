"""Create verification, comparison, and sample query files for the final graph."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
if (_here / "_helpers.py").exists():
    sys.path.insert(0, str(_here))

from _helpers import OUTPUT_DIR, read_jsonl

KG_DIR     = OUTPUT_DIR / "kg-jsonl"
CYPHER_DIR = OUTPUT_DIR / "cypher"
REPORTS_DIR = OUTPUT_DIR / "reports"
CYPHER_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ENT_PATH = KG_DIR / "final_entities.jsonl"
REL_PATH = KG_DIR / "final_relations.jsonl"


# Cypher queries grouped by purpose.

VERIFY_COUNTS = """\
// ============================================================
// 01_verify_counts.cypher
// Basic sanity checks — run these first after loading
// ============================================================

// 1. Node counts by label
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count
ORDER BY count DESC;

// 2. Relation counts by type
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count
ORDER BY count DESC;

// 3. Organisations in graph
MATCH (o:Organisation) RETURN o.name, o.geography;

// 4. Guidelines in graph
MATCH (o:Organisation)-[:PUBLISHES]->(g:Guideline)
RETURN o.name AS org, g.name AS guideline, g.year;

// 5. Orphaned nodes (no connections — should be 0)
MATCH (n) WHERE NOT (n)--()
RETURN labels(n)[0] AS label, n.id AS id, n.name AS name
LIMIT 20;

// 6. Broken relations (endpoints missing — should be 0)
MATCH (a)-[r]->(b)
WHERE a IS NULL OR b IS NULL
RETURN count(r) AS broken_relations;
"""

CROSS_GUIDELINE = """\
// ============================================================
// 02_cross_guideline.cypher
// Cross-guideline comparison queries
// These demonstrate the core scientific value of the KG
// ============================================================

// 1. Shared diagnostic thresholds (EQUIVALENT_TO)
//    Which guidelines agree on the same numeric cutpoint?
MATCH (a:GlycaemicTarget)-[r:EQUIVALENT_TO]->(b:GlycaemicTarget)
RETURN
  a.organization AS org_A,
  a.metric       AS metric,
  a.op           AS op,
  a.value        AS value,
  a.units        AS units,
  b.organization AS org_B,
  r.value_mmol_L AS normalised_mmol_L
ORDER BY metric, normalised_mmol_L;

// 2. Drugs mentioned across multiple guidelines
//    Which drugs does more than one guideline recommend?
MATCH (d:Drug)-[:MENTIONED_IN]->(g:Guideline)
WITH d, collect(g.organization) AS orgs, count(g) AS n_guidelines
WHERE n_guidelines > 1
RETURN d.name AS drug, orgs, n_guidelines
ORDER BY n_guidelines DESC
LIMIT 30;

// 3. Guidelines that agree on first-line treatments
MATCH (d:Drug)-[r:FIRST_LINE_FOR]->(c)
WITH d.name AS drug, collect(DISTINCT r.organization) AS orgs
WHERE size(orgs) > 1
RETURN drug, orgs
ORDER BY size(orgs) DESC;

// 4. Glycaemic targets by organisation — side by side
MATCH (c:Condition)-[:HAS_TARGET]->(t:GlycaemicTarget)
RETURN
  t.organization AS guideline,
  t.metric       AS metric,
  t.op           AS operator,
  t.value        AS value,
  t.units        AS units,
  t.value_mmol_L AS mmol_L_normalised
ORDER BY metric, t.organization;

// 5. Which conditions appear in all 4 guidelines?
MATCH (c:ClinicalCondition)-[:MENTIONED_IN]->(g:Guideline)
WITH c, collect(DISTINCT g.organization) AS orgs
WHERE size(orgs) = 4
RETURN c.name AS condition, orgs
ORDER BY c.name;

// 6. Which conditions are unique to one guideline?
MATCH (c:ClinicalCondition)-[:MENTIONED_IN]->(g:Guideline)
WITH c, collect(DISTINCT g.organization) AS orgs
WHERE size(orgs) = 1
RETURN c.name AS condition, orgs[0] AS only_in_guideline
ORDER BY only_in_guideline, condition
LIMIT 30;
"""

CLINICAL_QUERIES = """\
// ============================================================
// 03_clinical_queries.cypher
// Sample clinical queries demonstrating KG capabilities
// ============================================================

// --- DIAGNOSIS ---

// 1. All diagnostic thresholds for diabetes
MATCH (c:Condition {name: 'Type 2 Diabetes'})-[:HAS_TARGET]->(t:GlycaemicTarget)
RETURN t.organization AS guideline, t.metric, t.op, t.value, t.units
ORDER BY t.metric, t.organization;

// 2. HbA1c targets across guidelines
MATCH (c:Condition)-[:HAS_TARGET]->(t:GlycaemicTarget)
WHERE t.metric CONTAINS 'HbA1c' OR t.metric CONTAINS 'A1C'
RETURN t.organization, t.metric, t.op, t.value, t.units,
       t.value_mmol_mol AS mmol_mol
ORDER BY t.organization;

// --- TREATMENT ---

// 3. First-line treatments for Type 2 Diabetes
MATCH (d:Drug)-[r:FIRST_LINE_FOR]->(c)
RETURN d.name AS drug, r.organization AS guideline, c.name AS condition
ORDER BY guideline, drug
LIMIT 40;

// 4. What does each guideline recommend for heart failure patients?
MATCH (d:Drug)-[r:RECOMMENDED_FOR]->(c:ClinicalCondition)
WHERE toLower(c.name) CONTAINS 'heart failure'
RETURN r.organization AS guideline, d.name AS drug, r.source_page AS page
ORDER BY guideline, drug;

// 5. Drugs contraindicated in CKD
MATCH (d:Drug)-[r:CONTRAINDICATED_IN]->(c:ClinicalCondition)
WHERE toLower(c.name) CONTAINS 'kidney' OR toLower(c.name) CONTAINS 'ckd'
   OR toLower(c.name) CONTAINS 'renal'
RETURN r.organization AS guideline, d.name AS drug, c.name AS condition
ORDER BY guideline, drug;

// 6. Drug combinations used together (USED_WITH)
MATCH (a:Drug)-[r:USED_WITH]->(b:Drug)
RETURN r.organization AS guideline, a.name AS drug_A, b.name AS drug_B
ORDER BY guideline
LIMIT 30;

// --- MONITORING ---

// 7. CGM and Time-in-Range protocols
MATCH (t:Test {name: 'CGM'})-[:HAS_PROTOCOL]->(mp:MonitoringProtocol)
RETURN mp.organization AS guideline, mp.parameter,
       mp.tir_op AS tir_operator, mp.tir_value AS tir_percent_target;

// 8. What tests are used for monitoring?
MATCH (t:Test)-[:HAS_PROTOCOL]->(mp:MonitoringProtocol)
RETURN t.name AS test, mp.organization AS guideline, mp.parameter
ORDER BY t.name;

// --- RECOMMENDATIONS ---

// 9. All clinical recommendations from ADA 2026
MATCH (g:Guideline {organization: 'ADA'})-[:HAS_RECOMMENDATION]->(r:Recommendation)
RETURN r.label, r.page_start AS page, r.text
ORDER BY r.page_start, r.label
LIMIT 20;

// 10. Recommendations that specify a GlycaemicTarget
MATCH (r:Recommendation)-[:SPECIFIES_TARGET]->(t:GlycaemicTarget)
RETURN r.organization AS guideline, r.label,
       t.metric, t.op, t.value, t.units, r.page_start AS page
ORDER BY guideline, r.page_start
LIMIT 30;

// --- POPULATIONS ---

// 11. What populations does NICE specifically address?
MATCH (d:Drug)-[r:RECOMMENDED_FOR]->(p:Population)
WHERE r.organization = 'NICE'
RETURN d.name AS drug, p.name AS population
ORDER BY drug
LIMIT 30;

// 12. Elderly patient recommendations
MATCH (d)-[r]->(p:Population)
WHERE toLower(p.name) CONTAINS 'elder' OR toLower(p.name) CONTAINS 'older adult'
RETURN type(r) AS relation, labels(d)[0] AS entity_type,
       d.name AS entity, r.organization AS guideline, p.name AS population
LIMIT 30;
"""

QUESTION_GEN = """\
// ============================================================
// 04_question_gen.cypher
// Seed queries for question generation
// Each query returns (subject, relation, object) triples
// that map directly to question templates
// ============================================================

// Template: "What is the recommended HbA1c target according to [guideline]?"
// Pattern: Condition -HAS_TARGET-> GlycaemicTarget
MATCH (c:Condition)-[:HAS_TARGET]->(t:GlycaemicTarget)
RETURN
  'HbA1c target'       AS question_topic,
  c.name               AS subject,
  t.organization       AS guideline,
  t.metric             AS metric,
  t.op + ' ' + toString(t.value) + ' ' + t.units AS target_value,
  'what'               AS question_type;

// Template: "Which drug is recommended as first-line therapy for [condition]?"
// Pattern: Drug -FIRST_LINE_FOR-> Condition/ClinicalCondition
MATCH (d:Drug)-[r:FIRST_LINE_FOR]->(c)
RETURN
  'first-line therapy' AS question_topic,
  d.name               AS subject,
  type(r)              AS relation,
  c.name               AS object,
  r.organization       AS guideline,
  'which'              AS question_type
LIMIT 50;

// Template: "Is [drug] contraindicated in [condition]?"
// Pattern: Drug -CONTRAINDICATED_IN-> ClinicalCondition
MATCH (d:Drug)-[r:CONTRAINDICATED_IN]->(c:ClinicalCondition)
RETURN
  'contraindication'   AS question_topic,
  d.name               AS subject,
  c.name               AS object,
  r.organization       AS guideline,
  'yes_no'             AS question_type
LIMIT 50;

// Template: "How do ADA and NICE differ in their [metric] recommendations?"
// Pattern: EQUIVALENT_TO links where orgs differ
MATCH (a:GlycaemicTarget)-[r:EQUIVALENT_TO]->(b:GlycaemicTarget)
RETURN
  'cross-guideline comparison' AS question_topic,
  a.metric                     AS metric,
  a.organization               AS org_A,
  a.value                      AS value_A,
  a.units                      AS units_A,
  b.organization               AS org_B,
  b.value                      AS value_B,
  b.units                      AS units_B,
  'how'                        AS question_type;

// Template: "Which guidelines recommend [drug] for [condition]?"
// Pattern: Drug -RECOMMENDED_FOR-> condition, grouped by guideline
MATCH (d:Drug)-[r:RECOMMENDED_FOR]->(c)
WITH d.name AS drug, c.name AS condition,
     collect(DISTINCT r.organization) AS guidelines
RETURN
  'cross-guideline coverage'  AS question_topic,
  drug, condition, guidelines,
  size(guidelines)            AS n_guidelines,
  'which'                     AS question_type
ORDER BY n_guidelines DESC
LIMIT 40;

// Template: "What monitoring is recommended for patients using [drug]?"
// Pattern: Drug/Test -REQUIRES_MONITORING-> ...
MATCH (d:Drug)-[r:REQUIRES_MONITORING]->(t)
RETURN
  'monitoring requirement'    AS question_topic,
  d.name                      AS drug,
  labels(t)[0]                AS monitoring_type,
  t.name                      AS monitoring_target,
  r.organization              AS guideline,
  'what'                      AS question_type
LIMIT 30;
"""


# Build a local graph summary from the JSONL files.
def build_local_report(entities, relations) -> str:
    lines = []
    lines.append("KG VERIFICATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Total entities:  {len(entities)}")
    lines.append(f"Total relations: {len(relations)}")
    lines.append("")

    etype_counts = Counter(e.get("type", "?") for e in entities)
    lines.append("Entity types:")
    for etype, cnt in sorted(etype_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {etype:<30} {cnt:>5}")

    lines.append("")
    rtype_counts = Counter(r.get("type", "?") for r in relations)
    lines.append("Relation types:")
    for rtype, cnt in sorted(rtype_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {rtype:<30} {cnt:>5}")

    lines.append("")
    lines.append("Organisations:")
    for e in entities:
        if e.get("type") == "Organisation":
            lines.append(f"  {e['name']} ({e.get('geography', '')})")

    lines.append("")
    lines.append("Cross-guideline EQUIVALENT_TO:")
    eid_map = {e["id"]: e for e in entities}
    for r in relations:
        if r.get("type") == "EQUIVALENT_TO":
            a = eid_map.get(r["src"], {})
            b = eid_map.get(r["dst"], {})
            lines.append(
                f"  {a.get('organization','?'):18} {a.get('metric','?')} "
                f"{a.get('op','')} {a.get('value','')} {a.get('units','')}"
            )
            lines.append(
                f"  ≡ {b.get('organization','?'):15} {b.get('metric','?')} "
                f"{b.get('op','')} {b.get('value','')} {b.get('units','')}"
            )

    lines.append("")
    lines.append("Graph ready for Neo4j ingestion.")
    lines.append("Run 06_load_neo4j.py to load.")
    return "\n".join(lines)


# Main pipeline.
def main():
    # Save the Cypher query files.
    files = {
        "01_verify_counts.cypher":  VERIFY_COUNTS,
        "02_cross_guideline.cypher": CROSS_GUIDELINE,
        "03_clinical_queries.cypher": CLINICAL_QUERIES,
        "04_question_gen.cypher":   QUESTION_GEN,
    }
    for fname, content in files.items():
        path = CYPHER_DIR / fname
        path.write_text(content, encoding="utf-8")
        print(f"[wrote] {path}")

    # Build a local summary when the graph files are available.
    if ENT_PATH.exists() and REL_PATH.exists():
        entities  = list(read_jsonl(ENT_PATH))
        relations = list(read_jsonl(REL_PATH))
        report    = build_local_report(entities, relations)
        report_path = REPORTS_DIR / "08_kg_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"[wrote] {report_path}")
        print()
        print(report)
    else:
        print(f"[warn] {ENT_PATH} not found — run script 06 first for local report")

    print()
    print("[next] Steps:")
    print("  1. Run: python src/07_load_to_neo4j.py")
    print("     → generates outputs/cypher/load_entities.cypher + load_relations.cypher")
    print("  2. In Neo4j Browser: File → Open → load_entities.cypher → Run All")
    print("     Then: File → Open → load_relations.cypher → Run All")
    print("  3. Verify with: outputs/cypher/01_verify_counts.cypher")
    print("  4. Explore with: outputs/cypher/02_cross_guideline.cypher")


if __name__ == "__main__":
    main()