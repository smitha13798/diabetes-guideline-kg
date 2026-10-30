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
