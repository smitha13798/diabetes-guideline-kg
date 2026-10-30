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
