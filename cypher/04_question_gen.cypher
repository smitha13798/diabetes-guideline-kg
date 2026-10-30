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
