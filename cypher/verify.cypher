// ---- basic counts ----
MATCH (n) RETURN labels(n) AS labels, count(*) AS count ORDER BY count DESC;
MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC;

// ---- orphans (unconnected nodes) ----
MATCH (n) WHERE size((n)--()) = 0 RETURN labels(n) AS labels, n.id AS id, n LIMIT 50;

// ---- domain checks (match your schema labels) ----
// 1) Glycaemic targets and their tests
MATCH (c:Condition {name:'Diabetes'})-[:HAS_TARGET]->(t:GlycaemicTarget)-[:APPLIES_TO_TEST]->(test:Test)
RETURN t.metric, t.units, t.lb, t.ub, t.op, t.value, test.name LIMIT 25;

// 2) Diagnostic criteria with provenance
MATCH (c:Condition)-[:DEFINED_BY]->(dc:DiagnosticCriterion)-[:MEASURED_BY]->(test:Test)
RETURN c.name, test.name, dc.comparator_min, dc.cutpoint_min, dc.comparator_max, dc.cutpoint_max, dc.units, dc.page_id, dc.table_id LIMIT 25;

// 3) Test requirements matrix
MATCH (test:Test)-[:REQUIRES]->(req:TestRequirement)
RETURN test.name, req.name, req.value LIMIT 50;

// 4) CGM / Time-in-Range references
MATCH (x)-[:HAS_PROTOCOL]->(mp:MonitoringProtocol)
WHERE toLower(mp.parameter) CONTAINS 'cgm' OR toLower(mp.parameter) CONTAINS 'time in range'
RETURN labels(x)[0] AS parent_type, x.id AS parent_id, mp.parameter, mp.value_text, mp.units LIMIT 25;

// 5) Treatments
MATCH (c:Condition {name:'Diabetes'})-[:RECOMMENDS]->(tr:TreatmentRecommendation)
RETURN tr.setting, tr.recommendation, tr.page_id, tr.table_id LIMIT 50;

// 6) Flow diagrams (if any)
MATCH (a:FlowNode)-[:FLOWS_TO]->(b:FlowNode)
RETURN a.text AS from, b.text AS to LIMIT 25;