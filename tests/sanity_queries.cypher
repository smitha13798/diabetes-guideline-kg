// tests/sanity_queries.cypher
// Basic sanity queries for the final knowledge graph.
// Run in Neo4j Browser one block at a time. Expected values are from the final

// 1. Total nodes.
MATCH (n) RETURN count(n) AS nodes;

// 2. Total relationships. 
// Neo4j may show 3484: one FIRST_LINE_FOR relation has a source id (llm_3da9d197b3)
// that is not in the entities file, so the loader cannot create it.
MATCH ()-[r]->() RETURN count(r) AS relationships;

// 3. Nodes per type.
MATCH (n) RETURN labels(n)[0] AS type, count(*) AS n ORDER BY n DESC;

// 4. Relationships per type.

MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY n DESC;

// 5. Targets per organisation. 
MATCH (t:GlycaemicTarget) RETURN t.organization AS org, count(*) AS targets ORDER BY targets DESC;

// 6. HAS_TARGET only goes from Condition to GlycaemicTarget. 
MATCH (a)-[r:HAS_TARGET]->(b)
RETURN labels(a)[0] AS from_type, labels(b)[0] AS to_type, count(*) AS n;

// 7. Equivalence edges.
MATCH ()-[r:EQUIVALENT_TO]->() RETURN count(r) AS equivalent_to_edges;

// 8. Spot check: 126 mg/dL becomes 7.0 mmol/L. 
MATCH (t:GlycaemicTarget {units: "mg/dL", value: 126})
RETURN t.organization, t.metric, t.value, t.value_mmol_L LIMIT 5;

// 9. The three-way FPG agreement.
MATCH (t:GlycaemicTarget {metric: "FPG", op: "≥"})
WHERE round(t.value_mmol_L * 10) / 10.0 = 7.0
RETURN t.organization, t.value, t.units, t.value_mmol_L ORDER BY t.organization;

//10. all HbA1c targets
MATCH (t:GlycaemicTarget {metric: "HbA1c"})
RETURN t.organization, t.op, t.value, t.units, t.value_mmol_mol, t.page_id, t.source_quote
ORDER BY t.organization, t.value;


