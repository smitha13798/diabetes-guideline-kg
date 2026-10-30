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
