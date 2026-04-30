"""
Term-First strategy — extract key entities/specs from query and resolve them
against Neo4j Term nodes for authoritative full names and source specs.

Migrated and adapted from the parent project's hybrid_retriever.TermDefinitionResolver
+ SemanticQueryAnalyzer._extract_potential_terms.
"""
import re
from typing import Optional


# Known 3GPP network functions (abbreviations) — used for prioritising the primary term
NETWORK_FUNCTIONS = {
    "AMF", "SMF", "UPF", "NRF", "AUSF", "UDM", "PCF", "NSSF", "AF",
    "NEF", "SEPP", "SCP", "UDR", "CHF", "UDSF", "NWDAF", "LMF",
}

# Pattern to match TS/TR spec references like "TS 23.501", "TR 38.300"
SPEC_PATTERN = re.compile(r"\b(TS|TR)\s*(\d{2}\.\d{3})\b", re.IGNORECASE)

# Pattern for abbreviations as written in the query (preserves original casing)
# Optional 0-2 lowercase suffix handles plurals: UEs → UE, APIs → API, NEFs → NEF
ABBREV_PATTERN = re.compile(r"\b([A-Z][A-Z0-9]{1,5})(?:[a-z]{0,2})?\b")

# Common English words that match the abbreviation regex but aren't 3GPP terms
COMMON_WORDS = {
    "IN", "IS", "ON", "AT", "TO", "OF", "A", "AN", "OR", "IT", "AS", "BY",
    "NO", "SO", "ME", "BE", "DO", "IF", "UP", "WE",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW",
    "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "GET", "HIM", "LET",
    "PUT", "SAY", "SHE", "TOO", "USE",
    "WHAT", "WHICH", "ABOUT", "BETWEEN", "WHERE", "WHEN", "WHY", "HOW",
}

# Generic network terms that aren't specific entities (downweighted vs NF abbreviations)
GENERIC_TERMS = {"5G", "4G", "LTE", "NR", "3GPP", "5GS", "5GC", "3G", "2G"}


def _extract_potential_terms(query: str) -> list[str]:
    """Pull candidate abbreviations from the query, filtering out common English words."""
    matches = ABBREV_PATTERN.findall(query)
    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        if m in COMMON_WORDS or m in GENERIC_TERMS:
            continue
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


class TermFirstStrategy:
    """Pure-text term extraction. The optional resolver argument lets callers
    enrich extraction results with KG-verified full names + source specs.
    """

    def __init__(self, resolver: Optional["TermResolver"] = None):
        self._resolver = resolver

    def extract(self, query: str) -> dict:
        """
        Extract structured terms from query.

        Returns:
            {
                'network_functions': [NF names found in NETWORK_FUNCTIONS],
                'spec_refs':         ['TS 23.501', ...],
                'abbreviations':     [other uppercase tokens, deduped],
                'primary_term':      best single string for graph lookup,
                'resolved':          {abbrev: {full_name, specs}} (only if resolver given),
            }
        """
        # Network functions found verbatim in the query
        nfs = [nf for nf in NETWORK_FUNCTIONS if re.search(rf"\b{nf}\b", query)]
        # Spec references like "TS 23.501"
        specs = [f"{m.group(1).upper()} {m.group(2)}" for m in SPEC_PATTERN.finditer(query)]
        # Other uppercase abbreviations (excludes NFs to avoid duplicates)
        candidates = [t for t in _extract_potential_terms(query) if t not in NETWORK_FUNCTIONS]

        # Primary term: NF > spec > abbreviation > first word
        first_word = query.split()[0] if query.strip() else ""
        primary = (
            nfs[0] if nfs
            else specs[0] if specs
            else candidates[0] if candidates
            else first_word
        )

        result: dict = {
            "network_functions": nfs,
            "spec_refs": specs,
            "abbreviations": candidates,
            "primary_term": primary,
        }

        # If a resolver is wired, look up authoritative definitions from the KG
        if self._resolver is not None:
            to_resolve = list(dict.fromkeys(nfs + candidates))  # dedup, preserve order
            if to_resolve:
                result["resolved"] = self._resolver.resolve(to_resolve)
            else:
                result["resolved"] = {}

        return result


class TermResolver:
    """
    Resolve extracted abbreviations against Neo4j Term nodes for authoritative
    full names and the specs they're defined in.

    Migrated from hybrid_retriever.TermDefinitionResolver. The original used
    `MATCH (t:Term {abbreviation: $abbrev})` — schema may store the property
    under `name` instead, so we try a few fallbacks.
    """

    # Try these property names in order when matching Term nodes by their short form
    _ABBREV_PROPS = ("abbreviation", "name", "term", "key")

    def __init__(self, neo4j_driver):
        self._driver = neo4j_driver

    def resolve(self, entities: list[str]) -> dict[str, dict]:
        """
        Look up each entity in the KG.

        Returns:
            {
                'SCP': {
                    'full_name': 'Service Communication Proxy',
                    'specs': ['TS 23.501', 'TS 29.500'],
                    'matched_property': 'abbreviation',
                },
                ...
            }
            Entities not found are simply omitted from the result.
        """
        if not entities:
            return {}

        out: dict[str, dict] = {}
        with self._driver.session() as session:
            for entity in entities:
                clean = entity.strip().upper()
                hit = self._lookup(session, clean)
                if hit:
                    out[entity] = hit
        return out

    def _lookup(self, session, clean_entity: str) -> Optional[dict]:
        # Try each candidate property name; first hit wins.
        for prop in self._ABBREV_PROPS:
            try:
                cypher = (
                    f"MATCH (t:Term {{ {prop}: $val }}) "
                    "OPTIONAL MATCH (t)-[:DEFINED_IN]->(d:Document) "
                    "RETURN coalesce(t.full_name, t.fullName, t.definition, t.name) AS full_name, "
                    "       collect(DISTINCT d.spec_id) AS specs"
                )
                rec = session.run(cypher, val=clean_entity).single()
                if rec and rec["full_name"]:
                    return {
                        "full_name": rec["full_name"],
                        "specs": [s for s in (rec["specs"] or []) if s],
                        "matched_property": prop,
                    }
            except Exception:
                # Bad property name on this schema — try the next one
                continue
        return None
