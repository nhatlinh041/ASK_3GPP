"""
7 intent-specific prompt templates for 3GPP Q&A.
Each template slots in {context} (retrieved chunks) and {question}.

All templates share the same hard grounding rules: every claim must cite a
chunk from the context, and named entities (service operations, interfaces,
parameters) that don't appear verbatim in the context must NOT be invented.
"""

# Hard grounding rules prepended to every intent template. These are the
# anti-hallucination guard rails — they fire on EVERY claim, every name, every
# spec reference. Tightened after observing repeated service-name fabrications
# (e.g. "Nnwdaf_EventSubscription" instead of "Nnwdaf_EventsSubscription") and
# uncited generic prose in compare-style answers.
_GROUNDING_RULES = """# Grounding rules — read carefully

1. **Cite every claim.** Every factual sentence MUST end with `[spec_id §section]`
   matching a chunk from the context above. Format example:
       "NWDAF collects data from 5GC NFs [ts_23.288 §4.1]."
   If a sentence has no supporting chunk, DO NOT WRITE IT.

2. **Never invent named entities.** Service operations (Nxxx_Yyy), interface
   names (Nxxx), parameter names (e.g. S-NSSAI, DNN), and procedure names
   appear in the context EXACTLY or NOT AT ALL. If you want to mention a name
   and you cannot find it in the context, write:
       "the context does not specify the exact <service operation | interface | parameter> name"
   Do NOT guess "Nnwdaf_Analytics" or "Nnef_External" or any plausible-looking
   variant. Only emit names you can copy-paste from a chunk.

3. **No synthesis from training knowledge.** If the context lacks evidence for
   a sub-question, write one line: "Context does not cover <sub-question>."
   Then move on. Do NOT fill the gap with general 5G knowledge.

4. **Prefer concrete over generic.** When two chunks support a claim, cite the
   one with the most specific section (e.g. §6.2.18 over §4.1).

5. **Quote when in doubt.** If a chunk explicitly defines something, quote
   the defining sentence (in quotes) before your paraphrase, with the citation."""


# Per-intent role + structure instructions. The full prompt is assembled by
# concatenating: grounding rules → intent-specific instructions → context →
# question. Keeping rules first means the LLM reads them before any context.
_INTENT_INSTRUCTIONS: dict[str, str] = {
    "definition": (
        "You are a 3GPP technical expert. Define the term or network function asked about,"
        " using ONLY the provided context. Cover: full name, purpose, key interfaces, and"
        " relevant spec references — but ONLY include each item if a chunk supports it."
        " If the context does not specify an item, write \"context does not specify\" for"
        " that item rather than skipping or fabricating."
    ),
    "procedure": (
        "You are a 3GPP technical expert. Describe the procedure step by step using ONLY"
        " the provided context. Number each step and end each step with the supporting"
        " citation `[spec_id §section]`. If a step cannot be supported by any chunk, do"
        " not include it; instead, after listing the supported steps, add: \"Context does"
        " not specify subsequent steps.\""
    ),
    "comparison": (
        "You are a 3GPP technical expert. Compare the entities asked about using ONLY the"
        " provided context. Structure as a markdown table with one row per attribute and"
        " one column per entity. Every cell must contain either a cited fact"
        " `[spec_id §section]` OR the literal string \"context does not specify\". Do not"
        " leave cells blank and do not fabricate values to make rows symmetric."
    ),
    "reference": (
        "You are a 3GPP technical expert. List the specification references mentioned in"
        " the context. For each reference: spec id, full title (only if a chunk provides"
        " it), and the citation. If the chunk does not give the title, write the spec id"
        " alone — do not guess the title."
    ),
    "network_function": (
        "You are a 3GPP technical expert. Describe the network function using ONLY the"
        " provided context. Cover three areas with separate paragraphs: (1) role in 5G"
        " architecture, (2) interfaces / service operations, (3) key procedures. For (2),"
        " ONLY name an interface or service operation that appears verbatim in a chunk;"
        " if no chunk names them, write \"Context does not specify the exact interface"
        " names.\" For (3), only list procedures the context explicitly describes."
    ),
    "relationship": (
        "You are a 3GPP technical expert. Explain how the entities relate using ONLY the"
        " provided context. Describe each interaction the chunks document, with a citation"
        " per interaction. Do not infer interactions from general 5G knowledge — if the"
        " context only shows entity A and entity B separately without describing their"
        " interaction, say so."
    ),
    "general": (
        "You are a 3GPP technical expert. Answer the question using ONLY the provided"
        " context from 3GPP specifications. If the context does not contain enough"
        " information, say so clearly per missing aspect."
    ),
}


PROMPT_TEMPLATES: dict[str, str] = {
    intent: f"""{_GROUNDING_RULES}

# Task
{instructions}

Context:
{{context}}

Question: {{question}}

Answer:"""
    for intent, instructions in _INTENT_INSTRUCTIONS.items()
}


def build_prompt(intent: str, context: str, question: str) -> str:
    """Fill in a prompt template for the given intent."""
    template = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES["general"])
    return template.format(context=context, question=question)
