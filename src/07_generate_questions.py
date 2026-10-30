"""Generate graph-grounded questions with an LLM and validate them against the Knowledge Graph."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Optional .env support.
# from dotenv import load_dotenv
# load_dotenv()

_here = Path(__file__).resolve().parent
if (_here / "_helpers.py").exists():
    sys.path.insert(0, str(_here))

from _helpers import OUTPUT_DIR, read_jsonl

KG_DIR   = OUTPUT_DIR / "kg-jsonl"
Q_DIR    = OUTPUT_DIR / "questions"
Q_DIR.mkdir(parents=True, exist_ok=True)

ENT_PATH = KG_DIR / "final_entities.jsonl"
REL_PATH = KG_DIR / "final_relations.jsonl"

OUT_PATH     = Q_DIR / "questions_hybrid.jsonl"
FLAGGED_PATH = Q_DIR / "questions_hybrid_flagged.jsonl"
SUMMARY_PATH = Q_DIR / "questions_hybrid_summary.json"

# Model settings.
# Choose the LLM provider.
# Use OpenAI by default; set LLM_PROVIDER=anthropic for Claude.
PROVIDER     = os.environ.get("LLM_PROVIDER", "openai").lower()   # "openai" | "anthropic"

# OpenAI model.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
# Claude model.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")

# Model used for the selected provider.
MODEL        = CLAUDE_MODEL if PROVIDER == "anthropic" else OPENAI_MODEL
TEMPERATURE  = 0.4          # low enough to suppress invention, high enough for variety
MAX_TOKENS   = 512
RETRY_LIMIT  = 3
RETRY_DELAY  = 2            # seconds between retries

# Maximum questions generated for each relation type.
# Most relation types use all available anchors; MENTIONED_IN is capped and filtered.
SAMPLE_LIMITS: dict[str, int] = {
    "FIRST_LINE_FOR":      170,  # exhaust — every first-line recommendation
    "SECOND_LINE_FOR":      69,  # exhaust
    "CONTRAINDICATED_IN":  109,  # exhaust — safety-critical, all matter
    "RECOMMENDED_FOR":     546,  # exhaust — each is a distinct recommendation
    "HAS_TARGET":           46,  # exhaust
    "EQUIVALENT_TO":         8,  # exhaust
    "MANAGES":             265,  # exhaust
    "USED_WITH":            67,  # exhaust
    "REQUIRES_MONITORING":  53,  # exhaust
    "MENTIONED_IN":        200,  # capped + cross-guideline filter in main()
    "ASSOCIATED_WITH":     156,  # exhaust
}

# Question type for each relation.
REL_TO_QTYPE: dict[str, str] = {
    "FIRST_LINE_FOR":      "which",
    "SECOND_LINE_FOR":     "which",
    "CONTRAINDICATED_IN":  "yes_no",
    "RECOMMENDED_FOR":     "which",
    "HAS_TARGET":          "what",
    "EQUIVALENT_TO":       "how",
    "MANAGES":             "how",
    "USED_WITH":           "what",
    "REQUIRES_MONITORING": "what",
    "MENTIONED_IN":        "which",
    "ASSOCIATED_WITH":     "what",
}

# One primary generation method for each relation.
REL_TO_METHOD: dict[str, str] = {
    "FIRST_LINE_FOR":      "grounded_prompt",
    "SECOND_LINE_FOR":     "grounded_prompt",
    "CONTRAINDICATED_IN":  "grounded_prompt",
    "RECOMMENDED_FOR":     "grounded_prompt",
    "HAS_TARGET":          "grounded_prompt",
    "EQUIVALENT_TO":       "grounded_prompt",
    "MANAGES":             "vignette",
    "USED_WITH":           "grounded_prompt",
    "REQUIRES_MONITORING": "grounded_prompt",
    "MENTIONED_IN":        "paraphrase",
    "ASSOCIATED_WITH":     "grounded_prompt",
}

# Relations that can also receive MCQ options.
MCQ_RELATIONS = {
    "FIRST_LINE_FOR", "SECOND_LINE_FOR",
    "CONTRAINDICATED_IN", "HAS_TARGET",
}

# Clinical category for each relation.
REL_TO_CATEGORY: dict[str, str] = {
    "FIRST_LINE_FOR":      "treatment",
    "SECOND_LINE_FOR":     "treatment",
    "CONTRAINDICATED_IN":  "safety",
    "RECOMMENDED_FOR":     "treatment",
    "HAS_TARGET":          "diagnosis",
    "EQUIVALENT_TO":       "cross_guideline",
    "MANAGES":             "treatment",
    "USED_WITH":           "treatment",
    "REQUIRES_MONITORING": "monitoring",
    "MENTIONED_IN":        "cross_guideline",
    "ASSOCIATED_WITH":     "comorbidity",
}

# Simple filters for unsuitable graph terms.
SPURIOUS_CONDITIONS = {
    "adverse events", "adverse effects", "side effects",
    "hyperglycemia", "hypoglycemia",
}
POPULATION_PREFIXES = (
    "people with", "patients with", "adults with",
    "individuals with", "children with", "those with",
)


# Data loading

def load_graph() -> tuple[dict[str, dict], list[dict]]:
    """Load the final graph from JSONL files."""
    entities  = list(read_jsonl(ENT_PATH))
    relations = list(read_jsonl(REL_PATH))
    eid_map   = {e["id"]: e for e in entities}
    return eid_map, relations


def smart_condition_name(raw: str) -> str:
    """Remove common patient-group prefixes from condition names."""
    s = raw.strip()
    lower = s.lower()
    for prefix in POPULATION_PREFIXES:
        if lower.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    return s


def is_spurious(name: str) -> bool:
    return any(sp in name.lower() for sp in SPURIOUS_CONDITIONS)


# Remove prompt-related phrases that sometimes appear in generated text.
_META_PHRASE_RE = re.compile(
    r"\s*(,?\s*)"
    r"(according to (the )?(provided |given )?fact[s]?( provided| given)?"
    r"|based on (the )?(provided |given )?fact[s]?( provided| given)?"
    r"|as (stated|described|mentioned) in (the )?(fact|context|passage)"
    r"|from (the )?(provided |given )?(fact|context|passage)"
    r"|in the (provided |given )?(fact|context|passage))"
    r"\s*",
    re.IGNORECASE,
)

# Phrases used to detect model refusals or meta-responses.
REFUSAL_MARKERS = [
    "i'm not able to", "i am not able to", "i cannot write", "i can't write",
    "does not contain sufficient", "does not specify", "the fact provided does not",
    "please provide", "could you please provide", "please provide a fact",
    "falls outside the stated instructions", "not able to write this question",
    "factually incorrect", "would require adding information",
    "i would be happy to help",
]

def looks_like_refusal(q: str) -> bool:
    """Return True when the model response is not a usable question."""
    if not q:
        return True
    ql = q.lower()
    if any(m in ql for m in REFUSAL_MARKERS):
        return True
    # Long, prose-like output that talks about the 'fact' and asks to 'provide'
    if len(q) > 400 and q.count("?") <= 1 and "fact" in ql and "provide" in ql:
        return True
    return False


def clean_question_text(q: str) -> str:
    """Clean prompt-related wording from a generated question."""
    if not q:
        return q
    q = _META_PHRASE_RE.sub(" ", q)
    q = re.sub(r"\s{2,}", " ", q).strip()
    # Fix spacing left before terminal punctuation, and restore a final '?'
    q = re.sub(r"\s+([?.:,])", r"\1", q)
    if q and not q.endswith("?"):
        q = q.rstrip(".") + "?"
    # Recapitalise first letter if cleanup exposed a new leading word
    if q:
        q = q[0].upper() + q[1:]
    return q


# Prompt builders

def build_grounded_prompt(rel: dict, src: dict, dst: dict) -> tuple[str, str]:
    """Build a question prompt from a fixed graph fact."""
    rel_type  = rel["type"]
    guideline = rel.get("organization", "unknown guideline")
    page      = rel.get("source_page",  "")
    src_name  = src.get("name", "")
    dst_name  = smart_condition_name(dst.get("name", ""))

    # Build the graph fact supplied to the model.
    if rel_type == "HAS_TARGET":
        # Relation: Condition -HAS_TARGET-> GlycaemicTarget
        # src = Condition node (name only), dst = GlycaemicTarget node (has metric/op/value/units)
        metric  = dst.get("metric", "")
        op      = dst.get("op", "")
        value   = dst.get("value", "")
        units   = dst.get("units", "")
        context = (
            f"FACT (from {guideline} guidelines"
            + (f", p.{page}" if page else "") + "):\n"
            f"  Condition: {src_name}\n"
            f"  Metric:    {metric}\n"
            f"  Target:    {op} {value} {units}"
        )
        answer_key = f"{op} {value} {units}".strip()

    elif rel_type == "EQUIVALENT_TO":
        org_a = src.get("organization", "")
        org_b = dst.get("organization", "")
        metric = src.get("metric", "")
        val_a  = f"{src.get('op','')} {src.get('value','')} {src.get('units','')}".strip()
        val_b  = f"{dst.get('op','')} {dst.get('value','')} {dst.get('units','')}".strip()
        context = (
            f"FACT (cross-guideline comparison):\n"
            f"  Metric:   {metric}\n"
            f"  {org_a}: {val_a}\n"
            f"  {org_b}: {val_b}"
        )
        answer_key = f"{org_a}: {val_a}; {org_b}: {val_b}"

    else:
        # State the graph relation in the intended direction.
        prov = f" (from {guideline} guidelines" + (f", p.{page}" if page else "") + ")"

        if rel_type == "FIRST_LINE_FOR":
            fact = f"{src_name} is recommended as first-line therapy for {dst_name}."
            ask  = f"which drug is recommended as first-line therapy for {dst_name}"
            answer_key = src_name
        elif rel_type == "SECOND_LINE_FOR":
            fact = f"{src_name} is recommended as second-line therapy for {dst_name}."
            ask  = f"which drug is recommended as second-line therapy for {dst_name}"
            answer_key = src_name
        elif rel_type == "RECOMMENDED_FOR":
            fact = f"{src_name} is recommended for {dst_name}."
            ask  = f"which drug is recommended for {dst_name}"
            answer_key = src_name
        elif rel_type == "CONTRAINDICATED_IN":
            fact = f"{src_name} is contraindicated in {dst_name}."
            ask  = f"whether {src_name} is contraindicated in {dst_name}"
            answer_key = src_name
        elif rel_type == "USED_WITH":
            fact = f"{src_name} is used in combination with {dst_name}."
            ask  = f"what {src_name} is used in combination with"
            answer_key = dst_name
        elif rel_type == "REQUIRES_MONITORING":
            fact = f"{src_name} requires monitoring of {dst_name}."
            ask  = f"what monitoring is required for {src_name}"
            answer_key = dst_name
        elif rel_type == "MENTIONED_IN":
            fact = f"{src_name} is discussed in the {guideline} guideline."
            ask  = f"which guideline discusses {src_name}"
            answer_key = guideline or dst_name
        elif rel_type == "ASSOCIATED_WITH":
            fact = f"{src_name} is associated with {dst_name}."
            ask  = f"what is associated with {src_name}"
            answer_key = dst_name
        else:
            fact = f"{src_name} is related to {dst_name}."
            ask  = f"the relationship between {src_name} and {dst_name}"
            answer_key = dst_name

        context = f"FACT{prov}:\n  {fact}"
        # Tell the model exactly what the question should ask.
        rel = {**rel, "_ask": ask}

    system = (
        "You are a clinical medical education assistant. "
        "You write clear, accurate questions for healthcare students studying diabetes management. "
        "Use only the clinical information in the FACT block; do not add drug names, "
        "thresholds, or guidelines that are not stated there. "
        "Write the question as a standalone exam question. "
        "Never refer to 'the fact', 'the provided fact', 'the context', 'the passage', "
        "or the guideline source inside the question. A student reading the question "
        "must not be able to tell it was generated from a supplied fact."
    )

    qtype = REL_TO_QTYPE.get(rel_type, "what")
    ask_focus = rel.get("_ask", "")
    focus_line = f"- The question should ask {ask_focus}\n" if ask_focus else ""
    user = f"""{context}

Task: Write ONE clinical {qtype} question that tests knowledge of this fact.
{focus_line}- The question must be answerable from the FACT above
- Do not mention page numbers, guideline names, or 'the provided fact'
- Write in clear, professional clinical language
- Output ONLY the question text, nothing else"""

    return system, user, answer_key


def build_vignette_prompt(rel: dict, src: dict, dst: dict,
                          path_node: dict | None = None) -> tuple[str, str, str]:
    """Build a short clinical vignette from a MANAGES relation."""
    rel_type  = rel["type"]
    guideline = rel.get("organization", "unknown guideline")
    src_name  = src.get("name", "")
    dst_name  = smart_condition_name(dst.get("name", ""))

    context = (
        f"CLINICAL FACT (from {guideline} guidelines):\n"
        f"  {src_name} is used to manage {dst_name}."
    )
    if path_node:
        context += f"\n  Additionally: {path_node.get('name', '')} is associated with {dst_name}."

    answer_key = src_name

    system = (
        "You are a clinical medical education assistant writing case-based questions "
        "for diabetes management training. "
        "IMPORTANT: Base the clinical scenario ONLY on the facts provided. "
        "You may describe the patient's condition and symptoms in words, "
        "but must NOT include any numerical values (no ages, no HbA1c values, "
        "no glucose readings, no doses) and must NOT introduce drug names, "
        "targets, or guidelines beyond what is stated in the CLINICAL FACT block."
    )

    user = f"""{context}

Task: Write a short clinical vignette question (2–3 sentences).
- Describe a realistic patient with {dst_name} using only words (no numbers)
- End with a question about pharmacological management
- The correct answer must be {src_name} (but do not give it away in the question)
- Do NOT include any numerical values in the question
- Output ONLY the question text, nothing else"""

    return system, user, answer_key


def build_paraphrase_prompt(canonical: str, rel_type: str) -> tuple[str, str, str]:
    """Build recall, application, and analysis versions of a question."""
    system = (
        "You are a clinical medical education assistant. "
        "You rephrase questions for different difficulty levels without changing the factual content."
    )

    user = f"""Original question: "{canonical}"

Task: Rewrite this question at three difficulty levels.
Return ONLY a JSON object with this exact structure:
{{
  "recall":      "direct factual question (easiest)",
  "application": "clinical scenario requiring application of the fact",
  "analysis":    "comparative or reasoning question (hardest)"
}}
Do not change any drug names, numerical values, guideline names, or clinical facts."""

    return system, user, canonical   # answer_key = original canonical


def build_mcq_distractor_prompt(question: str, correct_answer: str,
                                  rel_type: str,
                                  drug_pool: list[str]) -> tuple[str, str]:
    """Build a prompt for three plausible MCQ distractors."""
    pool_sample = random.sample(drug_pool, min(8, len(drug_pool)))
    pool_str    = ", ".join(pool_sample)

    system = (
        "You are a clinical medical education expert creating multiple-choice questions "
        "for diabetes pharmacology assessment."
    )

    user = f"""Question: {question}
Correct answer: {correct_answer}

Task: Generate exactly 3 wrong but plausible distractor options for this multiple-choice question.
Rules:
- Distractors must be real diabetes medications or clinical values (choose from: {pool_str})
- Do NOT include "{correct_answer}" as a distractor
- Distractors should be clinically plausible (a student could reasonably consider them)
- Return ONLY a JSON object with this exact structure: {{"distractors": ["option1", "option2", "option3"]}}"""

    return system, user


# LLM calls

def call_openai(system: str, user: str,
                response_format: str = "text") -> str | None:
    """Call OpenAI and return the generated text."""
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. "
            "Set it with: $env:OPENAI_API_KEY = 'sk-...'"
        )

    payload: dict[str, Any] = {
        "model":       MODEL,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    if response_format == "json":
        payload["response_format"] = {"type": "json_object"}

    data    = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(RETRY_LIMIT):
        try:
            req  = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            print(f"    [warn] API call failed (attempt {attempt+1}/{RETRY_LIMIT}): {exc}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def call_claude(system: str, user: str,
                response_format: str = "text") -> str | None:
    """Call Claude and return the generated text."""
    import urllib.request, urllib.error

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. "
            "Set it with: $env:ANTHROPIC_API_KEY = 'sk-ant-...'"
        )

    user_content = user
    if response_format == "json":
        user_content = (
            user
            + "\n\nReturn ONLY the JSON object, with no surrounding prose, "
              "explanation, or markdown code fences."
        )

    payload: dict[str, Any] = {
        "model":      CLAUDE_MODEL,
        "max_tokens": MAX_TOKENS,
        "system":     system,
        "messages": [
            {"role": "user", "content": user_content},
        ],
    }
    # Claude is called without a temperature setting.

    data    = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }

    for attempt in range(RETRY_LIMIT):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=data, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                # Join returned text blocks.
                parts = [
                    b.get("text", "")
                    for b in body.get("content", [])
                    if b.get("type") == "text"
                ]
                text = "".join(parts).strip()
                # Remove code fences from JSON responses.
                if text.startswith("```"):
                    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
                    text = re.sub(r"\n?```$", "", text).strip()
                return text or None
        except urllib.error.HTTPError as exc:
            # Print the API error message when available.
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                err_body = "(no body)"
            print(f"    [warn] Claude API {exc.code} (attempt {attempt+1}/{RETRY_LIMIT}): {err_body[:300]}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
        except Exception as exc:
            print(f"    [warn] Claude API call failed (attempt {attempt+1}/{RETRY_LIMIT}): {exc}")
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def call_llm(system: str, user: str,
             response_format: str = "text") -> str | None:
    """Send a prompt to the selected LLM provider."""
    if PROVIDER == "anthropic":
        return call_claude(system, user, response_format)
    return call_openai(system, user, response_format)


# Validator

def validate(question: str, answer_key: str,
             rel: dict, src: dict, dst: dict,
             eid_map: dict) -> tuple[bool, str]:
    """Check answer grounding, unsupported numbers, and guideline names."""
    guideline = rel.get("organization", "")
    rel_type  = rel.get("type", "")

    # Check that the answer is present in the graph.
    # Numeric targets are checked against the target node value.
    if rel_type == "HAS_TARGET":
        dst_value = str(dst.get("value", ""))
        dst_units = str(dst.get("units", ""))
        # Compare the answer value with the target node.
        answer_numbers = set(re.findall(r'\d+\.?\d*', answer_key))
        dst_numbers    = set(re.findall(r'\d+\.?\d*', dst_value))
        answer_in_graph = bool(answer_numbers & dst_numbers) or dst_value in answer_key
    else:
        answer_lower = answer_key.lower().strip()
        entity_names = {e.get("name", "").lower() for e in eid_map.values()}
        answer_in_graph = bool(answer_lower) and any(
            answer_lower in name or name in answer_lower
            for name in entity_names
            if name
        )

    # Check that the question does not add unsupported numbers.
    q_numbers = set(re.findall(r'\b\d+\.?\d*\b', question))
    allowed_numbers: set[str] = set()
    for field in ["value", "value_mmol_L", "tir_value"]:
        v = str(src.get(field, ""))
        if v and v != "None":
            allowed_numbers.add(v)
            allowed_numbers.update(re.findall(r'\d+\.?\d*', v))
        v = str(dst.get(field, ""))
        if v and v != "None":
            allowed_numbers.add(v)
            allowed_numbers.update(re.findall(r'\d+\.?\d*', v))
    # Allow numbers already present in the target node.
    if rel_type == "HAS_TARGET":
        for field in ["op", "value", "units", "metric"]:
            v = str(dst.get(field, ""))
            allowed_numbers.update(re.findall(r'\d+\.?\d*', v))
    safe_numbers = {"2", "3", "4", "1", "0"}
    hallucinated = q_numbers - allowed_numbers - safe_numbers
    no_hallucinated_numbers = len(hallucinated) == 0

    # Check guideline names when they appear in the question.
    guideline_ok = True
    if guideline and guideline in {"ADA", "NICE", "IDF", "Diabetes Canada"}:
        other_guidelines = {
            "ADA", "NICE", "IDF", "Diabetes Canada"
        } - {guideline}
        for other in other_guidelines:
            if other in question:
                guideline_ok = False
                break

    passed = answer_in_graph and no_hallucinated_numbers and guideline_ok

    note_parts = []
    if not answer_in_graph:
        note_parts.append(f"answer '{answer_key}' not found in entity names")
    if not no_hallucinated_numbers:
        note_parts.append(f"hallucinated numbers: {hallucinated}")
    if not guideline_ok:
        note_parts.append("wrong guideline name in question")

    note = "; ".join(note_parts) if note_parts else "ok"
    return passed, note


# Question records

def make_record(qid: str, question: str, qtype: str, method: str,
                rel: dict, src: dict, dst: dict,
                answer_key: str, validator_passed: bool,
                validator_note: str, canonical: str,
                difficulty: str = "recall",
                mcq_options: list | None = None,
                correct_option: str = "",
                reformulations: list | None = None) -> dict:
    return {
        "id":               qid,
        "question":         question,
        "question_type":    qtype,
        "difficulty":       difficulty,
        "clinical_category": REL_TO_CATEGORY.get(rel["type"], "general"),
        "generation_method": method,
        "graph_anchor": {
            "src_id":    src.get("id", ""),
            "src_name":  src.get("name", ""),
            "rel_type":  rel["type"],
            "dst_id":    dst.get("id", ""),
            "dst_name":  dst.get("name", ""),
            "guideline": rel.get("organization", ""),
            "page":      rel.get("source_page", None),
        },
        "canonical_question":  canonical,
        "answer_key":          answer_key,
        "validator_passed":    validator_passed,
        "validator_note":      validator_note,
        "mcq_options":         mcq_options or [],
        "correct_option":      correct_option,
        "reformulations":      reformulations or [],
    }


# Rule-based baseline questions

def canonical_question(rel: dict, src: dict, dst: dict) -> str:
    """Create a baseline question directly from a graph relation."""
    rel_type  = rel["type"]
    guideline = rel.get("organization", "")
    src_name  = src.get("name", "")
    dst_name  = smart_condition_name(dst.get("name", ""))
    g_suffix  = f" according to {guideline} guidelines" if guideline else ""

    templates = {
        "FIRST_LINE_FOR":
            f"Which drug is recommended as first-line therapy for {dst_name}{g_suffix}?",
        "SECOND_LINE_FOR":
            f"Which drug is recommended as second-line therapy for {dst_name}{g_suffix}?",
        "CONTRAINDICATED_IN":
            f"Is {src_name} contraindicated in patients with {dst_name}{g_suffix}?",
        "RECOMMENDED_FOR":
            f"Which treatments are recommended for {dst_name}{g_suffix}?",
        "HAS_TARGET":
            (lambda: (
                f"What is the {src.get('metric','glycaemic')} target for {dst_name}{g_suffix}?"
            ))(),
        "EQUIVALENT_TO":
            (lambda: (
                f"How do {src.get('organization','')} and {dst.get('organization','')} "
                f"compare in their {src.get('metric','')} recommendations?"
            ))(),
        "MANAGES":
            f"How is {dst_name} managed with {src_name}{g_suffix}?",
        "USED_WITH":
            f"What is {src_name} commonly used in combination with{g_suffix}?",
        "REQUIRES_MONITORING":
            f"What monitoring is required for patients using {src_name}{g_suffix}?",
        "MENTIONED_IN":
            f"Which guidelines address {src_name}?",
        "ASSOCIATED_WITH":
            f"What conditions are associated with {src_name}{g_suffix}?",
    }
    return templates.get(rel_type, f"What is the relationship between {src_name} and {dst_name}?")


# Main pipeline

def main() -> None:
    # Load the final graph.
    print("[load] Reading graph JSONL...")
    eid_map, relations = load_graph()
    print(f"       {len(eid_map)} entities, {len(relations)} relations")

    # Build a drug list for MCQ distractors.
    drug_pool = sorted({
        e["name"] for e in eid_map.values()
        if e.get("type") == "Drug" and e.get("name")
    })
    print(f"       {len(drug_pool)} drugs in pool for MCQ distractors")

    # Group usable relations by type.
    rel_by_type: dict[str, list[dict]] = defaultdict(list)
    for rel in relations:
        rtype = rel.get("type", "")
        if rtype not in SAMPLE_LIMITS:
            continue
        src = eid_map.get(rel.get("src", ""), {})
        dst = eid_map.get(rel.get("dst", ""), {})
        if not src or not dst:
            continue
        if is_spurious(src.get("name", "")) or is_spurious(dst.get("name", "")):
            continue
        rel_by_type[rtype].append(rel)

    # Keep MENTIONED_IN anchors that occur in at least two guidelines.
    if "MENTIONED_IN" in rel_by_type:
        from collections import Counter as _Counter
        src_guideline_count = _Counter(
            rel.get("src", "") for rel in rel_by_type["MENTIONED_IN"]
        )
        rel_by_type["MENTIONED_IN"] = [
            rel for rel in rel_by_type["MENTIONED_IN"]
            if src_guideline_count.get(rel.get("src", ""), 0) >= 2
        ]
        print(f"[filter] MENTIONED_IN after cross-guideline filter: "
              f"{len(rel_by_type['MENTIONED_IN'])} triples")

    # Keep RECOMMENDED_FOR anchors and sort them consistently.
    if "RECOMMENDED_FOR" in rel_by_type:
        rel_by_type["RECOMMENDED_FOR"] = sorted(
            rel_by_type["RECOMMENDED_FOR"],
            key=lambda r: (r.get("src",""), r.get("organization",""))
        )

    print("\n[plan] Relations to process:")
    total_planned = 0
    for rtype, limit in SAMPLE_LIMITS.items():
        available = len(rel_by_type.get(rtype, []))
        selected  = min(available, limit)
        total_planned += selected
        print(f"       {rtype:<25} available={available:>4}  selected={selected:>4}")
    print(f"       Total planned: {total_planned}")

    # Track generation counts.
    counters   = Counter()
    q_counter  = 0
    seen_questions: set[str] = set()   # exact-duplicate filter
    out_f      = OUT_PATH.open("w", encoding="utf-8")
    flag_f     = FLAGGED_PATH.open("w", encoding="utf-8")

    # Generate questions relation by relation.
    for rtype, rel_list in rel_by_type.items():
        limit  = SAMPLE_LIMITS[rtype]
        method = REL_TO_METHOD[rtype]
        qtype  = REL_TO_QTYPE[rtype]
        sample = rel_list[:limit]   # already deterministic; shuffle if desired

        print(f"\n[gen]  {rtype} ({len(sample)} triples, method={method})")

        for idx, rel in enumerate(sample):
            src = eid_map.get(rel.get("src", ""), {})
            dst = eid_map.get(rel.get("dst", ""), {})
            if not src or not dst:
                continue

            q_counter += 1
            qid = f"q_{q_counter:04d}"

            # Build the rule-based baseline question.
            canon = canonical_question(rel, src, dst)

            # Generate the primary question using its assigned method.
            question_text  = None
            answer_key     = ""
            reformulations = []
            mcq_options    = []
            correct_option = ""
            difficulty     = "recall"

            if method == "grounded_prompt":
                sys_p, usr_p, answer_key = build_grounded_prompt(rel, src, dst)
                question_text = call_llm(sys_p, usr_p)

                # Generate recall, application, and analysis reformulations.
                if question_text:
                    sys_p2, usr_p2, _ = build_paraphrase_prompt(question_text, rtype)
                    variants_raw = call_llm(sys_p2, usr_p2, response_format="json")
                    if variants_raw:
                        try:
                            variants = json.loads(variants_raw)
                            reformulations = [
                                variants.get("recall", ""),
                                variants.get("application", ""),
                                variants.get("analysis", ""),
                            ]
                            reformulations = [r for r in reformulations if r]
                        except json.JSONDecodeError:
                            pass

            elif method == "vignette":
                sys_p, usr_p, answer_key = build_vignette_prompt(rel, src, dst)
                question_text = call_llm(sys_p, usr_p)
                difficulty    = "application"

            elif method == "paraphrase":
                sys_p, usr_p, answer_key = build_paraphrase_prompt(canon, rtype)
                variants_raw = call_llm(sys_p, usr_p, response_format="json")
                question_text = canon   # primary = canonical
                difficulty = "recall"
                if variants_raw:
                    try:
                        variants = json.loads(variants_raw)
                        reformulations = [
                            variants.get("recall", ""),
                            variants.get("application", ""),
                            variants.get("analysis", ""),
                        ]
                        reformulations = [r for r in reformulations if r]
                        if reformulations:
                            question_text = reformulations[0]
                    except json.JSONDecodeError:
                        pass
                answer_key = src.get("name", "") or dst.get("name", "")

            # Use the baseline question if generation fails.
            if not question_text:
                question_text = canon
                print(f"    [fallback] {qid} → using canonical")
                counters["llm_fallback"] += 1

            # Use the baseline question when the model returns a refusal.
            elif looks_like_refusal(question_text):
                print(f"    [refusal] {qid} → LLM declined, using canonical")
                counters["llm_refusal"] += 1
                question_text = canon
                reformulations = []

            # Clean generated question text.
            question_text  = clean_question_text(question_text)
            reformulations = [clean_question_text(r) for r in reformulations if r]
            reformulations = [r for r in reformulations if r]

            # Skip duplicate primary questions.
            dedup_key = question_text.strip().lower()
            if dedup_key in seen_questions:
                counters["duplicate_skipped"] += 1
                q_counter -= 1   # do not consume an id for a skipped duplicate
                continue
            seen_questions.add(dedup_key)

            # Add MCQ options for eligible relations.
            if rtype in MCQ_RELATIONS and question_text and drug_pool:
                sys_d, usr_d = build_mcq_distractor_prompt(
                    question_text, answer_key, rtype, drug_pool
                )
                distractors_raw = call_llm(sys_d, usr_d, response_format="json")
                if distractors_raw:
                    try:
                        parsed = json.loads(distractors_raw)
                        # Accept either supported distractor response format.
                        if isinstance(parsed, dict):
                            distractors = parsed.get("distractors", [])
                        elif isinstance(parsed, list):
                            distractors = parsed
                        else:
                            distractors = []
                        if len(distractors) >= 3:
                            # Randomise the correct-answer position.
                            options = [answer_key] + distractors[:3]
                            random.shuffle(options)
                            letters = ["A", "B", "C", "D"]
                            mcq_options    = [f"{letters[i]}. {o}" for i, o in enumerate(options)]
                            correct_option = letters[options.index(answer_key)]
                    except (json.JSONDecodeError, ValueError):
                        pass

            # Validate the generated question.
            passed, note = validate(
                question_text, answer_key, rel, src, dst, eid_map
            )

            # Save the question to the accepted or flagged file.
            record = make_record(
                qid           = qid,
                question      = question_text,
                qtype         = qtype,
                method        = method,
                rel           = rel,
                src           = src,
                dst           = dst,
                answer_key    = answer_key,
                validator_passed = passed,
                validator_note   = note,
                canonical     = canon,
                difficulty    = difficulty,
                mcq_options   = mcq_options if mcq_options else None,
                correct_option= correct_option,
                reformulations= reformulations,
            )

            line = json.dumps(record, ensure_ascii=False)
            if passed:
                out_f.write(line + "\n")
                counters["passed"] += 1
            else:
                flag_f.write(line + "\n")
                counters["flagged"] += 1

            counters[rtype]  += 1
            counters[qtype]  += 1
            counters[method] += 1
            if mcq_options:
                counters["mcq"] += 1

            # Print progress every 20 questions.
            if idx % 20 == 0:
                print(f"    {idx+1}/{len(sample)} — {qid}: "
                      f"{'✓' if passed else '✗'} {note[:60]}")

    out_f.close()
    flag_f.close()

    # Save generation statistics.
    summary = {
        "total_generated":   q_counter,
        "passed_validator":  counters["passed"],
        "flagged_validator": counters["flagged"],
        "llm_fallbacks":     counters["llm_fallback"],
        "by_relation":  {rt: counters[rt] for rt in SAMPLE_LIMITS},
        "by_qtype":     {qt: counters[qt] for qt in set(REL_TO_QTYPE.values())},
        "by_method":    {m: counters[m]  for m in set(REL_TO_METHOD.values())},
        "mcq_generated": counters["mcq"],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*60)
    print("HYBRID QUESTION GENERATION — COMPLETE")
    print("="*60)
    print(f"  Total generated  : {q_counter}")
    print(f"  Passed validator : {counters['passed']}")
    print(f"  Flagged          : {counters['flagged']}")
    print(f"  LLM fallbacks    : {counters['llm_fallback']}")
    print(f"  MCQ questions    : {counters['mcq']}")
    print(f"\n  Output  : {OUT_PATH}")
    print(f"  Flagged : {FLAGGED_PATH}")
    print(f"  Summary : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()