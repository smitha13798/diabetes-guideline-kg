import json
import re
from collections import Counter, defaultdict


input_file = r".\outputs\questions\questions_hybrid.jsonl"
output_file = r".\outputs\questions\questions_hybrid_detailed_summary.json"

segment_size = 50


# Load the data

data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))


# MSTTR

def tokenize(text):
    """Turn text into lowercase word tokens."""
    return re.findall(r"\b[\w']+\b", text.lower())


def calculate_msttr(texts, segment_size=50):
    """
    Calculate Mean Segmental Type-Token Ratio (MSTTR).
    The text is split into non-overlapping segments of equal size.
    """

    tokens = []

    for text in texts:
        tokens.extend(tokenize(text))

    if not tokens:
        return 0

    segments = []

    for i in range(0, len(tokens), segment_size):
        segment = tokens[i:i + segment_size]

        # Only use complete segments
        if len(segment) == segment_size:
            segments.append(segment)

    # This should only matter for very small datasets
    if not segments:
        return len(set(tokens)) / len(tokens)

    ttr_values = []

    for segment in segments:
        unique_words = len(set(segment))
        total_words = len(segment)

        ttr_values.append(unique_words / total_words)

    return sum(ttr_values) / len(ttr_values)


# Overall statistics

primary_count = len(data)

reformulation_count = sum(
    len(item.get("reformulations", []))
    for item in data
)

total_phrasings = primary_count + reformulation_count

# Mean phrasings per anchor
mean_phrasings = total_phrasings / primary_count


# Collect primary questions and reformulations for MSTTR
all_phrasings = []

for item in data:

    if item.get("question"):
        all_phrasings.append(item["question"])

    all_phrasings.extend(item.get("reformulations", []))


overall_msttr = calculate_msttr(
    all_phrasings,
    segment_size
)


# Generation method counts

method_counts = Counter(
    item.get("generation_method", "unknown")
    for item in data
)


# Relation statistics

relations = defaultdict(lambda: {
    "total": 0,
    "grounded_prompt": 0,
    "vignette": 0,
    "paraphrase": 0,
    "mcq": 0
})


for item in data:

    relation = item["graph_anchor"]["rel_type"]
    method = item.get("generation_method", "unknown")

    relations[relation]["total"] += 1

    if method in ["grounded_prompt", "vignette", "paraphrase"]:
        relations[relation][method] += 1

    # Count questions that contain MCQ options
    if item.get("mcq_options"):
        relations[relation]["mcq"] += 1


# Question type statistics

question_types = defaultdict(lambda: {
    "total": 0,
    "grounded_prompt": 0,
    "vignette": 0,
    "paraphrase": 0,
    "mcq": 0,
    "mcq_by_method": {
        "grounded_prompt": 0,
        "vignette": 0,
        "paraphrase": 0
    }
})


for item in data:

    qtype = item.get("question_type", "unknown")
    method = item.get("generation_method", "unknown")

    question_types[qtype]["total"] += 1

    # Count generation methods for this question type
    if method in ["grounded_prompt", "vignette", "paraphrase"]:
        question_types[qtype][method] += 1

    # Check whether the question has MCQ options
    if item.get("mcq_options"):

        question_types[qtype]["mcq"] += 1

        # Also record which generation method produced the MCQ
        if method in ["grounded_prompt", "vignette", "paraphrase"]:
            question_types[qtype]["mcq_by_method"][method] += 1


# Build the final summary

summary = {

    "overall": {
        "primary_questions": primary_count,
        "reformulations": reformulation_count,
        "total_phrasings": total_phrasings,
        "mean_phrasings_per_anchor": round(mean_phrasings, 2),
        "msttr": round(overall_msttr, 4),
        "msttr_segment_size": segment_size
    },

    "generation_methods": dict(method_counts),

    "by_relation": dict(relations),

    "by_question_type": dict(question_types)
}


# Save the summary

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


# Print the main results

print("\nOVERALL")
print("-" * 40)

print("Primary questions:", primary_count)
print("Reformulations:", reformulation_count)
print("Total phrasings:", total_phrasings)
print("Mean phrasings per anchor:", round(mean_phrasings, 2))
print("MSTTR:", round(overall_msttr, 4))


print("\nGENERATION METHODS")
print("-" * 40)

for method, count in method_counts.items():
    print(f"{method}: {count}")


print("\nBY RELATION")
print("-" * 40)

for relation, values in relations.items():

    print(f"\n{relation}")
    print("  Total:", values["total"])
    print("  Grounded prompt:", values["grounded_prompt"])
    print("  Vignette:", values["vignette"])
    print("  Paraphrase:", values["paraphrase"])
    print("  MCQ:", values["mcq"])


print("\nBY QUESTION TYPE")
print("-" * 40)

for qtype, values in question_types.items():

    print(f"\n{qtype}")
    print("  Total:", values["total"])
    print("  Grounded prompt:", values["grounded_prompt"])
    print("  Vignette:", values["vignette"])
    print("  Paraphrase:", values["paraphrase"])
    print("  MCQ:", values["mcq"])

    print("  MCQ by method:")
    print(
        "    Grounded prompt:",
        values["mcq_by_method"]["grounded_prompt"]
    )
    print(
        "    Vignette:",
        values["mcq_by_method"]["vignette"]
    )
    print(
        "    Paraphrase:",
        values["mcq_by_method"]["paraphrase"]
    )


print("\nSummary saved to:", output_file)