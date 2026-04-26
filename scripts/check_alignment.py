import argparse
import json
import os
import re
from collections import Counter, defaultdict


CLOSING_RE = re.compile(
    r"\b("
    r"deal(?!\s+with)|it's a deal|its a deal|we have a deal|you have a deal|got a deal|"
    r"done deal|sold|i accept|i will accept|i can accept|accepted|all yours|"
    r"see you soon|see you then|see you later|pick it up|come pick it up|"
    r"come get it|get it today|i'll be there|ill be there"
    r")\b"
)
POSITIVE_WILLING_RE = re.compile(
    r"\b("
    r"i can do that|i can do it|i can do \d|i can do pprriiccee|"
    r"that works|works for me|sounds good|sounds great|"
    r"i'll take it|i will take it|i can take it"
    r")\b"
)
PRICE_RE = re.compile(
    r"(PPRRIICCEE|\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\bfor\s+\d+(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(bucks|dollars)\b)",
    re.IGNORECASE,
)


def inspect_text(intent, text):
    if text is None:
        return []
    text = str(text)
    lowered = text.lower()
    reasons = []

    if intent in {"affirm", "agree-noprice", "counter-noprice", "inform", "confirm"} and CLOSING_RE.search(lowered):
        reasons.append("contains deal-closing language")

    if intent in {"agree-noprice", "counter-noprice", "affirm"} and PRICE_RE.search(text):
        reasons.append("mentions an explicit price for a no-price / weak-ack act")

    if intent in {"deny", "disagree"} and (CLOSING_RE.search(lowered) or POSITIVE_WILLING_RE.search(lowered)):
        reasons.append("sounds accepting for a negative act")

    if intent == "counter-noprice" and POSITIVE_WILLING_RE.search(lowered):
        reasons.append("sounds like acceptance rather than continued bargaining")

    if intent == "affirm" and POSITIVE_WILLING_RE.search(lowered):
        reasons.append("sounds stronger than simple affirmation")

    return reasons


def load_templates(path):
    with open(path, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    rows = []
    for category, intents in data.items():
        for intent, roles in intents.items():
            if not isinstance(roles, dict):
                continue
            for role, utterances in roles.items():
                for idx, text in enumerate(utterances):
                    rows.append(
                        {
                            "source": "template",
                            "category": category,
                            "role": role,
                            "intent": intent,
                            "text": text,
                            "location": "{} / {} / {} / {}".format(category, intent, role, idx),
                        }
                    )
    return rows


def load_log(path):
    rows = []
    pending_text = None

    with open(path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")

            if line.startswith("[") and "\tmessage\t" in line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    pending_text = "\t".join(parts[2:]).strip()
                else:
                    pending_text = None
                continue

            if pending_text is None:
                continue

            match = re.search(r"<([^>]+)>", line)
            if match:
                intent = match.group(1).strip()
                rows.append(
                    {
                        "source": "log",
                        "category": None,
                        "role": None,
                        "intent": intent,
                        "text": pending_text,
                        "location": os.path.basename(path),
                    }
                )
                pending_text = None

    return rows


def main():
    parser = argparse.ArgumentParser(description="Flag likely action-text mismatches in templates or dialogue logs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--templates", type=str, help="Path to nlg_templates_dict.json")
    group.add_argument("--log", type=str, help="Path to a generated *_example*.txt log")
    parser.add_argument("--max-examples", type=int, default=30, help="Maximum suspicious examples to print")
    args = parser.parse_args()

    rows = load_templates(args.templates) if args.templates else load_log(args.log)

    suspicious = []
    by_intent = Counter()
    by_reason = Counter()
    examples_by_intent = defaultdict(list)

    for row in rows:
        reasons = inspect_text(row["intent"], row["text"])
        if not reasons:
            continue
        suspicious.append((row, reasons))
        by_intent[row["intent"]] += 1
        for reason in reasons:
            by_reason[reason] += 1
        if len(examples_by_intent[row["intent"]]) < 3:
            examples_by_intent[row["intent"]].append((row, reasons))

    print("checked:", len(rows))
    print("suspicious:", len(suspicious))
    if rows:
        print("suspicious_rate: {:.4f}".format(len(suspicious) / float(len(rows))))

    print("\nby_intent:")
    for intent, count in by_intent.most_common():
        print("  {:>16}  {}".format(intent, count))

    print("\nby_reason:")
    for reason, count in by_reason.most_common():
        print("  {:>40}  {}".format(reason, count))

    print("\nexamples:")
    printed = 0
    for intent, items in sorted(examples_by_intent.items()):
        for row, reasons in items:
            if printed >= args.max_examples:
                return
            print("- [{}] {} :: {}".format(intent, row["location"], "; ".join(reasons)))
            print("  {}".format(row["text"]))
            printed += 1


if __name__ == "__main__":
    main()
