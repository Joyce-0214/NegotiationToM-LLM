import argparse
import csv
import os
from collections import Counter


PRICE_COLUMNS = [
    "raw_price_before_safety",
    "safe_price",
    "planned_price",
    "final_price_used_by_lf",
    "seller_price",
    "last_buyer_price",
    "last_seller_price",
    "buyer_limit",
]

UNSAFE_ACCEPT_INTENTS = set(["accept", "agree", "agree-noprice"])
COUNTER_ACCEPT_PHRASES = (
    "deal",
    "i'll take it",
    "ill take it",
    "i will take it",
    "take it for",
    "sounds good",
    "i agree",
    "accepted",
    "that works",
    "i accept",
)


def as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def present(value):
    return value is not None and str(value).strip() != ""


def close(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(a - b) <= max(tol, abs(a) * tol, abs(b) * tol)


def utterance_text(value):
    if value is None:
        return ""
    return str(value).strip()


def detect_act_text_mismatch(row):
    intent = row.get("final_intent_used_by_lf") or row.get("raw_intent")
    text = utterance_text(row.get("utterance")).lower()
    if not text:
        return False, None
    if intent == "counter":
        for phrase in COUNTER_ACCEPT_PHRASES:
            if phrase in text:
                return True, "counter utterance contains accept-like phrase: {}".format(phrase)
    return False, None


def row_id(index, row):
    dialogue_id = row.get("dialogue_id") or "<empty-dialogue-id>"
    turn_id = row.get("turn_id") or "?"
    return "csv_line={} dialogue_id={} turn_id={}".format(index + 2, dialogue_id, turn_id)


def print_examples(title, rows, limit):
    print("{}: {}".format(title, len(rows)))
    for index, row, reason in rows[:limit]:
        print("  - {} {}".format(row_id(index, row), reason))


def print_distribution(title, values):
    counter = Counter(v for v in values if present(v))
    if not counter:
        print("{}: no values".format(title))
        return
    print("{}: {}".format(title, dict(counter.most_common())))


def main():
    parser = argparse.ArgumentParser(description="Analyze turn trace price-safety/planner invariants.")
    parser.add_argument("csv_path", help="Path to turn_trace CSV.")
    parser.add_argument("--examples", type=int, default=5, help="Number of example rows to print per issue.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print("MISSING: {}".format(args.csv_path))
        return 2

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    buyer_rows = [(i, r) for i, r in enumerate(rows) if r.get("role") == "buyer"]
    seller_rows = [(i, r) for i, r in enumerate(rows) if r.get("role") == "seller"]

    print("rows: {}".format(len(rows)))
    print("buyer_rows: {}".format(len(buyer_rows)))
    print("seller_rows: {}".format(len(seller_rows)))
    print("dialogue_ids: {}".format(len(set(r.get("dialogue_id") for r in rows if r.get("dialogue_id")))))

    empty_dialogue = [(i, r, "dialogue_id is empty") for i, r in enumerate(rows) if not r.get("dialogue_id")]
    print_examples("empty_dialogue_id_rows", empty_dialogue, args.examples)

    planned_mismatch = []
    planned_present_rows = 0
    planned_safe_equal = 0
    planned_safe_diff = 0
    planner_compared = 0
    planner_no_effect = []
    for i, row in buyer_rows:
        planned = as_float(row.get("planned_price"))
        final_price = as_float(row.get("final_price_used_by_lf"))
        safe = as_float(row.get("safe_price"))

        if planned is not None:
            planned_present_rows += 1
            if not close(planned, final_price):
                planned_mismatch.append(
                    (i, row, "planned_price={} final_price_used_by_lf={}".format(planned, final_price))
                )
        if planned is not None and safe is not None:
            planner_compared += 1
            if close(planned, safe):
                planned_safe_equal += 1
                planner_no_effect.append((i, row, "planned_price == safe_price == {}".format(planned)))
            else:
                planned_safe_diff += 1

    print("buyer_rows_with_planned_price: {}".format(planned_present_rows))
    print_examples("planned_price_final_lf_mismatch_rows", planned_mismatch, args.examples)
    print("planned_price_equals_safe_price_rows: {}".format(planned_safe_equal))
    print("planned_price_differs_from_safe_price_rows: {}".format(planned_safe_diff))
    ratio = (float(len(planner_no_effect)) / planner_compared) if planner_compared else 0.0
    print("planner_no_effect_rows: {} of {} ({:.2%})".format(len(planner_no_effect), planner_compared, ratio))

    unsafe_intents = []
    final_above_seller = []
    final_above_limit = []
    for i, row in buyer_rows:
        intent = row.get("final_intent_used_by_lf") or row.get("raw_intent")
        seller_ask = as_float(row.get("last_seller_price"))
        buyer_limit = as_float(row.get("buyer_limit"))
        final_price = as_float(row.get("final_price_used_by_lf"))
        if intent in UNSAFE_ACCEPT_INTENTS and seller_ask is not None and buyer_limit is not None:
            if seller_ask > buyer_limit:
                unsafe_intents.append(
                    (i, row, "intent={} last_seller_price={} buyer_limit={}".format(intent, seller_ask, buyer_limit))
                )
        if final_price is not None and seller_ask is not None and final_price > seller_ask:
            final_above_seller.append(
                (i, row, "final_price_used_by_lf={} last_seller_price={}".format(final_price, seller_ask))
            )
        if final_price is not None and buyer_limit is not None and final_price > buyer_limit:
            final_above_limit.append(
                (i, row, "final_price_used_by_lf={} buyer_limit={}".format(final_price, buyer_limit))
            )
    print_examples("unsafe_buyer_accept_or_agree_rows", unsafe_intents, args.examples)
    print_examples("final_above_last_seller_price_rows", final_above_seller, args.examples)
    print_examples("final_above_buyer_limit_rows", final_above_limit, args.examples)

    historical_mismatch = []
    last_seller_price_by_dialogue = {}
    for i, row in enumerate(rows):
        dialogue_id = row.get("dialogue_id")
        if row.get("role") == "seller":
            seller_price = as_float(row.get("final_price_used_by_lf"))
            if seller_price is None:
                seller_price = as_float(row.get("seller_price"))
            if seller_price is not None:
                last_seller_price_by_dialogue[dialogue_id] = (seller_price, i, row)
        elif row.get("role") == "buyer" and dialogue_id in last_seller_price_by_dialogue:
            expected, seller_i, _ = last_seller_price_by_dialogue[dialogue_id]
            observed = as_float(row.get("last_seller_price"))
            if observed is None:
                historical_mismatch.append(
                    (i, row, "previous seller price={} at csv_line={} but last_seller_price is missing".format(
                        expected, seller_i + 2))
                )
            elif not close(expected, observed):
                historical_mismatch.append(
                    (i, row, "previous seller price={} at csv_line={} but buyer last_seller_price={}".format(
                        expected, seller_i + 2, observed))
                )
    print_examples("historical_price_mismatch_rows", historical_mismatch, args.examples)

    suspicious_tiny = []
    for i, row in enumerate(rows):
        final_price = as_float(row.get("final_price_used_by_lf"))
        buyer_limit = as_float(row.get("buyer_limit"))
        if final_price is not None and buyer_limit is not None:
            if 1.0 <= final_price <= 3.0 and buyer_limit > 100.0:
                suspicious_tiny.append(
                    (i, row, "final_price_used_by_lf={} buyer_limit={}".format(final_price, buyer_limit))
                )
    print_examples("suspicious_tiny_final_price_rows", suspicious_tiny, args.examples)

    empty_utterance = [
        (i, r, "utterance is empty")
        for i, r in enumerate(rows)
        if not utterance_text(r.get("utterance"))
    ]
    print_examples("empty_utterance_rows", empty_utterance, args.examples)

    act_text_mismatch = []
    for i, row in enumerate(rows):
        if "act_text_mismatch" in fieldnames and truthy(row.get("act_text_mismatch")):
            reason = row.get("act_text_mismatch_reason") or "trace flagged mismatch"
            act_text_mismatch.append((i, row, reason))
            continue
        mismatch, reason = detect_act_text_mismatch(row)
        if mismatch:
            act_text_mismatch.append((i, row, reason))
    print_examples("act_text_mismatch_rows", act_text_mismatch, args.examples)

    source_fields_available = (
        "real_price_source_missing" in fieldnames
        or "last_seller_price_source" in fieldnames
        or "last_buyer_price_source" in fieldnames
    )
    real_price_source_missing = []
    for i, row in enumerate(rows):
        if truthy(row.get("real_price_source_missing")):
            real_price_source_missing.append((i, row, "trace real_price_source_missing=True"))
        elif source_fields_available and row.get("role") == "buyer":
            last_seller = as_float(row.get("last_seller_price"))
            last_buyer = as_float(row.get("last_buyer_price"))
            if last_seller is not None and not present(row.get("last_seller_price_source")):
                real_price_source_missing.append((i, row, "last_seller_price has no source"))
            elif last_buyer is not None and not present(row.get("last_buyer_price_source")):
                real_price_source_missing.append((i, row, "last_buyer_price has no source"))
    print_examples("real_price_source_missing_rows", real_price_source_missing, args.examples)

    print("price_column_stats:")
    for col in PRICE_COLUMNS:
        vals = [as_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            print("  - {}: no numeric values".format(col))
            continue
        norm_like = sum(1 for v in vals if 0.0 <= v <= 1.0)
        tiny_like = sum(1 for v in vals if 1.0 <= v <= 3.0)
        real_like = sum(1 for v in vals if abs(v) > 3.0)
        print(
            "  - {}: count={} min={} max={} norm_like_0_to_1={} tiny_like_1_to_3={} real_like_abs_gt_3={}".format(
                col, len(vals), min(vals), max(vals), norm_like, tiny_like, real_like)
        )

    mixed_scale_rows = []
    for i, row in enumerate(rows):
        vals = [as_float(row.get(col)) for col in PRICE_COLUMNS]
        vals = [v for v in vals if v is not None]
        if any(0.0 <= v <= 1.0 for v in vals) and any(abs(v) > 3.0 for v in vals):
            mixed_scale_rows.append((i, row, "same row has both 0..1-like and >3 price values"))
        elif any(1.0 <= v <= 3.0 for v in vals) and as_float(row.get("buyer_limit")) is not None:
            buyer_limit = as_float(row.get("buyer_limit"))
            if buyer_limit and buyer_limit > 100.0:
                mixed_scale_rows.append((i, row, "same row has 1..3 tiny price with buyer_limit={}".format(buyer_limit)))
    print_examples("mixed_scale_suspicious_rows", mixed_scale_rows, args.examples)

    print_distribution("buyer_strategy_distribution", [r.get("buyer_strategy") for _, r in buyer_rows])
    print_distribution("seller_tactic_label_distribution", [r.get("seller_tactic_label") for _, r in seller_rows])
    seller_switch_vals = [
        as_float(r.get("seller_switch_prob"))
        for _, r in seller_rows
        if as_float(r.get("seller_switch_prob")) is not None
    ]
    if seller_switch_vals:
        print(
            "seller_switch_prob_stats: count={} min={} max={} avg={}".format(
                len(seller_switch_vals),
                min(seller_switch_vals),
                max(seller_switch_vals),
                sum(seller_switch_vals) / len(seller_switch_vals),
            )
        )
    else:
        print("seller_switch_prob_stats: no values")

    failures = (
        empty_dialogue
        or planned_mismatch
        or unsafe_intents
        or final_above_seller
        or final_above_limit
        or historical_mismatch
        or suspicious_tiny
        or empty_utterance
        or act_text_mismatch
        or real_price_source_missing
        or mixed_scale_rows
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
