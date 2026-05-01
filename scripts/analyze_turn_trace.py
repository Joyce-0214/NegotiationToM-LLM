import argparse
import csv
import os


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


def as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def present(value):
    return value is not None and value != ""


def close(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def row_id(index, row):
    dialogue_id = row.get("dialogue_id") or "<empty-dialogue-id>"
    turn_id = row.get("turn_id") or "?"
    return "csv_line={} dialogue_id={} turn_id={}".format(index + 2, dialogue_id, turn_id)


def print_examples(title, rows, limit):
    print("{}: {}".format(title, len(rows)))
    for index, row, reason in rows[:limit]:
        print("  - {} {}".format(row_id(index, row), reason))


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
            if close(planned, safe):
                planned_safe_equal += 1
            else:
                planned_safe_diff += 1

    print("buyer_rows_with_planned_price: {}".format(planned_present_rows))
    print_examples("planned_price_final_lf_mismatch_rows", planned_mismatch, args.examples)
    print("planned_price_equals_safe_price_rows: {}".format(planned_safe_equal))
    print("planned_price_differs_from_safe_price_rows: {}".format(planned_safe_diff))

    unsafe_intents = []
    for i, row in buyer_rows:
        intent = row.get("final_intent_used_by_lf") or row.get("raw_intent")
        seller_ask = as_float(row.get("last_seller_price"))
        buyer_limit = as_float(row.get("buyer_limit"))
        if intent in UNSAFE_ACCEPT_INTENTS and seller_ask is not None and buyer_limit is not None:
            if seller_ask > buyer_limit:
                unsafe_intents.append(
                    (i, row, "intent={} last_seller_price={} buyer_limit={}".format(intent, seller_ask, buyer_limit))
                )
    print_examples("unsafe_buyer_accept_or_agree_rows", unsafe_intents, args.examples)

    print("price_column_stats:")
    for col in PRICE_COLUMNS:
        vals = [as_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            print("  - {}: no numeric values".format(col))
            continue
        norm_like = sum(1 for v in vals if 0.0 <= v <= 1.0)
        real_like = sum(1 for v in vals if abs(v) > 2.0)
        print(
            "  - {}: count={} min={} max={} norm_like_0_to_1={} real_like_abs_gt_2={}".format(
                col, len(vals), min(vals), max(vals), norm_like, real_like)
        )

    mixed_scale_rows = []
    for i, row in enumerate(rows):
        vals = [as_float(row.get(col)) for col in PRICE_COLUMNS]
        vals = [v for v in vals if v is not None]
        if any(0.0 <= v <= 1.0 for v in vals) and any(abs(v) > 2.0 for v in vals):
            mixed_scale_rows.append((i, row, "same row has both 0..1-like and >2 price values"))
    print_examples("mixed_scale_suspicious_rows", mixed_scale_rows, args.examples)

    return 1 if empty_dialogue or planned_mismatch or unsafe_intents or mixed_scale_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
