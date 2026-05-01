import csv
import json
import os


def _stringify(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def append_turn_trace(path, row):
    if not path:
        return
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    row = dict((k, _stringify(v)) for k, v in row.items())
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    fieldnames = [
        "dialogue_id", "turn_id", "role", "raw_intent",
        "final_intent_used_by_lf", "price_unit",
        "raw_price_before_safety", "safe_price", "planned_price",
        "final_price_used_by_lf", "last_buyer_price", "last_seller_price",
        "buyer_limit", "price_safety_changed", "price_safety_violations",
        "seller_price", "seller_tactic_label", "seller_tactic_dist",
        "seller_switch_prob", "seller_concession", "price_gap_to_buyer",
        "buyer_strategy", "planner_reason", "intent_safety_changed",
        "intent_safety_reason", "lf_price_sync_changed", "utterance",
    ]
    for key in row.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
