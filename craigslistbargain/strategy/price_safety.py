from __future__ import print_function


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_buyer_limit(context):
    """Best-effort buyer limit lookup.

    CraigslistBargain stores the buyer's maximum acceptable price in
    kb.facts["personal"]["Target"] in the current data. The fallback order is
    explicit context first, then known KB/scenario locations, so callers do not
    need to hard-code one field name.
    """
    for key in ("buyer_limit", "buyer_max_willingness"):
        val = _as_float(context.get(key))
        if val is not None:
            return val, key

    kb = context.get("kb")
    if kb is not None:
        facts = getattr(kb, "facts", None)
        if isinstance(facts, dict):
            personal = facts.get("personal", {})
            if personal.get("Role") == "buyer":
                val = _as_float(personal.get("Target"))
                if val is not None:
                    return val, 'kb.facts["personal"]["Target"]'

    scenario = context.get("scenario")
    kbs = getattr(scenario, "kbs", None)
    if kbs is None and isinstance(scenario, dict):
        kbs = scenario.get("kbs")
    if kbs:
        for kb_item in kbs:
            facts = getattr(kb_item, "facts", kb_item)
            if not isinstance(facts, dict):
                continue
            personal = facts.get("personal", {})
            if personal.get("Role") == "buyer":
                val = _as_float(personal.get("Target"))
                if val is not None:
                    return val, 'scenario.kbs[*]["personal"]["Target"]'

    return None, None


class BuyerPriceSafetyFilter(object):
    def __init__(self, debug=False):
        self.debug = bool(debug)

    def check_and_fix(self, raw_price, context):
        role = context.get("role")
        violations = []
        warnings = []

        if role != "buyer":
            return {
                "safe_price": raw_price,
                "raw_price": raw_price,
                "changed": False,
                "violations": violations,
                "reason": "role is not buyer",
            }

        if raw_price is None:
            return {
                "safe_price": None,
                "raw_price": raw_price,
                "changed": False,
                "violations": ["raw_price_none"],
                "reason": "raw_price is None",
            }

        price = _as_float(raw_price)
        if price is None:
            return {
                "safe_price": None,
                "raw_price": raw_price,
                "changed": True,
                "violations": ["raw_price_not_numeric"],
                "reason": "raw_price is not numeric",
            }

        buyer_limit, limit_source = resolve_buyer_limit(context)
        last_seller_price = _as_float(context.get("last_seller_price"))
        last_buyer_price = _as_float(context.get("last_buyer_price"))
        allow_price_decrease = bool(context.get("allow_price_decrease", False))

        lower_bound = None
        upper_bound = None
        if last_buyer_price is not None and not allow_price_decrease:
            lower_bound = last_buyer_price
        if buyer_limit is not None:
            upper_bound = buyer_limit
        if last_seller_price is not None:
            upper_bound = last_seller_price if upper_bound is None else min(upper_bound, last_seller_price)

        if buyer_limit is not None and price > buyer_limit:
            violations.append("above_buyer_limit")
        if last_seller_price is not None and price > last_seller_price:
            violations.append("above_last_seller_price")
        if lower_bound is not None and price < lower_bound:
            violations.append("below_last_buyer_price")

        safe_price = price
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            warnings.append("lower_bound_above_upper_bound")
            if upper_bound is not None:
                safe_price = min(price, upper_bound)
            elif last_buyer_price is not None:
                safe_price = last_buyer_price
        else:
            if lower_bound is not None and safe_price < lower_bound:
                safe_price = lower_bound
            if upper_bound is not None and safe_price > upper_bound:
                safe_price = upper_bound

        changed = safe_price != price
        reason = "ok"
        if violations:
            reason = "clamped: " + ",".join(violations)
        if warnings:
            reason = reason + "; warnings: " + ",".join(warnings)

        if self.debug:
            print(
                "[price-safety] role={} raw={} safe={} buyer_limit={} source={} "
                "last_buyer={} last_seller={} violations={} warnings={}".format(
                    role, raw_price, safe_price, buyer_limit, limit_source,
                    last_buyer_price, last_seller_price, violations, warnings,
                )
            )

        return {
            "safe_price": safe_price,
            "raw_price": raw_price,
            "changed": changed,
            "violations": violations + warnings,
            "reason": reason,
        }
