from craigslistbargain.strategy.price_safety import BuyerPriceSafetyFilter


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RuleBasedBuyerOfferPlanner(object):
    def __init__(self, safety_filter=None, debug=False):
        self.safety_filter = safety_filter or BuyerPriceSafetyFilter(debug=debug)
        self.debug = bool(debug)

    def plan(self, raw_price, safe_price, context, buyer_strategy_info, tactic_state):
        if context.get("role") != "buyer":
            return {
                "planned_price": raw_price,
                "raw_price": raw_price,
                "safe_price": safe_price,
                "buyer_strategy": None,
                "reason": "role is not buyer",
                "changed": False,
            }
        if raw_price is None:
            return {
                "planned_price": None,
                "raw_price": raw_price,
                "safe_price": safe_price,
                "buyer_strategy": None,
                "reason": "raw_price is None",
                "changed": False,
            }

        last_seller = _as_float(context.get("last_seller_price"))
        last_buyer = _as_float(context.get("last_buyer_price"))
        safe = _as_float(safe_price)
        strategy = buyer_strategy_info.get("buyer_strategy", "neutral")
        scale = _as_float(buyer_strategy_info.get("concession_scale")) or 0.20

        if last_buyer is None or last_seller is None:
            planned = safe
            reason = "missing last buyer or seller price; start from safe price"
        else:
            gap = max(last_seller - last_buyer, 0.0)
            minimal_step = max(abs(last_seller) * 0.01, 1.0)
            if strategy == "firm":
                scale = min(scale, 0.05)
            elif strategy == "skeptical":
                scale = min(scale, 0.10)
            elif strategy == "reciprocal":
                seller_concession = _as_float((tactic_state or {}).get("features", {}).get("seller_concession"))
                if seller_concession is not None and seller_concession > 0:
                    scale = min(0.50, max(scale, seller_concession / max(gap, 1.0)))
            elif strategy == "close":
                scale = max(scale, 0.50)

            base_step = max(gap * scale, minimal_step if gap > 0 else 0.0)
            planned = last_buyer + base_step
            if safe is not None and strategy in ("firm", "skeptical"):
                planned = min(planned, safe)
            reason = "gap={} scale={} strategy={}".format(gap, scale, strategy)

        safety = self.safety_filter.check_and_fix(planned, context)
        planned = safety["safe_price"]
        if safety.get("violations"):
            reason = reason + "; final clamp: " + ",".join(safety["violations"])

        out = {
            "planned_price": planned,
            "raw_price": raw_price,
            "safe_price": safe_price,
            "buyer_strategy": strategy,
            "reason": reason,
            "changed": planned != raw_price,
        }
        if self.debug:
            print("[offer-planner] {}".format(out))
        return out
