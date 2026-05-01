class RuleBuyerResponsePolicy(object):
    def __init__(self, debug=False):
        self.debug = bool(debug)

    def select(self, tactic_state, context):
        dist = (tactic_state or {}).get("tactic_dist", {}) or {}

        def high(*labels):
            return max([dist.get(label, 0.0) for label in labels] or [0.0])

        if high("RNC", "NCR") >= 0.20:
            strategy, scale, reason = "firm", 0.05, "seller is not conceding or raised price"
        elif high("AEO", "REO") >= 0.20:
            strategy, scale, reason = "skeptical", 0.10, "seller is holding an aggressive position"
        elif high("RC", "CSC") >= 0.20:
            strategy, scale, reason = "reciprocal", 0.35, "seller is making reciprocal or smaller concessions"
        elif high("LIC", "CLOSING") >= 0.20:
            strategy, scale, reason = "close", 0.50, "seller appears open to closing"
        else:
            strategy, scale, reason = "neutral", 0.20, "no strong seller tactic"

        out = {
            "buyer_strategy": strategy,
            "concession_scale": scale,
            "reason": reason,
        }
        if self.debug:
            print("[buyer-response] {}".format(out))
        return out
