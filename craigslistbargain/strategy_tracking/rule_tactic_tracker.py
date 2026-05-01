TACTICS = ("AEO", "LIC", "CSC", "RC", "NCR", "RNC", "REO", "CLOSING", "NEUTRAL")


def _num(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class RuleSellerTacticTracker(object):
    def __init__(self, debug=False):
        self.debug = bool(debug)
        self.reset()

    def reset(self):
        self.feature_history = []
        self.tactic_history = []
        self.last_tactic = None
        self.last_tactic_dist = None

    def _score(self, features):
        scores = dict((t, 0.05) for t in TACTICS)
        seller_price = _num(features.get("seller_price"), None)
        last_seller = _num(features.get("last_seller_price"), None)
        buyer_last = _num(features.get("buyer_last_price"), None)
        seller_concession = _num(features.get("seller_concession"), 0.0)
        buyer_concession = _num(features.get("buyer_concession"), 0.0)
        gap = _num(features.get("price_gap_to_buyer"), 0.0)

        ref = max(abs(seller_price or 0.0), abs(last_seller or 0.0), abs(buyer_last or 0.0), 1.0)
        small = 0.02 * ref
        large = 0.10 * ref

        if seller_price is not None and last_seller is not None and seller_price > last_seller + small:
            scores["RNC"] += 2.0
        if features.get("is_early_round") and gap > large and seller_concession <= small:
            scores["AEO"] += 1.5
        if features.get("is_early_round") and seller_concession > large:
            scores["LIC"] += 1.6
        if seller_concession > small and features.get("seller_concession_delta") is not None and features["seller_concession_delta"] < -small:
            scores["CSC"] += 1.4
        if buyer_concession > small and seller_concession > small:
            scores["RC"] += 1.5
        if buyer_concession > small and abs(seller_concession) <= small:
            scores["NCR"] += 1.5
        if gap > 2.0 * large and seller_concession <= small:
            scores["REO"] += 1.2
        if features.get("is_late_round") and (seller_concession > large or abs(gap) <= small):
            scores["CLOSING"] += 1.7

        if max(scores.values()) <= 0.1:
            scores["NEUTRAL"] += 1.0
        return scores

    def update(self, features):
        scores = self._score(features)
        total = sum(scores.values()) or 1.0
        tactic_dist = dict((k, scores[k] / total) for k in TACTICS)
        tactic_label = max(tactic_dist, key=tactic_dist.get)

        if self.last_tactic is None:
            switch_prob = 0.0
        elif tactic_label != self.last_tactic and tactic_dist[tactic_label] > 0.45:
            switch_prob = 0.75
        else:
            l1 = 0.0
            for tactic in TACTICS:
                prev = self.last_tactic_dist.get(tactic, 0.0) if self.last_tactic_dist else 0.0
                l1 += abs(tactic_dist.get(tactic, 0.0) - prev)
            switch_prob = min(1.0, l1 / 2.0)

        state = {
            "tactic_dist": tactic_dist,
            "tactic_label": tactic_label,
            "switch_prob": switch_prob,
            "features": features,
        }
        self.feature_history.append(features)
        self.tactic_history.append(state)
        self.last_tactic = tactic_label
        self.last_tactic_dist = tactic_dist

        if self.debug:
            print("[tactic-tracker] label={} switch={} dist={} features={}".format(
                tactic_label, switch_prob, tactic_dist, features))
        return state

    def get_state(self):
        if self.tactic_history:
            return self.tactic_history[-1]
        return {
            "tactic_dist": dict((t, 0.0) for t in TACTICS),
            "tactic_label": "NEUTRAL",
            "switch_prob": 0.0,
            "features": {},
        }
