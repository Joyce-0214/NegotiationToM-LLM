def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_price(event):
    if event is None:
        return None
    if isinstance(event, dict):
        data = event
    else:
        data = getattr(event, "metadata", None)
    if isinstance(data, dict):
        return _as_float(data.get("price"))
    return None


def _event_intent(event):
    if event is None:
        return None
    data = event if isinstance(event, dict) else getattr(event, "metadata", None)
    if isinstance(data, dict):
        return data.get("intent")
    return None


def _event_utterance(event):
    if event is None:
        return None
    if isinstance(event, dict):
        return event.get("utterance") or event.get("data")
    return getattr(event, "data", None)


def extract_turn_features(history, current_event, role, market_price=None, round_id=None, max_round=None):
    history = history or []
    seller_price = _event_price(current_event)
    raw_utterance = _event_utterance(current_event)
    intent = _event_intent(current_event)

    seller_prices = []
    buyer_prices = []
    for item in history:
        if isinstance(item, dict):
            item_role = item.get("role")
            item_price = _as_float(item.get("price"))
        else:
            item_role = item[0] if len(item) > 0 else None
            item_price = _as_float(item[1]) if len(item) > 1 else None
        if item_price is None:
            continue
        if item_role == "seller":
            seller_prices.append(item_price)
        elif item_role == "buyer":
            buyer_prices.append(item_price)

    last_seller_price = seller_prices[-1] if seller_prices else None
    prev_seller_price = seller_prices[-2] if len(seller_prices) > 1 else None
    buyer_last_price = buyer_prices[-1] if buyer_prices else None
    buyer_prev_price = buyer_prices[-2] if len(buyer_prices) > 1 else None

    seller_concession = None
    if last_seller_price is not None and seller_price is not None:
        seller_concession = last_seller_price - seller_price

    prev_seller_concession = None
    if prev_seller_price is not None and last_seller_price is not None:
        prev_seller_concession = prev_seller_price - last_seller_price

    seller_concession_delta = None
    if seller_concession is not None and prev_seller_concession is not None:
        seller_concession_delta = seller_concession - prev_seller_concession

    buyer_concession = None
    if buyer_last_price is not None and buyer_prev_price is not None:
        # Buyer concessions move upward in a single-price bargain.
        buyer_concession = buyer_last_price - buyer_prev_price

    price_gap_to_buyer = None
    if seller_price is not None and buyer_last_price is not None:
        price_gap_to_buyer = seller_price - buyer_last_price

    rid = round_id if round_id is not None else len(history)
    if max_round:
        round_ratio = float(rid) / float(max(max_round, 1))
    else:
        round_ratio = 0.0

    return {
        "role": role,
        "seller_price": seller_price,
        "last_seller_price": last_seller_price,
        "buyer_last_price": buyer_last_price,
        "buyer_prev_price": buyer_prev_price,
        "seller_concession": seller_concession,
        "buyer_concession": buyer_concession,
        "seller_concession_delta": seller_concession_delta,
        "price_gap_to_buyer": price_gap_to_buyer,
        "market_price": _as_float(market_price),
        "round_id": rid,
        "round_ratio": round_ratio,
        "is_early_round": round_ratio <= 0.33 if max_round else rid <= 2,
        "is_late_round": round_ratio >= 0.67 if max_round else False,
        "raw_utterance": raw_utterance,
        "intent": intent,
        "pressure_score": 0.0,
        "evasion_score": 0.0,
        "deception_score": 0.0,
    }
