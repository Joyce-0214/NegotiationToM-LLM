"""
Strategy Annotation Script for CraigslistBargain Dataset
Annotates each seller turn with:
1. Pricing Strategy: OPENING_HIGH, FIRM, GRADUAL, FLEXIBLE, HOLD, NONE
2. Language Style: ASSERTIVE, COOPERATIVE, PERSUASIVE, NEUTRAL
3. Switch Detection: marks turns where strategy/style changes from previous turn

Concession ratio uses g_t = (p_{t-1} - p_t) / (p_open - p_target)
This ensures consistency: same absolute concession is evaluated relative to
total bargaining space, making strategies comparable across dialogue stages.
"""

import json
import re
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# ============== Pricing Strategy Detection ==============

PRICE_PATTERN = re.compile(r'\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)')

def extract_price(text: str) -> Optional[float]:
    """Extract price from text, return None if no price found."""
    matches = PRICE_PATTERN.findall(text)
    if matches:
        # Take the last mentioned price (usually the offer)
        price_str = matches[-1].replace(',', '')
        return float(price_str)
    return None

def classify_price_strategy(
    current_price: Optional[float],
    prev_price: Optional[float],
    opening_price: Optional[float],
    listing_price: float,
    target_price: float,
    is_first_offer: bool,
    role: str
) -> str:
    """
    Classify pricing strategy for a single turn.

    For sellers:
    - OPENING_HIGH: first offer >= listing_price or >= target + 0.8*(listing-target)
    - FIRM: concession < 5% of remaining bargaining space
    - GRADUAL: concession 5-15% of remaining space
    - FLEXIBLE: concession > 15% of remaining space
    - HOLD: price unchanged or concession < 2%

    Concession ratio g_t = (p_{t-1} - p_t) / (p_open - p_target)
    This ensures consistency across dialogue stages.

    Returns: OPENING_HIGH, FIRM, GRADUAL, FLEXIBLE, HOLD, or NONE
    """
    if current_price is None:
        return "NONE"

    if role.lower() != 'seller':
        return "NONE"  # Only annotate seller strategies

    bargaining_space = listing_price - target_price
    if bargaining_space <= 0:
        bargaining_space = listing_price * 0.3  # fallback

    # First offer check - OPENING_HIGH
    if is_first_offer:
        high_threshold = target_price + 0.8 * bargaining_space
        if current_price >= listing_price or current_price >= high_threshold:
            return "OPENING_HIGH"
        return "FIRM"

    # Subsequent offers - use g_t (relative to total bargaining space)
    if prev_price is None or opening_price is None:
        return "HOLD"

    # Calculate remaining space from opening price
    total_space = opening_price - target_price
    if total_space <= 0:
        total_space = bargaining_space  # fallback

    concession = prev_price - current_price

    # g_t = concession / total_bargaining_space
    epsilon = 1.0  # avoid division by zero
    g_t = concession / max(epsilon, total_space)

    # Classification thresholds
    if concession <= 0 or g_t < 0.02:
        return "HOLD"  # No concession or negligible
    elif g_t < 0.05:
        return "FIRM"
    elif g_t < 0.15:
        return "GRADUAL"
    else:
        return "FLEXIBLE"

# ============== Language Style Detection ==============

STYLE_KEYWORDS = {
    'ASSERTIVE': [
        'no', 'cannot', "can't", 'firm', 'lowest', 'final', 'sorry',
        'not possible', 'unable', 'must', 'need to', "won't", 'never'
    ],
    'COOPERATIVE': [
        'sure', 'okay', 'yes', 'maybe', "let's", 'understand', 'agree',
        'work with', 'help', 'consider', 'fair', 'deal', 'sounds good'
    ],
    'PERSUASIVE': [
        'quality', 'worth', 'value', 'great', 'excellent', 'condition',
        'brand new', 'original', 'rare', 'perfect', 'amazing', 'best',
        'bargain', 'steal', 'compare', 'retail', 'features'
    ]
}

def classify_language_style(text: str) -> str:
    """
    Classify language style based on keyword presence.
    Returns: ASSERTIVE, COOPERATIVE, PERSUASIVE, or NEUTRAL
    """
    text_lower = text.lower()

    scores = {style: 0 for style in STYLE_KEYWORDS}

    for style, keywords in STYLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[style] += 1

    # Get highest scoring style
    max_score = max(scores.values())

    if max_score == 0:
        return "NEUTRAL"

    for style, score in scores.items():
        if score == max_score:
            return style

    return "NEUTRAL"

# ============== Main Annotation Logic ==============

def annotate_dialogue(dialogue: Dict) -> Dict:
    """
    Annotate a single dialogue with per-turn strategy labels.

    Returns dialogue with added 'annotations' field containing:
    - turn_annotations: list of {turn_id, agent, pricing_strategy, language_style}
    - switch_points: list of turn_ids where strategy/style changed
    - seller_trajectory: sequence of (pricing, style) tuples for seller
    """
    events = dialogue.get('events', [])
    scenario = dialogue.get('scenario', {})

    # Extract scenario info
    kbs = scenario.get('kbs', [{}, {}])

    # Find seller's KB
    seller_idx = None
    for i, kb in enumerate(kbs):
        role = kb.get('personal', {}).get('Role', '') or kb.get('role', '')
        if role.lower() == 'seller':
            seller_idx = i
            break

    if seller_idx is None:
        seller_idx = 0  # fallback

    seller_kb = kbs[seller_idx]
    listing_price = seller_kb.get('item', {}).get('Price', 0)
    if isinstance(listing_price, str):
        listing_price = float(listing_price.replace('$', '').replace(',', ''))

    target_price = seller_kb.get('personal', {}).get('Target', listing_price * 0.7)
    if isinstance(target_price, str):
        target_price = float(target_price.replace('$', '').replace(',', ''))

    # Track state
    turn_annotations = []
    seller_prev_price = None
    seller_opening_price = None  # First seller offer price
    seller_opening_type = None   # HIGH or NORMAL
    seller_first_offer = True
    prev_seller_pricing = None
    prev_seller_style = None
    switch_points = []
    seller_trajectory = []

    for turn_id, event in enumerate(events):
        if event.get('action') != 'message':
            continue

        agent_id = event.get('agent', 0)
        text = event.get('data', '')

        # Determine role
        agent_role = 'seller' if agent_id == seller_idx else 'buyer'

        # Extract price
        price = extract_price(text)

        # Classify
        if agent_role == 'seller':
            raw_pricing = classify_price_strategy(
                price, seller_prev_price, seller_opening_price,
                listing_price, target_price, seller_first_offer, agent_role
            )

            # Separate opening_type from concession_strategy
            if seller_first_offer and price is not None:
                if raw_pricing == "OPENING_HIGH":
                    seller_opening_type = "HIGH"
                    pricing_strategy = "FIRM"  # First turn defaults to FIRM for concession tracking
                else:
                    seller_opening_type = "NORMAL"
                    pricing_strategy = "FIRM"
            else:
                pricing_strategy = raw_pricing

            language_style = classify_language_style(text)

            # Update tracking
            if price is not None:
                if seller_first_offer:
                    seller_opening_price = price  # Record opening price
                seller_prev_price = price
                seller_first_offer = False

            # If this seller turn didn't mention a price, inherit previous pricing strategy.
            # This avoids breaking the seller's pricing trajectory on non-price turns.
            if pricing_strategy == "NONE" and prev_seller_pricing is not None:
                pricing_strategy = prev_seller_pricing

            # Detect switch - separate price, style, and joint switches
            if prev_seller_pricing is not None and prev_seller_style is not None:
                price_changed = (pricing_strategy != prev_seller_pricing)
                style_changed = (language_style != prev_seller_style)

                if price_changed or style_changed:
                    # Classify switch type
                    if price_changed and style_changed:
                        switch_type = "JOINT"
                    elif price_changed:
                        switch_type = "PRICE"
                    else:
                        switch_type = "STYLE"

                    switch_points.append({
                        'turn_id': turn_id,
                        'switch_type': switch_type,  # PRICE, STYLE, or JOINT
                        'price_changed': price_changed,
                        'style_changed': style_changed,
                        'from': {'pricing': prev_seller_pricing, 'style': prev_seller_style},
                        'to': {'pricing': pricing_strategy, 'style': language_style}
                    })

            if pricing_strategy != "NONE":
                prev_seller_pricing = pricing_strategy
            prev_seller_style = language_style
            seller_trajectory.append((pricing_strategy, language_style))
        else:
            pricing_strategy = "N/A"
            language_style = classify_language_style(text)

        turn_annotations.append({
            'turn_id': turn_id,
            'agent': agent_id,
            'role': agent_role,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'extracted_price': price,
            'pricing_strategy': pricing_strategy,
            'language_style': language_style
        })

    # Compute summary stats
    has_switch = len(switch_points) > 0
    num_switches = len(switch_points)

    return {
        **dialogue,
        'annotations': {
            'turn_annotations': turn_annotations,
            'switch_points': switch_points,
            'seller_trajectory': seller_trajectory,
            'has_strategy_switch': has_switch,
            'num_switches': num_switches,
            'listing_price': listing_price,
            'target_price': target_price,
            'seller_opening_price': seller_opening_price,
            'seller_opening_type': seller_opening_type  # HIGH or NORMAL
        }
    }

def annotate_dataset(input_path: str, output_path: str):
    """Annotate entire dataset and save."""
    print(f"Loading data from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both list format and dict format
    if isinstance(data, dict):
        dialogues = data.get('dialogues', data.get('examples', []))
    else:
        dialogues = data

    print(f"Annotating {len(dialogues)} dialogues...")

    annotated = []
    stats = {
        'total': len(dialogues),
        'with_switches': 0,
        'switch_counts': defaultdict(int),
        'switch_type_counts': defaultdict(int),  # PRICE, STYLE, JOINT
        'opening_type_counts': defaultdict(int),  # HIGH, NORMAL
        'pricing_dist': defaultdict(int),
        'style_dist': defaultdict(int)
    }

    for i, dialogue in enumerate(dialogues):
        annotated_dialogue = annotate_dialogue(dialogue)
        annotated.append(annotated_dialogue)

        # Collect stats
        ann = annotated_dialogue['annotations']
        if ann['has_strategy_switch']:
            stats['with_switches'] += 1
        stats['switch_counts'][ann['num_switches']] += 1

        # Count switch types
        for switch in ann['switch_points']:
            stats['switch_type_counts'][switch['switch_type']] += 1

        # Count opening types
        if ann['seller_opening_type']:
            stats['opening_type_counts'][ann['seller_opening_type']] += 1

        for turn in ann['turn_annotations']:
            if turn['role'] == 'seller':
                stats['pricing_dist'][turn['pricing_strategy']] += 1
                stats['style_dist'][turn['language_style']] += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(dialogues)}")

    # Save annotated data
    print(f"Saving to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotated, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n" + "="*50)
    print("ANNOTATION STATISTICS")
    print("="*50)
    print(f"Total dialogues: {stats['total']}")
    print(f"Dialogues with strategy switches: {stats['with_switches']} ({100*stats['with_switches']/stats['total']:.1f}%)")
    print(f"\nOpening type distribution:")
    for otype, count in sorted(stats['opening_type_counts'].items()):
        print(f"  {otype}: {count}")
    print(f"\nSwitch count distribution:")
    for n, count in sorted(stats['switch_counts'].items()):
        print(f"  {n} switches: {count} dialogues")
    print(f"\nSwitch type distribution:")
    for stype, count in sorted(stats['switch_type_counts'].items()):
        print(f"  {stype}: {count}")
    print(f"\nPricing strategy distribution (seller turns):")
    for strategy, count in sorted(stats['pricing_dist'].items(), key=lambda x: -x[1]):
        print(f"  {strategy}: {count}")
    print(f"\nLanguage style distribution (seller turns):")
    for style, count in sorted(stats['style_dist'].items(), key=lambda x: -x[1]):
        print(f"  {style}: {count}")

    # Save stats
    stats_path = output_path.replace('.json', '_stats.json')
    stats['switch_counts'] = dict(stats['switch_counts'])
    stats['switch_type_counts'] = dict(stats['switch_type_counts'])
    stats['opening_type_counts'] = dict(stats['opening_type_counts'])
    stats['pricing_dist'] = dict(stats['pricing_dist'])
    stats['style_dist'] = dict(stats['style_dist'])
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate CraigslistBargain dialogues with strategy labels')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path')
    args = parser.parse_args()

    annotate_dataset(args.input, args.output)
