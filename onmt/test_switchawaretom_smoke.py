import torch

from SwitchAwareToM import StepBatch, SwitchAwareExplicitToM


def main():
    torch.manual_seed(7)

    B = 4
    n_acts = 12
    n_price_bins = 100
    d_text_in = 64
    d_hist = 128
    d_ctx = 128
    d_buyer = 128
    f_num = 11

    model = SwitchAwareExplicitToM(
        n_acts=n_acts,
        n_price_bins=n_price_bins,
        n_intents=5,
        n_styles=4,
        d_text_in=d_text_in,
        f_num=f_num,
        d_hist=d_hist,
        d_ctx=d_ctx,
        d_buyer=d_buyer,
        d_obs=256,
        d_h=256,
        k_type=4,
        d_core=256,
        dropout=0.0,
    )
    model.eval()

    batch = StepBatch(
        seller_act_id=torch.randint(0, n_acts, (B,)),
        seller_price_bin=torch.randint(0, n_price_bins, (B,)),
        seller_num_feats=torch.randn(B, f_num),
        seller_text_emb=torch.randn(B, d_text_in),
        hist_state=torch.randn(B, d_hist),
        scenario_ctx=torch.randn(B, d_ctx),
        buyer_state=torch.randn(B, d_buyer),
        prev_seller_num_feats=torch.randn(B, f_num),
    )

    with torch.no_grad():
        out = model.step(batch)

    expected = {
        'intent_logits': (B, 5),
        'price_logits': (B, 100),
        'style_logits': (B, 4),
        'switch_logit': (B, 1),
        'switch_prob': (B, 1),
        'belief_type_probs': (B, 4),
        'belief_cont_mu': (B, 16),
        'belief_cont_logvar': (B, 16),
        'belief_confidence': (B, 1),
        'belief_alpha': (B, 1),
        'seller_obs': (B, 256),
        'seller_hidden': (B, 256),
        'prev_seller_hidden': (B, 256),
        'buyer_core': (B, 256),
        'core_price': (B, 256),
        'core_style': (B, 256),
    }

    print('=== Output shape check ===')
    for k, shape in expected.items():
        actual = tuple(out[k].shape)
        ok = actual == shape
        print(f'{k:20s} expected={shape} actual={actual} ok={ok}')
        assert ok, f'{k} shape mismatch: expected {shape}, got {actual}'

    print('\n=== Value sanity check ===')
    tp = out['belief_type_probs']
    sp = out['switch_prob']
    conf = out['belief_confidence']
    alpha = out['belief_alpha']

    print('belief_type row sums:', tp.sum(dim=-1))
    print('switch_prob min/max:', sp.min().item(), sp.max().item())
    print('belief_confidence min/max:', conf.min().item(), conf.max().item())
    print('belief_alpha min/max:', alpha.min().item(), alpha.max().item())

    assert torch.allclose(tp.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert torch.all((sp >= 0.0) & (sp <= 1.0))
    assert torch.all((conf >= 0.0) & (conf <= 1.0))
    assert torch.all((alpha >= 0.05) & (alpha <= 0.95))

    print('\nSMOKE TEST PASSED')


if __name__ == '__main__':
    main()
