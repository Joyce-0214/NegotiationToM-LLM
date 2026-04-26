import torch
from types import SimpleNamespace
from craigslistbargain.neural.rl_model_builder import make_rl_model


class DummyVocab:
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size


def make_fake_opt():
    return SimpleNamespace(
        model='tom',
        mappings='dummy',
        use_utterance=True,
        word_vec_size=64,
        hidden_size=64,
        hidden_depth=1,
        tom_hidden_size=64,
        tom_hidden_depth=3,
        id_hidden_size=64,
        id_hidden_depth=1,
        tom_model='switch_aware',
        state_length=2,
        dia_num=0,
        dropout=0.0,
        param_init=0.1,
        sa_d_hist=128,
        sa_d_ctx=128,
        sa_d_buyer=128,
        sa_d_obs=256,
        sa_d_h=256,
        sa_k_type=4,
        sa_d_core=256,
        sa_f_num=11,
        num_intents=5,
    )


def main():
    torch.manual_seed(11)

    opt = make_fake_opt()

    mappings = {
        'lf_vocab': DummyVocab(5),
        'utterance_vocab': DummyVocab(1000),
        'tgt_vocab': DummyVocab(1000),
    }

    actor, critic, tom = make_rl_model(
        model_opt=opt,
        mappings=mappings,
        gpu=False,
        checkpoint=None,
        load_type='from_rl',
        exclude={},
    )

    print(type(tom))

    B = 4
    intent_size = mappings['lf_vocab'].size
    state_dim = (intent_size + 2) * 2   # 和 builder 里一致
    extra_dim = 3
    text_dim = opt.word_vec_size

    # ===== 按 trainer 风格构造输入 =====
    # uttr：这里先直接构造 [B, D] tensor，wrapper 内部直接当 emb 用
    uttr = torch.randn(B, text_dim)

    # identity_state：前7维动作 one-hot，第8维价格
    identity_state = torch.zeros(B, 8)
    fake_act_ids = torch.tensor([0, 1, 2, 3])
    identity_state[torch.arange(B), fake_act_ids] = 1.0
    identity_state[:, 7] = torch.tensor([0.10, 0.45, 0.70, 0.95])

    # state / extra
    state = torch.randn(B, state_dim)
    extra = torch.rand(B, extra_dim)

    print("\n=== First call ===")
    with torch.no_grad():
        predictions, hidden_next, identity_next = tom(
            uttr, identity_state, state, extra, None, None
        )

    intent_logits, price_logits = predictions
    print("intent_logits:", tuple(intent_logits.shape))
    print("price_logits :", tuple(price_logits.shape))
    print("identity_next:", tuple(identity_next.shape))

    h_next, belief_next = hidden_next
    print("seller_hidden:", tuple(h_next.shape))
    print("belief.type_probs:", tuple(belief_next.type_probs.shape))
    print("belief.cont_mu:", tuple(belief_next.cont_mu.shape))
    print("belief.cont_logvar:", tuple(belief_next.cont_logvar.shape))
    print("belief.confidence:", tuple(belief_next.confidence.shape))

    assert intent_logits.shape == (B, 5)
    assert price_logits.shape == (B, 100)
    assert identity_next.shape == (B, 37)

    print("\n=== Second call with hidden_next ===")
    with torch.no_grad():
        predictions2, hidden_next2, identity_next2 = tom(
            uttr, identity_state, state, extra, hidden_next, None
        )

    intent_logits2, price_logits2 = predictions2
    print("intent_logits2:", tuple(intent_logits2.shape))
    print("price_logits2 :", tuple(price_logits2.shape))
    print("identity_next2:", tuple(identity_next2.shape))

    assert intent_logits2.shape == (B, 5)
    assert price_logits2.shape == (B, 100)
    assert identity_next2.shape == (B, 37)

    print("\nFORWARD TEST PASSED")


if __name__ == "__main__":
    main()