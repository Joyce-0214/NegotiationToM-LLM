import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from onmt.RLModels import SwitchAwareHistoryModel
from craigslistbargain.neural.a2c_trainer import RLTrainer
import craigslistbargain.neural.a2c_trainer as trainer_mod


class FakeBatch(SimpleNamespace):
    pass


def make_fake_opt():
    return SimpleNamespace(
        num_intents=5,
        sa_d_hist=128,
        sa_d_ctx=128,
        sa_d_buyer=128,
        sa_d_obs=256,
        sa_d_h=256,
        sa_k_type=4,
        sa_d_core=256,
        sa_f_num=11,
        dropout=0.0,
    )


def make_fake_tom_batch(batch_size, state_dim, extra_dim, text_dim, n_intents=5):
    uttr = torch.randn(batch_size, text_dim)

    identity_state = torch.zeros(batch_size, 8)
    act_ids = torch.randint(0, 4, (batch_size,))
    identity_state[torch.arange(batch_size), act_ids] = 1.0
    identity_state[:, 7] = torch.rand(batch_size)

    state = torch.randn(batch_size, state_dim)
    extra = torch.rand(batch_size, extra_dim)

    act_intent = torch.randint(0, n_intents, (batch_size,))
    act_price = torch.randint(0, 100, (batch_size,))
    act_price_mask = torch.ones(batch_size, 1)

    strategy = torch.randint(0, 5, (batch_size,))

    return FakeBatch(
        size=batch_size,
        uttr=uttr,
        identity_state=identity_state,
        state=state,
        extra=extra,
        act_intent=act_intent,
        act_price=act_price,
        act_price_mask=act_price_mask,
        strategy=strategy,
    )


def bind_fake_compute_loss(trainer):
    def _compute_loss(self, batch, policy, price, loss=None):
        loss_intent = F.cross_entropy(
            policy, batch.act_intent.reshape(-1), reduction='none'
        )
        loss_price = F.cross_entropy(
            price, batch.act_price.reshape(-1), reduction='none'
        )
        loss_price = loss_price.reshape(-1, 1) * batch.act_price_mask
        return loss_intent, loss_price.reshape(-1), None

    trainer._compute_loss = types.MethodType(_compute_loss, trainer)


def main():
    torch.manual_seed(13)

    opt = make_fake_opt()

    state_dim = 14
    extra_dim = 3
    text_dim = 64

    tom = SwitchAwareHistoryModel(
        opt=opt,
        state_dim=state_dim,
        extra_dim=extra_dim,
        text_dim=text_dim,
        n_intents=5,
    )
    tom.eval()

    # 如果你还没在 RLModels.py 里补 hidden_vec，这里先补一个，避免 trainer 日志逻辑报错
    if not hasattr(tom, "hidden_vec"):
        tom.hidden_vec = None

    trainer = RLTrainer.__new__(RLTrainer)
    trainer.tom = tom
    trainer.tom_loss = None
    trainer.hidden_vec = []
    trainer.hidden_stra = []
    trainer.tom_identity_loss = torch.nn.CrossEntropyLoss(reduction='none')

    bind_fake_compute_loss(trainer)

    batch1 = make_fake_tom_batch(batch_size=4, state_dim=state_dim, extra_dim=extra_dim, text_dim=text_dim)
    batch2 = make_fake_tom_batch(batch_size=2, state_dim=state_dim, extra_dim=extra_dim, text_dim=text_dim)

    # monkeypatch：让 ToMBatch.from_raw 直接返回我们的 fake batch
    old_from_raw = trainer_mod.ToMBatch.from_raw
    trainer_mod.ToMBatch.from_raw = staticmethod(lambda raw, strategy: raw)

    try:
        losses, accus, logs = trainer._tom_gradient_accumulation(
            batch_iter=[batch1, batch2],
            strategy=[0, 1, 2, 3],
            model=tom,
            ret_table={'id': False, 'tom': True},
            id_gt=False,
        )
    finally:
        trainer_mod.ToMBatch.from_raw = old_from_raw

    print("=== trainer tom smoke ===")
    print("intent loss steps:", len(losses['tom'][0]))
    print("price  loss steps:", len(losses['tom'][1]))
    print("intent accu steps:", len(accus['tom'][0]))

    assert len(losses['tom'][0]) == 2
    assert len(losses['tom'][1]) == 2
    assert len(accus['tom'][0]) == 2

    pred_identity, pred_intent, pred_price, strategies = logs
    print("pred_intent steps:", len(pred_intent))
    print("pred_price  steps:", len(pred_price))

    assert len(pred_intent) == 2
    assert len(pred_price) == 2

    print("TRAINER TOM SMOKE TEST PASSED")


if __name__ == "__main__":
    main()