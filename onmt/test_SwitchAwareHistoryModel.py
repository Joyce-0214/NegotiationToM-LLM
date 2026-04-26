import torch
from types import SimpleNamespace
from onmt.RLModels import SwitchAwareHistoryModel

opt = SimpleNamespace(
    num_intents=5,
    sa_d_hist=128,
    sa_d_ctx=128,
    sa_d_buyer=128,
    sa_d_obs=256,
    sa_d_h=256,
    sa_k_type=4,
    sa_d_core=256,
    dropout=0.0,
)

B = 4
state_dim = 18
extra_dim = 3
text_dim = 64

model = SwitchAwareHistoryModel(opt, state_dim, extra_dim, text_dim)
model.eval()

state = torch.randn(B, state_dim)
extra = torch.rand(B, extra_dim)
uttr_emb = torch.randn(B, text_dim)

identity_state = torch.zeros(B, 8)
identity_state[torch.arange(B), torch.tensor([0, 1, 2, 3])] = 1.0
identity_state[:, 7] = torch.tensor([0.2, 0.5, 0.7, 0.9])

with torch.no_grad():
    predictions, hidden_next, identity_next = model(
        state=state,
        identity_state=identity_state,
        extra=extra,
        uttr_emb=uttr_emb,
        hidden=None,
    )

print(predictions["intent_logits"].shape)
print(predictions["price_logits"].shape)
print(predictions["style_logits"].shape)
print(predictions["switch_prob"].shape)
print(identity_next.shape)