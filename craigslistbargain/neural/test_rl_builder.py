import torch
from types import SimpleNamespace
from craigslistbargain.neural.rl_model_builder import make_rl_model

class DummyVocab:
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size

opt = SimpleNamespace(
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
)

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
    load_type='from_sl',
    exclude={}
)

print(type(actor))
print(type(critic))
print(type(tom))