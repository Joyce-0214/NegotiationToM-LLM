"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.RLModels import \
    HistoryEncoder, HistoryIDEncoder, CurrentEncoder, HistoryIdentity, \
    HistoryModel, CurrentModel, \
    MixedPolicy, SinglePolicy
from onmt.Utils import use_gpu

from cocoa.io.utils import read_pickle
from craigslistbargain.neural import make_model_mappings

from onmt.RLModels import SwitchAwareHistoryModel

def make_embeddings(opt, word_dict, emb_length, for_encoder=True):
    return nn.Embedding(len(word_dict), emb_length)


def make_identity(opt, intent_size, hidden_size, hidden_depth, identity_dim=2, emb=None):
    diaact_size = (intent_size+1+1)
    extra_size = 3
    if hidden_size is None:
        hidden_size = opt.hidden_size
    identity = HistoryIdentity(diaact_size * 2, hidden_size, extra_size,
                               identity_dim=identity_dim, hidden_depth=hidden_depth,
                               uttr_emb=emb)

    return identity

def make_encoder(opt, embeddings, intent_size, output_size, use_history=False, hidden_depth=1, identity=None,
                 hidden_size=None):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    # encoder = StateEncoder(intent_size=intent_size, output_size=output_size,
    #                     state_length=opt.state_length, extra_size=3 if opt.dia_num>0 else 0 )

    # intent + price
    diaact_size = (intent_size+1)
    extra_size = 3 + 2
    if hidden_size is None:
        hidden_size = opt.hidden_size
    if not opt.use_utterance:
        embeddings = None
    if use_history:
        extra_size = 3
        # + pmask
        diaact_size += 1
        if identity is None:
            encoder = HistoryIDEncoder(None, diaact_size * 2, extra_size, embeddings, output_size,
                                       hidden_depth=hidden_depth, rnn_state=True)
        else:
            # encoder = HistoryIDEncoder(identity, diaact_size*2+extra_size, embeddings, output_size,
            #                            hidden_depth=hidden_depth)
            encoder = HistoryIDEncoder(identity, diaact_size * 2, extra_size, embeddings, output_size,
                                       hidden_depth=hidden_depth, rnn_state=True)
    else:
        if identity is None:
            encoder = CurrentEncoder(diaact_size*opt.state_length+extra_size, embeddings, output_size,
                                     hidden_depth=hidden_depth)
        else:
            extra_size = 3
            # + pmask
            diaact_size += 1
            encoder = HistoryIDEncoder(identity, diaact_size * opt.state_length, extra_size, embeddings, output_size,
                                       hidden_depth=hidden_depth)

    return encoder


def make_decoder(opt, encoder_size, intent_size, hidden_size, price_action=False, output_value=False, hidden_depth=2):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if output_value:
        return SinglePolicy(encoder_size, intent_size, hidden_size=hidden_size, hidden_depth=hidden_depth)
    if price_action:
        return MixedPolicy(encoder_size, intent_size, 4, hidden_size=hidden_size, hidden_depth=hidden_depth)
    return MixedPolicy(encoder_size, intent_size, 1, hidden_size=hidden_size, hidden_depth=hidden_depth)
    # return PolicyDecoder(encoder_size=encoder_size, intent_size=intent_size)


def load_test_model(model_path, opt, dummy_opt, new_opt, model_type='sl', load_type=None, exclude={}):
    if model_path is not None:
        print('Load model from {}.'.format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        model_opt = checkpoint['opt']
        for arg in dummy_opt:
            if (arg in new_opt) or (arg not in model_opt):
                if model_opt.__dict__.get(arg) != dummy_opt[arg]:
                    print('update: {}:{} -> {}'.format(arg, model_opt.__dict__.get(arg), dummy_opt[arg]))
                model_opt.__dict__[arg] = dummy_opt[arg]
    else:
        print('Build model from scratch.')
        checkpoint = None
        model_opt = opt

    mappings = read_pickle('{}/vocab.pkl'.format(model_opt.mappings))
    # mappings = read_pickle('{0}/{1}/vocab.pkl'.format(model_opt.mappings, model_opt.model))
    mappings = make_model_mappings(model_opt.model, mappings)

    if model_type == 'sl':
        model = make_sl_model(model_opt, mappings, use_gpu(opt), checkpoint)
        model.eval()

        return mappings, model, model_opt
    else:
        actor, critic, tom = make_rl_model(model_opt, mappings, use_gpu(opt), checkpoint, load_type, exclude=exclude)
        actor.eval()
        critic.eval()
        tom.eval()
        return mappings, (actor, critic, tom), model_opt


def select_param_from(params, names):
    selected = {}
    for k in params:
        for name in names:
            if k.find(name) == 0:
                selected[k] = params[k]
                break
    return selected


def transfer_critic_model(model, checkpoint, model_opt, model_name='model'):
    # Load encoder and init decoder.
    print('Transfer sl parameters to {}.'.format(model_name))
    model_dict = model.state_dict()
    pretrain_dict = select_param_from(checkpoint[model_name], ['encoder'])
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    if model_opt.param_init != 0.0:
        for p in model.decoder.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)


def transfer_actor_model(model, checkpoint, model_opt, model_name='model'):
    # Load encoder and init decoder.
    print('Transfer sl parameters to {}.'.format(model_name))
    model_dict = model.state_dict()
    pretrain_dict = select_param_from(checkpoint[model_name], ['encoder', 'decoder.common_net', 'decoder.intent_net'])
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    if model_opt.param_init != 0.0:
        for p in model.decoder.price_net.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)

def init_model(model, checkpoint, model_opt, model_name='model'):

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint[model_name])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                # don't init embedding
                if p.requires_grad:
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)

def make_rl_model(model_opt, mappings, gpu, checkpoint=None, load_type='from_sl', exclude={}):
    intent_size = mappings['lf_vocab'].size

    # ===== actor / critic 公共部分 =====
    src_dict = mappings['utterance_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)

    rl_encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.hidden_size)
    rl_encoder.fix_emb = True
    src_embeddings.weight.requires_grad_(False)

    actor_decoder = make_decoder(
        model_opt, model_opt.hidden_size, intent_size,
        model_opt.hidden_size, price_action=True
    )
    critic_decoder = make_decoder(
        model_opt, model_opt.hidden_size, 1,
        model_opt.hidden_size, output_value=True
    )

    actor_model = CurrentModel(rl_encoder, actor_decoder, fix_encoder=True)
    critic_model = CurrentModel(rl_encoder, critic_decoder, fix_encoder=True)

    # ===== ToM 分支 =====
    if model_opt.tom_model == 'switch_aware':
        # 复用旧 history path 的结构尺寸假设：
        # diaact_size = intent_size + 1(price) + 1(pmask)
        # state_dim = diaact_size * 2
        state_dim = (intent_size + 2) * 2
        extra_dim = 3
        text_dim = model_opt.word_vec_size

        tom_model = SwitchAwareHistoryModel(
            opt=model_opt,
            state_dim=state_dim,
            extra_dim=extra_dim,
            text_dim=text_dim,
            n_intents=intent_size,
        )
        use_tom_checkpoint = False

    else:
        if model_opt.tom_model in ['history', 'naive']:
            tom_encoder = make_encoder(
                model_opt, src_embeddings, intent_size, model_opt.tom_hidden_size,
                use_history=(model_opt.tom_model == 'history'),
                hidden_depth=model_opt.tom_hidden_depth,
                identity=None,
                hidden_size=model_opt.tom_hidden_size
            )
        else:
            id_emb, tom_emb = None, src_embeddings
            if model_opt.tom_model in ['uttr_id_history_tom', 'uttr_fid_history_tom', 'uttr_id']:
                id_emb, tom_emb = src_embeddings, None

            tom_identity = make_identity(
                model_opt, intent_size, model_opt.id_hidden_size,
                hidden_depth=model_opt.id_hidden_depth,
                identity_dim=7,
                emb=id_emb
            )

            tom_encoder = make_encoder(
                model_opt, tom_emb, intent_size, model_opt.tom_hidden_size,
                use_history=('history' in model_opt.tom_model),
                hidden_depth=model_opt.tom_hidden_depth,
                identity=tom_identity,
                hidden_size=model_opt.tom_hidden_size
            )

            if model_opt.tom_model == 'uttr_fid_history_tom':
                tom_encoder.fix_identity = False

        tom_encoder.fix_emb = True
        if getattr(tom_encoder, 'identity', None) is not None:
            tom_encoder.identity.fix_emb = True

        tom_decoder = make_decoder(
            model_opt, model_opt.tom_hidden_size, intent_size,
            model_opt.tom_hidden_size, hidden_depth=1
        )
        tom_model = HistoryModel(tom_encoder, tom_decoder)
        use_tom_checkpoint = (not exclude.get('tom'))

    # ===== 初始化 / 加载 =====
    print('load type:', load_type)

    if checkpoint is None:
        print('No checkpoint provided; fall back to random initialization.')
        load_type = 'from_rl'
    elif load_type == 'from_sl' and checkpoint.get('tom') is not None:
        print('In fact, load from rl!')
        load_type = 'from_rl'

    if checkpoint is not None and load_type == 'from_sl' and checkpoint.get('tom') is not None:
        print('In fact, load from rl!')
        load_type = 'from_rl'

    random_init_rl = False

    if load_type == 'from_sl':
        if random_init_rl:
            transfer_critic_model(actor_model, checkpoint, model_opt, 'model')
        else:
            transfer_actor_model(actor_model, checkpoint, model_opt, 'model')
        transfer_critic_model(critic_model, checkpoint, model_opt, 'model')

        # switch_aware v1 默认不加载旧 tom checkpoint
        init_model(tom_model, None if not use_tom_checkpoint else checkpoint, model_opt, 'tom')

    else:
        init_model(actor_model, checkpoint, model_opt, 'model')
        init_model(critic_model, checkpoint, model_opt, 'critic')
        init_model(tom_model, checkpoint if use_tom_checkpoint else None, model_opt, 'tom')

    if gpu:
        actor_model.cuda()
        critic_model.cuda()
        tom_model.cuda()
    else:
        actor_model.cpu()
        critic_model.cpu()
        tom_model.cpu()

    return actor_model, critic_model, tom_model

def make_sl_model(model_opt, mappings, gpu, checkpoint=None):
    intent_size = mappings['lf_vocab'].size

    # Make encoder.
    src_dict = mappings['utterance_vocab']
    src_embeddings = make_embeddings(model_opt, src_dict, model_opt.word_vec_size)
    encoder = make_encoder(model_opt, src_embeddings, intent_size, model_opt.hidden_size)
    # print('encoder', encoder)

    # Make decoder.
    tgt_dict = mappings['tgt_vocab']

    decoder = make_decoder(model_opt, model_opt.hidden_size, intent_size, model_opt.hidden_size)
    # print('decoder', decoder)

    model = CurrentModel(encoder, decoder)

    init_model(model, checkpoint, model_opt)

    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model



# def build_tom_model(opt, state_dim, extra_dim, text_dim):
#     if opt.tom_model == "history":
#         return HistoryModel(opt)
#     elif opt.tom_model == "naive":
#         return NaiveModel(opt)
#     elif opt.tom_model == "switch_aware":
#         return SwitchAwareHistoryModel(
#             opt=opt,
#             state_dim=state_dim,
#             extra_dim=extra_dim,
#             text_dim=text_dim,
#         )
#     else:
#         raise ValueError(f"Unknown tom_model: {opt.tom_model}")