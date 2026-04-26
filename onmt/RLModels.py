import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from onmt.SwitchAwareToM import StepBatch, BeliefState, SwitchAwareExplicitToM

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq

from .Models import RNNEncoder

"""
利用pytorch搭建“强化学习/对话策略模型”的零件。
首先，把当前输入、历史状态、话语文本、身份信息编码成向量，再交给策略网络输出动作。
文件包括：编码器encoder、策略头policy、模型封装model wrapper

文本utterance + 额外特征extra + 对话行为/历史状态 -> encoder -> embedding -> policy -> 输出动作logits
"""



class MultilayerPerceptron(nn.Module):
    """
        final_output:
            If true, last layer also have activation function.
    """
    def __init__(self, input_size, layer_size, layer_depth, final_output=None):
        super(MultilayerPerceptron, self).__init__()

        last_size = input_size
        hidden_layers = []
        if layer_depth == 0:
            self.hidden_layers = nn.Identity()
        else:
            for i in range(layer_depth):   # 按层数循环构建网络
                if final_output is not None and i == layer_depth-1:   # final_output不为空，并且当前是最后一层
                    hidden_layers += [nn.Linear(last_size, final_output)]
                else:
                    hidden_layers += [nn.Linear(last_size, layer_size), nn.ReLU(layer_size)]
                last_size = layer_size
            self.hidden_layers = nn.Sequential(*hidden_layers)   # 收集到的层按顺序封装成一个顺序网络

    def forward(self, input):   # 前向传播
        return self.hidden_layers(input)


class CurrentEncoder(nn.Module):
    # intent | price | roles | number of history
    # 意图、价格、角色、历史条数
    def __init__(self, input_size, embeddings, output_size, hidden_size=64, hidden_depth=2):
        # 额外特征维度、词嵌入层、编码输出维度、RNN隐层维度、后面MLP的层数
        super(CurrentEncoder, self).__init__()

        self.fix_emb = False   # 默认不冻结embedding梯度

        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            self.uttr_emb = embeddings
            self.uttr_lstm = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)   # 构建一个LSTM来编码utterance

            hidden_input = input_size + hidden_size
            # 最后 MLP 的输入维度 = extra 维度 + utterance 编码维度

        else:
            hidden_input = input_size   # 如果没有 utterance embedding，那就只看 extra

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size, hidden_depth)   # 用 MLP 把拼接后的特征映射到目标空间

    def forward(self, uttr, extra):
        batch_size = extra.shape[0]

        if uttr is not None:
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_lstm(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)

            hidden_input = torch.cat([uttr_emb, extra], dim=-1)
        else:
            hidden_input = extra

        emb = self.hidden_layer(hidden_input)

        return emb


class HistoryIdentity(nn.Module):

    """
        从历史中编码”identity“
    """

    def __init__(self, diaact_size, last_lstm_size, extra_size, identity_dim,
                 uttr_emb=None,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn'):
        super(HistoryIdentity, self).__init__()

        self.fix_emb = False
        self.identity_dim = identity_dim
        self.uttr_emb = uttr_emb


        if rnn_type == 'lstm':
            self.rnnh_number = 2
            self.dia_rnn = torch.nn.LSTMCell(input_size=diaact_size, hidden_size=last_lstm_size)
        else:
            self.rnnh_number = 1
            self.dia_rnn = torch.nn.RNNCell(input_size=diaact_size, hidden_size=last_lstm_size)

        hidden_input = last_lstm_size + extra_size

        # language part
        if uttr_emb:
            uttr_emb_size = uttr_emb.embedding_dim
            # if rnn_type == 'lstm':
            self.uttr_rnn = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            # else:
            #     self.uttr_rnn = torch.nn.RNN(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            hidden_input += hidden_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, hidden_size, hidden_depth, final_output=identity_dim)

    def _uttr_forward(self, uttr, batch_size):
        # Uttrance part
        # uttr: [tensor, tensor, ...], each tensor in shape (len_str, 1).
        uttr = uttr.copy()
        with torch.set_grad_enabled(not self.fix_emb):
            for i, u in enumerate(uttr):
                if u.dtype != torch.int64:
                    print('uttr_emb:', uttr)
                # print('uttr_emb', next(self.uttr_emb.parameters()).device, u.device)
                uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
            # print(uttr[i].shape)
        uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
        # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

        _, output = self.uttr_rnn(uttr)

        # For LSTM case, output=(h_1, c_1)
        if isinstance(output, tuple):
            output = output[0]

        uttr_emb = output.reshape(batch_size, -1)
        return uttr_emb

    def forward(self, diaact, extra, last_hidden, uttr=None):
        batch_size = diaact.shape[0]
        next_hidden = self.dia_rnn(diaact, last_hidden)
        if isinstance(next_hidden, tuple):
            # For LSTM
            dia_emb = next_hidden[0].reshape(batch_size, -1)
        else:
            # For RNN
            dia_emb = next_hidden.reshape(batch_size, -1)

        encoder_input = [dia_emb, extra]

        if self.uttr_emb is not None:
            encoder_input.append(self._uttr_forward(uttr, batch_size))

        hidden_input = torch.cat(encoder_input, dim=-1)

        emb = self.hidden_layer(hidden_input)

        return emb, next_hidden


class HistoryEncoder(nn.Module):
    """RNN Encoder
        把历史状态+utterance编码成状态向量
    """
    def __init__(self, diaact_size, extra_size, embeddings, output_size,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn', fix_identity=True):
        super(HistoryEncoder, self).__init__()

        last_lstm_size = hidden_size

        if rnn_type == 'lstm':
            self.dia_rnn = torch.nn.LSTMCell(input_size=diaact_size, hidden_size=last_lstm_size)
        else:
            self.dia_rnn = torch.nn.RNNCell(input_size=diaact_size, hidden_size=last_lstm_size, nonlinearity='relu')

        # hidden_input = last_lstm_size + extra_size
        #
        # self.hidden_layer = MultilayerPerceptron(hidden_input, hidden_size, hidden_depth, final_output=identity_dim)

        self.fix_emb = False
        self.fix_identity = fix_identity
        self.ban_identity = False

        self.uttr_emb = embeddings

        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            if rnn_type == 'lstm':
                self.uttr_rnn = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            else:
                self.uttr_rnn = torch.nn.RNN(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input = hidden_size + last_lstm_size + extra_size
        else:
            hidden_input = last_lstm_size + extra_size

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size, hidden_depth)

    def forward(self, uttr, state, identity_state):
        # State Encoder
        diaact, extra, last_hidden = identity_state
        # identity, next_hidden = self.identity(*identity_state)
        batch_size = diaact.shape[0]
        next_hidden = self.dia_rnn(diaact, last_hidden)
        if isinstance(next_hidden, tuple):
            # For LSTM
            dia_emb = next_hidden[0].reshape(batch_size, -1)
        else:
            # For RNN
            dia_emb = next_hidden.reshape(batch_size, -1)

        # Uttr Encoder
        batch_size = state.shape[0]
        if uttr is not None:
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    # print('uttr_emb', next(self.uttr_emb.parameters()).device, u.device)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_rnn(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)

            hidden_input = torch.cat([uttr_emb, extra, dia_emb], dim=-1)
        else:

            hidden_input = torch.cat([extra, dia_emb], dim=-1)

        emb = self.hidden_layer(hidden_input)

        return emb, next_hidden


class HistoryIDEncoder(nn.Module):
    """
        ID move to the last layer
        Final output is [hidden, identity], so the size of hidden is (output_size-identity_size).
        同时编码状态和identity，并把identity拼到输出里
    """
    def __init__(self, identity, state_size, extra_size, embeddings, output_size,
                 hidden_size=64, hidden_depth=2, rnn_type='rnn', fix_identity=True, rnn_state=False):
        super(HistoryIDEncoder, self).__init__()

        self.fix_emb = False
        self.fix_identity = fix_identity
        self.ban_identity = False
        self.rnn_state = rnn_state
        # For split input rnn hidden
        self.id_rnnh_number = 0
        self.state_rnnh_number = 0

        self.identity = identity
        self.uttr_emb = embeddings
        hidden_input = extra_size

        if identity:
            identity_size = identity.identity_dim
        else:
            identity_size = 0

        if rnn_state:
            self.state_rnnh_number = 1
            last_lstm_size = hidden_size
            if rnn_type == 'lstm':
                self.dia_rnn = torch.nn.LSTMCell(input_size=state_size, hidden_size=last_lstm_size)
            else:
                self.dia_rnn = torch.nn.RNNCell(input_size=state_size, hidden_size=last_lstm_size, nonlinearity='relu')
            hidden_input += last_lstm_size
        else:
            hidden_input += state_size


        if embeddings is not None:
            uttr_emb_size = embeddings.embedding_dim
            # if rnn_type == 'lstm':
            self.uttr_rnn = torch.nn.LSTM(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)
            # else:
            #     self.uttr_rnn = torch.nn.RNN(input_size=uttr_emb_size, hidden_size=hidden_size, batch_first=True)

            hidden_input += hidden_size

        if identity:
            self.id_rnnh_number = self.identity.rnnh_number

        self.hidden_layer = MultilayerPerceptron(hidden_input, output_size - identity_size, hidden_depth)

    def forward(self, uttr, dia_act, state, extra, rnn_hiddens, id_gt=None):
        encoder_input = [extra]
        next_rnnh = ()
        batch_size = dia_act.shape[0]

        # split rnn_hiddens
        if not isinstance(rnn_hiddens, tuple):
            rnn_hiddens = (rnn_hiddens,)
        id_rnnh = rnn_hiddens[-self.id_rnnh_number:]
        state_rnnh = rnn_hiddens[:self.state_rnnh_number]
        if self.id_rnnh_number == 1: id_rnnh = id_rnnh[0]
        if self.state_rnnh_number == 1: state_rnnh = state_rnnh[0]

        # State Part
        if self.rnn_state:
            next_hidden = self.dia_rnn(dia_act, state_rnnh)
            if isinstance(next_hidden, tuple):
                # For LSTM
                dia_emb = next_hidden[0].reshape(batch_size, -1)
                next_rnnh = next_hidden
            else:
                # For RNN
                dia_emb = next_hidden.reshape(batch_size, -1)
                next_rnnh = (next_hidden,)
            encoder_input.append(dia_emb)
        else:
            encoder_input.append(state)

        # Uttrance part
        if self.uttr_emb is not None:
            # uttr: [tensor, tensor, ...], each tensor in shape (len_str, 1).
            uttr = uttr.copy()
            with torch.set_grad_enabled(not self.fix_emb):
                for i, u in enumerate(uttr):
                    if u.dtype != torch.int64:
                        print('uttr_emb:', uttr)
                    # print('uttr_emb', next(self.uttr_emb.parameters()).device, u.device)
                    uttr[i] = self.uttr_emb(u).reshape(-1, self.uttr_emb.embedding_dim)
                # print(uttr[i].shape)
            uttr = torch.nn.utils.rnn.pack_sequence(uttr, enforce_sorted=False)
            # uttr = torch.nn.utils.rnn.pack_padded_sequence(uttr, lengths, batch_first=True, enforce_sorted=False)

            _, output = self.uttr_rnn(uttr)

            # For LSTM case, output=(h_1, c_1)
            if isinstance(output, tuple):
                output = output[0]

            uttr_emb = output.reshape(batch_size, -1)
            encoder_input.append(uttr_emb)

        # Main part
        hidden_input = torch.cat(encoder_input, dim=-1)
        emb = self.hidden_layer(hidden_input)

        # Identity part
        identity = None
        if self.identity:
            if id_gt is not None:
                identity = id_gt
                _identity = id_gt
                next_rnnh = next_rnnh + (torch.zeros_like(id_gt),)
            else:
                identity, next_hidden = self.identity(dia_act, extra, id_rnnh, uttr)
                if isinstance(next_hidden, tuple): next_rnnh = next_rnnh + next_hidden
                else: next_rnnh = next_rnnh + (next_hidden,)

                if self.fix_identity:
                    _identity = identity.detach()
                else:
                    _identity = identity
                if self.ban_identity:
                    _identity.fill_(0)
                _identity = torch.softmax(_identity, dim=1)
            emb = torch.cat([emb, _identity], dim=-1)

        return emb, next_rnnh, identity


class SinglePolicy(nn.Module):
    """
        单头输出
    """
    def __init__(self, input_size, output_size, hidden_depth=1, hidden_size=128, ):
        super(SinglePolicy, self).__init__()

        self.hidden_layers = MultilayerPerceptron(input_size, hidden_size, hidden_depth)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, emb):
        hidden_state = self.hidden_layers(emb)
        output = self.output_layer(hidden_state)
        return output

        
# For RL Agent
# Intent + Price(action)
class MixedPolicy(nn.Module):
    """
        双头输出，一个输出intent，一个输出price
    """

    def __init__(self, input_size, intent_size, price_size, hidden_size=64, hidden_depth=2, price_extra=0):
        super(MixedPolicy, self).__init__()
        self.common_net = MultilayerPerceptron(input_size, hidden_size, hidden_depth)
        self.intent_net = SinglePolicy(hidden_size, intent_size, hidden_depth=1, hidden_size=hidden_size)
        self.price_net = SinglePolicy(hidden_size + price_extra, price_size, hidden_depth=1, hidden_size=hidden_size//2)
        self.intent_size = intent_size
        self.price_size = price_size

    def forward(self, state_emb, price_extra=None):
        common_state = self.common_net(state_emb)

        intent_output = self.intent_net(common_state)

        price_input = [common_state]
        if price_extra:
            price_input.append(price_extra)
        price_input = torch.cat(price_input, dim=-1)
        price_output = self.price_net(price_input)

        return intent_output, price_output


class CurrentModel(nn.Module):

    """
        encoder + decoder
    """
    def __init__(self, encoder, decoder, fix_encoder=False):
        super(CurrentModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fix_encoder = fix_encoder

    def forward(self, *input):
        with torch.set_grad_enabled(not self.fix_encoder):
            emb = self.encoder(*input)
        output = self.decoder(emb)
        return output


class HistoryModel(nn.Module):
    """
        encoder + decoder，并保存历史向量
    """
    def __init__(self, encoder, decoder, fix_encoder=False):
        super(HistoryModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fix_encoder = fix_encoder
        self.hidden_vec = None
        self.hidden_stra = None

    def forward(self, *input):
        with torch.set_grad_enabled(not self.fix_encoder):
            # return emb, next_hidden, (identity)
            e_output = self.encoder(*input)

        if e_output is not None and len(e_output) > 0 and e_output[-1] is not None:
            self.hidden_vec = e_output[-1].detach().cpu().numpy()
        else:
            self.hidden_vec = None

        d_output = self.decoder(e_output[0])
        return (d_output,) + e_output[1:]



class SwitchAwareHistoryModel(nn.Module):
    def __init__(self, opt, state_dim, extra_dim, text_dim, n_intents, token_embeddings=None):
        super().__init__()
        self.opt = opt
        self.n_intents = n_intents
        self.text_dim = text_dim
        self.token_embeddings = token_embeddings
        self.hidden_vec = None
        self.hidden_stra = None
        self._warned_missing_text_encoder = False
        # self.state_obs_dim = state_dim        # 应对应 batch.state.shape[-1]
        # self.extra_ctx_dim = extra_dim        # 应对应 batch.extra.shape[-1]

        # 新模型本体
        self.core = SwitchAwareExplicitToM(
            n_acts=n_intents,                       # 先按你的 identity_state 假设
            n_price_bins=100,
            n_intents=n_intents,
            n_styles=4,
            d_text_in=text_dim,
            f_num=getattr(opt, 'sa_f_num', 11),
            d_hist=opt.sa_d_hist,
            d_ctx=opt.sa_d_ctx,
            d_buyer=opt.sa_d_buyer,
            d_obs=opt.sa_d_obs,
            d_h=opt.sa_d_h,
            k_type=opt.sa_k_type,
            d_core=opt.sa_d_core,
            dropout=opt.dropout,
        )

        # 维度桥接层：把旧工程的 state/extra 映射到新模型需要的维度;旧工程张量 -> 新模型所需维度
        self.hist_adapter = nn.LazyLinear(opt.sa_d_hist)
        self.ctx_adapter = nn.LazyLinear(opt.sa_d_ctx)
        self.buyer_adapter = nn.LazyLinear(opt.sa_d_buyer)

    def _extract_seller_from_identity(self, identity_state):
        """
        真实语义对齐版：
        identity_state = [buyer_intent_onehot, seller_intent_onehot, buyer_price, seller_price, buyer_pmask, seller_pmask]
                    = [2*intent_size + 4]

        seller 是最近两步中的第 2 步（index 1）。
        """
        B, D = identity_state.shape
        assert D >= 6, f"identity_state dim too small: got {D}"

        # D = 2 * intent_size + 4
        assert (D - 4) % 2 == 0, \
            f"identity_state shape mismatch: expected 2*intent_size+4, got {identity_state.shape}"

        intent_size = (D - 4) // 2

        buyer_intent_onehot = identity_state[:, :intent_size]
        seller_intent_onehot = identity_state[:, intent_size:2 * intent_size]

        buyer_price_raw = identity_state[:, 2 * intent_size + 0].clamp(0.0, 1.0)
        seller_price_raw = identity_state[:, 2 * intent_size + 1].clamp(0.0, 1.0)

        buyer_pmask = identity_state[:, 2 * intent_size + 2]
        seller_pmask = identity_state[:, 2 * intent_size + 3]

        # v1: 先直接沿用旧 intent 空间，不做 20->7 coarse mapping
        seller_act_id = seller_intent_onehot.argmax(dim=-1).long()

        seller_price_bin = torch.where(
            seller_pmask > 0.5,
            (seller_price_raw * 99.0).clamp(0.0, 99.0).round().long(),
            torch.zeros_like(seller_price_raw, dtype=torch.long),
        )

        return {
            "intent_size": intent_size,
            "buyer_intent_onehot": buyer_intent_onehot,
            "seller_intent_onehot": seller_intent_onehot,
            "buyer_price_raw": buyer_price_raw,
            "seller_price_raw": seller_price_raw,
            "buyer_pmask": buyer_pmask,
            "seller_pmask": seller_pmask,
            "seller_act_id": seller_act_id,
            "seller_price_bin": seller_price_bin,
        }

    def _extract_prev_seller_price_from_state(self, state, intent_size):
        """
        从完整历史 state[2] 中回溯：
        - 当前 seller 所在列：最后一步 => L-1
        - previous seller 所在列：L-3, L-5, ...
        返回：
            seller_price_prev: [B]
            prev_seller_valid: [B]
        """
        B = state.size(0)
        device = state.device
        dtype = state.dtype

        step_width = intent_size + 2
        L = state.size(1) // step_width

        price_hist = state[:, L * intent_size : L * (intent_size + 1)]   # [B, L]
        pmask_hist = state[:, L * (intent_size + 1) : ]                  # [B, L]

        # seller 是最后一步，因此 seller 历史位是 L-1, L-3, L-5, ...
        seller_cols = list(range(L - 1, -1, -2))

        seller_price_prev = torch.zeros(B, device=device, dtype=dtype)
        prev_seller_valid = torch.zeros(B, device=device, dtype=dtype)

        # 从“上一个 seller 步”开始往前找第一个有效报价
        for col in seller_cols[1:]:
            valid = (pmask_hist[:, col] > 0.5).float()
            fill_mask = (prev_seller_valid < 0.5) * valid

            seller_price_prev = torch.where(
                fill_mask > 0.5,
                price_hist[:, col].clamp(0.0, 1.0),
                seller_price_prev,
            )
            prev_seller_valid = torch.maximum(prev_seller_valid, valid)

        return seller_price_prev, prev_seller_valid

    def _build_seller_num_feats(
        self,
        parsed_id,
        state,
        extra,
        seller_price_prev=None,
        prev_seller_valid=None,
    ):
        """
        11维对齐版（先兼容当前 f_num=11）：

        0  seller_price_raw
        1  delta_price
        2  delta2_price
        3  buyer_price_raw
        4  gap = seller_price_raw - buyer_price_raw
        5  turn_ratio
        6  is_offer
        7  is_accept
        8  is_reject
        9  is_question
        10 seller_has_price
        """
        seller_act_id = parsed_id["seller_act_id"]
        seller_price_raw = parsed_id["seller_price_raw"]
        buyer_price_raw = parsed_id["buyer_price_raw"]
        seller_pmask = parsed_id["seller_pmask"]

        B = seller_act_id.size(0)
        device = seller_act_id.device
        dtype = seller_price_raw.dtype

        # extra = [role_0, role_1, turn_ratio]
        turn_ratio = (
            extra[:, 2].clamp(0.0, 1.0)
            if extra is not None and extra.size(-1) > 2
            else torch.zeros(B, device=device, dtype=dtype)
        )

        cur_valid = (seller_pmask > 0.5).float()

        if seller_price_prev is None:
            seller_price_prev = torch.zeros(B, device=device, dtype=dtype)
        else:
            seller_price_prev = seller_price_prev.clamp(0.0, 1.0)

        if prev_seller_valid is None:
            prev_seller_valid = torch.zeros(B, device=device, dtype=dtype)
        else:
            prev_seller_valid = prev_seller_valid.float()

        # 只有当前 seller 有价 且 previous seller 也有价，delta 才有意义
        pair_valid = cur_valid * prev_seller_valid

        delta_price = torch.where(
            pair_valid > 0.5,
            seller_price_raw - seller_price_prev,
            torch.zeros_like(seller_price_raw),
        )

        # v1 先不跨 step 维护 prev_delta，先置 0
        delta2_price = torch.zeros_like(delta_price)

        gap = seller_price_raw - buyer_price_raw

        is_offer = (seller_act_id == 0).float()
        is_accept = (seller_act_id == 1).float()
        is_reject = (seller_act_id == 2).float()
        is_question = (seller_act_id == 3).float()

        seller_has_price = cur_valid

        seller_num_feats = torch.stack([
            seller_price_raw,      # 0
            delta_price,           # 1
            delta2_price,          # 2
            buyer_price_raw,       # 3
            gap,                   # 4
            turn_ratio,            # 5
            is_offer,              # 6
            is_accept,             # 7
            is_reject,             # 8
            is_question,           # 9
            seller_has_price,      # 10
        ], dim=-1)

        # v1 最小改法：至少把“上一轮 seller 的价格存在 prev_num_feats 里”
        prev_num_feats = seller_num_feats.clone()
        prev_num_feats[:, 0] = seller_price_prev
        prev_num_feats[:, 1] = torch.zeros_like(delta_price)
        prev_num_feats[:, 2] = torch.zeros_like(delta2_price)
        prev_num_feats[:, 4] = torch.where(
            prev_seller_valid > 0.5,
            seller_price_prev - buyer_price_raw,
            torch.zeros_like(gap),
        )
        prev_num_feats[:, 10] = prev_seller_valid

        return seller_num_feats, prev_num_feats

    # def _build_step_batch(self, state, identity_state, extra, uttr_emb):
    #     parsed_id = self._extract_seller_from_identity(identity_state)

    #     seller_price_prev, prev_seller_valid = self._extract_prev_seller_price_from_state(
    #         state=state,
    #         intent_size=parsed_id["intent_size"],
    #     )

    #     seller_num_feats, prev_seller_num_feats = self._build_seller_num_feats(
    #         parsed_id=parsed_id,
    #         state=state,
    #         extra=extra,
    #         seller_price_prev=seller_price_prev,
    #         prev_seller_valid=prev_seller_valid,
    #     )

    #     hist_state = self.hist_adapter(state)
    #     scenario_ctx = self.ctx_adapter(extra)
    #     buyer_state = self.buyer_adapter(state)

    #     return StepBatch(
    #         seller_act_id=parsed_id["seller_act_id"],
    #         seller_price_bin=parsed_id["seller_price_bin"],
    #         seller_num_feats=seller_num_feats,
    #         seller_text_emb=uttr_emb,
    #         hist_state=hist_state,
    #         scenario_ctx=scenario_ctx,
    #         buyer_state=buyer_state,
    #         prev_seller_num_feats=prev_seller_num_feats,
    #     )
    
    def _prepare_uttr_emb(self, uttr, batch_size, device, dtype):
        """
        将 trainer 传进来的 uttr 统一整理成 seller_text_emb。

        v1 最小兼容策略：
        1) 如果 uttr 已经是 [B, D_text] 的 Tensor，直接返回
        2) 如果 uttr 是 [B, T, D_text] 的 Tensor，做 mean pooling
        3) 如果 uttr 是旧项目常见的 list[tensor] token 序列，先退化成全零向量
        （后续再接真实 utterance encoder）
        """
        # 先把 text embedding 维度保存成成员变量，别从 core 猜
        text_dim = self.text_dim

        if isinstance(uttr, torch.Tensor):
            if uttr.dim() == 2:
                # [B, D_text]
                if uttr.size(0) != batch_size:
                    raise ValueError(f"uttr batch mismatch: expected {batch_size}, got {uttr.size(0)}")
                return uttr.to(device=device, dtype=dtype)

            elif uttr.dim() == 3:
                # [B, T, D_text]
                if uttr.size(0) != batch_size:
                    raise ValueError(f"uttr batch mismatch: expected {batch_size}, got {uttr.size(0)}")
                return uttr.to(device=device, dtype=dtype).mean(dim=1)

        if isinstance(uttr, (list, tuple)) and self.token_embeddings is not None:
            pooled = []
            for seq in uttr:
                if not torch.is_tensor(seq):
                    seq = torch.as_tensor(seq, dtype=torch.long, device=device)
                else:
                    seq = seq.to(device=device, dtype=torch.long)

                seq = seq.reshape(-1)
                if seq.numel() == 0:
                    pooled.append(torch.zeros(text_dim, device=device, dtype=dtype))
                    continue

                token_emb = self.token_embeddings(seq)
                pooled.append(token_emb.mean(dim=0).to(dtype=dtype))

            if len(pooled) != batch_size:
                raise ValueError(f"uttr batch mismatch: expected {batch_size}, got {len(pooled)}")
            return torch.stack(pooled, dim=0)

        if isinstance(uttr, (list, tuple)) and not self._warned_missing_text_encoder:
            print(
                "[SwitchAwareHistoryModel] warning: list[tensor] utterances received without "
                "token_embeddings; text features fall back to zeros."
            )
            self._warned_missing_text_encoder = True

        # trainer 真实路径里 uttr 往往是 list[tensor]，先用零向量占位
        return torch.zeros(batch_size, text_dim, device=device, dtype=dtype)

    def _init_hidden_and_belief(self, batch_size, device, dtype):
        h_prev = self.core.init_hidden(batch_size, device=device, dtype=dtype)
        belief_prev = self.core.init_belief(batch_size, device=device, dtype=dtype)
        return h_prev, belief_prev

    def forward(
        self,
        uttr,
        identity_state,
        state,
        extra,
        hidden=None,
        id_gt=None,
        last_price=None,
        force_switch_prob=None,
    ):
        B = state.size(0)
        device = state.device
        dtype = state.dtype

        if hidden is None:
            h_prev, belief_prev = self._init_hidden_and_belief(B, device, dtype)
        else:
            h_prev, belief_prev = hidden

        uttr_emb = self._prepare_uttr_emb(uttr, B, device, dtype)

        parsed_id = self._extract_seller_from_identity(identity_state)

        seller_price_prev, prev_seller_valid = self._extract_prev_seller_price_from_state(
            state=state,
            intent_size=parsed_id["intent_size"],
        )

        seller_num_feats, prev_seller_num_feats = self._build_seller_num_feats(
            parsed_id=parsed_id,
            state=state,
            extra=extra,
            seller_price_prev=seller_price_prev,
            prev_seller_valid=prev_seller_valid,
        )

        hist_state = self.hist_adapter(state)
        scenario_ctx = self.ctx_adapter(extra)
        buyer_state = self.buyer_adapter(state)

        step_batch = StepBatch(
            seller_act_id=parsed_id["seller_act_id"],
            seller_price_bin=parsed_id["seller_price_bin"],
            seller_num_feats=seller_num_feats,
            seller_text_emb=uttr_emb,
            hist_state=hist_state,
            scenario_ctx=scenario_ctx,
            buyer_state=buyer_state,
            prev_seller_num_feats=prev_seller_num_feats,
        )

        out = self.core.step(
            batch=step_batch,
            h_prev=h_prev,
            belief_prev=belief_prev,
            force_switch_prob=force_switch_prob,
        )

        predictions = (out["intent_logits"], out["price_logits"])

        belief_next = BeliefState(
            type_probs=out["belief_type_probs"],
            cont_mu=out["belief_cont_mu"],
            cont_logvar=out["belief_cont_logvar"],
            confidence=out["belief_confidence"],
        )

        hidden_next = (out["seller_hidden"], belief_next)

        if "belief_vector" in out:
            identity_next = out["belief_vector"]
        else:
            identity_next = torch.cat([
                out["belief_type_probs"],
                out["belief_cont_mu"],
                out["belief_cont_logvar"],
                out["belief_confidence"],
            ], dim=-1)

        self.hidden_vec = identity_next.detach().cpu().numpy()
        self.last_aux_outputs = out

        return predictions, hidden_next, identity_next
