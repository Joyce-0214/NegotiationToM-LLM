# Data: _batch_iters, (_rewards, strategies), examples, verbose_strs
# Out: (pred_identity, pred_intent, pred_price, strategies)

import argparse
import random
import json
import numpy as np
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from craigslistbargain.neural.debug_tom_batch import inspect_tom_batch
from craigslistbargain.core.controller import Controller
from craigslistbargain.neural.utterance import UtteranceBuilder
from craigslistbargain.core.price_tracker import PriceScaler

from tensorboardX import SummaryWriter
import pickle as pkl

from craigslistbargain.neural.batcher_rl import RLBatch, RawBatch, ToMBatch
from craigslistbargain.neural.rl_trainer import RLTrainer as BaseTrainer
from craigslistbargain.neural.sl_trainer import Statistics, SimpleLoss
from craigslistbargain.neural.generator import LFSampler

import math, time, sys


class RLStatistics(Statistics):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, reward=0, n_words=0):
        self.loss = loss
        self.n_words = n_words
        self.n_src_words = 0
        self.reward=reward
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.reward += stat.reward

    def mean_loss(self):
        return self.loss / self.n_words

    def mean_reward(self):
        return self.reward / self.n_words

    def elapsed_time(self):
        return time.time() - self.start_time

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def str_loss(self):
        return "loss: %6.4f reward: %6.4f;" % (self.mean_loss(), self.mean_reward())

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d;" + self.str_loss() +
               "%6.0f s elapsed") %
              (epoch, batch,  n_batches,
               time.time() - start))
        sys.stdout.flush()

class SimpleCriticLoss(nn.Module):
    def __init__(self):
        super(SimpleCriticLoss, self).__init__()
        self.criterion = nn.MSELoss()

    # def _get_correct_num(self, enc_policy, tgt_intents):
    #     enc_policy = enc_policy.argmax(dim=1)
    #     tmp = (enc_policy == tgt_intents).cpu().numpy()
    #     tgt = tgt_intents.data.cpu().numpy()
    #     tmp[tgt==19] = 1
    #     import numpy as np
    #     return np.sum(tmp)

    def forward(self, pred, oracle, pmask=None):
        loss = self.criterion(pred, oracle)
        stats = self._stats(loss, pred.shape[0])
        return loss, stats

    def _stats(self, loss, data_num):
        return RLStatistics(loss=loss.item(), n_words=data_num)

class RLTrainer(BaseTrainer):
    def __init__(self, agents, scenarios, train_loss, optim, training_agent=0, reward_func='margin',
                 cuda=False, args=None):
        super(RLTrainer, self).__init__(agents, scenarios, train_loss, optim,
                                        training_agent, reward_func, cuda, args)
        # print('training_agent', training_agent)

        self.critic = agents[training_agent].env.critic
        self.tom = agents[training_agent].env.tom_model   # trainer只关心self.tom()怎么调用；以及返回值格式
        self.vocab = agents[training_agent].env.vocab
        self.lf_vocab = agents[training_agent].env.lf_vocab
        self.model_type = args.model_type
        self.use_utterance = False
        self.tom_identity_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.hidden_vec = None
        self.args = args
        print("\n========== [RLTrainer init args] ==========")
        print("model_type =", getattr(args, "model_type", None))
        print("tom class  =", getattr(agents[training_agent].env.tom_model, "__class__", type(None)).__name__)
        print("sa_lambda_price        =", getattr(args, "sa_lambda_price", None))
        print("sa_lambda_switch       =", getattr(args, "sa_lambda_switch", None))
        print("sa_switch_infer_thresh =", getattr(args, "sa_switch_infer_thresh", None))
        print("sa_switch_pos_weight   =", getattr(args, "sa_switch_pos_weight", None))
        print("==========================================\n")
        self.last_tom_epoch_stats = {}

    def _run_batch_a2c(self, batch):
        value = self._run_batch_critic(batch)
        policy, price = self._run_batch(batch)
        # print('max price', torch.max(price))
        return value, policy, price

    def _slice_tom_hidden(self, hidden_state, batch_size):
        # 加入的hidden slicing helper
        if hidden_state is None:
            return None

        if isinstance(hidden_state, tuple):
            # switch_aware: (seller_hidden, belief_state)
            if len(hidden_state) == 2 and hasattr(hidden_state[1], 'type_probs'):
                h_prev, belief_prev = hidden_state
                return (
                    h_prev[:batch_size, :],
                    type(belief_prev)(
                        type_probs=belief_prev.type_probs[:batch_size, :],
                        cont_mu=belief_prev.cont_mu[:batch_size, :],
                        cont_logvar=belief_prev.cont_logvar[:batch_size, :],
                        confidence=belief_prev.confidence[:batch_size, :],
                    )
                )
            else:
                return tuple(x[:batch_size, :] for x in hidden_state)

        if isinstance(hidden_state, torch.Tensor):
            return hidden_state[:batch_size, :]

        raise TypeError(f"Unsupported hidden_state type: {type(hidden_state)}")

    def _run_batch_tom_identity(
        self,
        batch,
        hidden_state,
        only_identity=False,
        id_gt=False,
        force_switch_prob=None,
    ):
        if id_gt:
            id_gt = batch.strategy
        else:
            id_gt = None

        if only_identity:
            identity, next_hidden = \
                self.tom.encoder.identity(batch.identity_state, batch.extra, hidden_state, uttr=batch.uttr)
            predictions = None
        else:
            output = self.tom(
                batch.uttr,
                batch.identity_state,
                batch.state,
                batch.extra,
                hidden_state,
                id_gt,
                batch.last_price,
                force_switch_prob=force_switch_prob,
            )
            if len(output) == 3:
                predictions, next_hidden, identity = output
            else:
                predictions, next_hidden = output
                identity = None

        return predictions, next_hidden, identity

    def _tom_gradient_accumulation(self, batch_iter, strategy, model, ret_table, id_gt=False):
        model.train()

        h = None
        identity_loss = []
        tom_intent_loss = []
        tom_price_loss = []
        tom_switch_loss = []
        batch_stats_list = []
        identity_accu = []
        identity_accu2 = []
        tom_intent_accu = []
        strategies = []

        pred_intent = []
        pred_price = []
        pred_identity = []

        for i, batch in enumerate(batch_iter):
            tom_batch = ToMBatch.from_raw(batch, strategy[:batch.size])

            if os.environ.get("DEBUG_TOM_REALBATCH", "0") == "1" and i == 0:
                self.debug_realbatch_forward_once(batch, strategy)
                raise SystemExit(0)

            if os.environ.get("DEBUG_TOM_SWITCH_INTERVENTION", "0") == "1" and i == 0:
                self.debug_switch_intervention_once(batch, strategy)
                raise SystemExit(0)

            if os.environ.get("DEBUG_TOM_BATCH", "0") == "1" and i == 0:
                inspect_tom_batch(tom_batch, name="real_tom_batch")
                raise SystemExit(0)
            # if h is not None:
            #     if isinstance(h, tuple):
            #         h = tuple(map(lambda x: x[:batch.size, :], h))
            #         # h = (h[0][:batch.size, :], h[1][:batch.size, :])
            #     elif isinstance(h, torch.Tensor):
            #         h = h[:batch.size, :]
            h = self._slice_tom_hidden(h, batch.size)
            pred, h, identity = self._run_batch_tom_identity(tom_batch, hidden_state=h,
                                                             only_identity=(not ret_table['tom']), id_gt=id_gt)

            self.hidden_vec.append(self.tom.hidden_vec)
            s = np.array([strategy[:batch.size]]).T
            s = np.concatenate([s, tom_batch.identity_state.cpu().data.numpy(), tom_batch.extra.cpu().data.numpy(),], axis=1)
            self.hidden_stra.append(s)

            # Identity Loss
            if ret_table['id']:
                s = torch.tensor(strategy[:batch.size], dtype=torch.int64, device=identity.device)
                loss = self.tom_identity_loss(identity, s)
                # accu = torch.gather(torch.softmax(identity, dim=1), 1, s.reshape(-1, 1))
                id_p = torch.softmax(identity, dim=1)
                accu = id_p.argmax(dim=-1).reshape(-1, 1) == s.reshape(-1, 1)
                accu = accu.to(dtype=torch.float32)
                accu2 = (id_p.topk(3, dim=-1).indices == s.reshape(-1, 1)).max(dim=-1).values.reshape(-1, 1)
                accu2 = accu2.to(dtype=torch.float32)
                identity_loss.append(loss.reshape(-1))
                identity_accu.append(accu.reshape(-1))
                identity_accu2.append(accu2.reshape(-1))
                pred_identity.append(identity.reshape(1, -1).detach())

            # ToM Loss
            if ret_table['tom']:
                intent, price = pred

                if getattr(self.tom, '__class__', None).__name__ == 'SwitchAwareHistoryModel':
                    tom_batch = copy.copy(tom_batch)

                    loss0, loss1, loss2, batch_stats, intent_accu = self._compute_switchaware_tom_loss(
                        tom_batch, intent, price
                    )
                    batch_stats_list.append(batch_stats)
                    tom_switch_loss.append(loss2.reshape(-1))
                else:
                    loss0, loss1, batch_stats = self._compute_loss(
                        tom_batch, policy=intent, price=price, loss=self.tom_loss
                    )
                    intent_accu = torch.gather(
                        torch.softmax(intent, dim=1),
                        1,
                        tom_batch.act_intent.reshape(-1, 1),
                    )

                tom_intent_accu.append(intent_accu)
                tom_intent_loss.append(loss0.reshape(-1))
                tom_price_loss.append(loss1.reshape(-1))
                pred_intent.append(intent.reshape(1, -1).detach())
                pred_price.append(price.reshape(1, -1).detach())
            # losses.append(loss.reshape(-1))
            # accus.append(accu.reshape(-1))
            # strategies.append(s.detach())

            # preds.append(pred.reshape(1, -1).detach())


        # preds = torch.cat(preds, dim=0)
        # strategy = torch.tensor([strategy]*preds.shape[0], dtype=torch.int64, device=preds.device)
        # (-1,), (-1, 1) -> (-1,) *2
        # print('loss & accu:', loss, accu)
        return (
            {'id': [identity_loss], 'tom': [tom_intent_loss, tom_price_loss, tom_switch_loss]},
            {'id': [identity_accu, identity_accu2], 'tom': [tom_intent_accu]},
            (pred_identity, pred_intent, pred_price, strategies),
            batch_stats_list,
        )

    def _sort_merge_batch(self, batch_iters, batch_size, device=None):
        sorted_id = [i for i in range(len(batch_iters))]
        sorted_id.sort(key=lambda i: len(batch_iters[i]), reverse=True)
        batch_iters = sorted(batch_iters, reverse=True, key=lambda l: len(l))
        batch_length = [len(b) for b in batch_iters]

        if device is None:
            if self.cuda:
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

        def merge_batch(one_batch):
            batches = [[] for i in range(len(one_batch[0]))]
            for bi in one_batch:
                for i, b in enumerate(bi):
                    batches[i].append(b)
            for i, b in enumerate(batches):
                # print('merge batch:', i, len(b))
                batches[i] = RawBatch.merge(b)
                batches[i].to(device)
            return batches

        # Split by size
        right = 0
        bs, ids, bl = [], [], []
        while True:
            left = right
            right = min(right+batch_size, len(batch_iters))
            # print('merge: ', left, right)
            bs.append(merge_batch(batch_iters[left: right]))
            ids.append(sorted_id[left: right])
            bl.append(batch_length[left: right])
            if right >= len(batch_iters):
                break

        return bs, ids, bl

    @staticmethod
    def split_by_strategy(t, s, s_num=5):
        if s is None:
            return t
        s = torch.tensor(s, dtype=torch.int64, device=t.device).reshape(-1, 1)
        ret = [0]*s_num
        for i in range(s_num):
            ret[i] = torch.masked_select(t, (s == i).bool())
        # ret[0]=torch.masked_select(t, s)
        # ret[1]=torch.masked_select(t, 1-s)
        return ret

    def add_strategy_in_language(self, batch_iters, strategies):
        for i in range(2):
            if i == 0:
                continue
            # for each dialogue
            for j in range(len(batch_iters[i])):
                c = self.vocab.size-1-strategies[i][j]
                # for each sentences
                for k, b in enumerate(batch_iters[i][j]):
                    if random.randint(0, 5) > 0:
                        continue
                    tmp = b.uttr[0].cpu().numpy()
                    l = np.prod(tmp.shape)
                    tmp = np.insert(tmp, random.randint(2, l-1), c, axis=1)
                    b.uttr[0] = torch.tensor(tmp, device=b.uttr[0].device)

    def update_tom(self, args, batch_iters, strategy, model,
               update_table=None, ret_table=None, dump_name=None):
        switchaware = getattr(self.tom, '__class__', None).__name__ == 'SwitchAwareHistoryModel'

        global_stats = {
            "intent_loss_sum": 0.0,
            "intent_count": 0.0,
            "intent_correct_sum": 0.0,

            "price_loss_sum": 0.0,
            "price_active_count": 0.0,
            "price_correct_sum": 0.0,

            "switch_loss_sum": 0.0,
            "switch_count": 0.0,
            "switch_correct_sum": 0.0,

            "switch_positive_count": 0.0,
            "switch_prob_mean_sum": 0.0,
            "switch_logit_mean_sum": 0.0,

            "switch_pred_positive_count": 0.0,
            "switch_tp": 0.0,
            "switch_fp": 0.0,
            "switch_tn": 0.0,
            "switch_fn": 0.0,
        }

        for th in [0.5, 0.7, 0.8, 0.9, 0.95]:
            key = str(th).replace(".", "p")
            global_stats[f"switch_scan_{key}_pred_pos"] = 0.0
            global_stats[f"switch_scan_{key}_tp"] = 0.0
            global_stats[f"switch_scan_{key}_fp"] = 0.0
            global_stats[f"switch_scan_{key}_tn"] = 0.0
            global_stats[f"switch_scan_{key}_fn"] = 0.0

        # 初始化 threshold-scan 聚合槽位
        scan_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        for th in scan_thresholds:
            key = str(th).replace(".", "p")
            global_stats[f"switch_scan_{key}_pred_pos"] = 0.0
            global_stats[f"switch_scan_{key}_tp"] = 0.0
            global_stats[f"switch_scan_{key}_fp"] = 0.0
            global_stats[f"switch_scan_{key}_tn"] = 0.0
            global_stats[f"switch_scan_{key}_fn"] = 0.0

        if switchaware:
            update_table = {'id': False, 'tom': True}
            ret_table = {'id': False, 'tom': True}

        cur_t = time.time()

        if not update_table:
            update_table = {'id': True, 'tom': True}
        if not ret_table:
            ret_table = {'id': True, 'tom': True}

        batch_iters, sorted_id, batch_length = batch_iters
        split_by_strategy = False

        model.zero_grad()

        if switchaware:
            loss = {'id': [[]], 'tom': [[], [], []]}
            step_loss = {'id': [[]], 'tom': [[], [], []]}
        else:
            loss = {'id': [[]], 'tom': [[], []]}
            step_loss = {'id': [[]], 'tom': [[], []]}

        accu = {'id': [[], []], 'tom': [[]]}
        step_accu = {'id': [[], []], 'tom': [[]]}
        output_data = []

        def add_list(step, one, s=None):
            for j, o in enumerate(one):
                if isinstance(o, list):
                    for k in range(len(o)):
                        if k >= len(step[j]):
                            step[j].append([])
                        step[j][k].append(self.split_by_strategy(o[k], s))
                else:
                    step[j].append(o)

        self.hidden_vec = []
        self.hidden_stra = []

        for i, b in enumerate(batch_iters):
            stra = [strategy[j] for j in sorted_id[i]]
            l, a, logs, batch_stats_list = self._tom_gradient_accumulation(
                b, stra, model, ret_table=ret_table, id_gt=args.idgt
            )

            if switchaware:
                for bs in batch_stats_list:
                    global_stats["intent_loss_sum"] += float(bs["intent_loss_sum"].item())
                    global_stats["intent_count"] += float(bs["intent_count"].item())
                    global_stats["intent_correct_sum"] += float(bs["intent_correct_sum"].item())

                    global_stats["price_loss_sum"] += float(bs["price_loss_sum"].item())
                    global_stats["price_active_count"] += float(bs["price_active_count"].item())
                    global_stats["price_correct_sum"] += float(bs["price_correct_sum"].item())

                    global_stats["switch_loss_sum"] += float(bs["switch_loss_sum"].item())
                    global_stats["switch_count"] += float(bs["switch_count"].item())
                    global_stats["switch_correct_sum"] += float(bs["switch_correct_sum"].item())

                    global_stats["switch_positive_count"] += float(bs["switch_positive_count"].item())

                    sc = max(float(bs["switch_count"].item()), 1.0)
                    global_stats["switch_prob_mean_sum"] += float(bs["switch_prob_mean"].item()) * sc
                    global_stats["switch_logit_mean_sum"] += float(bs["switch_logit_mean"].item()) * sc

                    global_stats["switch_pred_positive_count"] += float(bs["switch_pred_positive_count"].item())
                    global_stats["switch_tp"] += float(bs["switch_tp"].item())
                    global_stats["switch_fp"] += float(bs["switch_fp"].item())
                    global_stats["switch_tn"] += float(bs["switch_tn"].item())
                    global_stats["switch_fn"] += float(bs["switch_fn"].item())

                    for th in scan_thresholds:
                        key = str(th).replace(".", "p")
                        global_stats[f"switch_scan_{key}_pred_pos"] += float(bs[f"switch_scan_{key}_pred_pos"].item())
                        global_stats[f"switch_scan_{key}_tp"] += float(bs[f"switch_scan_{key}_tp"].item())
                        global_stats[f"switch_scan_{key}_fp"] += float(bs[f"switch_scan_{key}_fp"].item())
                        global_stats[f"switch_scan_{key}_tn"] += float(bs[f"switch_scan_{key}_tn"].item())
                        global_stats[f"switch_scan_{key}_fn"] += float(bs[f"switch_scan_{key}_fn"].item())

            output_data.append(logs)

            if not split_by_strategy:
                stra = None

            for key in l:
                l_key = l[key]
                if not ret_table[key]:
                    continue
                for j, ll in enumerate(l_key):
                    tmp = torch.cat(ll, dim=0)
                    loss[key][j].append(self.split_by_strategy(tmp, stra))
                add_list(step_loss[key], l_key, stra)

            for key in a:
                a_key = a[key]
                if not ret_table[key]:
                    continue
                for j, aa in enumerate(a_key):
                    tmp = torch.cat(aa, dim=0)
                    accu[key][j].append(self.split_by_strategy(tmp, stra))
                add_list(step_accu[key], a_key, stra)

        step_num = None
        for key in ['id', 'tom']:
            if not ret_table[key]:
                continue

            if step_num is None:
                if split_by_strategy:
                    step_num = [[np.sum([dd[i].shape[0] for dd in d]) for d in step_loss[key][0]]
                                for i in range(2)]
                else:
                    step_num = [np.sum([dd.shape[0] for dd in d]) for d in step_loss[key][0]]

            for i, l in enumerate(loss[key]):
                loss[key][i] = torch.cat(l, dim=0).mean()
                if split_by_strategy:
                    step_loss[key][i] = [
                        [torch.cat([dd[j] for dd in d], dim=0).mean().item() if len(d) > 0 else None
                        for d in step_loss[key][i]]
                        for j in range(2)
                    ]
                else:
                    step_loss[key][i] = [
                        torch.cat(d, dim=0).mean().item() if len(d) > 0 else None
                        for d in step_loss[key][i]
                    ]

            for i, a in enumerate(accu[key]):
                accu[key][i] = torch.cat(a, dim=0).mean().item()
                step_accu[key][i] = [
                    torch.cat(d, dim=0).mean().item() if len(d) > 0 else None
                    for d in step_accu[key][i]
                ]

        if dump_name is not None:
            with open(dump_name, 'wb') as f:
                pkl.dump(output_data, f)

        price_w = getattr(args, "sa_lambda_price", 1.0)
        switch_w = getattr(args, "sa_lambda_switch", 0.5)
        print(f"[update_tom] price_w={price_w}, switch_w={switch_w}")

        if update_table['id']:
            loss['id'][0].backward()
            if self.optim.get('tom_identity') is not None:
                self.optim['tom_identity'].step()
            else:
                print('[Warning] update identity, but no identity exists.')

        price_w = getattr(args, "sa_lambda_price", 1.0)
        switch_w = getattr(args, "sa_lambda_switch", 0.0)

        if update_table['tom']:
            l = loss['tom'][0] + price_w * loss['tom'][1] + switch_w * loss['tom'][2]
            l.backward()
            self.optim['tom'].step()

        for key in ['id', 'tom']:
            for i, l in enumerate(loss[key]):
                if isinstance(loss[key][i], torch.Tensor):
                    loss[key][i] = loss[key][i].item()
                else:
                    loss[key][i] = None

        loss = loss['id'] + loss['tom']
        accu = accu['id'][:1] + accu['tom'] + accu['id'][1:]
        step_loss = step_loss['id'] + step_loss['tom']
        step_accu = step_accu['id'][:1] + step_accu['tom'] + step_accu['id'][1:]

        if switchaware:
            def safe_div(a, b):
                return a / max(b, 1.0)

            pred_pos = global_stats["switch_pred_positive_count"]
            switch_tp = global_stats["switch_tp"]
            switch_fp = global_stats["switch_fp"]
            switch_tn = global_stats["switch_tn"]
            switch_fn = global_stats["switch_fn"]

            epoch_stats = {
                "intent_loss_mean": safe_div(global_stats["intent_loss_sum"], global_stats["intent_count"]),
                "intent_acc_mean": safe_div(global_stats["intent_correct_sum"], global_stats["intent_count"]),

                "price_loss_active_mean": safe_div(global_stats["price_loss_sum"], global_stats["price_active_count"]),
                "price_acc_active_mean": safe_div(global_stats["price_correct_sum"], global_stats["price_active_count"]),
                "price_active_count": global_stats["price_active_count"],

                "switch_loss_mean": safe_div(global_stats["switch_loss_sum"], global_stats["switch_count"]),
                "switch_acc_mean": safe_div(global_stats["switch_correct_sum"], global_stats["switch_count"]),
                "switch_count": global_stats["switch_count"],

                "switch_positive_count": global_stats["switch_positive_count"],
                "switch_positive_ratio": safe_div(global_stats["switch_positive_count"], global_stats["switch_count"]),
                "switch_prob_mean": safe_div(global_stats["switch_prob_mean_sum"], global_stats["switch_count"]),
                "switch_logit_mean": safe_div(global_stats["switch_logit_mean_sum"], global_stats["switch_count"]),

                "switch_pred_positive_count": pred_pos,
                "switch_pred_positive_ratio": safe_div(pred_pos, global_stats["switch_count"]),
                "switch_tp": switch_tp,
                "switch_fp": switch_fp,
                "switch_tn": switch_tn,
                "switch_fn": switch_fn,
                "switch_precision": safe_div(switch_tp, switch_tp + switch_fp),
                "switch_recall": safe_div(switch_tp, switch_tp + switch_fn),
                "switch_tnr": safe_div(switch_tn, switch_tn + switch_fp),
            }

            for th in [0.5, 0.7, 0.8, 0.9, 0.95]:
                key = str(th).replace(".", "p")
                pred_pos_th = global_stats[f"switch_scan_{key}_pred_pos"]
                tp_th = global_stats[f"switch_scan_{key}_tp"]
                fp_th = global_stats[f"switch_scan_{key}_fp"]
                tn_th = global_stats[f"switch_scan_{key}_tn"]
                fn_th = global_stats[f"switch_scan_{key}_fn"]

                epoch_stats[f"switch_scan_{key}_pred_pos"] = pred_pos_th
                epoch_stats[f"switch_scan_{key}_tp"] = tp_th
                epoch_stats[f"switch_scan_{key}_fp"] = fp_th
                epoch_stats[f"switch_scan_{key}_tn"] = tn_th
                epoch_stats[f"switch_scan_{key}_fn"] = fn_th

                epoch_stats[f"switch_scan_{key}_pred_pos_ratio"] = safe_div(pred_pos_th, global_stats["switch_count"])
                epoch_stats[f"switch_scan_{key}_precision"] = safe_div(tp_th, tp_th + fp_th)
                epoch_stats[f"switch_scan_{key}_recall"] = safe_div(tp_th, tp_th + fn_th)
                epoch_stats[f"switch_scan_{key}_tnr"] = safe_div(tn_th, tn_th + fp_th)

            self.last_tom_epoch_stats = epoch_stats

            # 继续兼容旧 multi_manager_debug.py 的前三个槽位
            loss = [
                None,
                self.last_tom_epoch_stats["intent_loss_mean"],
                self.last_tom_epoch_stats["price_loss_active_mean"],
                self.last_tom_epoch_stats["switch_loss_mean"],
            ]
            accu = [
                None,
                self.last_tom_epoch_stats["intent_acc_mean"],
                None,
            ]
            return loss, accu, (step_loss, step_accu, step_num)

        return loss, accu, (step_loss, step_accu, step_num)

    def _gradient_accumulation(self, batch_iter, reward, model, critic, discount=1):
        # Compute losses
        model.train()
        critic.train()

        values = []
        losses = [[], []]
        ents = [[], []]

        # batch_iter gives a dialogue
        policy_stats = Statistics()
        # For value: deprecated
        for_value = False

        # In one batch, from sentence 1 to n.
        for i, batch in enumerate(batch_iter):
            # print("batch: \nencoder{}\ndecoder{}\ntitle{}\ndesc{}".format(batch.encoder_inputs.shape, batch.decoder_inputs.shape, batch.title_inputs.shape, batch.desc_inputs.shape))
            # batch.mask_last_price()
            rlbatch = RLBatch.from_raw(batch, None, None)
            value, i_pred, p_pred = self._run_batch_a2c(rlbatch)

            # The last sentence may only need to predict value?
            # if not for_value:
                # intent_loss, price_stats, batch_stats = self._compute_loss(rlbatch, policy=policy, price=price, loss=self.train_loss)
            # print('it', i_pred, rlbatch.act_intent)
            # print('price', p_pred, rlbatch.act_price)
            intent_loss = F.cross_entropy(i_pred, rlbatch.act_intent.reshape(-1), reduction='none')
            pact_loss = F.cross_entropy(p_pred, rlbatch.act_price.reshape(-1), reduction='none')
                # print('policy_loss is:', policy_loss)
                # policy_stats.update(pl_stats)

            # entropy_loss, _ = self._compute_loss(rlbatch, policy=policy, price=price, loss=self.entropy_loss)

            intent_ent = self.entropy_loss(i_pred)
            if torch.isnan(intent_ent.mean()):
                isnan = torch.isnan(intent_ent.mean()).reshape(-1)
                intent_ent = intent_ent.reshape(-1)
                for j in range(isnan.shape[0]):
                    if isnan[j] == 1:
                        print('nan: ', i_pred[j])
                        print('nan2: ', rlbatch.act_intent.reshape(-1)[j], intent_loss.reshape(-1)[j])
                quit()
            pact_ent = self.entropy_loss(p_pred)
            pact_loss = pact_loss.reshape(-1, 1)*rlbatch.act_price_mask
            pact_ent = pact_ent.reshape(-1, 1)*rlbatch.act_price_mask

            # policy_loss = intent_loss + pact_loss
            # entropy_loss = intent_ent + pact_ent


            # penalty = ((price-1)**2).mul((price>2).float()) + ((price-0)**2).mul((price<0.5).float())
            # penalty = ((price > 2).float()).mul((price - 1) ** 2) + ((price < 0.5).float()).mul((price - 0) ** 2)
            # penalty = ((price > 2).float()).mul(0.1) + ((price < 0.5).float()).mul(0.1)
            # penalty = torch.zeros_like(price, device=price.device)

            # if not for_value:
            # penalties.append(penalty.view(-1))
            losses[0].append(intent_loss.reshape(-1))
            losses[1].append(pact_loss.reshape(-1))
            ents[0].append(intent_ent.reshape(-1))
            ents[1].append(pact_ent.reshape(-1))
            values.append(value.view(-1))

        # regular = torch.cat(penalties)
        regular = None

        value_loss = []
        pg_losses = ([], [])
        ret = torch.tensor([], device=values[0].device, dtype=torch.float)
        cur_size = 0
        # td_error = adv = (discount*v[s']+r - v[s])
        for i in range(len(batch_iter)-1, -1, -1):
            # mid reward = 0,
            # discount*v[s']+r = discount*v[s']
            if ret.shape[0] > 0:
                ret = discount * values[i+1][:ret.shape[0]].detach()

            # s' do not exist
            # discount*v[s']+r = r
            if cur_size < batch_iter[i].size:
                step = batch_iter[i].size - cur_size
                tmp = torch.tensor(reward[cur_size:cur_size+step], device=value[0].device, dtype=torch.float)
                ret = torch.cat([ret, tmp])
                cur_size += step
            # value loss
            value_loss.append(F.mse_loss(values[i], ret, reduction='none'))
            # self._compute_loss(None, value=values[i], oracle=ret[:cur_size], loss=self.critic_loss)
            # policy loss
            adv = ret-values[i].detach()
            # print('infos', ret[:cur_size].shape, values[i].shape, losses[0][i].shape, losses[1][i].shape, adv.shape)
            pg_losses[0].append(adv*losses[0][i])
            pg_losses[1].append(adv*losses[1][i])

        value_loss = torch.cat(value_loss, dim=0)
        pg_losses = tuple(torch.cat(pl, dim=0) for pl in pg_losses)
        ents = tuple(torch.cat(e, dim=0) for e in ents)
        losses = tuple(torch.cat(e, dim=0) for e in losses)

        return pg_losses, ents, value_loss, regular, (losses, policy_stats)

    def update_a2c(self, args, batch_iters, rewards, model, critic, discount=1, update_table=None):
        if update_table is None:
            update_table = {'value': False, 'policy': False}
        pg_losses, e_losses, value_loss, p_losses = None, None, None, None
        policy_stats = Statistics()
        cur = 0
        for i, bi in enumerate(batch_iters):
            p, e, v, _, info = self._gradient_accumulation(bi, rewards[cur: cur+bi[0].size], model, critic, discount)
            if pg_losses is None:
                pg_losses, e_losses, value_loss = p, e, v
                p_losses = info[0]
            else:
                pg_losses = tuple(torch.cat([pg_losses[i], p[i]], dim=-1) for i in range(2))
                e_losses = tuple(torch.cat([e_losses[i], e[i]], dim=-1) for i in range(2))
                value_loss = torch.cat([value_loss, v], dim=-1)
                p_losses = tuple(torch.cat([p_losses[i], info[0][i]], dim=-1) for i in range(2))
            policy_stats.update(info[1])

        # Update step
        # p_losses = p_losses.mean()
        # e_losses = e_losses.mean()
        value_loss = value_loss.mean()

        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
        # print('pgl', pg_losses[0])
        # print('el', e_losses[0])
        model_loss = tuple(pg_losses[i].mean() - self.ent_coef * e_losses[i].mean() for i in range(2))
        critic_loss = self.val_coef * value_loss
        pg_loss = model_loss[0] + model_loss[1]
        total_loss = pg_loss + critic_loss

        # print('all loss', final_loss, p_losses, e_losses, value_loss)
        nan_str = "nan: {}, {}\n{}, {}\n{}\n{}, {}".\
            format(torch.isnan(pg_losses[0].mean()), torch.isnan(pg_losses[1].mean()),
                   torch.isnan(e_losses[0].mean()), torch.isnan(e_losses[1].mean()),
                   torch.isnan(value_loss.mean()),
                   torch.isnan(p_losses[0].mean()), torch.isnan(p_losses[1].mean()))
        assert not torch.isnan(total_loss), nan_str
        # final_loss.backward()
        # model_loss.backward()
        # critic_loss.backward()
        # nn.utils.clip_grad_norm(critic.parameters(), 1.)
        # nn.utils.clip_grad_norm(model.parameters(), 1.)
        # self.optim.step()

        # if not self.model_type == "reinforce":
        if not args.only_run:
            if update_table['value']:
                critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                self.optim['critic'].step()

            if update_table['policy']:
                model.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                self.optim['model'].step()

        ret = {'total_loss': total_loss,
               'pg_loss': pg_loss,
               'pg_loss0': model_loss[0],
               'pg_loss1': model_loss[1],
               'value_loss': critic_loss,
               'entropy0': e_losses[0],
               'entropy1': e_losses[1],
               'policy_loss0': p_losses[0],
               'policy_loss1': p_losses[1],
        }
        return {k: ret[k].reshape(1, -1).cpu().data.numpy() for k in ret}

    def validate(self, args, valid_size, valid_critic=False, start=0, split='dev', exchange=None,
                 print_dialogues=0):
        """
        Args:
            print_dialogues: 打印前 N 个完整谈判对话（0=不打印）
        """
        rate = 0.5
        if exchange is not None:
            if exchange:
                rate = 1
            else:
                rate = 0
        self.model.eval()
        self.critic.eval()
        total_stats = RLStatistics()
        oppo_total_stats = RLStatistics()
        valid_size = min(valid_size, 200)
        # print('='*20, 'VALIDATION', '='*20)
        examples = []
        verbose_str = []
        controllers = []
        all_rewards = []
        all_strategies = []
        for sid, scenario in enumerate(self.scenarios[split][start:start+valid_size]):
            controller = self._get_controller(scenario, split=split, rate=rate)
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose)
            session = controller.sessions[self.training_agent]
            reward = self.get_reward(example, session)
            rewards = [self.get_reward(example, controller.sessions[i]) for i in range(2)]
            stats = RLStatistics(reward=rewards[0], n_words=1)
            oppo_stats = RLStatistics(reward=rewards[1], n_words=1)
            total_stats.update(stats)
            oppo_total_stats.update(oppo_stats)
            examples.append(example)
            controllers.append(controller)
            all_rewards.append(rewards)
            stra = [controller.sessions[i].price_strategy for i in range(2)]
            all_strategies.append(stra)
            verbose_str.append(self.example_to_str(example, controller, rewards, sid+start, stra))

        # 打印前 N 个完整谈判对话
        if print_dialogues > 0:
            for idx in range(min(print_dialogues, len(examples))):
                self.print_full_negotiation(
                    examples[idx], controllers[idx],
                    rewards=all_rewards[idx], sid=idx + start,
                    strategies=all_strategies[idx],
                )

        # print('='*20, 'END VALIDATION', '='*20)
        self.model.train()
        self.critic.train()
        return [total_stats, oppo_total_stats], examples, verbose_str

    def save_best_checkpoint(self, checkpoint, opt, valid_stats, score_type='accu'):

        if self.best_valid_reward is None:
            better = True
        else:
            if score_type == 'accu' or score_type == 'reward':
                better = valid_stats > self.best_valid_reward
            else:
                better = valid_stats < self.best_valid_reward

        if better:
            path = '{root}/{model}_best.pt'.format(
                        root=opt.model_path,
                        model=opt.model_filename)
            print('[Info] Update new best model({}:{:.4f}) at {}.'.format(score_type, valid_stats, path))
            self.best_valid_reward = valid_stats
        # if path is not None:
        #     print('[Info] Save best checkpoint {path}'.format(path=path))
            torch.save(checkpoint, path)

    def checkpoint_path(self, episode, opt, stats, score_type='loss'):
        path = '{root}/{model}_{score_type}{score:.4f}_e{episode:d}.pt'.format(
                    root=opt.model_path,
                    model=opt.model_filename,
                    score_type=score_type,
                    score=stats,
                    episode=episode)
        assert path is not None
        return path

    def update_opponent(self, type=None):
        if type is None:
            types = ['policy', 'critic']
        elif not isinstance(type, list):
            types = [type]
        else:
            types = type

        print('update opponent model for {}.'.format(types))
        if 'policy' in types:
            tmp_model_dict = self.agents[self.training_agent].env.model.state_dict()
            self.agents[self.training_agent^1].env.model.load_state_dict(tmp_model_dict)
        if 'critic' in types:
            tmp_model_dict = self.agents[self.training_agent].env.critic.state_dict()
            self.agents[self.training_agent^1].env.critic.load_state_dict(tmp_model_dict)

    def get_temperature(self, epoch, batch_size, args):
        # deprecated
        return 1
        if args.only_run or args.warmup_epochs == 0:
            return 1
        half = args.num_dialogues // batch_size / 2
        t_s, t_e = 0.3, 1
        i_s, i_e = 0, half
        return min(t_e, t_s + (t_e - t_s) * 1. * epoch / args.warmup_epochs)
        # return min(1., 1.*epoch/half)

    @staticmethod
    def merge_policy(i_policy, p_policy):
        actions = getattr(LFSampler, "_rl_actions", None)

        if actions is None or i_policy is None or p_policy is None:
            return None

        policy = torch.zeros(len(actions), dtype=torch.float32, device=i_policy.device)
        i_policy = i_policy.reshape(-1)
        p_policy = p_policy.reshape(-1)

        for i, act in enumerate(actions):
            policy[i] = i_policy[act[0]]
            if act[1] is not None:
                policy[i] = policy[i] * p_policy[act[1]]

        return policy

    # @staticmethod
    def sort_policy(self, policy, actions, display_num=-1, to_word=str):
        flat_policy = policy.reshape(-1)
        n = min(flat_policy.shape[0], len(actions))

        scored_actions = [(flat_policy[i].data.item(), actions[i]) for i in range(n)]
        scored_actions = sorted(scored_actions, reverse=True, key=lambda x: x[0])

        if display_num == -1:
            return scored_actions

        display_num = min(display_num, len(scored_actions))
        s = ""
        for i in range(display_num):
            sp, sa = scored_actions[i]
            if isinstance(sa, tuple):
                act = self.lf_vocab.to_word(sa[0])
                if sa[1] is not None:
                    act = act + "," + str(sa[1])
            else:
                act = to_word(sa)
            s = s + "{}:{:.3f} ".format(act, sp)

        return scored_actions, s

    def append_policy_info(self, e, ret, prefix="", display_num=3):
        output_data = e.metadata['output_data']

        p_policy = output_data.get('p_policy')
        i_policy = output_data.get('policy')
        rl_actions = getattr(LFSampler, "_rl_actions", None)
        use_tom = output_data.get('tominf_p') is not None

        # sl agent or missing price head
        if p_policy is None:
            if i_policy is not None:
                _, s = self.sort_policy(i_policy, list(range(LFSampler.INTENT_NUM)),
                                        display_num, self.lf_vocab.to_word)
                ret.append(prefix + "policy: " + s)
            else:
                ret.append(prefix + "policy: <unavailable>")
            return

        pact_size = np.prod(p_policy.shape)
        if pact_size == 1:
            _, s = self.sort_policy(i_policy, list(range(LFSampler.INTENT_NUM)),
                                    display_num, self.lf_vocab.to_word)
            ret.append(prefix + "policy: " + s)
        else:
            _, s = self.sort_policy(i_policy, list(range(LFSampler.INTENT_NUM)),
                                    display_num, self.lf_vocab.to_word)
            ret.append(prefix + "i_policy: " + s)

            _, s = self.sort_policy(p_policy, list(range(LFSampler.PACT_NUM)), display_num)
            ret.append(prefix + "p_policy: " + s)

            merged = RLTrainer.merge_policy(i_policy, p_policy)
            if merged is not None and rl_actions is not None:
                _, s = self.sort_policy(merged, rl_actions, display_num * 2)
                ret.append(prefix + "policy: " + s)
            else:
                ret.append(prefix + "policy: <merge unavailable>")

            if use_tom:
                if output_data.get('tominf_p2') is not None and rl_actions is not None:
                    _, s = self.sort_policy(output_data['tominf_p2'], rl_actions, display_num * 2)
                    ret.append(prefix + "tom_p2: " + s)
                if output_data.get('tominf_p') is not None and rl_actions is not None:
                    _, s = self.sort_policy(output_data['tominf_p'], rl_actions, display_num * 2)
                    ret.append(prefix + "tom_p: " + s)
                if output_data.get('tom_ev') is not None and rl_actions is not None:
                    _, s = self.sort_policy(output_data['tom_ev'], rl_actions, display_num * 2)
                    ret.append(prefix + "tom_ev: " + s)

    def example_to_text(self, example):
        ret = []
        for i, e in enumerate(example.events):
            if "real_uttr" in e.metadata.keys():
                ret.append("[{}: {}]\t{}\t{}\t\"{}\"".format(e.time, e.agent, e.action, e.data, e.metadata["real_uttr"]))
            else:
                intent = e.metadata.get('intent')
                intent = self.lf_vocab.to_word(intent)
                ret.append("[{}: {}]\t{}\t{}".format(e.time, e.agent, e.action, e.data))
                ret.append("        <{}>\t{}\t{}".format(intent, e.metadata.get('price'), e.metadata.get('price_act')))
                self.append_policy_info(e, ret, "  ")
                # ret.append("        <{}>\t{}\t{}".format(, e.metadata.get('price'), e.metadata.get('price_act')))
        return ret 

    def example_to_str(self, example, controller, rewards, sid=None, strategies=None):
        if strategies is None:
            strategies = [None, None]
        verbose_str = []
        
        if sid is not None:
            verbose_str.append('[Scenario id: {}]'.format(sid))
        for session_id, session in enumerate(controller.sessions):
            bottom, top = PriceScaler.get_price_range(session.kb)
            s = 'Agent[{}: {}], bottom ${}, top ${}'.format(session_id, session.kb.role, bottom, top)
            verbose_str.append(s)
        verbose_str.append("They are negotiating for "+session.kb.facts['item']['Category'])
        verbose_str.append("strategy: {}, {}".format(strategies[0], strategies[1]))

        strs = self.example_to_text(example)
        for str in strs:
            verbose_str.append(str)
        s = "reward: [0]{}\nreward: [1]{}".format(rewards[0], rewards[1])
        verbose_str.append(s)
        return verbose_str

    def get_eval_dict(self, examples, strategies):
        eval_dict, separate_edict = {}, [{} for _ in range(10)]
        # len, s_rate, utility, fairness
        for i, e in enumerate(examples):
            role = e.scenario.kbs[0].role
            l = len(e.events)
            srate = self._is_agreed(e)
            reward = self._margin_reward(e)[role]
            ut = reward / 2 + 0.5
            fa = 1 - abs(reward)
            tmp_dict = {'length': l, 'success_rate': srate, 'reward': reward}
            if srate:
                tmp_dict['utility'] = ut
                tmp_dict['fairness'] = fa
            for k in tmp_dict:
                if eval_dict.get(k) is None:
                    eval_dict[k] = []
                if separate_edict[strategies[i]].get(k) is None:
                    separate_edict[strategies[i]][k] = []
                eval_dict[k].append(tmp_dict[k])
                separate_edict[strategies[i]][k].append(tmp_dict[k])

        return eval_dict, separate_edict

    def sample_data(self, i, sample_size, args, real_batch=None, batch_size=128, eval=False):
        if real_batch is None:
            real_batch = sample_size
        rewards = [0]*2
        s_rewards = [0]*2
        _batch_iters = [[], []]
        _rewards = [[], []]
        examples = []
        verbose_strs = []
        strategies = [[], []]

        dialogue_batch = [[], []]
        last_t = time.time()
        for j in range(real_batch):
            # Rollout
            if eval:
                scenario, sid = self._get_scenario(scenario_id=j)
                controller = self._get_controller(scenario, split='train', rate=0)
            else:
                scenario, sid = self._get_scenario()
                controller = self._get_controller(scenario, split='train')
            controller.sessions[0].set_controller(controller)
            controller.sessions[1].set_controller(controller)
            example = controller.simulate(args.max_turns, verbose=args.verbose, temperature=self.get_temperature(i, sample_size, args))

            for session_id, session in enumerate(controller.sessions):
                # if args.only_run != True and session_id != self.training_agent:
                #     continue
                # Compute reward
                reward = self.get_reward(example, session)
                # Standardize the reward
                all_rewards = self.all_rewards[session_id]
                all_rewards.append(reward)
                s_reward = (reward - np.mean(all_rewards)) / max(1e-4, np.std(all_rewards))

                rewards[session_id] = reward
                s_rewards[session_id] = s_reward
                _rewards[session_id].append(reward)
                strategies[session_id].append(session.price_strategy_label)

            for session_id, session in enumerate(controller.sessions):
                # dialogue_batch[session_id].append(session.dialogue)
                # if len(dialogue_batch[session_id]) == batch_size or j == real_batch-1:
                batch_iter = session.iter_batches()
                T = next(batch_iter)
                _batch_iters[session_id].append(list(batch_iter))

            stra = [controller.sessions[i].price_strategy for i in range(2)]
            examples.append(example)

            debug_batch_mode = (
                os.environ.get("DEBUG_TOM_BATCH", "0") == "1" or
                os.environ.get("DEBUG_TOM_REALBATCH", "0") == "1"
            )

            if debug_batch_mode:
                verbose_str = ["[debug] skip example_to_str while inspecting real ToMBatch"]
            else:
                verbose_str = self.example_to_str(example, controller, rewards, sid, stra)

            if args.verbose:
                for s in verbose_str:
                    print(s)
            verbose_strs.append(verbose_str)

            # print('t: ', time.time() - last_t)
            # last_t=time.time()

        return _batch_iters, (_rewards, strategies), examples, verbose_strs

    def learn(self, args):
        rewards = [None]*2
        s_rewards = [None]*2

        critic_report_stats = RLStatistics()
        critic_stats = RLStatistics()
        last_time = time.time()

        tensorboard_every = 1
        save_every = 100

        history_train_losses = [[],[]]

        batch_size = 100

        pretrain_rounds = 3
        if args.only_run:
            batch_size = 1
            pretrain_rounds = 0

        save_every = max(1, save_every // batch_size)
        report_every = max(1, args.report_every // batch_size)

        print("\n========== [before training loop] ==========")
        print("num_dialogues =", getattr(args, "num_dialogues", None))
        print("report_every  =", getattr(args, "report_every", None))
        print("only_run      =", getattr(args, "only_run", None))
        print("sa_lambda_price  =", getattr(args, "sa_lambda_price", None))
        print("sa_lambda_switch =", getattr(args, "sa_lambda_switch", None))
        print("===========================================\n")

        for i in range(args.num_dialogues // batch_size):
            _batch_iters, _rewards, example, train_ex_str = self.sample_data(i, batch_size, args)
            # print('reward is:', _rewards)
            # print(np.mean(_rewards[0]), np.mean(_rewards[1]))
            # print(np.mean(self.all_rewards[0][-tensorboard_every*batch_size:]), np.mean(self.all_rewards[1][-tensorboard_every*batch_size:]))

            path_txt = '{root}/{model}_example{epoch}.txt'.format(
                root=args.model_path,
                model=args.name,
                epoch=i)
            with open(path_txt, 'w') as f:
                for ex in train_ex_str:
                    f.write('-' * 7 + '\n')
                    for s in ex:
                        f.write(s + '\n')

                # if train_policy:
                #     self.update(batch_iter, reward, self.model, discount=args.discount_factor)
                #
                # if train_critic:
                #     stats = self.update_critic(batch_iter, reward, self.critic, discount=args.discount_factor)
                #     critic_report_stats.update(stats)
                #     critic_stats.update(stats)
            k = -1
            for k in range(pretrain_rounds):
                loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                       discount=args.discount_factor, fix_policy=True)
                # if (k+1)%5 == 0:
                #     _batch_iters, _rewards, example, _ = self.sample_data(i, batch_size, args)
                # if loss[0,3].item() < 0.2:
                #     break
            if k >=0:
                print('Pretrained value function for {} rounds, and the final loss is {}.'.format(k+1, loss[0,3].item()))
            # if loss[0, 3].item() >= 0.3:
            #     print('Try to initialize critic parameters.')
            #     for p in self.critic.parameters():
            #         p.data.uniform_(-args.param_init, args.param_init)
            #     for k in range(20):
            #         loss = self.update_a2c(args, _batch_iters, _rewards, self.model, self.critic,
            #                                discount=args.discount_factor, fix_policy=True)
            #         if (k + 1) % 5 == 0:
            #             _batch_iters, _rewards, controller, example = self.sample_data(i, batch_size, args)
            #         if loss[0, 3].item() < 0.2:
            #             break
            #     print('Pretrained value function for {} rounds, and the final loss is {}.'.format(k + 1,
            #                                                                                       loss[0, 3].item()))
            loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                   discount=args.discount_factor)
            for k in range(pretrain_rounds):
                loss = self.update_a2c(args, _batch_iters, _rewards[self.training_agent], self.model, self.critic,
                                       discount=args.discount_factor, fix_policy=True)
            history_train_losses[self.training_agent].append(loss)

            # print('verbose: ', args.verbose)

                    # print("Standard reward: [0]{} [1]{}".format(s_rewards[0], s_rewards[1]))

            # Save logs on tensorboard
            if (i + 1) % tensorboard_every == 0:
                ii = (i+1)*batch_size
                for j in range(2):
                    self.writer.add_scalar('agent{}/reward'.format(j), np.mean(self.all_rewards[j][-tensorboard_every*batch_size:]), ii)
                    if len(history_train_losses[j]) >= tensorboard_every*batch_size:
                        tmp = np.concatenate(history_train_losses[j][-tensorboard_every*batch_size:], axis=0)
                        tmp = np.mean(tmp, axis=0)
                        self.writer.add_scalar('agent{}/total_loss'.format(j), tmp[0], ii)
                        self.writer.add_scalar('agent{}/policy_loss'.format(j), tmp[1], ii)
                        self.writer.add_scalar('agent{}/entropy_loss'.format(j), tmp[2], ii)
                        self.writer.add_scalar('agent{}/value_loss'.format(j), tmp[3], ii)
                        self.writer.add_scalar('agent{}/intent_loss'.format(j), tmp[4], ii)
                        self.writer.add_scalar('agent{}/price_loss'.format(j), tmp[5], ii)
                        self.writer.add_scalar('agent{}/logp_loss'.format(j), tmp[6], ii)


            if ((i + 1) % report_every) == 0:
                import seaborn as sns
                import matplotlib.pyplot as plt
                if args.histogram:
                    sns.set_style('darkgrid')

                # if train_policy:
                for j in range(2):
                    print('agent={}'.format(j), end=' ')
                    print('step:', i, end=' ')
                    print('reward:', rewards[j], end=' ')
                    print('scaled reward:', s_rewards[j], end=' ')
                    print('mean reward:', np.mean(self.all_rewards[j][-args.report_every:]))
                    if args.histogram:
                        self.agents[j].env.dialogue_generator.get_policyHistogram()

                # if train_critic:
                #     critic_report_stats.output(i+1, 0, 0, last_time)
                #     critic_report_stats = RLStatistics()

                print('-'*10)
                if args.histogram:
                    plt.show()

                last_time = time.time()

            # Save model
            if (i+1) % save_every == 0:
                # TODO: valid in dev set
                n_print = getattr(args, 'print_dialogues', 0)
                if n_print == 0 and args.verbose:
                    n_print = 3
                valid_stats, _, valid_verbose = self.validate(
                    args, 50 if args.only_run else 200,
                    print_dialogues=n_print,
                )
                valid_stats = valid_stats[0]

                # SwitchAware: dev intervention detail
                if getattr(args, 'print_dev_detail', False) and hasattr(self, 'tom'):
                    switchaware = getattr(self.tom, '__class__', None).__name__ == 'SwitchAwareHistoryModel'
                    if switchaware:
                        # 需要 batch_iters，从 sample_data 获取
                        try:
                            dev_batch_iters, dev_rewards_stra, _, dev_verbose = self.sample_data(
                                i, min(20, 50 if args.only_run else 200), args, eval=True,
                            )
                            dev_strategy = dev_rewards_stra[1][self.training_agent]
                            dev_bi = self._sort_merge_batch(
                                dev_batch_iters[self.training_agent],
                                len(dev_batch_iters[self.training_agent]),
                            )
                            self.print_dev_intervention_detail(
                                dev_bi, dev_strategy,
                                split_name="dev", max_batches=2, max_steps=3, max_samples=4,
                                verbose_strs=dev_verbose,
                            )
                        except Exception as e:
                            print(f"[warn] print_dev_detail failed: {e}")
                if not args.only_run:
                    self.drop_checkpoint(args, i+1, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                    if args.update_oppo:
                        print('update oppo!')
                        self.update_opponent(['policy', 'critic'])
                else:
                    print('valid ', valid_stats.str_loss())

                # if train_policy:
                #     valid_stats, _ = self.validate(args)
                #     self.drop_checkpoint(args, i, valid_stats, model_opt=self.agents[self.training_agent].env.model_args)
                #     self.update_opponent('policy')
                #
                # elif train_critic:
                #     # TODO: reverse!
                #     self.drop_checkpoint(args, i, critic_stats, model_opt=self.agents[self.training_agent].env.model_args)
                #     critic_stats = RLStatistics()
                # else:
                #     valid_stats, _ = self.validate(args)
                #     print('valid result: ', valid_stats.str_loss())

    def debug_realbatch_forward_once(self, raw_batch, strategy):
        tom_batch = ToMBatch.from_raw(raw_batch, strategy[:raw_batch.size])

        pred1, h1, identity1 = self._run_batch_tom_identity(
            tom_batch, hidden_state=None, only_identity=False, id_gt=False
        )

        print("\n=== REALBATCH FORWARD #1 ===")
        print("intent_logits:", tuple(pred1[0].shape))
        print("price_logits :", tuple(pred1[1].shape))
        print("identity     :", tuple(identity1.shape) if identity1 is not None else None)

        pred2, h2, identity2 = self._run_batch_tom_identity(
            tom_batch, hidden_state=h1, only_identity=False, id_gt=False
        )

        print("\n=== REALBATCH FORWARD #2 ===")
        print("intent_logits:", tuple(pred2[0].shape))
        print("price_logits :", tuple(pred2[1].shape))
        print("identity     :", tuple(identity2.shape) if identity2 is not None else None)

    def debug_switch_intervention_once(self, raw_batch, strategy):
        tom_batch = ToMBatch.from_raw(raw_batch, strategy[:raw_batch.size])

        runs = [
            ("normal", None),
            ("force_off", 0.0),
            ("force_on", 1.0),
        ]

        saved = {}

        # 关键：临时切到 eval，关掉 dropout；同时 no_grad 保证纯推理对照
        was_training = self.tom.training
        self.tom.eval()

        try:
            with torch.no_grad():
                for tag, override in runs:
                    pred, _, identity = self._run_batch_tom_identity(
                        tom_batch,
                        hidden_state=None,          # 同一输入、同一初始状态
                        only_identity=False,
                        id_gt=False,
                        force_switch_prob=override,
                    )

                    aux = copy.copy(getattr(self.tom, "last_aux_outputs", {}) or {})
                    saved[tag] = {
                        "pred": pred,
                        "identity": identity,
                        "aux": aux,
                    }
        finally:
            if was_training:
                self.tom.train()

        def _belief_from_saved(item):
            aux = item["aux"]
            if aux.get("belief_vector") is not None:
                return aux["belief_vector"].detach().cpu()
            if item["identity"] is not None:
                return item["identity"].detach().cpu()
            return None

        def _preview_tensor(x, n=8):
            if x is None:
                return None
            x = x.detach().cpu()
            if x.dim() == 0:
                return float(x.item())
            return x[: min(n, x.size(0))].tolist()

        normal_belief = _belief_from_saved(saved["normal"])
        off_belief = _belief_from_saved(saved["force_off"])
        on_belief = _belief_from_saved(saved["force_on"])

        if normal_belief is not None and off_belief is not None:
            diff_off_vs_normal = (off_belief - normal_belief).norm(dim=-1)
        else:
            diff_off_vs_normal = None

        if normal_belief is not None and on_belief is not None:
            diff_on_vs_normal = (on_belief - normal_belief).norm(dim=-1)
        else:
            diff_on_vs_normal = None

        if off_belief is not None and on_belief is not None:
            diff_on_vs_off = (on_belief - off_belief).norm(dim=-1)
        else:
            diff_on_vs_off = None

        print("\n================ SWITCH INTERVENTION DEBUG ================")
        for tag in ["normal", "force_off", "force_on"]:
            pred = saved[tag]["pred"]
            aux = saved[tag]["aux"]

            intent_logits, price_logits = pred
            switch_prob_raw = aux.get("switch_prob_raw", None)
            switch_prob_used = aux.get("switch_prob", None)

            print(f"\n--- {tag} ---")
            intent_ids = intent_logits.argmax(dim=-1)
            print("intent_argmax   :", _preview_tensor(intent_ids))
            print("intent_decode   :", self._decode_intent_ids(intent_ids))
            print("price_argmax    :", _preview_tensor(price_logits.argmax(dim=-1)))
            print("switch_prob_raw :", _preview_tensor(switch_prob_raw))
            print("switch_prob_used:", _preview_tensor(switch_prob_used))

        normal_intent = saved["normal"]["pred"][0].argmax(dim=-1)
        off_intent = saved["force_off"]["pred"][0].argmax(dim=-1)
        on_intent = saved["force_on"]["pred"][0].argmax(dim=-1)

        normal_price = saved["normal"]["pred"][1].argmax(dim=-1)
        off_price = saved["force_off"]["pred"][1].argmax(dim=-1)
        on_price = saved["force_on"]["pred"][1].argmax(dim=-1)

        print("intent_change off_vs_normal:", float((off_intent != normal_intent).float().mean().item()))
        print("intent_change on_vs_normal :", float((on_intent != normal_intent).float().mean().item()))
        print("price_change off_vs_normal :", float((off_price != normal_price).float().mean().item()))
        print("price_change on_vs_normal  :", float((on_price != normal_price).float().mean().item()))

        def _kl_from_logits(p_logits, q_logits):
            p_logprob = torch.log_softmax(p_logits, dim=-1)
            q_logprob = torch.log_softmax(q_logits, dim=-1)
            p_prob = p_logprob.exp()
            return (p_prob * (p_logprob - q_logprob)).sum(dim=-1)

        

        normal_intent_logits, normal_price_logits = saved["normal"]["pred"]
        off_intent_logits, off_price_logits = saved["force_off"]["pred"]
        on_intent_logits, on_price_logits = saved["force_on"]["pred"]

        print("intent_l2 off_vs_normal:",
            ((off_intent_logits - normal_intent_logits).norm(dim=-1)[:8]).detach().cpu().tolist())
        print("intent_l2 on_vs_normal :",
            ((on_intent_logits - normal_intent_logits).norm(dim=-1)[:8]).detach().cpu().tolist())

        print("price_l2 off_vs_normal:",
            ((off_price_logits - normal_price_logits).norm(dim=-1)[:8]).detach().cpu().tolist())
        print("price_l2 on_vs_normal :",
            ((on_price_logits - normal_price_logits).norm(dim=-1)[:8]).detach().cpu().tolist())

        print("intent_kl off||normal:",
            _kl_from_logits(off_intent_logits, normal_intent_logits)[:8].detach().cpu().tolist())
        print("intent_kl on||normal :",
            _kl_from_logits(on_intent_logits, normal_intent_logits)[:8].detach().cpu().tolist())

        print("price_kl off||normal:",
            _kl_from_logits(off_price_logits, normal_price_logits)[:8].detach().cpu().tolist())
        print("price_kl on||normal :",
            _kl_from_logits(on_price_logits, normal_price_logits)[:8].detach().cpu().tolist())

        print("\nbelief_diff ||force_off - normal|| :", _preview_tensor(diff_off_vs_normal))
        print("belief_diff ||force_on  - normal|| :", _preview_tensor(diff_on_vs_normal))
        print("belief_diff ||force_on  - force_off|| :", _preview_tensor(diff_on_vs_off))
        print("==========================================================\n")


    def _kl_from_logits(self, p_logits, q_logits):
        p_logprob = F.log_softmax(p_logits, dim=-1)
        q_logprob = F.log_softmax(q_logits, dim=-1)
        p_prob = p_logprob.exp()
        return (p_prob * (p_logprob - q_logprob)).sum(dim=-1)

    def _topk_set_change(self, a_logits, b_logits, k):
        k = min(k, a_logits.size(-1), b_logits.size(-1))
        a_idx = torch.topk(a_logits, k=k, dim=-1).indices
        b_idx = torch.topk(b_logits, k=k, dim=-1).indices

        a_idx, _ = torch.sort(a_idx, dim=-1)
        b_idx, _ = torch.sort(b_idx, dim=-1)

        return (a_idx != b_idx).any(dim=-1).float()


    def _top1_margin(self, logits):
        top2 = torch.topk(logits, k=2, dim=-1).values
        return top2[:, 0] - top2[:, 1]


    def _expected_bin(self, logits):
        probs = F.softmax(logits, dim=-1)
        idx = torch.arange(
            logits.size(-1),
            device=logits.device,
            dtype=probs.dtype,
        )
        return (probs * idx.unsqueeze(0)).sum(dim=-1)


    def debug_switch_intervention_eval(
        self,
        batch_iters,
        strategy,
        split_name="dev",
        max_merged_batches=None,
        max_steps_per_batch=None,
        decode_examples=False,
        verbose_strs=None,
    ):
        """
        在 dev / train 的 split_batch 输出上，批量做 intervention eval。

        Args:
            batch_iters: worker.split_batch(...) 的返回值
                        = (merged_batches, sorted_id, batch_length)
            strategy:    对应 split 前的 strategy 列表
            split_name:  "dev" / "train"
            max_merged_batches: 最多评估多少个 merged batch
            max_steps_per_batch: 每个 merged batch 最多跑多少个 step
            decode_examples: 是否打印少量 intent decode 示例
            verbose_strs: 对应 split 的 example_to_str 文本列表，用于 case 记录

        Returns:
            一个 dict，里面是聚合后的 intervention 指标。
        """
        merged_batches, sorted_id, batch_length = batch_iters

        stats = {
            "num_steps": 0.0,
            "num_examples": 0.0,

            "intent_flip_off_sum": 0.0,
            "intent_flip_on_sum": 0.0,
            "price_flip_off_sum": 0.0,
            "price_flip_on_sum": 0.0,

            "intent_l2_off_sum": 0.0,
            "intent_l2_on_sum": 0.0,
            "price_l2_off_sum": 0.0,
            "price_l2_on_sum": 0.0,

            "intent_kl_off_sum": 0.0,
            "intent_kl_on_sum": 0.0,
            "price_kl_off_sum": 0.0,
            "price_kl_on_sum": 0.0,

            "belief_diff_off_sum": 0.0,
            "belief_diff_on_sum": 0.0,
            "belief_diff_on_off_sum": 0.0,

            "switch_prob_raw_sum": 0.0,
            "switch_prob_used_normal_sum": 0.0,
            "switch_prob_used_off_sum": 0.0,
            "switch_prob_used_on_sum": 0.0,

            "price_top3_change_off_sum": 0.0,
            "price_top3_change_on_sum": 0.0,
            "price_top5_change_off_sum": 0.0,
            "price_top5_change_on_sum": 0.0,

            "price_margin_delta_off_sum": 0.0,
            "price_margin_delta_on_sum": 0.0,

            "price_expect_shift_off_sum": 0.0,
            "price_expect_shift_on_sum": 0.0,
        }

        first_price_flip_on = None
        first_price_top5_change_on = None

        was_training = self.tom.training
        self.tom.eval()

        try:
            with torch.no_grad():
                for mb_idx, merged_batch in enumerate(merged_batches):
                    if max_merged_batches is not None and mb_idx >= max_merged_batches:
                        break

                    stra = [strategy[j] for j in sorted_id[mb_idx]]

                    # 正常轨迹下的 recurrent hidden，跨 step 递推
                    h_normal = None

                    for step_idx, raw_batch in enumerate(merged_batch):
                        if max_steps_per_batch is not None and step_idx >= max_steps_per_batch:
                            break

                        tom_batch = ToMBatch.from_raw(raw_batch, stra[:raw_batch.size])

                        # 三次前向共用同一份输入 hidden
                        h_in = self._slice_tom_hidden(h_normal, raw_batch.size)

                        # ----- normal -----
                        normal_pred, normal_h_next, normal_identity = self._run_batch_tom_identity(
                            tom_batch,
                            hidden_state=h_in,
                            only_identity=False,
                            id_gt=False,
                            force_switch_prob=None,
                        )
                        normal_aux = copy.copy(getattr(self.tom, "last_aux_outputs", {}) or {})

                        # ----- force off -----
                        off_pred, _, off_identity = self._run_batch_tom_identity(
                            tom_batch,
                            hidden_state=h_in,
                            only_identity=False,
                            id_gt=False,
                            force_switch_prob=0.0,
                        )
                        off_aux = copy.copy(getattr(self.tom, "last_aux_outputs", {}) or {})

                        # ----- force on -----
                        on_pred, _, on_identity = self._run_batch_tom_identity(
                            tom_batch,
                            hidden_state=h_in,
                            only_identity=False,
                            id_gt=False,
                            force_switch_prob=1.0,
                        )
                        on_aux = copy.copy(getattr(self.tom, "last_aux_outputs", {}) or {})

                        # 下一 step 只沿用 normal rollout 的 hidden
                        h_normal = normal_h_next

                        normal_intent_logits, normal_price_logits = normal_pred
                        off_intent_logits, off_price_logits = off_pred
                        on_intent_logits, on_price_logits = on_pred

                        normal_intent = normal_intent_logits.argmax(dim=-1)
                        off_intent = off_intent_logits.argmax(dim=-1)
                        on_intent = on_intent_logits.argmax(dim=-1)

                        normal_price = normal_price_logits.argmax(dim=-1)
                        off_price = off_price_logits.argmax(dim=-1)
                        on_price = on_price_logits.argmax(dim=-1)

                        # 一定要先算 top-k change，再去用它找案例
                        price_top3_change_off = self._topk_set_change(normal_price_logits, off_price_logits, 3)
                        price_top3_change_on = self._topk_set_change(normal_price_logits, on_price_logits, 3)

                        price_top5_change_off = self._topk_set_change(normal_price_logits, off_price_logits, 5)
                        price_top5_change_on = self._topk_set_change(normal_price_logits, on_price_logits, 5)

                        active_dialogue_ids = sorted_id[mb_idx][:raw_batch.size]

                        flip_on_idx = (on_price != normal_price).nonzero(as_tuple=False).reshape(-1)
                        if first_price_flip_on is None and flip_on_idx.numel() > 0:
                            j = int(flip_on_idx[0].item())
                            dialogue_idx = int(active_dialogue_ids[j])

                            first_price_flip_on = self._build_intervention_case_record(
                                split_name=split_name,
                                mb_idx=mb_idx,
                                step_idx=step_idx,
                                local_idx=j,
                                dialogue_idx=dialogue_idx,
                                verbose_strs=verbose_strs,
                                normal_aux=normal_aux,
                                on_aux=on_aux,
                                normal_intent=normal_intent,
                                on_intent=on_intent,
                                normal_price=normal_price,
                                on_price=on_price,
                                normal_price_logits=normal_price_logits,
                                on_price_logits=on_price_logits,
                            )

                        top5_on_idx = (price_top5_change_on > 0.5).nonzero(as_tuple=False).reshape(-1)
                        if first_price_top5_change_on is None and top5_on_idx.numel() > 0:
                            j = int(top5_on_idx[0].item())
                            dialogue_idx = int(active_dialogue_ids[j])

                            first_price_top5_change_on = self._build_intervention_case_record(
                                split_name=split_name,
                                mb_idx=mb_idx,
                                step_idx=step_idx,
                                local_idx=j,
                                dialogue_idx=dialogue_idx,
                                verbose_strs=verbose_strs,
                                normal_aux=normal_aux,
                                on_aux=on_aux,
                                normal_intent=normal_intent,
                                on_intent=on_intent,
                                normal_price=normal_price,
                                on_price=on_price,
                                normal_price_logits=normal_price_logits,
                                on_price_logits=on_price_logits,
                            )

                        normal_price_margin = self._top1_margin(normal_price_logits)
                        off_price_margin = self._top1_margin(off_price_logits)
                        on_price_margin = self._top1_margin(on_price_logits)

                        normal_price_expect = self._expected_bin(normal_price_logits)
                        off_price_expect = self._expected_bin(off_price_logits)
                        on_price_expect = self._expected_bin(on_price_logits)

                        def _belief_from(identity, aux):
                            if aux.get("belief_vector") is not None:
                                return aux["belief_vector"]
                            return identity

                        normal_belief = _belief_from(normal_identity, normal_aux)
                        off_belief = _belief_from(off_identity, off_aux)
                        on_belief = _belief_from(on_identity, on_aux)

                        B = float(raw_batch.size)
                        stats["num_steps"] += 1.0
                        stats["num_examples"] += B

                        stats["intent_flip_off_sum"] += float((off_intent != normal_intent).float().sum().item())
                        stats["intent_flip_on_sum"] += float((on_intent != normal_intent).float().sum().item())
                        stats["price_flip_off_sum"] += float((off_price != normal_price).float().sum().item())
                        stats["price_flip_on_sum"] += float((on_price != normal_price).float().sum().item())

                        stats["intent_l2_off_sum"] += float((off_intent_logits - normal_intent_logits).norm(dim=-1).sum().item())
                        stats["intent_l2_on_sum"] += float((on_intent_logits - normal_intent_logits).norm(dim=-1).sum().item())
                        stats["price_l2_off_sum"] += float((off_price_logits - normal_price_logits).norm(dim=-1).sum().item())
                        stats["price_l2_on_sum"] += float((on_price_logits - normal_price_logits).norm(dim=-1).sum().item())

                        stats["intent_kl_off_sum"] += float(self._kl_from_logits(off_intent_logits, normal_intent_logits).sum().item())
                        stats["intent_kl_on_sum"] += float(self._kl_from_logits(on_intent_logits, normal_intent_logits).sum().item())
                        stats["price_kl_off_sum"] += float(self._kl_from_logits(off_price_logits, normal_price_logits).sum().item())
                        stats["price_kl_on_sum"] += float(self._kl_from_logits(on_price_logits, normal_price_logits).sum().item())

                        if normal_belief is not None and off_belief is not None:
                            stats["belief_diff_off_sum"] += float((off_belief - normal_belief).norm(dim=-1).sum().item())
                        if normal_belief is not None and on_belief is not None:
                            stats["belief_diff_on_sum"] += float((on_belief - normal_belief).norm(dim=-1).sum().item())
                        if off_belief is not None and on_belief is not None:
                            stats["belief_diff_on_off_sum"] += float((on_belief - off_belief).norm(dim=-1).sum().item())

                        switch_prob_raw = normal_aux.get("switch_prob_raw", None)
                        switch_prob_used_normal = normal_aux.get("switch_prob", None)
                        switch_prob_used_off = off_aux.get("switch_prob", None)
                        switch_prob_used_on = on_aux.get("switch_prob", None)

                        if switch_prob_raw is not None:
                            stats["switch_prob_raw_sum"] += float(switch_prob_raw.reshape(-1).sum().item())
                        if switch_prob_used_normal is not None:
                            stats["switch_prob_used_normal_sum"] += float(switch_prob_used_normal.reshape(-1).sum().item())
                        if switch_prob_used_off is not None:
                            stats["switch_prob_used_off_sum"] += float(switch_prob_used_off.reshape(-1).sum().item())
                        if switch_prob_used_on is not None:
                            stats["switch_prob_used_on_sum"] += float(switch_prob_used_on.reshape(-1).sum().item())

                        stats["price_top3_change_off_sum"] += float(price_top3_change_off.sum().item())
                        stats["price_top3_change_on_sum"] += float(price_top3_change_on.sum().item())
                        stats["price_top5_change_off_sum"] += float(price_top5_change_off.sum().item())
                        stats["price_top5_change_on_sum"] += float(price_top5_change_on.sum().item())

                        stats["price_margin_delta_off_sum"] += float((off_price_margin - normal_price_margin).abs().sum().item())
                        stats["price_margin_delta_on_sum"] += float((on_price_margin - normal_price_margin).abs().sum().item())

                        stats["price_expect_shift_off_sum"] += float((off_price_expect - normal_price_expect).abs().sum().item())
                        stats["price_expect_shift_on_sum"] += float((on_price_expect - normal_price_expect).abs().sum().item())

                        if decode_examples and mb_idx == 0 and step_idx == 0:
                            print(f"\n[{split_name} intervention preview]")
                            print("normal intent decode:", self._decode_intent_ids(normal_intent))
                            print("off    intent decode:", self._decode_intent_ids(off_intent))
                            print("on     intent decode:", self._decode_intent_ids(on_intent))

        finally:
            if was_training:
                self.tom.train()

        N = max(stats["num_examples"], 1.0)

        ret = {
            "split_name": split_name,
            "num_steps": stats["num_steps"],
            "num_examples": stats["num_examples"],

            "intent_flip_off_rate": stats["intent_flip_off_sum"] / N,
            "intent_flip_on_rate": stats["intent_flip_on_sum"] / N,
            "price_flip_off_rate": stats["price_flip_off_sum"] / N,
            "price_flip_on_rate": stats["price_flip_on_sum"] / N,

            "intent_l2_off_mean": stats["intent_l2_off_sum"] / N,
            "intent_l2_on_mean": stats["intent_l2_on_sum"] / N,
            "price_l2_off_mean": stats["price_l2_off_sum"] / N,
            "price_l2_on_mean": stats["price_l2_on_sum"] / N,

            "intent_kl_off_mean": stats["intent_kl_off_sum"] / N,
            "intent_kl_on_mean": stats["intent_kl_on_sum"] / N,
            "price_kl_off_mean": stats["price_kl_off_sum"] / N,
            "price_kl_on_mean": stats["price_kl_on_sum"] / N,

            "belief_diff_off_mean": stats["belief_diff_off_sum"] / N,
            "belief_diff_on_mean": stats["belief_diff_on_sum"] / N,
            "belief_diff_on_off_mean": stats["belief_diff_on_off_sum"] / N,

            "switch_prob_raw_mean": stats["switch_prob_raw_sum"] / N,
            "switch_prob_used_normal_mean": stats["switch_prob_used_normal_sum"] / N,
            "switch_prob_used_off_mean": stats["switch_prob_used_off_sum"] / N,
            "switch_prob_used_on_mean": stats["switch_prob_used_on_sum"] / N,

            "price_top3_change_off_rate": stats["price_top3_change_off_sum"] / N,
            "price_top3_change_on_rate": stats["price_top3_change_on_sum"] / N,
            "price_top5_change_off_rate": stats["price_top5_change_off_sum"] / N,
            "price_top5_change_on_rate": stats["price_top5_change_on_sum"] / N,

            "price_margin_delta_off_mean": stats["price_margin_delta_off_sum"] / N,
            "price_margin_delta_on_mean": stats["price_margin_delta_on_sum"] / N,

            "price_expect_shift_off_mean": stats["price_expect_shift_off_sum"] / N,
            "price_expect_shift_on_mean": stats["price_expect_shift_on_sum"] / N,

            "first_price_flip_on": first_price_flip_on,
            "first_price_top5_change_on": first_price_top5_change_on,
        }

        print("\n================ SWITCH INTERVENTION EVAL ================")
        for k, v in ret.items():
            if k in ("first_price_flip_on", "first_price_top5_change_on"):
                continue
            print(f"{k}: {v}")
        print("=========================================================\n")

        if first_price_flip_on is not None:
            print("\n[first_price_flip_on]")
            print(json.dumps(first_price_flip_on, indent=2, ensure_ascii=False))

        if first_price_top5_change_on is not None:
            print("\n[first_price_top5_change_on]")
            print(json.dumps(first_price_top5_change_on, indent=2, ensure_ascii=False))

        return ret

    def _price_to_bin_target(self, act_price, act_price_mask):
        price = act_price.reshape(-1).clamp(0.0, 1.0)
        mask = (act_price_mask.reshape(-1) > 0.5)

        bins = torch.zeros_like(price, dtype=torch.long)
        bins[mask] = (price[mask] * 99.0).round().clamp(0.0, 99.0).long()
        return bins.reshape(-1, 1)

    def _compute_switchaware_tom_loss(self, tom_batch, intent_logits, price_logits):
        """
        返回两套东西：
        1) 给 backward 用的 per-sample vectors
        2) 给 epoch 汇总用的 global-safe batch_stats
        """
        tgt_intent = tom_batch.act_intent.reshape(-1).long()  # [B]
        tgt_price_bin = self._price_to_bin_target(
            tom_batch.act_price, tom_batch.act_price_mask
        ).reshape(-1).long()  # [B]

        pmask = (tom_batch.act_price_mask.reshape(-1) > 0.5).float()  # [B]
        B = pmask.numel()

        # ===== intent =====
        intent_ce_vec = F.cross_entropy(intent_logits, tgt_intent, reduction='none')  # [B]
        intent_pred = intent_logits.argmax(dim=1)
        intent_correct_vec = (intent_pred == tgt_intent).float()  # [B]

        intent_loss_sum = intent_ce_vec.sum()
        intent_count = torch.tensor(float(B), device=intent_logits.device)
        intent_correct_sum = intent_correct_vec.sum()

        # ===== price (active-only) =====
        price_ce_vec = F.cross_entropy(price_logits, tgt_price_bin, reduction='none')  # [B]
        price_active_count = pmask.sum()
        price_active_count_safe = price_active_count.clamp_min(1.0)

        # backward 用：保持 downstream mean() 等价于 active-only mean
        loss_price_vec = price_ce_vec * pmask * (B / price_active_count_safe)

        price_pred = price_logits.argmax(dim=1)
        price_correct_sum = ((price_pred == tgt_price_bin).float() * pmask).sum()
        price_loss_sum = (price_ce_vec * pmask).sum()

        # ===== switch =====
        scan_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        scan_stats = {}

        aux = getattr(self.tom, "last_aux_outputs", None)
        if aux is not None and aux.get("switch_logit") is not None:
            switch_logit = aux["switch_logit"].reshape(-1)  # [B]
            switch_target, switch_mask = self._build_switch_pseudo_label(tom_batch)

            masked_pos = (switch_target * switch_mask).sum()
            masked_neg = ((1.0 - switch_target) * switch_mask).sum()

            fixed_pos_weight = getattr(self.args, "sa_switch_pos_weight", None)
            if fixed_pos_weight is not None and fixed_pos_weight > 0:
                pos_weight = torch.tensor(
                    float(fixed_pos_weight),
                    device=switch_logit.device,
                    dtype=switch_logit.dtype,
                )
            else:
                pos_weight = (masked_neg / masked_pos.clamp_min(1.0)).clamp(min=1.0, max=10.0)
                pos_weight = pos_weight.to(device=switch_logit.device, dtype=switch_logit.dtype)

            switch_bce_vec = F.binary_cross_entropy_with_logits(
                switch_logit,
                switch_target,
                reduction='none',
                pos_weight=pos_weight,
            )

            switch_count = switch_mask.sum()
            switch_count_safe = switch_count.clamp_min(1.0)

            # backward 用：保持 downstream mean() 等价于 masked-only mean
            loss_switch_vec = switch_bce_vec * switch_mask * (B / switch_count_safe)

            # 训练 loss 继续用 BCE；推理统计单独走 infer_thresh
            switch_prob_raw = aux.get("switch_prob_raw", torch.sigmoid(switch_logit)).reshape(-1)
            infer_thresh = getattr(self.args, "sa_switch_infer_thresh", 0.7)
            switch_pred = (switch_prob_raw > infer_thresh).float()

            for th in scan_thresholds:
                pred_th = (switch_prob_raw > th).float()

                tp_th = ((pred_th == 1).float() * (switch_target == 1).float() * switch_mask).sum()
                fp_th = ((pred_th == 1).float() * (switch_target == 0).float() * switch_mask).sum()
                tn_th = ((pred_th == 0).float() * (switch_target == 0).float() * switch_mask).sum()
                fn_th = ((pred_th == 0).float() * (switch_target == 1).float() * switch_mask).sum()
                pred_pos_th = (pred_th * switch_mask).sum()

                key = str(th).replace(".", "p")
                scan_stats[f"switch_scan_{key}_pred_pos"] = pred_pos_th.detach()
                scan_stats[f"switch_scan_{key}_tp"] = tp_th.detach()
                scan_stats[f"switch_scan_{key}_fp"] = fp_th.detach()
                scan_stats[f"switch_scan_{key}_tn"] = tn_th.detach()
                scan_stats[f"switch_scan_{key}_fn"] = fn_th.detach()

            switch_correct_sum = ((switch_pred == switch_target).float() * switch_mask).sum()
            switch_loss_sum = (switch_bce_vec * switch_mask).sum()

            switch_positive_count = (switch_target * switch_mask).sum()
            switch_positive_ratio = switch_positive_count / switch_count_safe

            switch_prob_mean = (switch_prob_raw * switch_mask).sum() / switch_count_safe
            switch_logit_mean = (switch_logit * switch_mask).sum() / switch_count_safe
            switch_pred_positive_count = (switch_pred * switch_mask).sum()

            tp = ((switch_pred == 1).float() * (switch_target == 1).float() * switch_mask).sum()
            fp = ((switch_pred == 1).float() * (switch_target == 0).float() * switch_mask).sum()
            tn = ((switch_pred == 0).float() * (switch_target == 0).float() * switch_mask).sum()
            fn = ((switch_pred == 0).float() * (switch_target == 1).float() * switch_mask).sum()

            # print("[switch loss debug]")
            # print("infer_thresh:", float(infer_thresh))
            # print("masked_pos:", float(masked_pos.item()))
            # print("masked_neg:", float(masked_neg.item()))
            # print("pos_weight:", float(pos_weight.item()))
            # print("pred_positive_count:", float(switch_pred_positive_count.item()))
            # print("tp:", float(tp.item()))
            # print("fp:", float(fp.item()))
            # print("tn:", float(tn.item()))
            # print("fn:", float(fn.item()))

        else:
            loss_switch_vec = torch.zeros_like(intent_ce_vec)
            switch_count = torch.zeros((), device=intent_logits.device)
            switch_correct_sum = torch.zeros((), device=intent_logits.device)
            switch_loss_sum = torch.zeros((), device=intent_logits.device)

            switch_positive_count = torch.zeros((), device=intent_logits.device)
            switch_positive_ratio = torch.zeros((), device=intent_logits.device)
            switch_prob_mean = torch.zeros((), device=intent_logits.device)
            switch_logit_mean = torch.zeros((), device=intent_logits.device)

            switch_pred_positive_count = torch.zeros((), device=intent_logits.device)
            tp = torch.zeros((), device=intent_logits.device)
            fp = torch.zeros((), device=intent_logits.device)
            tn = torch.zeros((), device=intent_logits.device)
            fn = torch.zeros((), device=intent_logits.device)

            for th in scan_thresholds:
                key = str(th).replace(".", "p")
                zero = torch.zeros((), device=intent_logits.device)
                scan_stats[f"switch_scan_{key}_pred_pos"] = zero
                scan_stats[f"switch_scan_{key}_tp"] = zero
                scan_stats[f"switch_scan_{key}_fp"] = zero
                scan_stats[f"switch_scan_{key}_tn"] = zero
                scan_stats[f"switch_scan_{key}_fn"] = zero

        batch_stats = {
            "intent_loss_sum": intent_loss_sum.detach(),
            "intent_count": intent_count.detach(),
            "intent_correct_sum": intent_correct_sum.detach(),

            "price_loss_sum": price_loss_sum.detach(),
            "price_active_count": price_active_count.detach(),
            "price_correct_sum": price_correct_sum.detach(),

            "switch_loss_sum": switch_loss_sum.detach(),
            "switch_count": switch_count.detach(),
            "switch_correct_sum": switch_correct_sum.detach(),

            "switch_positive_count": switch_positive_count.detach(),
            "switch_positive_ratio": switch_positive_ratio.detach(),
            "switch_prob_mean": switch_prob_mean.detach(),
            "switch_logit_mean": switch_logit_mean.detach(),

            "switch_pred_positive_count": switch_pred_positive_count.detach(),
            "switch_tp": tp.detach(),
            "switch_fp": fp.detach(),
            "switch_tn": tn.detach(),
            "switch_fn": fn.detach(),
        }
        batch_stats.update(scan_stats)

        # print("\n[check switch label]")
        # print("switch_count:", float(switch_count.item()))
        # print("switch_positive_count:", float(switch_positive_count.item()))
        # print("switch_positive_ratio:", float(switch_positive_ratio.item()))
        # print("switch_prob_mean:", float(switch_prob_mean.item()))
        # print("switch_logit_mean:", float(switch_logit_mean.item()))

        # 这里 intent_accu 改成真正 accuracy，而不是 gold prob
        intent_accu = intent_correct_vec.reshape(-1, 1)

        return intent_ce_vec, loss_price_vec, loss_switch_vec, batch_stats, intent_accu

    def _extract_prev_seller_act_from_state(self, state, intent_size):
        """
        从完整历史 state 中回溯 previous seller act id。

        state layout 假设与 _extract_prev_seller_price_from_state 一致：
        [L * intent_size onehot intents][L prices][L pmasks]

        seller 位次假设：
        最后一步 L-1 是当前 seller
        previous seller 依次在 L-3, L-5, ...

        返回：
        prev_seller_act_id: [B] long
        prev_seller_valid:  [B] float in {0,1}
        """
        B = state.size(0)
        device = state.device

        step_width = intent_size + 2
        L = state.size(1) // step_width

        intent_hist = state[:, :L * intent_size].reshape(B, L, intent_size)  # [B, L, intent_size]

        seller_cols = list(range(L - 1, -1, -2))

        prev_seller_act_id = torch.zeros(B, device=device, dtype=torch.long)
        prev_seller_valid = torch.zeros(B, device=device, dtype=state.dtype)

        # 从“上一个 seller 步”开始往前找第一个有效 act
        for col in seller_cols[1:]:
            act_onehot = intent_hist[:, col, :]  # [B, intent_size]
            valid = (act_onehot.abs().sum(dim=-1) > 0.5).float()
            fill_mask = (prev_seller_valid < 0.5) * valid

            act_id = act_onehot.argmax(dim=-1).long()
            prev_seller_act_id = torch.where(
                fill_mask > 0.5,
                act_id,
                prev_seller_act_id,
            )
            prev_seller_valid = torch.maximum(prev_seller_valid, valid)

        return prev_seller_act_id, prev_seller_valid

    def _build_switch_pseudo_label(self, tom_batch):
        """
        v2 baseline:
        switch_target = price_jump OR act_id_jump
        switch_mask   = act_pair_valid

        设计动机：
        1) 保留 price_jump，继续利用最稳的数值变化信号
        2) 加入 act_id_jump，减少纯 price-only 标签过稀疏的问题
        3) mask 扩到 act_pair_valid，允许“动作变了但没有有效 price 对”的样本进入监督
        """
        identity_state = tom_batch.identity_state
        B, D = identity_state.shape
        device = identity_state.device

        intent_size = (D - 4) // 2

        # ===== 当前 seller act / price =====
        seller_intent_onehot = identity_state[:, intent_size:2 * intent_size]
        seller_act_id = seller_intent_onehot.argmax(dim=-1).long()
        cur_act_valid = (seller_intent_onehot.abs().sum(dim=-1) > 0.5).float()

        seller_price_raw = identity_state[:, 2 * intent_size + 1].clamp(0.0, 1.0)
        cur_price_valid = (identity_state[:, 2 * intent_size + 3] > 0.5).float()

        # ===== previous seller act =====
        prev_seller_act_id, prev_act_valid = self._extract_prev_seller_act_from_state(
            tom_batch.state,
            intent_size,
        )
        act_pair_valid = cur_act_valid * prev_act_valid

        act_id_jump = ((seller_act_id != prev_seller_act_id).float()) * act_pair_valid

        # ===== previous seller price =====
        seller_price_prev, prev_price_valid = self._extract_prev_seller_price_from_state(
            tom_batch.state,
            intent_size,
        )
        price_pair_valid = cur_price_valid * prev_price_valid

        delta_abs = torch.where(
            price_pair_valid > 0.5,
            (seller_price_raw - seller_price_prev).abs(),
            torch.zeros_like(seller_price_raw),
        )

        thresh = getattr(self.args, "sa_switch_delta_thresh", 0.10)
        price_jump = ((delta_abs > thresh).float()) * price_pair_valid

        # ===== combine =====
        # mask: 只要存在 previous seller act，就允许监督
        switch_mask = act_pair_valid

        # target: price jump 或 act id jump 任一成立都记为 switch
        switch_target = torch.maximum(price_jump, act_id_jump)

        # 对无效位置清零，避免数值脏传播
        switch_target = switch_target * switch_mask

        # ===== debug =====
        switch_count = switch_mask.sum().clamp_min(1.0)

        price_jump_count = (price_jump * switch_mask).sum()
        act_jump_count = (act_id_jump * switch_mask).sum()
        overlap_count = ((price_jump > 0.5).float() * (act_id_jump > 0.5).float() * switch_mask).sum()
        switch_positive_count = (switch_target * switch_mask).sum()

        price_jump_002 = ((delta_abs > 0.02).float() * price_pair_valid).sum()
        price_jump_005 = ((delta_abs > 0.05).float() * price_pair_valid).sum()
        price_jump_010 = ((delta_abs > 0.10).float() * price_pair_valid).sum()
        price_jump_015 = ((delta_abs > 0.15).float() * price_pair_valid).sum()

        if switch_mask.sum() > 0:
            idx = (switch_mask > 0).nonzero(as_tuple=False).reshape(-1)[:5]
            # print("[sample seller acts/prices]")
            # print("cur_act_id   :", seller_act_id[idx].detach().cpu().tolist())
            # print("prev_act_id  :", prev_seller_act_id[idx].detach().cpu().tolist())
            # print("act_id_jump  :", act_id_jump[idx].detach().cpu().tolist())
            # print("cur_price    :", seller_price_raw[idx].detach().cpu().tolist())
            # print("prev_price   :", seller_price_prev[idx].detach().cpu().tolist())
            # print("delta_abs    :", delta_abs[idx].detach().cpu().tolist())
            # print("price_jump   :", price_jump[idx].detach().cpu().tolist())
            # print("switch_target:", switch_target[idx].detach().cpu().tolist())

        # print("\n[check switch label v2]")
        # print("switch_count:", float(switch_mask.sum().item()))
        # print("act_pair_valid_count:", float(act_pair_valid.sum().item()))
        # print("price_pair_valid_count:", float(price_pair_valid.sum().item()))
        # print("price_jump_count:", float(price_jump_count.item()))
        # print("act_jump_count:", float(act_jump_count.item()))
        # print("overlap_count:", float(overlap_count.item()))
        # print("switch_positive_count:", float(switch_positive_count.item()))
        # print("switch_positive_ratio:", float((switch_positive_count / switch_count).item()))
        # print("ratio@0.02:", float((price_jump_002 / price_pair_valid.sum().clamp_min(1.0)).item()))
        # print("ratio@0.05:", float((price_jump_005 / price_pair_valid.sum().clamp_min(1.0)).item()))
        # print("ratio@0.10:", float((price_jump_010 / price_pair_valid.sum().clamp_min(1.0)).item()))
        # print("ratio@0.15:", float((price_jump_015 / price_pair_valid.sum().clamp_min(1.0)).item()))

        return switch_target, switch_mask

    def _extract_prev_seller_price_from_state(self, state, intent_size):
        B = state.size(0)
        device = state.device
        dtype = state.dtype

        step_width = intent_size + 2
        L = state.size(1) // step_width

        price_hist = state[:, L * intent_size : L * (intent_size + 1)]   # [B, L]
        pmask_hist = state[:, L * (intent_size + 1) : ]                  # [B, L]

        seller_cols = list(range(L - 1, -1, -2))

        seller_price_prev = torch.zeros(B, device=device, dtype=dtype)
        prev_seller_valid = torch.zeros(B, device=device, dtype=dtype)

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

    # def _extract_prev_seller_act_from_state(self, state, intent_size):
    #     """
    #     从完整历史 state 中取 previous seller 的 act id。
    #     当前 seller 在最后一步 L-1，
    #     previous seller 在 L-3。
    #     """
    #     B = state.size(0)
    #     device = state.device
    #     dtype = state.dtype

    #     step_width = intent_size + 2
    #     L = state.size(1) // step_width

    #     # state = [intent_onehot, masked_price, pmask]
    #     # 前 L * intent_size 这一段是所有历史步的 intent onehot
    #     intent_hist = state[:, :L * intent_size].reshape(B, L, intent_size)

    #     if L < 3:
    #         prev_seller_act = torch.zeros(B, device=device, dtype=torch.long)
    #         prev_seller_exists = torch.zeros(B, device=device, dtype=dtype)
    #         return prev_seller_act, prev_seller_exists

    #     prev_col = L - 3
    #     prev_seller_act = intent_hist[:, prev_col, :].argmax(dim=-1).long()
    #     prev_seller_exists = torch.ones(B, device=device, dtype=dtype)

    #     return prev_seller_act, prev_seller_exists

    def _decode_intent_ids(self, ids, n=8):
        if ids is None:
            return None

        if torch.is_tensor(ids):
            ids = ids.detach().cpu().reshape(-1).tolist()

        ids = [int(x) for x in ids[:n]]

        ret = []
        for x in ids:
            try:
                name = self.lf_vocab.to_word(x)
            except Exception:
                name = str(x)
            ret.append((x, name))
        return ret

    def _build_intervention_case_record(
        self,
        split_name,
        mb_idx,
        step_idx,
        local_idx,
        dialogue_idx,
        verbose_strs,
        normal_aux,
        on_aux,
        normal_intent,
        on_intent,
        normal_price,
        on_price,
        normal_price_logits,
        on_price_logits,
    ):
        rec = {
            "split_name": split_name,
            "merged_batch_idx": int(mb_idx),
            "step_idx": int(step_idx),
            "local_sample_idx": int(local_idx),
            "dialogue_idx_in_split": int(dialogue_idx),

            "switch_prob_raw": (
                float(normal_aux["switch_prob_raw"][local_idx].item())
                if normal_aux.get("switch_prob_raw") is not None
                else None
            ),
            "switch_prob_used_normal": (
                float(normal_aux["switch_prob"][local_idx].item())
                if normal_aux.get("switch_prob") is not None
                else None
            ),
            "switch_prob_used_on": (
                float(on_aux["switch_prob"][local_idx].item())
                if on_aux.get("switch_prob") is not None
                else None
            ),

            "normal_intent": self._decode_intent_ids(
                normal_intent[local_idx:local_idx + 1], n=1
            )[0],
            "force_on_intent": self._decode_intent_ids(
                on_intent[local_idx:local_idx + 1], n=1
            )[0],
            "intent_changed_on": bool(
                on_intent[local_idx].item() != normal_intent[local_idx].item()
            ),

            "normal_price_argmax": int(normal_price[local_idx].item()),
            "force_on_price_argmax": int(on_price[local_idx].item()),
            "price_top1_flipped_on": bool(
                on_price[local_idx].item() != normal_price[local_idx].item()
            ),

            "normal_price_top5": torch.topk(
                normal_price_logits[local_idx],
                k=min(5, normal_price_logits.size(-1)),
                dim=-1,
            ).indices.detach().cpu().tolist(),

            "force_on_price_top5": torch.topk(
                on_price_logits[local_idx],
                k=min(5, on_price_logits.size(-1)),
                dim=-1,
            ).indices.detach().cpu().tolist(),

            "price_top5_changed_on": bool(
                self._topk_set_change(
                    normal_price_logits[local_idx:local_idx + 1],
                    on_price_logits[local_idx:local_idx + 1],
                    k=5,
                )[0].item() > 0.5
            ),

            "price_margin_normal": float(
                self._top1_margin(normal_price_logits[local_idx:local_idx + 1])[0].item()
            ),
            "price_margin_on": float(
                self._top1_margin(on_price_logits[local_idx:local_idx + 1])[0].item()
            ),
            "price_margin_abs_delta_on": float(
                (
                    self._top1_margin(on_price_logits[local_idx:local_idx + 1])[0]
                    - self._top1_margin(normal_price_logits[local_idx:local_idx + 1])[0]
                ).abs().item()
            ),

            "price_expect_normal": float(
                self._expected_bin(normal_price_logits[local_idx:local_idx + 1])[0].item()
            ),
            "price_expect_on": float(
                self._expected_bin(on_price_logits[local_idx:local_idx + 1])[0].item()
            ),
            "price_expect_abs_shift_on": float(
                (
                    self._expected_bin(on_price_logits[local_idx:local_idx + 1])[0]
                    - self._expected_bin(normal_price_logits[local_idx:local_idx + 1])[0]
                ).abs().item()
            ),
        }

        if verbose_strs is not None and dialogue_idx < len(verbose_strs):
            lines = verbose_strs[dialogue_idx]
            if isinstance(lines, list):
                rec["dialogue_text"] = "\n".join(lines)
            else:
                rec["dialogue_text"] = str(lines)

        return rec

    # ================================================================
    #  完整谈判对话打印 + dev 逐样本 intervention 诊断
    # ================================================================

    def print_full_negotiation(self, example, controller, rewards=None, sid=None, strategies=None):
        """
        打印一个完整的 seller/buyer 谈判对话，包括：
        - 场景信息（角色、价格区间、商品类别）
        - 每轮的 agent、intent、price、utterance
        - 最终 reward
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  FULL NEGOTIATION DIALOGUE")
        lines.append("=" * 60)

        if sid is not None:
            lines.append(f"[Scenario #{sid}]")

        for session_id, session in enumerate(controller.sessions):
            bottom, top = PriceScaler.get_price_range(session.kb)
            role = session.kb.role
            category = session.kb.facts['item']['Category']
            lines.append(f"  Agent[{session_id}]: role={role}, price=[${bottom:.0f}, ${top:.0f}], category={category}")

        if strategies:
            lines.append(f"  Strategies: agent0={strategies[0]}, agent1={strategies[1]}")

        lines.append("-" * 60)

        for turn_idx, e in enumerate(example.events):
            agent_id = e.agent
            role = controller.sessions[agent_id].kb.role
            action = e.action
            data = e.data

            # 提取 intent
            intent_str = "?"
            price_str = ""
            uttr_str = ""

            if e.metadata:
                intent_raw = e.metadata.get('intent')
                if intent_raw is not None:
                    try:
                        intent_str = self.lf_vocab.to_word(intent_raw)
                    except Exception:
                        intent_str = str(intent_raw)

                price_val = e.metadata.get('price')
                if price_val is not None:
                    price_str = f"  price={price_val}"

                if "real_uttr" in e.metadata:
                    uttr_str = f'  "{e.metadata["real_uttr"]}"'

                # SwitchAware aux info
                output_data = e.metadata.get('output_data', {})
                sa_aux = output_data.get('sa_aux', None) if output_data else None

            tag = "BUYER" if role == "buyer" else "SELLER"
            lines.append(f"  [{turn_idx:2d}] {tag}(agent{agent_id}) | {action} <{intent_str}>{price_str}{uttr_str}")

            if sa_aux is not None:
                sw_prob = sa_aux.get('switch_prob')
                belief_type = sa_aux.get('belief_type_probs')
                if sw_prob is not None:
                    sw_val = float(sw_prob.reshape(-1)[0].item()) if hasattr(sw_prob, 'item') else float(sw_prob)
                    lines.append(f"         switch_prob={sw_val:.3f}")
                if belief_type is not None:
                    bt = belief_type.reshape(-1).tolist() if hasattr(belief_type, 'tolist') else belief_type
                    bt_str = ", ".join(f"{v:.2f}" for v in bt[:4])
                    lines.append(f"         belief_type=[{bt_str}]")

        lines.append("-" * 60)
        if rewards is not None:
            lines.append(f"  Reward: agent0={rewards[0]:.4f}, agent1={rewards[1]:.4f}")
        lines.append("=" * 60)

        text = "\n".join(lines)
        print(text)
        return text

    def print_dev_intervention_detail(
        self,
        batch_iters,
        strategy,
        split_name="dev",
        max_batches=3,
        max_steps=5,
        max_samples=4,
        verbose_strs=None,
    ):
        """
        逐 batch / step / sample 打印 dev 上的 intervention 诊断：
        - 当前是 dev 的第几个 batch、第几个 step、第几个样本
        - normal 下的 intent 是什么
        - force_on 下 intent 有没有变
        - switch_prob、price 变化等

        用法：在 validate 之后调用，传入 batch_iters 和 strategy。
        """
        merged_batches, sorted_id, batch_length = batch_iters

        was_training = self.tom.training
        self.tom.eval()

        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  DEV INTERVENTION DETAIL (per batch / step / sample)")
        lines.append("=" * 70)

        try:
            with torch.no_grad():
                for mb_idx, merged_batch in enumerate(merged_batches):
                    if mb_idx >= max_batches:
                        break

                    stra = [strategy[j] for j in sorted_id[mb_idx]]
                    h_normal = None

                    for step_idx, raw_batch in enumerate(merged_batch):
                        if step_idx >= max_steps:
                            break

                        tom_batch = ToMBatch.from_raw(raw_batch, stra[:raw_batch.size])
                        h_in = self._slice_tom_hidden(h_normal, raw_batch.size)

                        # normal
                        normal_pred, normal_h_next, normal_identity = self._run_batch_tom_identity(
                            tom_batch, hidden_state=h_in, only_identity=False,
                            id_gt=False, force_switch_prob=None,
                        )
                        normal_aux = copy.copy(getattr(self.tom, "last_aux_outputs", {}) or {})

                        # force_on
                        on_pred, _, on_identity = self._run_batch_tom_identity(
                            tom_batch, hidden_state=h_in, only_identity=False,
                            id_gt=False, force_switch_prob=1.0,
                        )
                        on_aux = copy.copy(getattr(self.tom, "last_aux_outputs", {}) or {})

                        h_normal = normal_h_next

                        normal_intent = normal_pred[0].argmax(dim=-1)
                        on_intent = on_pred[0].argmax(dim=-1)
                        normal_price = normal_pred[1].argmax(dim=-1)
                        on_price = on_pred[1].argmax(dim=-1)

                        B = min(raw_batch.size, max_samples)

                        lines.append(f"\n--- [{split_name}] batch={mb_idx}, step={step_idx}, batch_size={raw_batch.size} ---")

                        for s in range(B):
                            dialogue_idx = sorted_id[mb_idx][s] if s < len(sorted_id[mb_idx]) else -1

                            n_intent_id = int(normal_intent[s].item())
                            o_intent_id = int(on_intent[s].item())
                            n_price_id = int(normal_price[s].item())
                            o_price_id = int(on_price[s].item())

                            try:
                                n_intent_name = self.lf_vocab.to_word(n_intent_id)
                            except Exception:
                                n_intent_name = str(n_intent_id)
                            try:
                                o_intent_name = self.lf_vocab.to_word(o_intent_id)
                            except Exception:
                                o_intent_name = str(o_intent_id)

                            intent_changed = "YES" if n_intent_id != o_intent_id else "no"
                            price_changed = "YES" if n_price_id != o_price_id else "no"

                            sw_prob_str = "?"
                            if normal_aux.get("switch_prob_raw") is not None:
                                sw_prob_str = f"{float(normal_aux['switch_prob_raw'].reshape(-1)[s].item()):.3f}"

                            belief_str = ""
                            if normal_aux.get("belief_type_probs") is not None:
                                bt = normal_aux["belief_type_probs"][s].detach().cpu().tolist()
                                belief_str = " belief=[" + ",".join(f"{v:.2f}" for v in bt[:4]) + "]"

                            lines.append(
                                f"  sample={s} (dia#{dialogue_idx}) | "
                                f"normal_intent={n_intent_name}({n_intent_id}) "
                                f"force_on_intent={o_intent_name}({o_intent_id}) "
                                f"intent_changed={intent_changed} | "
                                f"normal_price={n_price_id} force_on_price={o_price_id} "
                                f"price_changed={price_changed} | "
                                f"switch_prob={sw_prob_str}{belief_str}"
                            )

                            # 如果有 verbose_strs，打印对应对话的前几行
                            if verbose_strs is not None and dialogue_idx >= 0 and dialogue_idx < len(verbose_strs):
                                dia_lines = verbose_strs[dialogue_idx]
                                if isinstance(dia_lines, list):
                                    for dl in dia_lines[:6]:
                                        lines.append(f"    | {dl}")
                                    if len(dia_lines) > 6:
                                        lines.append(f"    | ... ({len(dia_lines) - 6} more lines)")

        finally:
            if was_training:
                self.tom.train()

        lines.append("=" * 70 + "\n")
        text = "\n".join(lines)
        print(text)
        return text