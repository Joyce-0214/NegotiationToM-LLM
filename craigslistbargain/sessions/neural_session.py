import math
import random
import re
import numpy as np
import torch
from onmt.Utils import use_gpu

from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity, Entity, CanonicalEntity

from craigslistbargain.core.event import Event
from craigslistbargain.sessions.session import Session
from craigslistbargain.neural.preprocess import markers, Dialogue
from craigslistbargain.neural.batcher_rl import RLBatch, ToMBatch, RawBatch
import copy
import time
import json


class NeuralSession(Session):
    # Number of types of price actions
    # in PytorchNeuralSession.generate()
    P_ACT_NUMBER = 4
    tominf_beta = 1

    def __init__(self, agent, kb, env, tom_session=False):
        """

        :param tom_session:
            False: SL session
            True: ToM session
            NeuralSession(): Using ToM
        """
        super(NeuralSession, self).__init__(agent)
        self.env = env
        self.kb = kb
        self.builder = env.utterance_builder
        self.generator = env.dialogue_generator
        self.cuda = env.cuda
        self.tom_session = tom_session

        self.force_switch_schedule = {}
        self.force_switch_persistent_start = None
        self.force_switch_persistent_value = None
        self.turn_idx = 0
        self.rollout_trace = []


        # utterance generator
        self.uttr_gen = env.nlg_module.gen

        self.batcher = self.env.dialogue_batcher
        self.dialogue = Dialogue(agent, kb, None)
        self.dialogue.kb_context_to_int()
        self.kb_context_batch = self.batcher.create_context_batch([self.dialogue], self.batcher.kb_pad)
        self.max_len = 100

        # SL agent type
        # min/expect
        # price strategy: high/low/decay
        self.tom_type = env.tom_type
        strategy_name = ['insist', 'decay']
        strategy_name = ['insist', 'decay', 'persuaded']
        strategy_name = ['insist', 'decay', 'persuaded', 'convex', 'concave', 'wait', 'sigmoid']
        strategy_name = ['insist', 'decay', 'persuaded', 'convex', 'concave', 'low', 'high']
        self.price_strategy_distribution = \
            {'name': strategy_name,
             'prob': [1./len(strategy_name)]*len(strategy_name)}
        self.price_strategy = env.price_strategy
        self.acpt_range = [0.4, 1]

        # Tom
        self.tom = False
        self.controller = None
        # self.tominf_beta = 1
        self.sa_hidden = None  # SwitchAware: (seller_h, belief_state) 跨轮传递
        if hasattr(env, 'usetom') and env.usetom:
            self.tom = True
            self.critic = env.critic
            if tom_session == True:
                self.model = env.tom_model
                self.generator = env.tom_generator
                self.tom_hidden = None
            else:
                self.model = env.model

        if env.name == 'pt-neural-r':
            self.sample_price_strategy()

    def sample_price_strategy(self):
        ps = self.price_strategy_distribution['name']
        p = [s for s in self.price_strategy_distribution['prob']]
        self.price_strategy = np.random.choice(ps, p=p)

    @property
    def price_strategy_label(self):
        for i, s in enumerate(self.price_strategy_distribution['name']):
            if s == self.price_strategy:
                return i
        return -1

    def set_controller(self, controller):
        self.controller = controller
        if not isinstance(self.tom_session, bool):
            self.tom_session.controller = controller

    # TODO: move this to preprocess?
    def convert_to_int(self):
        self.dialogue.lf_to_int()

    def receive_quit(self):
        e = Event.QuitEvent(self.agent ^ 1, time=self.timestamp())
        self.receive(e)

    # Using another dialogue with semi-event
    def receive(self, event, another_dia=None):
        if isinstance(event, Event) and event.action in Event.decorative_events:
            return
        # print(event.data)
        # Parse utterance
        lf = event.metadata
        utterance = self.env.preprocessor.process_event(event, self.kb)

        # e.g. sentences are here!!
        # when message is offer / accept / reject we do not have "real_uttr"
        # need to be added into state in your ways

        # Empty message
        if lf is None:
            return

        if lf.get('intent') is None:
            print('lf i is None: ', lf)
        if another_dia is None:
            self.dialogue.add_utterance(event.agent, utterance, lf=lf)
        else:
            another_dia.add_utterance(self.dialogue.agent ^ 1, utterance, lf=lf)


    def _has_entity(self, tokens):
        for token in tokens:
            if is_entity(token):
                return True
        return False

    def attach_punct(self, s):
        s = re.sub(r' ([.,!?;])', r'\1', s)
        s = re.sub(r'\.{3,}', r'...', s)
        s = re.sub(r" 's ", r"'s ", s)
        s = re.sub(r" n't ", r"n't ", s)
        return s

    def _intent_ind2word(self, ind):
        return self.env.lf_vocab.to_word(ind)

    def _to_semi_event(self, tokens):
        # From scale to real price
        # print('semi_event: {}->'.format(tokens[1]),end='')
        if tokens[1] is not None:
            if isinstance(tokens[1], float):
                tokens[1] = CanonicalEntity(type='price', value=tokens[1])
            tokens[1] = self.builder.get_price_number(tokens[1], self.kb)
        # print('{}.'.format(tokens[1]))
        return tokens

    def _to_event(self, utterance, lf, output_data):
        intent = lf.get('intent')
        intent = self.env.lf_vocab.to_word(intent)
        metadata = {**lf, 'output_data': output_data}
        metadata_nolf = {'output_data': output_data}
        if intent == markers.OFFER:
            return self.offer('offer', metadata)
        elif intent == markers.ACCEPT:
            return self.accept(metadata=metadata)
        elif intent == markers.REJECT:
            return self.reject(metadata=metadata)
        elif intent == markers.QUIT:
            return self.quit(metadata=metadata)
        return self.message(utterance, metadata=metadata)

    def get_value(self, all_events):
        all_dia = []
        lt0 = last_time = time.time()
        time_list = None
        a_r = [self.acc_idx, self.rej_idx]
        qt = [self.quit_idx]
        # all_events = [(tokens, uttr)]
        if all_events[0][0][0] in a_r:
            price = None
            # get offer price
            # TODO: token_turns

            if self.dialogue.lf_turns[-1].get('price') is not None:

                price = self.builder.get_price_number(self.dialogue.lf_turns[-1].get('price') , self.kb)
            values = []
            for e in all_events:
                r = self.controller.get_margin_reward(price=price, agent=self.agent, is_agreed=e[0][0] == self.acc_idx)
                values.append(r)

            return torch.tensor(values, device=next(self.critic.parameters()).device).view(-1,1)

        if all_events[0][0][0] in qt:
            is_agreed = (self.acc_idx == self.dialogue.lf_turns[-1].get('intent'))

            values = [self.controller.get_margin_reward(price=None, agent=self.agent, is_agreed=is_agreed)]

            return torch.tensor(values, device=next(self.critic.parameters()).device).view(-1,1)

        # Normal cases:
        attached_events = []
        attached_uttrs = []
        for e, u in all_events:
            e = self.env.preprocessor.process_event(e, self.kb)
            attached_uttrs.append(u)
            attached_events.append({'intent': e[0], 'price': e[1], 'original_price': None})


        batch = self._create_batch(attached_events=(attached_events, self.dialogue.agent^1, attached_uttrs))
        # print('create batch: ', time.time() - last_time)
        last_time = time.time()

        rlbatch = RLBatch.from_raw(batch, None, None)

        values = self.critic(rlbatch.uttr, rlbatch.state)

        return values

    def _to_real_price(self, price):
        if price is None: return None
        return self.builder.get_price_number(price, self.kb)

    def _raw_token_to_lf(self, tokens):
        if tokens[-1] is None:
            tokens = tokens[:-1]
        if len(tokens) > 1:
            price = self._to_real_price(tokens[1])
            # print('rt price', tokens[1], type(tokens[1]), price)
            return {'intent': tokens[0], 'price': price}
        return {'intent': tokens[0]}

    def _lf_to_utterance(self, lf, as_tokens=False, add_stra=None):
        role = self.kb.facts['personal']['Role']
        category = self.kb.facts['item']['Category']
        tmplf = lf.copy()
        tmplf['intent'] = self.env.lf_vocab.to_word(tmplf['intent'])
        utterance, uid = self.uttr_gen(tmplf, role, category, as_tokens=as_tokens, add_stra=add_stra)
        return utterance, uid

    @staticmethod
    def _pact_to_price(p_act, p_last):
        pmax, pmin = p_last[0], p_last[1]
        p = 1
        if p_act == 0:
            # insist
            p = pmax
        elif p_act == 1:
            p = pmin
        elif p_act == 2:
            p = (pmax + pmin) / 2
        elif p_act == 3:
            # FIX: Use relative decay instead of absolute
            p = pmax - 0.1 * (pmax - pmin)
        else:
            print('what\'s wrong?', p_act)

        p = min(p, pmax)
        p = max(p, pmin)
        return p

    # TODO: special token for different strategy.
    def _add_strategy_in_uttr(self, uttr):
        c = self.env.vocab.size - 1 - self.price_strategy_label
        uttr = uttr.copy()
        # for each sentences
        if random.randint(0, 5) > 0:
            return uttr
        uttr.insert(random.randint(0, len(uttr)), self.env.vocab.to_word(c))
        return uttr

    def _is_switch_aware(self):
        """检查当前 generator 的 model 是否是 SwitchAwareHistoryModel"""
        gen = self.generator
        if hasattr(gen, '_is_switch_aware'):
            return gen._is_switch_aware()
        return getattr(gen, 'model', None).__class__.__name__ == 'SwitchAwareHistoryModel'

    def _sa_detach_hidden(self, hidden):
        """Detach SwitchAware hidden state (seller_h, belief) 以切断计算图"""
        if hidden is None:
            return None
        if isinstance(hidden, tuple) and len(hidden) == 2:
            h, belief = hidden
            if hasattr(belief, 'detach'):
                return (h.detach(), belief.detach())
            return (h.detach(), belief)
        if hasattr(hidden, 'detach'):
            return hidden.detach()
        return hidden

    def reset_switch_state(self, clear_schedule=True):
        """
        显式重置 SwitchAware 在线推理状态。
        """
        self.sa_hidden = None
        self.turn_idx = 0
        self.rollout_trace = []

        if clear_schedule:
            self.force_switch_schedule = {}
            self.force_switch_persistent_start = None
            self.force_switch_persistent_value = None

        if hasattr(self, "tom_hidden"):
            self.tom_hidden = None

    def set_force_switch_from_turn(self, start_turn, switch_value):
        """
        从指定 turn 开始持续强制 switch。
        例子：
            set_force_switch_from_turn(2, 0.0)
            set_force_switch_from_turn(2, 1.0)
        """
        if switch_value is None:
            self.force_switch_persistent_start = None
            self.force_switch_persistent_value = None
            return

        self.force_switch_persistent_start = int(start_turn)
        self.force_switch_persistent_value = float(switch_value)

    def clear_force_switch_schedule(self):
        self.force_switch_schedule = {}
        self.force_switch_persistent_start = None
        self.force_switch_persistent_value = None

    def _build_turn_trace(self, output_data, lf, turn_idx=None):
        sa = output_data.get("sa_aux", {})
        if not sa:
            return None

        if turn_idx is None:
            turn_idx = self.turn_idx

        intent_val = lf.get("intent")
        if isinstance(intent_val, int):
            intent_name = self.env.lf_vocab.to_word(intent_val)
        else:
            intent_name = str(intent_val)

        trace = {
            "turn_idx": int(turn_idx),
            "intent": intent_name,
            "price": lf.get("price"),
            "price_act": lf.get("price_act"),
            "prob": lf.get("prob"),
            "switch_prob_raw": float(sa["switch_prob_raw"].reshape(-1)[0])
                if "switch_prob_raw" in sa else None,
            "switch_prob_used": float(sa["switch_prob"].reshape(-1)[0])
                if "switch_prob" in sa else None,
            "belief_type_probs": sa["belief_type_probs"][0].tolist()
                if "belief_type_probs" in sa else None,
            "belief_confidence": float(sa["belief_confidence"].reshape(-1)[0])
                if "belief_confidence" in sa else None,
        }

        # 新增：price top-k 诊断
        p_policy = output_data.get("p_policy", None)
        if torch.is_tensor(p_policy) and p_policy.numel() > 1:
            flat = p_policy[0].reshape(-1)
            k = min(3, flat.numel())
            topv, topi = torch.topk(flat, k=k)
            trace["price_top3"] = [
                {"bin": int(topi[j].item()), "prob": float(topv[j].item())}
                for j in range(k)
            ]
            if k >= 2:
                trace["price_top1_top2_gap"] = float((topv[0] - topv[1]).item())
            else:
                trace["price_top1_top2_gap"] = None
        else:
            trace["price_top3"] = None
            trace["price_top1_top2_gap"] = None

        return trace

    def print_turn_trace(self, output_data, lf, turn_idx=None):
        trace = self._build_turn_trace(output_data, lf, turn_idx=turn_idx)
        if trace is None:
            return

        self.rollout_trace.append(copy.deepcopy(trace))
        print("[turn-trace]", json.dumps(trace, ensure_ascii=False))

    def get_rollout_trace(self):
        return copy.deepcopy(self.rollout_trace)

    def build_rollout_summary(self, example):
        """
        给一条完整对话构造 rollout-level summary。
        用来比较：
            normal / force_off / force_on
        是否改变了最终谈判结果。
        """
        trace = self.get_rollout_trace()

        # 简单判断是否成交
        is_agreed = any(getattr(e, "action", None) == "accept" for e in example.events)

        # 从最后往前找一个非空 price，当作最后谈判价格的近似记录
        final_price = None
        for e in reversed(example.events):
            md = getattr(e, "metadata", None)
            if isinstance(md, dict) and md.get("price") is not None:
                final_price = md.get("price")
                break

        buyer_reward = None
        seller_reward = None
        if self.controller is not None and hasattr(self.controller, "get_margin_reward"):
            try:
                buyer_reward = self.controller.get_margin_reward(
                    price=final_price,
                    agent=0,
                    is_agreed=is_agreed,
                )
            except Exception:
                buyer_reward = None

            try:
                seller_reward = self.controller.get_margin_reward(
                    price=final_price,
                    agent=1,
                    is_agreed=is_agreed,
                )
            except Exception:
                seller_reward = None

        raw_vals = [x["switch_prob_raw"] for x in trace if x.get("switch_prob_raw") is not None]
        used_vals = [x["switch_prob_used"] for x in trace if x.get("switch_prob_used") is not None]

        summary = {
            "num_turns": len(trace),
            "num_events": len(example.events),
            "is_agreed": bool(is_agreed),
            "final_price": final_price,
            "buyer_reward": buyer_reward,
            "seller_reward": seller_reward,
            "avg_switch_prob_raw": (sum(raw_vals) / len(raw_vals)) if len(raw_vals) > 0 else None,
            "avg_switch_prob_used": (sum(used_vals) / len(used_vals)) if len(used_vals) > 0 else None,
            "forced_turns": dict(self.force_switch_schedule),
            "trace": trace,
        }
        return summary

    def tom_inference(self, tokens, output_data):
        # For the step of choosing U2
        # get parameters of normal distribution for price

        # get all actions
        all_actions = self.generator.rl_actions
        best_action = (None, None)
        print_list = []

        tom_policy = []
        tom_actions = []

        avg_time = []

        all_value = [-np.inf for i in range(len(all_actions))]

        # tominf_p = p1*p2
        p1 = torch.zeros(len(all_actions))
        p2 = torch.zeros(len(all_actions))
        tominf_p = torch.zeros(len(all_actions))
        tom_ev = torch.zeros(len(all_actions))
        evs = torch.zeros(len(all_actions))

        for i, act in enumerate(all_actions):
            if output_data['policy'][0, act[0]].item() < 1e-7:
                continue
            # use fake step to get opponent policy
            tmp_tokens = list(act)
            p_act = None

            if tmp_tokens[1] is not None:
                tmp_tokens[1] = self._pact_to_price(tmp_tokens[1], output_data['last_prices'])   #新增判断，避免归一化传参错误。先把price action index转成归一化price
            tmp_lf = self._raw_token_to_lf(tmp_tokens)
            psl = self.controller.sessions[1-self.agent].price_strategy_label
            tmp_u, uid = self._lf_to_utterance(tmp_lf,
                                               add_stra=self.env.vocab.to_word(self.env.vocab.size - 1 - psl))
            e = self._to_event(tmp_u, tmp_lf, output_data)
            tmp_u = self.env.preprocessor.process_event(e, self.kb)
            # tmp_u = self._add_strategy_in_uttr(tmp_u)

            self.dialogue.add_utterance(self.agent, tmp_u, lf=tmp_lf, price_act=p_act)

            # From [0,1] to real price

            tmp_time = time.time()
            # get sigma(P(U3)*V(U3)), estimated reward of taking U2.
            # 小作修改，打印info / evs的真实数值范围
            info = self.controller.fake_step(self.agent, e, self.tom_session)
            avg_time.append(time.time() - tmp_time)
            self.dialogue.delete_last_utterance()
            self.controller.step_back(self.agent, self.tom_session)

            ev_raw = (NeuralSession.tominf_beta * info).exp()

            # print(
            #     "DEBUG tom_info:",
            #     "act=", act,
            #     "intent_prob=", output_data['policy'][0, act[0]].item(),
            #     "p_prob=", (output_data['p_policy'][0, act[1]].item() if act[1] is not None else None),
            #     "info=", float(info),
            #     "beta=", NeuralSession.tominf_beta,
            #     "exp_beta_info=", float(ev_raw),
            # )

            evs[i] = ev_raw
            tom_ev[i] = info

        evs = evs / evs.sum()

        for i, act in enumerate(all_actions):
            if output_data['policy'][0, act[0]].item() < 1e-7:
                continue
            tmp_tokens = list(act)
            if tmp_tokens[1] is not None:
                tmp_tokens[1] = self._pact_to_price(tmp_tokens[1], output_data['last_prices'])   # 新增判断，保证fake-step时使用的是正确的归一化price，最终选择的动作也带着正确的price
            # pi = pi1 * pi2
            ev = evs[i]
            tmp = output_data['policy'][0, act[0]]
            if act[1] is not None:
                tmp = tmp * output_data['p_policy'][0, act[1]]

            p1[i] = tmp.cpu().data.item()
            p2[i] = ev.cpu().item()

            tmp = tmp * ev
            tominf_p[i] = tmp.cpu().data.item()

            tom_policy.append(tmp.item())
            tom_actions.append(tmp_tokens)

            print_list.append((self.env.lfint_map.int_to_text([act[0]]), act, tmp.item(), ev.item(),
                               output_data['policy'][0, act[0]].item()))

        # print('fake_step costs {} time.'.format(np.mean(avg_time)))
        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()
        tominf_p = tominf_p / tominf_p.sum()
        policy_info = {'tominf_p1': p1, 'tominf_p2': p2, 'tominf_p': tominf_p, 'tom_ev': tom_ev}

        # print('tom_policy', tom_policy)

        # Sample action from new policy
        final_action = torch.multinomial(torch.tensor(tom_policy, ), 1).item()
        tokens = list(tom_actions[final_action])

        self.dialogue.lf_to_int()
        # for s in self.dialogue.lfs:
        #     print(s)

        return tokens, policy_info

    def try_all_aa(self, tokens, output_data):
        # For the step of choosing U3
        p_mean = self._scalar_price_from_policy(output_data['p_policy'])
        # p_logstd = output_data['price_logstd']
        # get all
        num_price = 1
        all_actions = self.generator.get_sl_actions(p_mean, 0, num_price)
        all_events = []
        new_all_actions = []

        for act in all_actions:
            # TODO: remove continue for min/max tom
            if output_data['policy'][0, act[0]].item() < 1e-7:
                continue
            # Use semi-event here
            #   *For semi-events we only need to transform the price (keep intent as integer)
            # From [0,1] to real price

            tmp_tokens = list(act)
            # pact -> price

            tmp_lf = self._raw_token_to_lf(tmp_tokens)
            tmp_u, uid = self._lf_to_utterance(tmp_lf, as_tokens=True)
            tmp_u = self.dialogue._insert_markers(self.agent, tmp_u, True)
            tmp_u = self.dialogue.textint_map.text_to_int(tmp_u, uid=uid)

            e = self._to_semi_event(tmp_tokens)
            all_events.append((e, tmp_u))
            new_all_actions.append(act)
        all_actions = new_all_actions

        print_list = []

        # Get value functions from other one.
        values = self.controller.get_value(self.agent, all_events)

        probs = torch.ones_like(values, device=values.device)
        for i, act in enumerate(all_actions):
            # print('act: ',i ,output_data['policy'], act, probs.shape)
            if act[1] is not None:
                probs[i, 0] = output_data['policy'][0, act[0]].item() * 1
            else:
                probs[i, 0] = output_data['policy'][0, act[0]].item()

            print_list.append(
                (self.env.textint_map.int_to_text([act[0]]), act, probs[i, 0].item(), values[i, 0].item()))

        info = {'values': values, 'probs': probs}

        # For the min one
        minone = torch.zeros_like(probs, device=probs.device)
        minone[values.argmin().item(), 0] = 1
        maxone = torch.zeros_like(probs, device=probs.device)
        maxone[values.argmax().item(), 0] = 1

        # print('info', self.tom_type, values.sum(), probs.sum(), (values*probs).sum())
        if self.tom_type == 'expectation':
            # If use expectation here
            return (values.mul(probs)).sum()
        elif self.tom_type == 'competitive':
            # If use max here
            return (values.mul(minone)).sum()
        elif self.tom_type == 'cooperative':
            # If use max here
            return (values.mul(maxone)).sum()
        else:
            print('Unknown tom type: ', self.tom_type)
            assert NotImplementedError()
        # return info

    def send(self, temperature=1, is_fake=False):

        last_time = time.time()

        acpt_range = None
        if self.env.name == 'pt-neural-r':
            acpt_range = self.acpt_range
        tokens, output_data = self.generate(is_fake=is_fake, temperature=temperature, acpt_range=acpt_range)

        if is_fake:
            tmp_time = time.time()
            return self.try_all_aa(tokens, output_data)

        last_time=time.time()
        if self.tom:
            tokens, policy_info = self.tom_inference(tokens, output_data)
            output_data.update(policy_info)

        if tokens is None:
            return None

        lf = self._raw_token_to_lf(tokens)
        utterance, uid = self._lf_to_utterance(lf)
        lf['price_act'] = output_data.get('price_act')
        lf['prob'] = output_data.get('prob')

        event = self._to_event(utterance, lf, output_data)
        uttr = self.env.preprocessor.process_event(event, self.kb)
        if uttr is None:
            print('event', event.action, event.metadata)

        price_act = {'price_act': output_data.get('price_act'), 'prob': output_data.get('prob')}
        self.dialogue.add_utterance(self.agent, uttr, lf=lf, price_act=price_act)

        # 在 turn_idx 自增之前打印，这样第一轮是 0
        self.print_turn_trace(output_data, lf, turn_idx=self.turn_idx)

        self.turn_idx += 1
        # print('tokens', tokens)
        
        return event
        # return self._tokens_to_event(tokens, output_data)


    def step_back(self):
        # Delete utterance from receive
        self.dialogue.delete_last_utterance()

    def iter_batches(self):
        """Compute the logprob of each generated utterance.
        """

        self.convert_to_int()
        batches = self.batcher.create_batch([self.dialogue])

        yield len(batches)
        for batch in batches:
            # TODO: this should be in batcher
            batch = RawBatch.generate(batch['encoder_args'],
                          batch['decoder_args'],
                          batch['context_data'],
                          self.env.lf_vocab,
                          cuda=self.env.cuda,)
            yield batch

    def _scalar_price_from_policy(self, p_policy):
        """
        兼容：
        1) 旧标量 price
        2) 100-bin price distribution / logits
        返回一个 [0,1] 的归一化 price 标量
        """
        if not torch.is_tensor(p_policy):
            return float(p_policy)

        if p_policy.numel() == 1:
            return float(p_policy.item())

        if p_policy.dim() == 1:
            p_policy = p_policy.view(1, -1)

        probs = p_policy
        # 如果不像概率分布，就先 softmax
        row_sum = float(probs[0].sum().item())
        if probs.min().item() < 0 or probs.max().item() > 1.0 + 1e-6 or abs(row_sum - 1.0) > 1e-3:
            probs = torch.softmax(p_policy, dim=-1)

        idx = torch.arange(probs.size(-1), device=probs.device, dtype=probs.dtype)
        mean_bin = float((probs[0] * idx).sum().item())
        denom = max(probs.size(-1) - 1, 1)
        return mean_bin / float(denom)


class PytorchNeuralSession(NeuralSession):
    def __init__(self, agent, kb, env, tom_session=False):
        super(PytorchNeuralSession, self).__init__(agent, kb, env, tom_session)
        self.vocab = env.vocab
        self.lf_vocab = env.lf_vocab
        self.quit_idx = self.lf_vocab.to_ind('quit')
        self.acc_idx = self.lf_vocab.to_ind('accept')
        self.rej_idx = self.lf_vocab.to_ind('reject')

        self.new_turn = False
        self.end_turn = False

    def get_decoder_inputs(self):
        # Don't include EOS
        utterance = self.dialogue._insert_markers(self.agent, [], True)[:-1]
        inputs = self.env.textint_map.text_to_int(utterance, 'decoding')
        inputs = np.array(inputs, dtype=np.int32).reshape([1, -1])
        return inputs

    def _create_batch(self, other_dia=None, attached_events=None):
        num_context = Dialogue.num_context

        # All turns up to now
        self.convert_to_int()
        if other_dia is None:
            dias = [self.dialogue]
        else:
            dias = other_dia

        LF, TOKEN, PACT = Dialogue.LF, Dialogue.TOKEN, Dialogue.PACT
        ROLE = Dialogue.ROLE

        encoder_turns = self.batcher._get_turn_batch_at(dias, LF, -1, step_back=self.batcher.state_length, attached_events=attached_events)
        # print('encoder_turns', encoder_turns)
        encoder_tokens = self.batcher._get_turn_batch_at(dias, TOKEN, -1, attached_events=attached_events)
        roles = self.batcher._get_turn_batch_at(dias, ROLE, -1, attached_events=attached_events)

        encoder_intent, encoder_price, encoder_price_mask = self.batcher.get_encoder_inputs(encoder_turns)

        encoder_args = {
            'intent': encoder_intent,
            'price': encoder_price,
            'price_mask': encoder_price_mask,
            'tokens': encoder_tokens,
        }
        if attached_events is not None:
            i = len(dias[0].lf_turns)+1
        else:
            i = len(dias[0].lf_turns)
        extra = [r + [i / self.batcher.dia_num] + encoder_price[j][-2:] for j, r in enumerate(roles)]
        encoder_args['extra'] = extra

        decoder_args = None

        context_data = {
            'encoder_tokens': encoder_tokens,
            'agents': [self.agent],
            'kbs': [self.kb],
        }
        return RawBatch.generate(encoder_args, decoder_args, context_data,
                self.lf_vocab, cuda=self.cuda)

    @staticmethod
    def get_price(length, strategy, current_price):
        factor = length
        o_factor = factor

        # decay until 0.7
        factor = min(1., factor * (1/0.7))

        if strategy == 'insist':
            prange = [1., 1.]
            p = prange[0] * (1 - factor) + prange[1] * (factor)
        elif strategy == 'decay':
            prange = [1., 0.4]
            p = prange[0] * (1 - factor) + prange[1] * (factor)
            # print('pfactor', p, factor)
        elif strategy == 'persuaded':
            prange = [1., 0.4]
            step = 1. / 5
            if o_factor < 1:
                # decay
                p = current_price - step * (prange[0] - prange[1])
            else:
                p = current_price
        else:
            prange = [0.4, 1.]
            x = factor
            if strategy == 'convex':
                factor = 1-(1-(x-1)**2)**0.5
            elif strategy == 'concave':
                factor = (1-x**2)**0.5
            elif strategy == 'sigmoid':
                y = math.exp(-10*x+5)
                factor = y/(1+y)
            elif strategy == 'wait':
                if x < 0.5:
                    factor = (0.25-(x-0.5)**2)**0.5
                else:
                    factor = 1-(0.25-(x-0.5)**2)**0.5
            elif strategy == 'high':
                factor = 1 - x/2
            elif strategy == 'low':
                factor = (1-x) / 2
            else:
                # p = oldp
                print('Unknown price strategy: ', strategy)
                assert NotImplementedError()
            factor = max(min(factor, 1.), 0.)
            p = prange[0] * (1 - factor) + prange[1] * (factor)

        return p

    @staticmethod
    def get_acpt_range(length, strategy, current_price):
        factor = length
        # factor = batch.state[1][0, -3].item()
        if strategy == 'insist':
            acpt_range = [1., 0.7]
        elif strategy == 'decay':
            prange = [1., 0.4]
            p = prange[0]*(1-factor) + prange[1]*(factor)
            acpt_range = [1., p-0.1]
        #elif self.price_strategy == 'persuaded':
        else:
            acpt_range = [1., current_price]
        return acpt_range

    def generate(self, temperature=1, is_fake=False, acpt_range=None, hidden_state=None):
        if len(self.dialogue.agents) == 0:
            self.dialogue._add_utterance(1 - self.agent, [], lf={'intent': 'start'})
            # TODO: Should we add an empty state?
            # TODO: What is this?

        batch = self._create_batch()
        force_switch_prob = self.force_switch_schedule.get(self.turn_idx, None)

        if (
            force_switch_prob is None
            and self.force_switch_persistent_start is not None
            and self.turn_idx >= self.force_switch_persistent_start
        ):
            force_switch_prob = self.force_switch_persistent_value

        intents, prices = batch.get_pre_info(self.lf_vocab)
        last_prices = [prices[0, 0].item(), prices[0, 1].item()]

        # get acpt_range
        if self.env.name == 'pt-neural-r':
            factor = batch.state[1][0, -3].item()
            acpt_range = self.get_acpt_range(factor, self.price_strategy, last_prices[0])

        # ------------------------------------------------------------
        # 关键修复：
        # switch-aware 在线路径必须走 ToMBatch，而不是 RLBatch
        # ------------------------------------------------------------
        if self._is_switch_aware():
            # 在线 rollout 时给 ToMBatch 一个长度为 1 的 strategy 列表即可
            # 这里只是为了让 batch 结构完整，不改变主实验脚本
            stra_label = 0
            try:
                if hasattr(self, "price_strategy_label") and self.price_strategy_label >= 0:
                    stra_label = int(self.price_strategy_label)
            except Exception:
                stra_label = 0

            gen_batch = ToMBatch.from_raw(batch, [stra_label])
            sa_hidden = self.sa_hidden
        else:
            gen_batch = RLBatch.from_raw(batch, None, None)
            sa_hidden = None

        output_data = self.generator.generate_batch(
            gen_batch,
            enc_state=None,
            whole_policy=is_fake,
            temperature=temperature,
            acpt_range=acpt_range,
            hidden_state=sa_hidden,
            force_switch_prob=force_switch_prob,
        )

        # SwitchAware: 存储更新后的 hidden state 供下一轮使用
        if self._is_switch_aware() and output_data.get('rnn_hidden') is not None:
            self.sa_hidden = self._sa_detach_hidden(output_data['rnn_hidden'])

        # SL Agent with rule-based price action
        if self.env.name == 'pt-neural-r' and self.price_strategy != 'neural' and output_data['price'] is not None:
            oldp = output_data['price']
            if isinstance(oldp, float):
                output_data['price'] = self.get_price(
                    batch.state[1][0, -3].item(),
                    self.price_strategy,
                    oldp,
                )

        entity_tokens = self._output_to_tokens(output_data)
        output_data['last_prices'] = last_prices

        return entity_tokens, output_data

    def _is_valid(self, tokens):
        if not tokens:
            return False
        if Vocabulary.UNK in tokens:
            return False
        return True

    def _output_to_tokens(self, data):
        if isinstance(data['intent'], int):
            predictions = [data["intent"]]
        elif isinstance(data['intent'], torch.Tensor):
            predictions = [data["intent"].item()]

        if data.get('price') is not None:
            p = data['price']
            p = max(p, 0.0)
            p = min(p, 1.0)
            predictions += [p]
        else:
            predictions += [None]

        return predictions

