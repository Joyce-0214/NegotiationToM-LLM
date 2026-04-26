import argparse
import random
import json
import numpy as np
import json

import time

from onmt.Utils import use_gpu
import logging

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from cocoa.neural.loss import ReinforceLossCompute
import cocoa.options

from craigslistbargain.core.scenario import Scenario
from craigslistbargain.core.controller import Controller
from craigslistbargain.systems import get_system
from craigslistbargain.neural.rl_trainer import RLTrainer
from craigslistbargain.neural import build_optim
import craigslistbargain.options as options

from craigslistbargain.neural.a2c_trainer import RLStatistics

from tensorboardX import SummaryWriter

import os
from craigslistbargain.buffer import ReplayBuffer
import torch

try:
    import thread
except ImportError:
    import _thread as thread

import multiprocessing
import multiprocessing.connection
import math
import pickle as pkl
import numpy as np
import shutil

from craigslistbargain.multi_manager import MultiRunner, execute_runner

def init_dir(path, clean_all=False):
    if not os.path.exists(path):
        print('[Info] make dir {}'.format(path))
        os.makedirs(path, exist_ok=True)
    else:
        print('[Warning] path {} exists!'.format(path))
        if clean_all:
            print('[Warning] clean files in {}!'.format(path))
            shutil.rmtree(path, True)
            # Deal with delay on NAS
            while not os.path.exists(path):
                os.mkdir(path)
            print('[Info] remake dir {}'.format(path))


class MultiRunner:
    def __init__(self, args, addr):
        self.init_trainer(args)
        # self.addr = self.get_real_addr(addr)
        # self.conn = multiprocessing.connection.Client(self.addr)

    def init_trainer(self, args):
        if args.random_seed:
            random.seed(args.random_seed+os.getpid())
            np.random.seed(args.random_seed+os.getpid())

        schema = Schema(args.schema_path)
        scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path), Scenario)
        valid_scenario_db = ScenarioDB.from_dict(schema, read_json(args.valid_scenarios_path), Scenario)

        # if len(args.agent_checkpoints) == 0
        # assert len(args.agent_checkpoints) <= len(args.agents)
        if len(args.agent_checkpoints) < len(args.agents):
            ckpt = [None] * 2
        else:
            ckpt = args.agent_checkpoints

        systems = [get_system(name, args, schema, False, ckpt[i], id=i) for i, name in enumerate(args.agents)]

        rl_agent = 0
        system = systems[rl_agent]
        model = system.env.model
        loss = None
        # optim = build_optim(args, [model, system.env.critic], None)
        optim = {'model': build_optim(args, model, None),}
        if system.env.critic is not None:
            optim['critic'] = build_optim(args, system.env.critic, None)
            optim['critic']._set_rate(0.05)
        if system.env.tom_model is not None:
            optim['tom'] = build_optim(args, system.env.tom_model, None)

            # 只有旧式 explicit identity models 才有 encoder.identity
            if hasattr(system.env.tom_model, 'encoder') and \
                hasattr(system.env.tom_model.encoder, 'identity'):
                    optim['tom_identity'] = build_optim(
                        args, system.env.tom_model.encoder.identity, None
                    )
            # optim['tom']._set_rate(0.1)

        scenarios = {'train': scenario_db.scenarios_list, 'dev': valid_scenario_db.scenarios_list}
        from craigslistbargain.neural.a2c_trainer import RLTrainer as A2CTrainer
        trainer = A2CTrainer(systems, scenarios, loss, optim, rl_agent,
                             reward_func=args.reward, cuda=(len(args.gpuid) > 0), args=args)

        self.args = args
        self.trainer = trainer
        self.systems = systems

    def get_real_addr(self, addr):
        return addr

    def simulate(self, cmd):
        i, batch_size, real_batch = cmd
        data = self.trainer.sample_data(i, batch_size, self.args, real_batch=real_batch)
        return data

    def train(self, epoch, batches, rewards, train_mode):
        update_table = {'policy': True, 'value': True}
        with torch.autograd.set_detect_anomaly(True):
            if train_mode == 'normal':
                pretrain_number = 3
                update_table['policy'] = False
                for i in range(pretrain_number):
                    info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                                   discount=self.args.discount_factor, update_table=update_table)
                update_table['policy'] = True
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, update_table=update_table)
                update_table['policy'] = False
                for i in range(pretrain_number):
                    info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                                   discount=self.args.discount_factor, update_table=update_table)
            elif train_mode == 'fix_value':
                update_table['value'] = False
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, update_table=update_table)
            elif train_mode == 'fix_policy':
                update_table['policy'] = False
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, update_table=update_table)
            else:
                info = self.trainer.update_a2c(self.args, batches, rewards, self.trainer.model, self.trainer.critic,
                                               discount=self.args.discount_factor, update_table=update_table)
        return info

    def trainset_valid(self, i, batch_size, real_batch, ):
        data = self.trainer.sample_data(i, batch_size, self.args, real_batch=real_batch, eval=True)
        return data

    def get_eval_dict(self, examples, strategies):
        ret = self.trainer.get_eval_dict(examples, strategies)
        return ret

    def valid(self, start, length):
        infos = self.trainer.validate(self.args, length, start=start)
        return infos

    def save_model(self, i, valid_stats, score_type):
        self.trainer.drop_checkpoint(self.args, i + 1, valid_stats,
                                     model_opt=self.trainer.agents[self.trainer.training_agent].env.model_args,
                                     score_type=score_type)
        # if self.args.update_oppo:
        #     self.trainer.update_opponent(['policy', 'critic'])

    def save_best_model(self, i, valid_stats, score_type, best_only):

        self.trainer.drop_checkpoint(self.args, i + 1, valid_stats,
                                     model_opt=self.trainer.agents[self.trainer.training_agent].env.model_args,
                                     score_type=score_type, best_only=best_only)

    def update_model(self, cmd):
        model_idx, model_p, critic_p = cmd
        env = self.systems[model_idx].env

        env.model.load_state_dict(model_p)
        env.critic.load_state_dict(critic_p)

    def fetch_model(self, cmd):
        model_idx = cmd[0]
        env = self.systems[model_idx].env
        return env.model.state_dict(), env.critic.state_dict()

    def train_tom(self, model_idx, batch_iters, strategy, update_table=None, ret_table=None, dump_file=None):
        env = self.systems[model_idx].env
        # if learn_type == 'id':
        #     update = {'id': True, 'tom': False}
        #     ret = {'id': True, 'tom': False}
        # elif learn_type == 'id_tom':
        #     update = {'id': True, 'tom': True}
        #     ret = {'id': True, 'tom': True}
        # elif learn_type == 'fixed_id_tom':
        #     update = {'id': False, 'tom': True}
        #     ret = {'id': True, 'tom': True}
        # elif learn_type in ['history', 'naive']:
        #     update = {'id': False, 'tom': True}
        #     ret = {'id': False, 'tom': True}
        # else:
        #     raise NameError('unknown learn_type ')

        train_loss = self.trainer.update_tom(self.args, batch_iters, strategy, env.tom_model,
                                             update_table=update_table,
                                             ret_table=ret_table, dump_name=dump_file)
        return train_loss

    def valid_tom(self, model_idx, batch_iters, strategy, update_table=None, ret_table=None, dump_file=None):
        env = self.systems[model_idx].env
        update_table = {'id': False, 'tom': False}
        valid_loss = self.trainer.update_tom(self.args, batch_iters, strategy,
                                             env.tom_model, update_table=update_table,
                                             ret_table=ret_table, dump_name=dump_file)
        return valid_loss

    def split_batch(self, batch_iters, batch_size, device=None):
        ret = self.trainer._sort_merge_batch(batch_iters, batch_size, device=device)
        return ret

    def add_strategy_in_language(self, batch_iters, strategies):
        self.trainer.add_strategy_in_language(batch_iters, strategies)

    def send(self, cmd):
        if cmd[0] == 'quit':
            return
        elif cmd[0] == 'check':
            # self.conn.send(['done'])
            return ['done']
        elif cmd[0] == 'simulate':
            data = self.simulate(cmd[1:])
            # self.conn.send(['done', pkl.dumps(data)])
            return ['done', pkl.dumps(data)]
            # try:
            # except Exception, err:
            #     print(e)
            #     self.conn.send(['error'])
        elif cmd[0] == 'train':
            data = self.train(pkl.loads(cmd[1]))
            # self.conn.send(['done', pkl.dumps(data)])
            return ['done', pkl.dumps(data)]
            # try:
            #     data = self.train(pkl.loads(cmd[1]))
            #     self.conn.send(['done', pkl.dumps(data)])
            # except:
            #     self.conn.send(['error'])
        elif cmd[0] == 'update_model':
            self.update_model((cmd[1],) + pkl.loads(cmd[2]))
            return ['done']
            # self.conn.send(['done'])
            # try:
            #     self.update_model(pkl.loads(cmd[1]))
            #     self.conn.send(['done'])
            # except:
            #     self.conn.send(['error'])

        elif cmd[0] == 'fetch_model':

            data = self.fetch_model(cmd[1:])
            return ['done', pkl.dumps(data)]
            # self.conn.send(['done', pkl.dumps(data)])
            # try:
            #     data = self.fetch_model(cmd[1:])
            #     self.conn.send(['done', pkl.dumps(data)])
            # except:
            #     self.conn.send(['error'])
        elif cmd[0] == 'valid':
            data = self.valid(cmd[1])
            return ['done', pkl.dumps(data)]
            # self.conn.send(['done', pkl.dumps(data)])

        elif cmd[0] == 'save_model':
            self.save_model(pkl.loads(cmd[1]))
            return ['done']
            # self.conn.send(['done'])

        else:
            # Using Universal Formation
            if len(cmd) < 2:
                cmd.append([])
            else:
                cmd[1] = pkl.loads(cmd[1])
            if len(cmd) < 3:
                cmd.append({})
            else:
                cmd[2] = pkl.loads(cmd[2])

            # try:
            ret = getattr(self, cmd[0])(*cmd[1], **cmd[2])
            status = 'done'
            ret_data = ret
            # except Exception as e:
            #     status = 'failed'
            #     print('[failed] ', e)
            #     ret_data = str(e)

            ret_data = pkl.dumps(ret_data)
            return [status, ret_data]

    def local_send(self, cmd):
        if len(cmd) < 2:
            cmd.append([])
        if len(cmd) < 3:
            cmd.append({})

        # try:
        ret = getattr(self, cmd[0])(*cmd[1], **cmd[2])
        status = 'done'
        ret_data = ret
        # except Exception as e:
        #     status = 'failed'
        #     print('[failed] ', e)
        #     ret_data = str(e)
        # ret_data = pkl.dumps(ret_data)
        return [status, ret_data]
    
    def debug_switch_intervention_eval(self, batch_iters, strategy, **kwargs):
        return self.trainer.debug_switch_intervention_eval(
            batch_iters=batch_iters,
            strategy=strategy,
            **kwargs,
        )
    
    def rollout_switch_eval(
        self,
        scenario_idx=0,
        split="dev",
        intervene_agent=0,
        intervene_turn=2,
        max_turns=None,
        temperature=1.0,
        base_seed=1234,
    ):
        """
        在同一个 scenario 上跑三条完整 rollout：
            - normal
            - force_off
            - force_on

        这版专门为了“不改训练脚本”：
            1) 仍然复用当前训练入口和 args
            2) 只在 rollout 分支里，临时把目标 session 热切换到 switch-aware online path
            3) 不影响主实验
        """
        import random
        import numpy as np
        import torch

        trainer = self.trainer
        args = self.args

        if max_turns is None:
            max_turns = args.max_turns

        scenarios = trainer.scenarios[split]
        assert 0 <= scenario_idx < len(scenarios), \
            f"scenario_idx={scenario_idx} out of range for split={split}, size={len(scenarios)}"

        scenario = scenarios[scenario_idx]

        def _set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        def _session_diag_one(sess, tag):
            env = getattr(sess, "env", None)
            return {
                "tag": tag,
                "session_class": type(sess).__name__,
                "role": getattr(getattr(sess, "kb", None), "role", None),
                "tom_flag": bool(getattr(sess, "tom", False)),
                "is_switch_aware": bool(
                    sess._is_switch_aware() if hasattr(sess, "_is_switch_aware") else False
                ),
                "generator_class": type(getattr(sess, "generator", None)).__name__,
                "generator_model_class": type(
                    getattr(getattr(sess, "generator", None), "model", None)
                ).__name__,
                "env_tom_model_class": type(getattr(env, "tom_model", None)).__name__,
                "env_tom_generator_class": type(getattr(env, "tom_generator", None)).__name__,
            }

        def _promote_one_session_to_switchaware(sess):
            """
            不改训练脚本的前提下，在 rollout 分支里临时把 session 切到 switch-aware online path。
            关键点：
            - 只切换 model / generator
            - 不打开旧的 tom_inference 路径（即不要 cand.tom = True）
            - 这样 rollout 仍然能记录 sa_aux / trace，但不会再走 fake_step -> get_value -> critic
            """
            promoted_tags = []

            candidates = [("main_session", sess)]
            tom_sess = getattr(sess, "tom_session", None)
            if tom_sess is not None and not isinstance(tom_sess, bool):
                candidates.append(("tom_session", tom_sess))

            for tag, cand in candidates:
                env = getattr(cand, "env", None)
                tom_model = getattr(env, "tom_model", None)
                tom_generator = getattr(env, "tom_generator", None)

                if tom_model is None or tom_generator is None:
                    continue

                # 只接受真正的 switch-aware tom_model
                if tom_model.__class__.__name__ != "SwitchAwareHistoryModel":
                    continue

                # ------------------------------------------------------------
                # 关键修复：
                # rollout 分支只需要 switch-aware online generation + trace
                # 不要打开旧的 tom_inference / fake_step / critic 评估链
                # ------------------------------------------------------------
                cand.tom = False
                cand.model = tom_model
                cand.generator = tom_generator

                # 可选：如果 env 上本来有 critic，就挂上去；但正常不会再走到它
                if hasattr(env, "critic") and getattr(env, "critic", None) is not None:
                    cand.critic = env.critic

                if hasattr(cand, "tom_hidden"):
                    cand.tom_hidden = None
                if hasattr(cand, "sa_hidden"):
                    cand.sa_hidden = None
                if hasattr(cand, "reset_switch_aware_state"):
                    cand.reset_switch_aware_state(clear_schedule=True)

                promoted_tags.append(tag)

            return promoted_tags

        def _find_switchaware_target(sess):
            candidates = [("main_session", sess)]
            tom_sess = getattr(sess, "tom_session", None)
            if tom_sess is not None and not isinstance(tom_sess, bool):
                candidates.append(("tom_session", tom_sess))

            for tag, cand in candidates:
                if hasattr(cand, "_is_switch_aware") and cand._is_switch_aware():
                    return tag, cand

            return None, None

        def _prepare_controller(seed, fixed_strategies=None):
            _set_seed(seed)

            controller = trainer._get_controller(scenario, split=split, rate=0)

            for s in controller.sessions:
                if hasattr(s, "set_controller"):
                    s.set_controller(controller)
                if hasattr(s, "reset_switch_aware_state"):
                    s.reset_switch_aware_state(clear_schedule=True)

            # 固定三条 rollout 的策略，避免把 strategy 随机性也混进去
            if fixed_strategies is not None:
                for i, s in enumerate(controller.sessions):
                    if hasattr(s, "price_strategy"):
                        s.price_strategy = fixed_strategies[i]

            # 关键：只在 rollout 分支里尝试把目标 session 热切到 switch-aware online path
            target_main = controller.sessions[intervene_agent]
            promoted_tags = _promote_one_session_to_switchaware(target_main)

            return controller, promoted_tags

        # 先构造一个 baseline controller，锁定三条 rollout 共用的策略
        probe_controller, promoted_probe = _prepare_controller(base_seed, fixed_strategies=None)
        fixed_strategies = [probe_controller.sessions[i].price_strategy for i in range(2)]

        per_agent_diag = {}
        for ai, sess in enumerate(probe_controller.sessions):
            diags = [_session_diag_one(sess, f"agent{ai}.main_session")]
            tom_sess = getattr(sess, "tom_session", None)
            if tom_sess is not None and not isinstance(tom_sess, bool):
                diags.append(_session_diag_one(tom_sess, f"agent{ai}.tom_session"))
            per_agent_diag[f"agent_{ai}"] = diags

        target_probe_main = probe_controller.sessions[intervene_agent]
        chosen_tag, chosen_probe = _find_switchaware_target(target_probe_main)

        session_diag = {
            "intervene_agent": int(intervene_agent),
            "fixed_strategies": fixed_strategies,
            "all_session_diagnostics": per_agent_diag,
            "promoted_probe_tags": promoted_probe,
            "chosen_target_tag": chosen_tag,
        }

        if chosen_probe is None:
            raise RuntimeError(
                "No switch-aware session found EVEN AFTER rollout-time promotion. "
                f"diagnostics={session_diag}"
            )

        def _run_one(tag, forced_value):
            controller, promoted_tags = _prepare_controller(base_seed, fixed_strategies=fixed_strategies)

            target_session_main = controller.sessions[intervene_agent]
            chosen_tag, target_session = _find_switchaware_target(target_session_main)

            if target_session is None:
                raise RuntimeError(
                    f"rollout target for agent {intervene_agent} is still not switch-aware "
                    f"after rollout-time promotion"
                )

            if hasattr(target_session, "reset_switch_aware_state"):
                target_session.reset_switch_aware_state(clear_schedule=True)

            if forced_value is not None:
                if not hasattr(target_session, "set_force_switch_from_turn"):
                    raise AttributeError(
                        f"target session has no set_force_switch_from_turn(); chosen_tag={chosen_tag}"
                    )
                target_session.set_force_switch_from_turn(intervene_turn, forced_value)

            example = controller.simulate(
                max_turns,
                verbose=args.verbose,
                temperature=temperature,
            )

            rewards = [trainer.get_reward(example, controller.sessions[i]) for i in range(2)]
            strategies = [controller.sessions[i].price_strategy for i in range(2)]

            if hasattr(target_session, "build_rollout_summary"):
                summary = target_session.build_rollout_summary(example)
            else:
                summary = {
                    "num_events": len(example.events),
                    "forced_turns": {},
                    "trace": [],
                }

            summary["chosen_target_tag"] = chosen_tag
            summary["promoted_tags"] = promoted_tags
            summary["tag"] = tag
            summary["split"] = split
            summary["scenario_idx_in_split"] = int(scenario_idx)
            summary["intervene_agent"] = int(intervene_agent)
            summary["intervene_turn"] = int(intervene_turn)
            summary["temperature"] = float(temperature)
            summary["session_diag"] = dict(session_diag)
            summary["strategies"] = strategies
            summary["rewards"] = rewards
            summary["outcome"] = controller.get_outcome() if hasattr(controller, "get_outcome") else None
            summary["dialogue_text"] = "\n".join(
                trainer.example_to_str(example, controller, rewards, sid=scenario_idx, strategies=strategies)
            )

            return summary

        normal = _run_one("normal", None)
        force_off = _run_one("force_off", 0.0)
        force_on = _run_one("force_on", 1.0)

        def _last_price_from_trace(trace):
            last_price = None
            for step in trace:
                if step.get("price") is not None:
                    last_price = step.get("price")
            return last_price

        def _last_intent_from_trace(trace):
            last_intent = None
            for step in trace:
                if step.get("intent") is not None:
                    last_intent = step.get("intent")
            return last_intent

        def _first_diverge_turn(trace_a, trace_b):
            n = min(len(trace_a), len(trace_b))
            for i in range(n):
                a_intent = trace_a[i].get("intent")
                b_intent = trace_b[i].get("intent")
                a_price = trace_a[i].get("price")
                b_price = trace_b[i].get("price")
                if a_intent != b_intent or a_price != b_price:
                    return i
            if len(trace_a) != len(trace_b):
                return n
            return None

        def _buyer_path_summary(trace):
            rows = []
            for step in trace:
                rows.append({
                    "turn_idx": step.get("turn_idx"),
                    "intent": step.get("intent"),
                    "price": step.get("price"),
                    "switch_prob_used": step.get("switch_prob_used"),
                    "belief_confidence": step.get("belief_confidence"),
                    "price_top1_top2_gap": step.get("price_top1_top2_gap"),
                })
            return rows

        comparison = {
            "normal_is_agreed": normal.get("is_agreed"),
            "force_off_is_agreed": force_off.get("is_agreed"),
            "force_on_is_agreed": force_on.get("is_agreed"),

            "normal_final_price": normal.get("final_price"),
            "force_off_final_price": force_off.get("final_price"),
            "force_on_final_price": force_on.get("final_price"),

            "normal_num_turns": normal.get("num_turns"),
            "force_off_num_turns": force_off.get("num_turns"),
            "force_on_num_turns": force_on.get("num_turns"),

            "normal_buyer_reward": normal.get("buyer_reward"),
            "force_off_buyer_reward": force_off.get("buyer_reward"),
            "force_on_buyer_reward": force_on.get("buyer_reward"),

            "normal_seller_reward": normal.get("seller_reward"),
            "force_off_seller_reward": force_off.get("seller_reward"),
            "force_on_seller_reward": force_on.get("seller_reward"),

            "normal_avg_switch_prob_used": normal.get("avg_switch_prob_used"),
            "force_off_avg_switch_prob_used": force_off.get("avg_switch_prob_used"),
            "force_on_avg_switch_prob_used": force_on.get("avg_switch_prob_used"),

            # 新增：从 trace 提取的 buyer 最后动作
            "normal_buyer_last_price": _last_price_from_trace(normal.get("trace", [])),
            "force_off_buyer_last_price": _last_price_from_trace(force_off.get("trace", [])),
            "force_on_buyer_last_price": _last_price_from_trace(force_on.get("trace", [])),

            "normal_buyer_last_intent": _last_intent_from_trace(normal.get("trace", [])),
            "force_off_buyer_last_intent": _last_intent_from_trace(force_off.get("trace", [])),
            "force_on_buyer_last_intent": _last_intent_from_trace(force_on.get("trace", [])),

            # 新增：off / on 从哪一轮开始真正分叉
            "force_off_vs_on_first_diverge_turn": _first_diverge_turn(
                force_off.get("trace", []),
                force_on.get("trace", []),
            ),

            # 新增：最后一轮 buyer 是否真的不同
            "force_off_vs_on_last_price_diff": (
                _last_price_from_trace(force_off.get("trace", []))
                != _last_price_from_trace(force_on.get("trace", []))
            ),
            "force_off_vs_on_last_intent_diff": (
                _last_intent_from_trace(force_off.get("trace", []))
                != _last_intent_from_trace(force_on.get("trace", []))
            ),
        }

        return {
            "scenario_idx": int(scenario_idx),
            "split": split,
            "intervene_agent": int(intervene_agent),
            "intervene_turn": int(intervene_turn),
            "temperature": float(temperature),
            "session_diag": session_diag,
            "normal": normal,
            "force_off": force_off,
            "force_on": force_on,
            "comparison": comparison,
        }


class MultiManager():
    def __init__(self, num_cpu, args, worker_class):
        self.local_workers = []
        self.worker_addr = []
        self.trainer_addr = []
        self.args = args

        for i in range(num_cpu):
            addr = ('localhost', 7000 + i)
            worker = worker_class(args, addr)
            self.worker_addr.append(worker)
            # self.local_workers.append(multiprocessing.Process(target=execute_runner, args=(worker_class, args, addr)))
            self.local_workers.append(worker)
        # self.trainer = multiprocessing.Process(target=execute_runner, args=(trainer_class, args))
        self.trainer = self.local_workers[0]

        self.worker_listener = []
        for i, addr in enumerate(self.worker_addr):
            # self.worker_listener.append(multiprocessing.connection.Listener(addr))
            self.worker_listener.append(addr)
        self.worker_conn = []

        cache_path = 'cache/{}'.format(args.name)
        log_path = 'logs/' + args.name
        init_dir(cache_path)
        init_dir(log_path, clean_all=True)

        self.writer = SummaryWriter(logdir='logs/{}'.format(args.name))
        self.policies_log = [{}, {}]

    def run_local_workers(self):
        for w in self.local_workers:
            w.start()

    def update_worker_list(self):
        self.worker_conn = []
        for l in self.worker_listener:
            # self.worker_conn.append(l.accept())
            self.worker_conn.append(l)
        return len(self.worker_conn)

    @staticmethod
    def allocate_tasks(num_worker, batch_size):
        ret = []
        while num_worker > 0:
            ret.append(batch_size // num_worker)
            batch_size -= ret[-1]
            num_worker -= 1

        print('allocate: {} workers, {} tasks, final list:{}'.format(num_worker, batch_size, ret))
        return ret

    def _draw_tensorboard(self, ii, losses, all_rewards):
        # print(all_rewards)
        for j in range(2):
            self.writer.add_scalar('agent{}/reward'.format(j), np.mean(all_rewards[j]), ii)
            if len(losses[j]) > 0:
                for k in losses[j][0]:
                    tmp = []
                    for l in losses[j]:
                        tmp.append(l[k])
                    tmp = np.concatenate(tmp[j])
                    tmp = np.mean(tmp)
                    self.writer.add_scalar('agent{}/{}'.format(j, k), tmp, ii)
        self.writer.flush()

    def _draw_tensorboard_valid(self, ii, all_rewards):
        for j in range(2):
            self.writer.add_scalar('agent{}/dev_reward'.format(j), all_rewards[j], ii)

    def dump_examples(self, examples, verbose_strs, epoch, mode='train', other_path=None):
        # Dump with details
        args = self.args
        if other_path is None:
            path = args.model_path
        else:
            path = other_path
        path_txt = '{root}/{model}_{mode}_example{epoch}.txt'.format(
            root=path,
            model=args.name,
            mode=mode,
            epoch=epoch)
        path_pkl = '{root}/{model}_{mode}_example{epoch}.pkl'.format(
            root=path,
            model=args.name,
            mode=mode,
            epoch=epoch)

        print('Save examples at {} and {}.'.format(path_txt, path_pkl))
        with open(path_txt, 'w') as f:
            for ex in verbose_strs:
                f.write('-' * 7 + '\n')
                for s in ex:
                    f.write(s + '\n')
        with open(path_pkl, 'wb') as f:
            pkl.dump(examples, f)

    def evaluate(self):
        num_worker = self.update_worker_list()
        worker = self.worker_conn[0]
        args = self.args
        sample_size = args.batch_size
        max_epoch = args.epochs
        last_time = time.time()
        if args.debug:
            sample_size = 2

        eval_dict = {}
        separate_edict = [{} for _ in range(10)]

        # add dd to d
        def update_edict(d, dd):
            for k in dd:
                if d.get(k) is None:
                    d[k] = []
                d[k] = d[k] + dd[k]

        def get_result_dict(d):
            ret = {}
            if d.get('reward') is None:
                num = 0
            else:
                num = len(d.get('reward'))
            for k in d:
                ret[k] = np.mean(d[k])
                ret[k+'_std'] = np.std(d[k])
            return ret, num

        for epoch in range(max_epoch):
            last_time = time.time()

            info = worker.local_send(['trainset_valid', (epoch, sample_size, sample_size)])
            _batch_iters, batch_info, example, v_str = info[1]
            _rewards, strategies = batch_info

            data_pkl = 'cache/{}/data_{}.pkl'.format(args.name, epoch)
            with open(data_pkl, 'wb') as f:
                pkl.dump(info[1], f)

            self.dump_examples(example, v_str, epoch, other_path='logs/'+args.name)

            info = worker.local_send(['get_eval_dict', (example, strategies[1])])
            ed, sed = info[1]

            # log eval table as json file
            eval_json = 'logs/{}/eval_{}.json'.format(args.name, epoch)
            update_edict(eval_dict, ed)
            tmpd, _ = get_result_dict(eval_dict)
            tmpd['number'] = (epoch+1) * sample_size
            with open(eval_json, 'w') as f:
                json.dump(tmpd, f)

            print("tmpd keys:", list(tmpd.keys()))
            print("tmpd:", tmpd)
            print('=' * 5 + ' [reward: {}\t utility: {}\t success_rate: {}]'.
                  format(tmpd.get('reward', 'NA'), tmpd.get('utility', 'NA'), tmpd.get("success_rate", 'NA')))

            eval_json = 'logs/{}/eval_separate_{}.json'.format(args.name, epoch)
            tmpds = []
            for i in range(len(sed)):
                update_edict(separate_edict[i], sed[i])
                tmpd, num = get_result_dict(separate_edict[i])
                tmpd['number'] = num
                tmpd['strategy'] = i
                tmpds.append(tmpd)
            with open(eval_json, 'w') as f:
                json.dump(tmpds, f)

            print('=' * 5 + ' [Epoch {}/{}, {} dialogues for {:.3f}s.]'.
                  format(epoch + 1, max_epoch, (epoch+1)*sample_size, time.time() - last_time))


    def learn_identity(self):

        args = self.args
        save_every = 100

        batch_size = 100
        split_results = False
        if args.only_run:
            batch_size = 1

        # if args.tom_model == 'id':
        #     learn_type = 'identity'
        # elif args.tom_model in ['history', 'naive']:
        #     learn_type = 'tom'
        # else:# == 'idtom'
        #     learn_type = 'co-train'

        if args.tom_model in ['id', 'uttr_id']:
            update_table = {'id': True, 'tom': False}
            ret_table = {'id': True, 'tom': False}
        elif args.tom_model in ['uttr_fid_history_tom']:
            update_table = {'id': False, 'tom': True}
            ret_table = {'id': True, 'tom': True}
        elif args.tom_model in ['uttr_id_history_tom', 'id_tom', 'id_history_tom']:
            update_table = {'id': True, 'tom': True}
            ret_table = {'id': True, 'tom': True}
        elif args.tom_model in ['fixed_id_tom', 'fixed_id_history_tom']:
            update_table = {'id': False, 'tom': True}
            ret_table = {'id': True, 'tom': True}
        elif args.tom_model in ['history', 'naive', 'switch_aware']:
            update_table = {'id': False, 'tom': True}
            ret_table = {'id': False, 'tom': True}
        else:
            raise NameError('unknown learn_type ')

        if args.fix_id:
            update_table['id'] = False

        if args.only_run:
            update_table = {'id': False, 'tom': False}

        num_worker = self.update_worker_list()
        worker = self.worker_conn[0]
        train_agent = 0
        load_data = args.load_sample

        # Generate data samples or load from files
        data_pkl = 'cache/{}/data.pkl'.format(args.name)
        if load_data is None:
            print('[Info] Start sampling.')
            info = worker.send(['simulate', train_agent, args.num_dialogues, args.num_dialogues])
            with open(data_pkl, 'wb') as f:
                pkl.dump(pkl.loads(info[1]), f)
            _batch_iters, batch_info, example, v_str = pkl.loads(info[1])
        else:
            print('[Info] Load sample from {}'.format(load_data))
            info = ['done', None]
            with open(load_data, 'rb') as f:
                info[1] = pkl.load(f)
            _batch_iters, batch_info, example, v_str = info[1]

        _rewards, strategies = batch_info

        # Single Thread!
        if args.strategy_in_words:
            worker.local_send(
                ['add_strategy_in_language', (_batch_iters, strategies)]
            )
        self.dump_examples(example, v_str, 0)

        # Divide the training set
        train_size = round(len(_batch_iters[1-train_agent]) * 0.6)
        train_batch = _batch_iters[1-train_agent][:train_size]
        train_strategy = strategies[1-train_agent][:train_size]
        dev_batch = _batch_iters[1-train_agent][train_size:]
        dev_strategy = strategies[1-train_agent][train_size:]

        train_verbose_strs = v_str[:train_size]
        dev_verbose_strs = v_str[train_size:]

        # if not, only learn identifier
        if args.tom_model != 'id' and split_results:
            dev_batches = [[], []]
            dev_strategies = [[], []]
            for i, s in enumerate(dev_strategy):
                dev_batches[s].append(dev_batch[i])
                dev_strategies[s].append(s)
            dev_batch = dev_batches
            dev_strategy = dev_strategies
            dev_writer = [SummaryWriter(logdir='logs/{}/strategy_{}'.format(args.name, i)) for i in range(2)]

        print('[Info] Start training model.')
        step_range = 10
        step_writer = [SummaryWriter(logdir='logs/{}/step_{}'.format(args.name, i)) for i in range(step_range)]

        # split training batch
        _, train_batch_splited = worker.local_send(
            ['split_batch', (train_batch, 1024)])
        if args.tom_model != 'id' and split_results:
            dev_batch_splited = [None, None]
            _, dev_batch_splited[0] = worker.local_send(
                ['split_batch', (dev_batch[0], 1024)]
            )
            _, dev_batch_splited[1] = worker.local_send(
                ['split_batch', (dev_batch[1], 1024)]
            )
        else:
            _, dev_batch_splited = worker.local_send(
                ['split_batch', (dev_batch, 1024)]
            )

        if os.environ.get("DEBUG_TOM_ROLLOUT", "0") == "1":
            rollout_split = os.environ.get("ROLLOUT_SPLIT", "dev")
            rollout_scenario_idx = int(os.environ.get("ROLLOUT_SCENARIO_IDX", "0"))
            rollout_intervene_agent = int(os.environ.get("ROLLOUT_INTERVENE_AGENT", "0"))
            rollout_intervene_turn = int(os.environ.get("ROLLOUT_INTERVENE_TURN", "2"))
            rollout_temperature = float(os.environ.get("ROLLOUT_TEMPERATURE", "1.0"))
            rollout_max_turns_env = os.environ.get("ROLLOUT_MAX_TURNS", None)
            rollout_max_turns = None if rollout_max_turns_env is None else int(rollout_max_turns_env)

            info = worker.local_send([
                'rollout_switch_eval',
                (),
                {
                    'scenario_idx': rollout_scenario_idx,
                    'split': rollout_split,
                    'intervene_agent': rollout_intervene_agent,
                    'intervene_turn': rollout_intervene_turn,
                    'max_turns': rollout_max_turns,
                    'temperature': rollout_temperature,
                }
            ])

            result = info[1]

            def _buyer_path_summary_local(trace):
                rows = []
                for step in trace:
                    rows.append({
                        "turn_idx": step.get("turn_idx"),
                        "intent": step.get("intent"),
                        "price": step.get("price"),
                        "switch_prob_used": step.get("switch_prob_used"),
                        "belief_confidence": step.get("belief_confidence"),
                        "price_top1_top2_gap": step.get("price_top1_top2_gap"),
                    })
                return rows

            normal = result["normal"]
            force_off = result["force_off"]
            force_on = result["force_on"]

            print("\n================ SESSION-LEVEL ROLLOUT EVAL ================\n")

            print("[session_diag]")
            print(json.dumps(result.get("session_diag", {}), indent=2, ensure_ascii=False))

            print("\n[comparison]")
            print(json.dumps(result["comparison"], indent=2, ensure_ascii=False))

            print("\n[buyer_path.normal]")
            print(json.dumps(_buyer_path_summary_local(normal.get("trace", [])), indent=2, ensure_ascii=False))

            print("\n[buyer_path.force_off]")
            print(json.dumps(_buyer_path_summary_local(force_off.get("trace", [])), indent=2, ensure_ascii=False))

            print("\n[buyer_path.force_on]")
            print(json.dumps(_buyer_path_summary_local(force_on.get("trace", [])), indent=2, ensure_ascii=False))

            print("\n---------------- NORMAL ----------------")
            print(result["normal"]["dialogue_text"])

            print("\n---------------- FORCE OFF ----------------")
            print(result["force_off"]["dialogue_text"])

            print("\n---------------- FORCE ON ----------------")
            print(result["force_on"]["dialogue_text"])

            print("\n---------------- NORMAL TRACE ----------------")
            print(json.dumps(result["normal"].get("trace", []), indent=2, ensure_ascii=False))

            print("\n---------------- FORCE OFF TRACE ----------------")
            print(json.dumps(result["force_off"].get("trace", []), indent=2, ensure_ascii=False))

            print("\n---------------- FORCE ON TRACE ----------------")
            print(json.dumps(result["force_on"].get("trace", []), indent=2, ensure_ascii=False))

            def _to_jsonable(x):
                import torch
                import numpy as np

                if torch.is_tensor(x):
                    return x.detach().cpu().tolist()
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, (np.integer, np.floating)):
                    return x.item()
                if isinstance(x, dict):
                    return {k: _to_jsonable(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    return [_to_jsonable(v) for v in x]
                return x

            print("\n---------------- FULL RESULT JSON ----------------")
            print(json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False))

            print("\n============================================================\n")
            raise SystemExit(0)

        if os.environ.get("DEBUG_TOM_SWITCH_EVAL", "0") == "1":
            info = worker.local_send([
                'debug_switch_intervention_eval',
                (dev_batch_splited, dev_strategy),
                {
                    'split_name': 'dev',
                    'max_merged_batches': None,
                    'max_steps_per_batch': None,
                    'decode_examples': True,
                    'verbose_strs': dev_verbose_strs,
                }
            ])
            print(json.dumps(info[1], indent=2, ensure_ascii=False))
            raise SystemExit(0)

        def draw_dev_info(loss, accu, step_info, name, w, i):
            if ret_table['id']:
                w.add_scalar('identity{}/{}_loss'.format(train_agent, name), loss[0], i)
                w.add_scalar('identity{}/{}_accuracy'.format(train_agent, name), accu[0], i)
                w.add_scalar('identity{}/{}_accuracy2'.format(train_agent, name), accu[2], i)

            if ret_table['tom']:
                price_w = getattr(args, "sa_lambda_price", 1.0)
                switch_w = getattr(args, "sa_lambda_switch", 0.5)

                w.add_scalar('tom{}/{}_intent_loss'.format(train_agent, name), loss[1], i)
                w.add_scalar('tom{}/{}_intent_accuracy'.format(train_agent, name), accu[1], i)
                w.add_scalar('tom{}/{}_price_loss'.format(train_agent, name), loss[2], i)

                if len(loss) > 3 and loss[3] is not None:
                    w.add_scalar('tom{}/{}_switch_loss'.format(train_agent, name), loss[3], i)
                    total = loss[1] + price_w * loss[2] + switch_w * loss[3]
                else:
                    total = loss[1] + price_w * loss[2]

                w.add_scalar('tom{}/{}_total_loss'.format(train_agent, name), total, i)

            w.flush()

        # Draw outputs on the tensorboard
                # Draw outputs on the tensorboard
        def draw_info(loss, accu, step_info, name, i):
            draw_dev_info(loss, accu, None, name, self.writer, i)

            for j, w in enumerate(step_writer):
                if j >= len(step_info[2]):
                    break
                if math.isnan(step_info[2][j]) or step_info[2][j] == 0:
                    continue
                if ret_table['id']:
                    w.add_scalar('identity{}/{}_loss'.format(train_agent, name), step_info[0][0][j], i)
                    w.add_scalar('identity{}/{}_accuracy'.format(train_agent, name), step_info[1][0][j], i)
                    w.add_scalar('identity{}/{}_accuracy2'.format(train_agent, name), step_info[1][2][j], i)

                if ret_table['tom']:
                    w.add_scalar('tom{}/{}_intent_loss'.format(train_agent, name), step_info[0][1][j], i)
                    w.add_scalar('tom{}/{}_intent_accuracy'.format(train_agent, name), step_info[1][1][j], i)
                    w.add_scalar('tom{}/{}_price_loss'.format(train_agent, name), step_info[0][2][j], i)

                if ret_table['tom'] and len(step_info[0]) > 3:
                    w.add_scalar('tom{}/{}_switch_loss'.format(train_agent, name), step_info[0][3][j], i)
                w.flush()

        # ===== periodic reward eval helpers =====
        def _get_result_dict(d):
            ret = {}
            if d.get('reward') is None:
                num = 0
            else:
                num = len(d.get('reward'))
            for k in d:
                if len(d[k]) > 0:
                    ret[k] = float(np.mean(d[k]))
                    ret[k + '_std'] = float(np.std(d[k]))
                else:
                    ret[k] = None
                    ret[k + '_std'] = None
            return ret, num

        # def _run_periodic_reward_eval(epoch_idx, eval_dialogues):
        #     """
        #     复用 evaluate() 的评测口径：
        #       trainset_valid -> get_eval_dict -> 聚合 reward/utility/success_rate
        #     """
        #     info = worker.local_send(['trainset_valid', (epoch_idx, eval_dialogues, eval_dialogues)])
        #     _batch_iters_eval, batch_info_eval, example_eval, v_str_eval = info[1]
        #     _rewards_eval, strategies_eval = batch_info_eval

        #     info = worker.local_send(['get_eval_dict', (example_eval, strategies_eval[1])])
        #     ed, sed = info[1]

        #     tmpd, num = _get_result_dict(ed)
        #     tmpd['number'] = num

        #     # 持久化到日志，便于后面和 2000 epoch checkpoint 对比
        #     eval_json = 'logs/{}/periodic_eval_e{}.json'.format(args.name, epoch_idx + 1)
        #     with open(eval_json, 'w') as f:
        #         json.dump(tmpd, f)

        #     reward_str = '{:.6f}'.format(tmpd['reward']) if tmpd.get('reward') is not None else 'NA'
        #     utility_str = '{:.6f}'.format(tmpd['utility']) if tmpd.get('utility') is not None else 'NA'
        #     success_str = '{:.6f}'.format(tmpd['success_rate']) if tmpd.get('success_rate') is not None else 'NA'

        #     print(
        #         '\t[periodic-eval@e{}] reward {} utility {} success_rate {} n={}'.format(
        #             epoch_idx + 1,
        #             reward_str,
        #             utility_str,
        #             success_str,
        #             num,
        #         )
        #     )

        #     # 可选：同时打到 tensorboard
        #     if tmpd.get('reward') is not None:
        #         self.writer.add_scalar('periodic_eval/reward', tmpd['reward'], epoch_idx)
        #     if tmpd.get('utility') is not None:
        #         self.writer.add_scalar('periodic_eval/utility', tmpd['utility'], epoch_idx)
        #     if tmpd.get('success_rate') is not None:
        #         self.writer.add_scalar('periodic_eval/success_rate', tmpd['success_rate'], epoch_idx)
        #     self.writer.flush()

        #     return tmpd

        # def _run_periodic_reward_eval(epoch_idx, eval_dialogues):
        #     """
        #     复用 evaluate() 的评测口径：
        #     trainset_valid -> get_eval_dict -> 聚合 reward/utility/success_rate
        #     但不打印每条具体 dialogue。
        #     """
        #     old_verbose = worker.args.verbose
        #     worker.args.verbose = False
        #     try:
        #         info = worker.local_send(['trainset_valid', (epoch_idx, eval_dialogues, eval_dialogues)])
        #     finally:
        #         worker.args.verbose = old_verbose

        #     _batch_iters_eval, batch_info_eval, example_eval, v_str_eval = info[1]
        #     _rewards_eval, strategies_eval = batch_info_eval

        #     info = worker.local_send(['get_eval_dict', (example_eval, strategies_eval[1])])
        #     ed, sed = info[1]

        #     tmpd, num = _get_result_dict(ed)
        #     tmpd['number'] = num

        #     eval_json = 'logs/{}/periodic_eval_e{}.json'.format(args.name, epoch_idx + 1)
        #     with open(eval_json, 'w') as f:
        #         json.dump(tmpd, f)

        #     reward_str = '{:.6f}'.format(tmpd['reward']) if tmpd.get('reward') is not None else 'NA'
        #     utility_str = '{:.6f}'.format(tmpd['utility']) if tmpd.get('utility') is not None else 'NA'
        #     success_str = '{:.6f}'.format(tmpd['success_rate']) if tmpd.get('success_rate') is not None else 'NA'

        #     print(
        #         '\t[periodic-eval@e{}] reward {} utility {} success_rate {} n={}'.format(
        #             epoch_idx + 1,
        #             reward_str,
        #             utility_str,
        #             success_str,
        #             num,
        #         )
        #     )

        #     if tmpd.get('reward') is not None:
        #         self.writer.add_scalar('periodic_eval/reward', tmpd['reward'], epoch_idx)
        #     if tmpd.get('utility') is not None:
        #         self.writer.add_scalar('periodic_eval/utility', tmpd['utility'], epoch_idx)
        #     if tmpd.get('success_rate') is not None:
        #         self.writer.add_scalar('periodic_eval/success_rate', tmpd['success_rate'], epoch_idx)
        #     self.writer.flush()

        #     return tmpd

        def _run_periodic_reward_eval(epoch_idx, eval_dialogues):
            """
            复用 evaluate() 的评测口径：
            trainset_valid -> get_eval_dict -> 聚合 reward/utility/success_rate
            """
            old_verbose = worker.args.verbose
            worker.args.verbose = False
            try:
                info = worker.local_send(['trainset_valid', (epoch_idx, eval_dialogues, eval_dialogues)])
            finally:
                worker.args.verbose = old_verbose

            _batch_iters_eval, batch_info_eval, example_eval, v_str_eval = info[1]
            _rewards_eval, strategies_eval = batch_info_eval

            info = worker.local_send(['get_eval_dict', (example_eval, strategies_eval[1])])
            ed, sed = info[1]

            tmpd, num = _get_result_dict(ed)
            tmpd['number'] = num

            eval_json = 'logs/{}/periodic_eval_e{}.json'.format(args.name, epoch_idx + 1)
            with open(eval_json, 'w') as f:
                json.dump(tmpd, f)

            reward_str = '{:.6f}'.format(tmpd['reward']) if tmpd.get('reward') is not None else 'NA'
            utility_str = '{:.6f}'.format(tmpd['utility']) if tmpd.get('utility') is not None else 'NA'
            success_str = '{:.6f}'.format(tmpd['success_rate']) if tmpd.get('success_rate') is not None else 'NA'

            print(
                '\t[periodic-eval@e{}] reward {} utility {} success_rate {} n={}'.format(
                    epoch_idx + 1,
                    reward_str,
                    utility_str,
                    success_str,
                    num,
                )
            )

            if tmpd.get('reward') is not None:
                self.writer.add_scalar('periodic_eval/reward', tmpd['reward'], epoch_idx)
            if tmpd.get('utility') is not None:
                self.writer.add_scalar('periodic_eval/utility', tmpd['utility'], epoch_idx)
            if tmpd.get('success_rate') is not None:
                self.writer.add_scalar('periodic_eval/success_rate', tmpd['success_rate'], epoch_idx)
            self.writer.flush()

            return tmpd
        
        def _run_final_reward_eval(tag, eval_dialogues):
            """
            训练完成后做一次更大的最终评测。
            注意：这不是把 2000 个 epoch 的 periodic 点平均，
            而是“第 2000 epoch checkpoint 的一次大样本评测”。
            """
            old_verbose = worker.args.verbose
            worker.args.verbose = False
            try:
                info = worker.local_send(['trainset_valid', (args.epochs - 1, eval_dialogues, eval_dialogues)])
            finally:
                worker.args.verbose = old_verbose

            _batch_iters_eval, batch_info_eval, example_eval, v_str_eval = info[1]
            _rewards_eval, strategies_eval = batch_info_eval

            info = worker.local_send(['get_eval_dict', (example_eval, strategies_eval[1])])
            ed, sed = info[1]

            tmpd, num = _get_result_dict(ed)
            tmpd['number'] = num
            tmpd['tag'] = tag

            final_json = 'logs/{}/final_eval_{}.json'.format(args.name, tag)
            with open(final_json, 'w') as f:
                json.dump(tmpd, f)

            reward_str = '{:.6f}'.format(tmpd['reward']) if tmpd.get('reward') is not None else 'NA'
            utility_str = '{:.6f}'.format(tmpd['utility']) if tmpd.get('utility') is not None else 'NA'
            success_str = '{:.6f}'.format(tmpd['success_rate']) if tmpd.get('success_rate') is not None else 'NA'

            print(
                '\t[final-eval:{}] reward {} utility {} success_rate {} n={}'.format(
                    tag,
                    reward_str,
                    utility_str,
                    success_str,
                    num,
                )
            )

            if tmpd.get('reward') is not None:
                self.writer.add_scalar('final_eval/reward', tmpd['reward'], args.epochs)
            if tmpd.get('utility') is not None:
                self.writer.add_scalar('final_eval/utility', tmpd['utility'], args.epochs)
            if tmpd.get('success_rate') is not None:
                self.writer.add_scalar('final_eval/success_rate', tmpd['success_rate'], args.epochs)
            self.writer.flush()

            return tmpd

        periodic_eval_every = int(os.environ.get("PERIODIC_REWARD_EVAL_EVERY", "50"))
        periodic_eval_dialogues = int(os.environ.get("PERIODIC_REWARD_EVAL_DIALOGUES", "32"))
        if args.debug:
            periodic_eval_dialogues = min(periodic_eval_dialogues, 8)

        final_eval_dialogues = int(os.environ.get("FINAL_REWARD_EVAL_DIALOGUES", "500"))
        run_final_eval = os.environ.get("RUN_FINAL_REWARD_EVAL", "1") == "1"
        if args.debug:
            final_eval_dialogues = min(final_eval_dialogues, 32)


        # train model
        cur_t = time.time()
        for i in range(args.epochs):
            info = worker.local_send(
                ['train_tom', (train_agent, train_batch_splited,
                            train_strategy, update_table, ret_table,
                            'cache/{}/train_pred_{}.pkl'.format(args.name, i))])
            train_loss, train_accu, train_step_info = info[1]

            # 关键：在 train_tom 之后立刻保存 train epoch stats
            train_sw = dict(getattr(worker.trainer, "last_tom_epoch_stats", {}))

            if args.only_run:
                save_dir = 'logs/{}/hidden_vec_{}.pkl'.format(args.name, i)
                total_num = 0
                for j in range(len(worker.trainer.hidden_vec)):
                    assert worker.trainer.hidden_vec[j].shape[0] == worker.trainer.hidden_stra[j].shape[0], \
                        "miss match at {}, {} of {}".format(
                            worker.trainer.hidden_vec[j].shape,
                            worker.trainer.hidden_stra[j].shape,
                            j
                        )
                    total_num = total_num + len(worker.trainer.hidden_stra[j])

                with open(save_dir, "wb") as f:
                    pkl.dump([worker.trainer.hidden_vec, worker.trainer.hidden_stra], f)

                print("accu:", train_accu)
                print('[run{}/{}]\t num:{} \t time:{:.2f}s.'.format(
                    i + 1, args.epochs, total_num, time.time() - cur_t
                ))
                continue

            draw_info(train_loss, train_accu, train_step_info, 'train', i)

            if args.tom_model != 'id' and split_results:
                dev_loss = [0] * 4
                dev_accu = [0] * 2
                dev_sw = {
                    "switch_acc_weighted_sum": 0.0,
                    "switch_prob_weighted_sum": 0.0,
                    "switch_logit_weighted_sum": 0.0,

                    "switch_count": 0.0,
                    "switch_positive_count": 0.0,
                    "switch_pred_positive_count": 0.0,
                    "switch_tp": 0.0,
                    "switch_fp": 0.0,
                    "switch_tn": 0.0,
                    "switch_fn": 0.0,
                }

                for th in ["0p5", "0p7", "0p8", "0p9", "0p95"]:
                    dev_sw[f"switch_scan_{th}_pred_pos"] = 0.0
                    dev_sw[f"switch_scan_{th}_tp"] = 0.0
                    dev_sw[f"switch_scan_{th}_fp"] = 0.0
                    dev_sw[f"switch_scan_{th}_tn"] = 0.0
                    dev_sw[f"switch_scan_{th}_fn"] = 0.0

                for j in range(2):
                    ratio = len(dev_strategy[j]) / (len(dev_strategy[0]) + len(dev_strategy[1]))
                    info = worker.local_send(
                        ['valid_tom', (train_agent, dev_batch_splited[j],
                                    dev_strategy[j], update_table, ret_table,
                                    'cache/{}/dev{}_pred_{}.pkl'.format(args.name, j, i))])
                    tmp_loss, tmp_accu, dev_step_info = info[1]

                    # 关键：在 valid_tom 之后更新 dev epoch stats
                    tmp_sw = dict(getattr(worker.trainer, "last_tom_epoch_stats", {}))

                    sc = max(tmp_sw.get("switch_count", 0.0), 1.0)

                    dev_sw["switch_acc_weighted_sum"] += tmp_sw.get("switch_acc_mean", 0.0) * sc
                    dev_sw["switch_prob_weighted_sum"] += tmp_sw.get("switch_prob_mean", 0.0) * sc
                    dev_sw["switch_logit_weighted_sum"] += tmp_sw.get("switch_logit_mean", 0.0) * sc

                    dev_sw["switch_count"] += tmp_sw.get("switch_count", 0.0)
                    dev_sw["switch_positive_count"] += tmp_sw.get("switch_positive_count", 0.0)
                    dev_sw["switch_pred_positive_count"] += tmp_sw.get("switch_pred_positive_count", 0.0)
                    dev_sw["switch_tp"] += tmp_sw.get("switch_tp", 0.0)
                    dev_sw["switch_fp"] += tmp_sw.get("switch_fp", 0.0)
                    dev_sw["switch_tn"] += tmp_sw.get("switch_tn", 0.0)
                    dev_sw["switch_fn"] += tmp_sw.get("switch_fn", 0.0)

                    for th in ["0p5", "0p7", "0p8", "0p9", "0p95"]:
                        dev_sw[f"switch_scan_{th}_pred_pos"] += tmp_sw.get(f"switch_scan_{th}_pred_pos", 0.0)
                        dev_sw[f"switch_scan_{th}_tp"] += tmp_sw.get(f"switch_scan_{th}_tp", 0.0)
                        dev_sw[f"switch_scan_{th}_fp"] += tmp_sw.get(f"switch_scan_{th}_fp", 0.0)
                        dev_sw[f"switch_scan_{th}_tn"] += tmp_sw.get(f"switch_scan_{th}_tn", 0.0)
                        dev_sw[f"switch_scan_{th}_fn"] += tmp_sw.get(f"switch_scan_{th}_fn", 0.0)

                    for x in range(min(4, len(tmp_loss))):
                        if isinstance(tmp_loss[x], float):
                            dev_loss[x] += ratio * tmp_loss[x]
                        else:
                            if tmp_loss[x] not in ([], None):
                                print(tmp_loss[x])
                            dev_loss[x] = None

                    for x in range(2):
                        if isinstance(tmp_accu[x], float):
                            dev_accu[x] += ratio * tmp_accu[x]
                        else:
                            if tmp_accu[x] not in ([], None):
                                print(tmp_accu[x])
                            dev_accu[x] = None

                    draw_dev_info(tmp_loss, tmp_accu, dev_step_info, 'dev', dev_writer[j], i)

                def _safe_div(a, b):
                    return a / max(b, 1.0)

                dev_sw["switch_acc_mean"] = _safe_div(dev_sw["switch_acc_weighted_sum"], dev_sw["switch_count"])
                dev_sw["switch_prob_mean"] = _safe_div(dev_sw["switch_prob_weighted_sum"], dev_sw["switch_count"])
                dev_sw["switch_logit_mean"] = _safe_div(dev_sw["switch_logit_weighted_sum"], dev_sw["switch_count"])
                dev_sw["switch_positive_ratio"] = _safe_div(dev_sw["switch_positive_count"], dev_sw["switch_count"])
                dev_sw["switch_pred_positive_ratio"] = _safe_div(dev_sw["switch_pred_positive_count"], dev_sw["switch_count"])
                dev_sw["switch_precision"] = _safe_div(dev_sw["switch_tp"], dev_sw["switch_tp"] + dev_sw["switch_fp"])
                dev_sw["switch_recall"] = _safe_div(dev_sw["switch_tp"], dev_sw["switch_tp"] + dev_sw["switch_fn"])
                dev_sw["switch_tnr"] = _safe_div(dev_sw["switch_tn"], dev_sw["switch_tn"] + dev_sw["switch_fp"])

                for th in ["0p5", "0p7", "0p8", "0p9", "0p95"]:
                    pred_pos = dev_sw[f"switch_scan_{th}_pred_pos"]
                    tp = dev_sw[f"switch_scan_{th}_tp"]
                    fp = dev_sw[f"switch_scan_{th}_fp"]
                    tn = dev_sw[f"switch_scan_{th}_tn"]
                    fn = dev_sw[f"switch_scan_{th}_fn"]

                    dev_sw[f"switch_scan_{th}_pred_pos_ratio"] = _safe_div(pred_pos, dev_sw["switch_count"])
                    dev_sw[f"switch_scan_{th}_precision"] = _safe_div(tp, tp + fp)
                    dev_sw[f"switch_scan_{th}_recall"] = _safe_div(tp, tp + fn)
                    dev_sw[f"switch_scan_{th}_tnr"] = _safe_div(tn, tn + fp)

                draw_dev_info(dev_loss, dev_accu, None, 'dev', self.writer, i)

            else:
                info = worker.local_send(
                    ['valid_tom', (train_agent, dev_batch_splited,
                                dev_strategy, update_table, ret_table,
                                'cache/{}/dev_pred_{}.pkl'.format(args.name, i))])
                dev_loss, dev_accu, dev_step_info = info[1]

                # 关键：在 valid_tom 之后保存 dev epoch stats
                dev_sw = dict(getattr(worker.trainer, "last_tom_epoch_stats", {}))

                draw_info(dev_loss, dev_accu, dev_step_info, 'dev', i)

            if i == 0:
                print('train_step_info:', train_step_info[2])

            if not update_table['tom']:
                score = dev_accu[0]
                score_type = 'accu'
            else:
                price_w = getattr(args, "sa_lambda_price", 1.0)
                switch_w = getattr(args, "sa_lambda_switch", 0.5)

                score = dev_loss[1] + price_w * dev_loss[2] + switch_w * dev_loss[3]
                score_type = 'loss'

            print('[train{}/{}]\t time:{:.2f}s.'.format(
                i + 1, args.epochs, time.time() - cur_t
            ))
            cur_t = time.time()

            if ret_table['id']:
                print('\t<identity> train loss{:.5f} accu{:.5f}, valid loss{:.5f} accu{:.5f}'.format(
                    train_loss[0], train_accu[0], dev_loss[0], dev_accu[0]
                ))

            if ret_table['tom']:
                print(
                    '\t<tom> train iloss{:.5f} iacc{:.5f} ploss{:.5f} sloss{:.5f} train_switch_acc {:.5f}, '
                    'valid iloss{:.5f} iacc{:.5f} ploss{:.5f} sloss{:.5f} dev_switch_acc {:.5f}, '
                    'score{:.5f}'.format(
                        train_loss[1], train_accu[1], train_loss[2], train_loss[3],
                        train_sw.get("switch_acc_mean", 0.0),
                        dev_loss[1], dev_accu[1], dev_loss[2], dev_loss[3],
                        dev_sw.get("switch_acc_mean", 0.0),
                        score
                    )
                )
                print(
                    '\t[switch-debug] '
                    'train_switch_count {:.1f} train_pos_count {:.1f} train_pos_ratio {:.5f} train_prob_mean {:.5f} | '
                    'dev_switch_count {:.1f} dev_pos_count {:.1f} dev_pos_ratio {:.5f} dev_prob_mean {:.5f}'.format(
                        train_sw.get("switch_count", 0.0),
                        train_sw.get("switch_positive_count", 0.0),
                        train_sw.get("switch_positive_ratio", 0.0),
                        train_sw.get("switch_prob_mean", 0.0),

                        dev_sw.get("switch_count", 0.0),
                        dev_sw.get("switch_positive_count", 0.0),
                        dev_sw.get("switch_positive_ratio", 0.0),
                        dev_sw.get("switch_prob_mean", 0.0),
                    )
                )
                print(
                    '\t[switch-confusion] '
                    'train_pred_pos {:.1f} tp {:.1f} fp {:.1f} tn {:.1f} fn {:.1f} | '
                    'dev_pred_pos {:.1f} tp {:.1f} fp {:.1f} tn {:.1f} fn {:.1f}'.format(
                        train_sw.get("switch_pred_positive_count", 0.0),
                        train_sw.get("switch_tp", 0.0),
                        train_sw.get("switch_fp", 0.0),
                        train_sw.get("switch_tn", 0.0),
                        train_sw.get("switch_fn", 0.0),

                        dev_sw.get("switch_pred_positive_count", 0.0),
                        dev_sw.get("switch_tp", 0.0),
                        dev_sw.get("switch_fp", 0.0),
                        dev_sw.get("switch_tn", 0.0),
                        dev_sw.get("switch_fn", 0.0),
                    )
                )
                if (i + 1) % periodic_eval_every == 0:
                    _run_periodic_reward_eval(i, periodic_eval_dialogues)
                # for th in ["0p5", "0p7", "0p8", "0p9", "0p95"]:
                #     print(
                #         '\t[switch-scan@{}] dev pred_pos_ratio {:.5f} precision {:.5f} recall {:.5f} tnr {:.5f}'.format(
                #             th.replace("p", "."),
                #             dev_sw.get(f"switch_scan_{th}_pred_pos_ratio", 0.0),
                #             dev_sw.get(f"switch_scan_{th}_precision", 0.0),
                #             dev_sw.get(f"switch_scan_{th}_recall", 0.0),
                #             dev_sw.get(f"switch_scan_{th}_tnr", 0.0),
                #         )
                #     )

            if (i + 1) % 30 == 0:
                worker.local_send(['save_best_model', (i, score, score_type, True)])

            elif (i + 1) % 100 == 0:
                worker.local_send(['save_best_model', (i, score, score_type, False)])

        if run_final_eval:
            _run_final_reward_eval('e{}'.format(args.epochs), final_eval_dialogues)

    def _log_policy(self, examples, dump_result):
        policies = [{
            'i_policy': [], 'p_policy': []
        }, {
            'i_policy': [], 'p_policy': []
        }]
        for ex in examples:
            for e in ex.events:
                i = e.agent
                odata = e.metadata['output_data']
                policies[i]['i_policy'].append(odata['policy'].reshape(1, -1))
                if odata.get('p_policy') is not None:
                    policies[i]['p_policy'].append(odata['p_policy'].reshape(1, -1))

        for i in range(2):
            for k in policies[i]:
                if len(policies[i][k]) > 0:
                    policies[i][k] = torch.cat(policies[i][k], dim=0).mean(dim=0, keepdim=True)
                    if self.policies_log[i].get(k) is None:
                        self.policies_log[i][k] = []
                    self.policies_log[i][k].append(policies[i][k])
                    if dump_result:
                        logger = logging.getLogger('agent{}_plog_{}'.format(i, k))
                        tmp = torch.cat(self.policies_log[i][k], dim=0).mean(dim=0)
                        # tensor([x, x, x])
                        logger.info(str(tmp.data)[8:-2].replace("        ", "").replace("\n", ""))

    def _init_policy_logfiles(self, logdir):
        formatter = logging.Formatter('%(message)s')
        # stream_handler = logging.StreamHandler()
        # stream_handler.setLevel(logging.DEBUG)
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)
        for i in range(2):
            for name in ['i_policy', 'p_policy']:
                file_handler = logging.FileHandler(os.path.join(logdir, 'agent{}_plog_{}.log'.format(i, name)))
                file_handler.setLevel(level=logging.INFO)
                file_handler.setFormatter(formatter)

                logger = logging.getLogger('agent{}_plog_{}'.format(i, name))
                logger.setLevel(level=logging.INFO)
                logger.addHandler(file_handler)

    def learn(self):
        args = self.args
        rewards = [None] * 2
        s_rewards = [None] * 2

        critic_report_stats = RLStatistics()
        critic_stats = RLStatistics()
        last_time = time.time()

        tensorboard_every = 1
        save_every = 100

        history_train_losses = [[], []]

        batch_size = 50

        pretrain_rounds = 3
        if args.only_run:
            batch_size = 1
            pretrain_rounds = 0

        save_every = max(1, save_every // batch_size)
        report_every = max(1, args.report_every // batch_size)
        num_worker = self.update_worker_list()

        worker = self.worker_conn[0]
        max_epoch = args.num_dialogues // batch_size
        max_epoch = args.epochs
        batch_size = args.batch_size

        save_every = max(50, max_epoch // 100)
        report_every = max(1, max_epoch // 100)

        if args.debug:
            save_every = 1

        device = 'cpu'
        if len(args.gpuid) > 0:
            device = "cuda:{}".format(args.gpuid[0])

        policy_buffer = ReplayBuffer.get_instance('policy')
        value_buffer = ReplayBuffer.get_instance('value')
        self._init_policy_logfiles('logs/' + args.name)

        sample_size = 32
        train_size = 128

        for epoch in range(max_epoch):
            last_time = time.time()
            policy_buffer.empty()
            # _batch_iters, _rewards, example, _ = self.sample_data(i, batch_size, args)
            # print('=' * 5 + ' [Epoch {}/{} running.]'.format(epoch, max_epoch))
            tt = time.time()
            info = worker.send(['simulate', epoch, sample_size, sample_size])
            _batch_iters, batch_info, example, v_str = pkl.loads(info[1])
            _rewards, strategies = batch_info

            self._log_policy(example, (epoch+1) % save_every == 0)

            policy_buffer.add_batch_iters(_batch_iters[0],
                                          add_dict={'reward': _rewards[0], 'strategy': strategies[0]})
            value_buffer.add_batch_iters(_batch_iters[0],
                                         add_dict={'reward': _rewards[0], 'strategy': strategies[0]})

            # For debug
            # print("rewards:", np.mean(_rewards[0]), np.mean(_rewards[1]))
            # print("rewards_num:", len(_rewards[0]), len(_rewards[1]))

            tt = time.time()
            value_update = min(value_buffer.size//train_size, 5)
            for i in range(value_update):
                batch_iters, _, ret_add = value_buffer.sample_batch(train_size, add_info={'reward'}, to_device=device)
                worker.local_send(
                    ['train', (epoch, batch_iters, ret_add['reward'], 'fix_policy')])

            batch_iters, _, ret_add = policy_buffer.sample_batch(train_size, add_info={'reward'}, to_device=device)

            info = worker.local_send(
                ['train', (epoch, batch_iters, ret_add['reward'], '')])
            loss = info[1]
            print('train time:', time.time()-tt)

            # Draw outputs on the tensorboard
            self._draw_tensorboard((epoch + 1) , [[loss], []],
                                   _rewards)

            print('\t<train> reward{:.3f}, {:.3f} pg_loss {:.5f}, value_loss {:.5f}, value_update {}'
                  .format(np.mean(_rewards[0]), np.mean(_rewards[1]), loss['pg_loss'][0,0], loss['value_loss'][0,0], value_update))

            if (epoch+1)%save_every == 0:
                self._dump_buffer(value_buffer, epoch+1)

                self.dump_examples(example, v_str, epoch, 'train')
                valid_info = worker.local_send(['valid', (0, 200)])
                valid_stats, example, v_str = valid_info[1]
                self.dump_examples(example, v_str, epoch, 'dev')

                valid_reward = [vs.mean_reward() for vs in valid_stats]
                self._draw_tensorboard_valid((epoch + 1), valid_reward)
                print('\t<valid> reward{:.3f}, {:.3f}'.format(valid_reward[0], valid_reward[1]))
                worker.local_send(['save_model', (epoch, valid_reward[0], 'reward')])
            print('=' * 5 + ' [Epoch {}/{} for {:.3f}s.]'.format(epoch+1, max_epoch, time.time() - last_time))
            # # Save model
            # if (i + 1) % save_every == 0:
            #     # TODO: valid in dev set
            #     valid_stats, _, _ = self.validate(args, 50 if args.only_run else 200)
            #     if not args.only_run:
            #         self.drop_checkpoint(args, i + 1, valid_stats,
            #                              model_opt=self.agents[self.training_agent].env.model_args)
            #         if args.update_oppo:
            #             self.update_opponent(['policy', 'critic'])
            #     else:
            #         print('valid ', valid_stats.str_loss())

    def _dump_buffer(self, buffer, epoch, ):
        args = self.args
        path_pkl = '{root}/{model}_buffer{epoch}.pkl'.format(
            root=args.model_path,
            model=args.name,
            epoch=epoch)
        print('Save buffer at {}.'.format(path_pkl))
        with open(path_pkl, 'wb') as f:
            pkl.dump(buffer, f)
        # with open(path_pkl, 'rb') as f:
        #     bf = pkl.load(f)

    def run(self):
        # deprecated
        # self.run_local_workers()
        args = self.args
        rewards = [None] * 2
        s_rewards = [None] * 2
        tensorboard_every = 1
        save_every = 50

        history_train_losses = [[], []]

        batch_size = 100

        pretrain_rounds = 3

        save_every = max(1, save_every // batch_size)
        report_every = max(1, args.report_every // batch_size)

        max_epoch = args.num_dialogues // batch_size
        epoch = 0
        data_size = 0

        all_rewards = [[], []]

        num_worker = self.update_worker_list()
        last_time = time.time()
        for epoch in range(args.start_epoch, max_epoch):
            batches = []
            rewards = [[], []]

            task_lists = self.allocate_tasks(num_worker, batch_size)

            # Use workers to get trajectories
            train_examples = []
            train_ex_str = []
            for i, w in enumerate(self.worker_conn):
                info = w.send(['simulate', epoch, batch_size, task_lists[i]])
                if info[0] != 'done':
                    print('Error on {}: {}.'.format(i, info))
                data = pkl.loads(info[1])
                batches += data[0]
                rewards[0] += data[1][0]
                rewards[1] += data[1][1]
                train_examples += data[2]
                train_ex_str += data[3]

            self.dump_examples(train_examples, train_ex_str, epoch)

            # For debug
            print("rewards:", np.mean(rewards[0]), np.mean(rewards[1]))
            print("rewards_num:", len(rewards[0]), len(rewards[1]))

            # Train the model
            train_info = self.worker_conn[0].send(['train', pkl.dumps((epoch, batches, rewards[0], self.args.train_mode))])
            if train_info[0] != 'done':
                print('Error on {}: {}.'.format(i, train_info))

            # Draw outputs on the tensorboard
            self._draw_tensorboard((epoch + 1) * batch_size, [[pkl.loads(train_info[1])], []],
                                   rewards)

            # Get new model from trainer

            info = self.worker_conn[0].send(['fetch_model', 0])
            data = info[1]

            # Save local checkpoint

            # Update all the worker
            for i, w in enumerate(self.worker_conn):
                if i == 0:
                    continue
                w.send(['update_model', 0, data])

            # for i, w in enumerate(self.worker_conn):
            #     if i == 0:
            #         continue
            #     w.recv()

            # Valid new model
            task_lists = self.allocate_tasks(num_worker, 50)
            now = 0

            valid_stats = [RLStatistics(), RLStatistics()]
            valid_examples = []
            valid_ex_str = []
            for i, w in enumerate(self.worker_conn):
                valid_info = w.send(['valid', (now, task_lists[i])])
                now += task_lists[i]
                valid_info[1] = pkl.loads(valid_info[1])
                for j in range(2):
                    valid_stats[j].update(valid_info[1][0][j])
                valid_examples += valid_info[1][1]
                valid_ex_str += valid_info[1][2]

            self.dump_examples(valid_examples, valid_ex_str, epoch, 'dev')
            # Save the model
            self.worker_conn[0].send(['save_model', pkl.dumps((epoch, valid_stats[0]))])
            # self.worker_conn[0].recv()

            # Draw dev rewards on tensorboard
            dev_rewards = [valid_stats[j].mean_reward() for j in range(2)]
            self._draw_tensorboard_valid((epoch + 1) * batch_size, dev_rewards)

            print('=' * 5 + ' [Epoch {} for {:.3f}s.]'.format(epoch, time.time() - last_time))
            last_time = time.time()

        self.quit_all_workers()
        self.join_local_workers()

    def quit_all_workers(self):
        for w in self.worker_conn:
            w.send(['quit'])

    def join_local_workers(self):
        # for w in self.local_workers:
        #     w.join()
        pass