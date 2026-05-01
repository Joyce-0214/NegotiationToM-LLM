import argparse
import random
import json
import numpy as np
import torch
import sys
import cocoa.options
from onmt.Utils import use_gpu

from cocoa.core.util import read_json
from cocoa.core.schema import Schema
from cocoa.core.scenario_db import ScenarioDB
from cocoa.neural.loss import ReinforceLossCompute
from craigslistbargain.sessions.neural_session import NeuralSession


from craigslistbargain.core.scenario import Scenario
from craigslistbargain.core.controller import Controller
from craigslistbargain.systems import get_system
from craigslistbargain.neural.rl_trainer import RLTrainer
from craigslistbargain.neural import build_optim

from craigslistbargain.multi_manager_debug import MultiManager
from craigslistbargain.multi_trainer import MultiTrainer

from craigslistbargain.multi_manager_debug import MultiManager as MultiManager_DEBUG, MultiRunner as MultiTrainer_DEBUG

import craigslistbargain.options as options

from craigslistbargain.buffer import ReplayBuffer

def dump_args():
    # dump all the args at checkpoints
    fpath = '{}/{}.txt'.format(args.model_path, args.name)
    print('dumped args at {}'.format(fpath))
    with open(fpath, 'w') as f:
        for s in sys.argv:
            if s[0] == '-':
                f.write('\n')
            f.write(s+' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--agents', help='What kind of agent to use. The first agent is always going to be updated and the second is fixed.', nargs='*', required=True)
    parser.add_argument('--agent-checkpoints', default=[], nargs='+', help='Directory to learned models')
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--verbose', default=False, action='store_true', help='Whether or not to have verbose prints')

    parser.add_argument('--histogram', default=False, action='store_true', help='Whether or not to show histogram of policies')
    parser.add_argument('--valid-scenarios-path', help='Output path for the validation scenarios')

    parser.add_argument('--only-run', default=False, action='store_true', help='only sample trajectories.')

    parser.add_argument('--update-oppo', action='store_true', help='update opponent')

    parser.add_argument('--model-type', default='reinforce', choices=['reinforce', 'a2c', 'critic', 'tom'], help='choise rl algorithms')
    parser.add_argument('--load-critic-from', default=None, type=str, help='load critic model from another checkpoint')

    parser.add_argument('--name', default='rl', type=str, help='Name of this experiment.')

    parser.add_argument('--mappings', help='Path to vocab mappings')

    parser.add_argument('--tom-type', choices=['expectation', 'competitive', 'cooperative'], type=str, default='expectation',
                        help='tom inference type')
    parser.add_argument('--price-strategy', choices=['high', 'low', 'decay', 'neural'], type=str,
                        default='neural',
                        help='supervise agent price strategy.')
    # Initialization
    parser.add_argument('--pretrained-wordvec', nargs='+', default=['', ''],
                       help="""If a valid path is specified, then this will load
                           pretrained word embeddings, if list contains two embeddings,
                           then the second one is for item title and description""")
    parser.add_argument('--param-init', type=float, default=0.1,
                       help="""Parameters are initialized over uniform distribution
                           with support (-param_init, param_init).
                           Use 0 to not use initialization""")
    parser.add_argument('--fix-pretrained-wordvec',
                       action='store_true',
                       help="Fix pretrained word embeddings.")

    parser.add_argument('--train-mode', type=str, default='normal', choices=['normal', 'fix_value', 'fix_policy', 'none'],
                        help='choices for different training mode.')

    parser.add_argument('--num-cpus', type=int, default=1, help='number of cpu threads(worker)')

    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--start-epoch', type=int, default=0, help='start from specific epoch.')
    parser.add_argument('--warmup-epochs', type=int, default=25, help='epoch number of using randomness actions')
    parser.add_argument('--start-port', type=int, default=7000, help='start port address for socket transport')
    parser.add_argument('--use-utterance', action='store_true', default=False, help='use data generator')

    parser.add_argument('--nlg-dir', type=str, default='data/nlg_templates_dict.json',
                        help='directory of templates for IR-based NLG')
    parser.add_argument('--nlg-backend', type=str, default='template', choices=['template', 'llm'],
                        help='surface realization backend')
    parser.add_argument('--llm-nlg-model', type=str, default=None,
                        help='model name for the OpenAI-compatible NLG backend')
    parser.add_argument('--llm-nlg-api-base', type=str, default='https://api.openai.com/v1',
                        help='OpenAI-compatible API base URL for the LLM NLG backend')
    parser.add_argument('--llm-nlg-api-key-env', type=str, default='OPENAI_API_KEY',
                        help='environment variable that stores the API key for LLM NLG')
    parser.add_argument('--llm-nlg-temperature', type=float, default=0.2,
                        help='sampling temperature for LLM NLG')
    parser.add_argument('--llm-nlg-timeout', type=int, default=30,
                        help='request timeout in seconds for LLM NLG')
    parser.add_argument('--llm-nlg-max-retries', type=int, default=2,
                        help='maximum number of retries for LLM NLG requests')
    parser.add_argument('--llm-nlg-cache-path', type=str, default='cache/llm_nlg_cache.json',
                        help='cache file for LLM-generated utterance templates')
    parser.add_argument('--llm-nlg-fallback-to-template', action='store_true', default=False,
                        help='fallback to the old template NLG when the LLM backend fails')
    parser.add_argument('--sa-lambda-price', type=float, default=1.0,
                        help='weight for switch-aware ToM price supervision loss')
    parser.add_argument('--sa-lambda-switch', type=float, default=0.5,
                        help='weight for switch-aware ToM switch supervision loss')
    parser.add_argument('--strategy-ignore-surface-text', action='store_true', default=False,
                        help='keep generated surface text out of strategy state; only structured LF/price are encoded')
    parser.add_argument('--enforce-price-protocol', action='store_true', default=False,
                        help='enforce monotonic offers and close immediately when buyer bid crosses seller ask')
    parser.add_argument('--enable-buyer-price-safety', action='store_true', default=False,
                        help='enable buyer-side post-policy price safety filtering')
    parser.add_argument('--enable-seller-tactic-tracker', action='store_true', default=False,
                        help='enable rule-based seller turn-level tactic tracking')
    parser.add_argument('--enable-rule-offer-planner', action='store_true', default=False,
                        help='enable rule-based buyer offer planner after price safety')
    parser.add_argument('--allow-buyer-price-decrease', action='store_true', default=False,
                        help='allow buyer planned prices to decrease below the previous buyer offer')
    parser.add_argument('--price-safety-debug', action='store_true', default=False,
                        help='print buyer price safety diagnostics')
    parser.add_argument('--tactic-tracker-debug', action='store_true', default=False,
                        help='print seller tactic tracker diagnostics')
    parser.add_argument('--offer-planner-debug', action='store_true', default=False,
                        help='print buyer offer planner diagnostics')
    parser.add_argument('--turn-trace-path', default=None,
                        help='optional CSV path for buyer/seller turn trace logs')

    parser.add_argument('--fix-id', action='store_true', default=False, help='Fix identity')
    parser.add_argument('--strategy-in-words', action='store_true', default=False,
                        help='add strategy directly in language part.')

    parser.add_argument('--get-dialogues', default=False, action='store_true')
    parser.add_argument('--tom-test', default=False, action='store_true')
    parser.add_argument('--load-identity-from', default=None, type=str, help='load critic model from another checkpoint')
    parser.add_argument('--load-sample', default=None, type=str)

    parser.add_argument('--idgt', action='store_true', default=False, help="identity ground truth as model\'s input.")

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tom-beta', type=float, default=1)

    cocoa.options.add_scenario_arguments(parser)
    options.add_data_generator_arguments(parser)
    options.add_system_arguments(parser)
    options.add_rl_arguments(parser)
    options.add_model_arguments(parser)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set tom-beta
    NeuralSession.tominf_beta = args.tom_beta

    dump_args()

    # on policy for policy buffer
    ReplayBuffer.global_init('policy')
    ReplayBuffer.global_init('value')

    if args.tom_test:
        # Single thread
        print('[Info] Running in debug mode for identity test.')
        manager = MultiManager_DEBUG(args.num_cpus, args, MultiTrainer_DEBUG)
        manager.learn_identity()
    elif args.get_dialogues:
        manager = MultiManager_DEBUG(args.num_cpus, args, MultiTrainer_DEBUG)
        manager.evaluate()
    else:
        # manager = MultiManager(args.num_cpus, args, MultiTrainer)
        # manager.run()
        manager = MultiManager_DEBUG(args.num_cpus, args, MultiTrainer_DEBUG)
        manager.learn()

