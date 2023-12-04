import random
import numpy as np
import gym

from algorithm.dqn import *
from algorithm.ours import *
from algorithm.sac import *
from algorithm.sac_ours import *
from models import ReplayBuffer
from wrappers import *
import torch
import argparse
from utils import *
import sys
import copy
import warnings

warnings.filterwarnings("ignore") 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Continual Atari')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')
    parser.add_argument('--algorithm', type=str, default='dqn', 
                        help='Which algorithm to run (dqn, ours)')

    args = parser.parse_args()
    # If you have a checkpoint file, spend less time exploring
    if(args.load_checkpoint_file):
        eps_start= 0.01
    else:
        eps_start= 1

    hyper_params = {
        "seed": 42,  # which seed to use
        "replay-buffer-size": 10000,  # replay buffer size
        "learning-rate": 0.0003,  # learning rate for Adam optimizer
        "discount-factor": 0.95,  # discount factor
        "dqn_type":"neurips",
        # total number of steps to run the environment for
        "num-steps": int(2e5),
        "batch-size": 64,  # number of transitions to optimize at the same time
        "learning-starts": 5000,  # number of steps before learning starts
        "learning-freq": 4,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 3000,  # number of iterations between every target network update
        "eps-start": eps_start,  # e-greedy start threshold
        "eps-end": 0.025,  # e-greedy end threshold
        "eps-fraction": 0.30,  # fraction of num-steps
        "first": True,
        "actions": None,
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])
    
    #envs = ["ALE/Assault-v5", "ALE/Carnival-v5", "ALE/Centipede-v5", "ALE/Phoenix-v5"]
    #envs = ["AssaultDeterministic-v4", "CarnivalDeterministic-v4", "CentipedeDeterministic-v4", "PhoenixDeterministic-v4"]
    envs = ["AssaultNoFrameskip-v4", "CarnivalNoFrameskip-v4", "CentipedeNoFrameskip-v4", "PhoenixNoFrameskip-v4"]
    actions = []
    for env_name in envs:
        _, a_s = init_env(env_name, True)
        actions.append([i for i in range(a_s.n)])
        
    agent = None
    for env_name in envs:
        env, a_s = init_env(env_name, True)
        if agent == None:
            agent = init_agent(hyper_params, env.action_space, env, args)
        hyper_params["actions"] = actions[envs.index(env_name)]
        print(f"Training {env_name}")
        if env_name != envs[-1]:
            agent = getattr(sys.modules[__name__], 'train_'+args.algorithm)(agent, env, hyper_params)
            #test_agent = weight_updates_test(agent, a_s)
            test_all(agent, envs)
        