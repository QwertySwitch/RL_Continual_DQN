import random
import numpy as np
import gym

from algorithm.dqn import *
from algorithm.ours import *
from models import ReplayBuffer
from wrappers import *
import torch
import argparse
from utils import *
import sys

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
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "dqn_type":"neurips",
        # total number of steps to run the environment for
        "num-steps": int(1e6),
        "batch-size": 32,  # number of transitions to optimize at the same time
        "learning-starts": 5000,  # number of steps before learning starts
        "learning-freq": 1,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": eps_start,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])
    
    envs = ["ALE/Assault-v5", "ALE/Carnival-v5", "ALE/Centipede-v5", "ALE/DemonAttack-v5", "ALE/Phoenix-v5"]
    
    agent = None
    for env_name in envs:
        env = init_env(env_name)
        if agent == None:
            agent = init_agent(hyper_params, env, args)
            
        print(f"Training {env_name}")
        getattr(sys.modules[__name__], 'train_'+args.algorithm)(agent, env, hyper_params)
        test_all(agent, envs)
        