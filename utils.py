import random
import numpy as np
import gym

from algorithm.dqn import DQNAgent, train_dqn
from algorithm.ours import OursAgent
from models import ReplayBuffer
from wrappers import *
import torch
import argparse


def init_env(env_name, test):
    if test:
        env = gym.make(env_name, full_action_space=True, frameskip=1)
    else:
        env = gym.make(env_name, full_action_space=False, frameskip=1)
    env_ = gym.make(env_name, full_action_space=True, frameskip=1)
    env.seed(42)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    
    return env, env_.action_space

def weight_updates_test(agent, action_space):
    current_weight = agent.policy_network.classifier.weight.data
    current_bias = agent.policy_network.classifier.bias.data
    agent.policy_network.classifier = torch.nn.Linear(512, action_space.n).cuda()  
    for i in range(len(current_weight)):
        agent.policy_network.classifier.weight.data[i] = current_weight[i]
        agent.policy_network.classifier.bias.data[i] = current_bias[i]
        
    return agent
    
    
def init_agent(hyper_params, action_space, env, args):
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
    
    if args.algorithm == 'dqn':
        agent = DQNAgent(
            env.observation_space,
            action_space,
            replay_buffer,
            use_double_dqn=hyper_params["use-double-dqn"],
            lr=hyper_params['learning-rate'],
            batch_size=hyper_params['batch-size'],
            gamma=hyper_params['discount-factor'],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dqn_type=hyper_params["dqn_type"]
        )
    elif args.algorithm == 'ours':
        agent = OursAgent(
            env.observation_space,
            action_space,
            replay_buffer,
            use_double_dqn=hyper_params["use-double-dqn"],
            lr=hyper_params['learning-rate'],
            batch_size=hyper_params['batch-size'],
            gamma=hyper_params['discount-factor'],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dqn_type=hyper_params["dqn_type"]
        )

    if(args.load_checkpoint_file):
        print(f"Loading a policy - { args.load_checkpoint_file } ")
        agent.policy_network.load_state_dict(
            torch.load(args.load_checkpoint_file))
    return agent

def test_all(agent, envs):
    print('Inference Start')
    for env_n in envs:
        env, _ = init_env(env_n, True)
        reward = test(agent, env_n, env)
        print('/n')
        

def test(agent, name, env):
    rewards = []
    for i in range(5):
        reward_done = 0
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            reward_done += reward
            state = next_state
            print(f'Env Name : {name} | Current Reward : {reward_done}'"\r", end="", flush=True)
            if done == True:
                break
        rewards.append(reward_done)
        done = False
    print(f'Env Name : {name} | 100R Avg Reward : {sum(rewards) / len(rewards)}'"\r", end="", flush=True)
    return sum(rewards) / len(rewards)