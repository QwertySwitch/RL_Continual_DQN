import random
import numpy as np
import gym

from algorithm.dqn import DQNAgent, train_dqn
from algorithm.ours import OursAgent
from algorithm.sac import SACAgent
from algorithm.sac_ours import SACOursAgent
from models import ReplayBuffer
from wrappers import *
import torch
import argparse
import math

def init_env(env_name, test):
    #env = Environment(env_name)
    env = make_pytorch_env(env_name, clip_rewards=False)
    if test:
        env = make_pytorch_env(
            env_name, episode_life=False, clip_rewards=False)
    '''env = gym.make(env_name, full_action_space=True, frameskip=1)
    env_ = gym.make(env_name, full_action_space=True, frameskip=1)
    env.seed(42)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)'''
    
    return env, env.action_space

def weight_updates_test(agent, action_space):
    current_weight = agent.policy_network.classifier.weight.data
    current_bias = agent.policy_network.classifier.bias.data
    agent.policy_network.classifier = torch.nn.Linear(512, action_space.n).cuda()  
    for i in range(len(current_weight)):
        agent.policy_network.classifier.weight.data[i] = current_weight[i]
        agent.policy_network.classifier.bias.data[i] = current_bias[i]
        
    return agent

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

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
    elif args.algorithm == 'sac':
        agent = SACAgent(
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
    elif args.algorithm == 'sacours':
        agent = SACOursAgent(
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
        print('\n')
        

def test(agent, name, env):
    rewards = []
    for i in range(5):
        reward_done = 0
        env.reset()
        state, _, done, _ = env.step(env.action_space.sample())
        while True:
            action = agent.exploit(state)
            next_state, reward, done, info = env.step(action)
            reward_done += reward
            state = next_state
            print(f'Env Name : {name} | Current Reward : {reward_done}'"\r", end="", flush=True)
            if done == True:
                break
        rewards.append(reward_done)
        done = False
    print(f'Env Name : {name} | 5 Episode Avg Reward : {sum(rewards) / len(rewards)}'"\r", end="", flush=True)
    return sum(rewards) / len(rewards)