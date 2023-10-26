import random
import numpy as np
import gym

from agent import DQNAgent
from models import ReplayBuffer
from wrappers import *
import torch
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN Atari')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')

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
        "learning-starts": 10000,  # number of steps before learning starts
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
    
    envs = ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5"]
    agent = None
    for env_name in envs:
        env = gym.make(env_name, full_action_space=True, frameskip=1)
        env.seed(hyper_params["seed"])

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = WarpFrame(env)
        env = PyTorchFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, 4)
        if agent == None:
            replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])
            agent = DQNAgent(
                env.observation_space,
                env.action_space,
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

        eps_timesteps = hyper_params["eps-fraction"] * \
            float(hyper_params["num-steps"])
        episode_rewards = [-20.0, 0.0]
        print(f"Training {env_name}")
        state = env.reset()
        loss = 0
        for t in range(hyper_params["num-steps"]):
            fraction = min(1.0, float(t) / eps_timesteps)
            eps_threshold = hyper_params["eps-start"] + fraction * \
                (hyper_params["eps-end"] - hyper_params["eps-start"])
            sample = random.random()

            if(sample > eps_threshold):
                # Exploit
                action = agent.act(state)
            else:
                # Explore
                action = env.action_space.sample()
            
            next_state, reward, done, _, info = env.step(action)
            agent.memory.add(state, action, reward, next_state, float(done))
            state = next_state
            episode_rewards[-1] += reward
            if done:
                state = env.reset()
                episode_rewards.append(0.0)

            if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
                loss = agent.optimise_td_loss()

            if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
                agent.update_target_network()

            num_episodes = len(episode_rewards) - 1
            print(f' [Episode {num_episodes}] [Step {t}/{hyper_params["num-steps"]} Reward] : {episode_rewards[-1]} | [Max Reward] : {max(episode_rewards[:-1])} | [Action] : {action} | [Loss] : {loss:.4f} | [Eps] : {eps_threshold:.4f}'"\r", end="", flush=True)
            
        print('\n')
        done = False
        print('Inference Start')
        for env_n in envs:
            reward_done = 0
            env = gym.make(env_n, full_action_space=True, frameskip=1)
            env.seed(hyper_params["seed"])

            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            env = FireResetEnv(env)
            env = WarpFrame(env)
            env = PyTorchFrame(env)
            env = ClipRewardEnv(env)
            env = FrameStack(env, 4)
            state = env.reset()
            while True:
                action = agent.act(state)
                next_state, reward, done, _, info = env.step(action)
                reward_done += reward
                if done == True:
                    break
            done = False
            print(f'Env Name : {env_n} | Reward : {reward_done}')