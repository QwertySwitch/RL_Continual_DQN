from gym import spaces
import numpy as np
import random
from models import DQN, ReplayBuffer
import torch
import torch.nn.functional as F


def train_dqn(agent, env, hyper_params):
    state = env.reset()
    loss = 0
    eps_timesteps = hyper_params["eps-fraction"] * \
        float(hyper_params["num-steps"])
    episode_rewards = [0.0, 0.0]
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * \
            (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()

        if sample > eps_threshold:
            action = agent.act(state)
        else:
            action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
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
    return agent

class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 use_double_dqn,
                 lr,
                 batch_size,
                 gamma,
                 device=torch.device("cpu" ),
                 dqn_type="neurips"):

        self.memory = replay_buffer
        self.batch_size = batch_size
        self.use_double_dqn = use_double_dqn
        self.gamma = gamma

        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters()
            , lr=0.0000625, eps=1.5e-4)        
        self.device = device

    def optimise_td_loss(self):
        device = self.device

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            if self.use_double_dqn:
                next_action, _ = self.policy_network(next_states) 
                _, max_next_action = next_action.max(1)
                next_q_values, _ = self.target_network(next_states)
                max_next_q_values = next_q_values.gather(1, max_next_action.unsqueeze(1)).squeeze()
            else:
                next_q_values, _ = self.target_network(next_states)
                max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values, _ = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del states
        del next_states
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def exploit(self, state: np.ndarray):
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, _ = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()
    
    def act(self, state: np.ndarray):
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, _ = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()