from gym import spaces
import numpy as np
import random
from models import DQN, ReplayBuffer
import torch
import torch.nn.functional as F
import copy

def train_ours(agent, env, hyper_params):
    state =env.reset()
    loss = 0
    eps_timesteps = hyper_params["eps-fraction"] * \
        float(hyper_params["num-steps"])
    episode_rewards = [0.0, 0.0]
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * \
            (hyper_params["eps-end"] - hyper_params["eps-start"])
        sample = random.random()
        
        '''if(sample > eps_threshold):
            if t > hyper_params["learning-starts"] + hyper_params["num-steps"]:
                while True:
                    action = agent.act(state)
                    if action in hyper_params["actions"]:
                        break
            else:
                while True:
                    action = env.action_space.sample()
                    if action in hyper_params["actions"]:
                        break
        else:
            while True:
                action = env.action_space.sample()
                if action in hyper_params["actions"]:
                    break'''
                    
        if sample > eps_threshold:
            action = agent.act(state)
        else:
            action = env.action_space.sample()
            
        lives = env.ale.lives()
        next_state, reward, done, info = env.step(action)
        agent.memory.add(state, action, reward, next_state, done or (env.ale.lives() != lives))
        state = next_state
        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            if hyper_params["first"]:
                loss = agent.fr_optimise_loss()
            else:
                loss = agent.ar_optimise_loss()

        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            if hyper_params["first"]:
                agent.fr_update_target_network()
            else:
                agent.ar_update_target_network()

        num_episodes = len(episode_rewards) - 1
        print(f' [Episode {num_episodes}] [Step {t}/{hyper_params["num-steps"]} Reward] : {episode_rewards[-1]} | [Max Reward] : {max(episode_rewards[:-1])} | [Action] : {action} | [Loss] : {loss:.4f} | [Eps] : {eps_threshold:.4f}'"\r", end="", flush=True)
    hyper_params["first"] = False
    hyper_params['learning-starts'] = 0
    print('\n')
    
    return agent
    

class OursAgent:
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
        self.gamma = 0.8

        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.fr_update_target_network()
        self.target_network.eval()
        self.previous_policy_network = copy.deepcopy(self.policy_network)
        self.optimizer = torch.optim.RMSprop(self.policy_network.parameters()
            , lr=0.00625, eps=1.5e-4)  
        self.device = device
        self.ema_decay = 0.99
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.temperature = 0.5
        self.mu = 5.0

    def fr_optimise_loss(self):
        loss = self.calculate_td_loss()
        self.optimizer.zero_grad()
        loss.backward()
        for name, param in self.policy_network.named_parameters():
            if 'proj' in name:
                continue
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def ar_optimise_loss(self):
        loss1 = self.calculate_td_loss()
        loss2 = self.calculate_contrastive_loss()
        self.previous_policy_network = copy.deepcopy(self.policy_network)
        loss = loss1 + self.mu*loss2
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()
    
    def calculate_td_loss(self):
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

        del states
        del next_states
        
        return loss

    def calculate_contrastive_loss(self):
        
        device = self.device
        states, _, _, _, _ = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        states = torch.from_numpy(states).float().to(device)
        _, prev_feature =self.previous_policy_network(states)
        _, curr_feature = self.policy_network(states)
        _, target_feature = self.target_network(states)
        
        posi = self.cos(target_feature, curr_feature)
        logits = posi.reshape(-1, 1)
        nega = self.cos(curr_feature, prev_feature)
        logits = torch.cat((logits, nega.reshape(-1, 1)),dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(states.size(0)).to(device).long()
        loss = F.cross_entropy(logits, labels)

        return loss
        
    def fr_update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def ar_update_target_network(self):
        policy = copy.deepcopy(self.policy_network.state_dict())
        target = copy.deepcopy(self.target_network.state_dict())
        for key in policy.keys():
            target[key] = self.ema_decay * target[key] + (1-self.ema_decay) * policy[key]
        self.target_network.load_state_dict(target)
        self.ema_decay *= 0.99
    
    def act(self, state: np.ndarray):
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, _ = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()

    def exploit(self, state):
        device = self.device
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, _ = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()  