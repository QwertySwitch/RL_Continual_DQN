import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from models import TwinnedQNetwork, CateoricalPolicy, ReplayBuffer
import random
import numpy as np
from gym import spaces

def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()

def train_sac(agent, env, hyper_params):
    #env.reset()
    #state, _, done, _ = env.step(env.action_space.sample())
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

        if t > hyper_params["learning-starts"]:
            #action = agent.explore(state[0])
            action = agent.explore(state)
        else:
            action = env.action_space.sample()

        #next_state, reward, done, info = env.step(action)
        next_state, reward, done, info = env.step(action)
        #agent.memory.add(state[0], action, reward, next_state[0], float(done))
        agent.memory.add(state, action, reward, next_state, float(done))
        state = next_state
        episode_rewards[-1] += reward
        if done:
            #env.reset()
            #state, _, done, _ = env.step(env.action_space.sample())
            state = env.reset()
            episode_rewards.append(0.0)

        if t > hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
            loss = agent.learn()

        if t > hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
            agent.update_target()

        num_episodes = len(episode_rewards) - 1
        print(f' [Episode {num_episodes}] [Step {t}/{hyper_params["num-steps"]} Reward] : {episode_rewards[-1]} | [Max Reward] : {max(episode_rewards[:-1])} | [Action] : {action} | [Loss] : {loss:.4f} | [Eps] : {eps_threshold:.4f}'"\r", end="", flush=True)
        
    print('\n')
    return agent

class SACAgent:
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

        self.policy = CateoricalPolicy(action_space.n).to(device)
        self.online_critic = TwinnedQNetwork(action_space.n).to(device)
        self.target_critic = TwinnedQNetwork(action_space.n).to(device).eval()
        self.batch_size = batch_size
        self.target_critic.load_state_dict(self.online_critic.state_dict())
        self.device = device
        self.gamma = gamma
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)
        self.memory = replay_buffer
        self.target_entropy = \
            -np.log(1.0 / action_space.n) * 0.98

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)


    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.online_critic(states)
        curr_q1 = curr_q1.gather(1, actions.unsqueeze(1)).squeeze()
        curr_q2 = curr_q2.gather(1, actions.unsqueeze(1)).squeeze()
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True).squeeze()
        
        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def explore(self, state):
        # Act with randomness.
        state = np.array(state) / 255.0
        state = torch.FloatTensor(state[None, ...])
        state = state.to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.item()
    
    def exploit(self, state):
        # Act without randomness.
        state = np.array(state) / 255.0
        state = torch.FloatTensor(state[None, ...])
        state = state.to(self.device)
        with torch.no_grad():
            action = self.policy.act(state)
        return action.item()    
    
    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)
        
        batch = (states, actions, rewards, next_states, dones)
            # Set priority weights to 1 when we don't use PER.
        weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = \
            self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss)
        update_params(self.q2_optim, q2_loss)
        update_params(self.policy_optim, policy_loss)
        update_params(self.alpha_optim, entropy_loss)

        self.alpha = self.log_alpha.exp()
        
        return q1_loss.item() + q2_loss.item() + policy_loss.item() + entropy_loss.item()
    
    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.policy.save(os.path.join(save_dir, 'policy.pth'))
        self.online_critic.save(os.path.join(save_dir, 'online_critic.pth'))
        self.target_critic.save(os.path.join(save_dir, 'target_critic.pth'))