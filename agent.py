import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """与环境交互并学习的Agent。"""
    def __init__(self, state_size, action_space, seed=0, device='cuda'):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.seed = random.seed(seed)

        self.device = torch.device(device)

        self.qnetwork_local = QNetwork(state_size, self.action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, self.action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)

        self.memory = deque(maxlen=20000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 1e-3

    def step(self, state, action_index, reward, next_state, done):
        self.memory.append((state, action_index, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, action_indices, rewards, next_states, dones = zip(*experiences)
        
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        action_indices = torch.from_numpy(np.vstack(action_indices)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, action_indices)

        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReinforceAgent:
    def __init__(self, state_size, action_space, seed=0,device='cuda'):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.gamma = 0.99
        self.device = torch.device(device)

        self.policy = self._create_policy_network().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        
        self.saved_log_probs = []

    def _create_policy_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Softmax(dim=-1)
        )

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy(state)
        m = Categorical(probs)
        action_index = m.sample()
        self.saved_log_probs.append(m.log_prob(action_index))
        return action_index.item()

    def learn(self, final_reward):
        policy_loss = []
        # 为回合中的每一步都赋予相同的最终奖励
        # 在REINFORCE中，更标准的是使用discounted returns，但对于稀疏奖励问题，此方法更简单有效
        returns = torch.tensor([final_reward] * len(self.saved_log_probs), device=self.device)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空回合数据
        del self.saved_log_probs[:]

### --- 2.3 A2C Agent (Actor-Critic) ---
class A2CAgent:
    def __init__(self, state_size, action_space, seed=0, device='cuda'):
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.gamma = 0.99
        self.device = torch.device(device)
        
        # Actor Network
        self.actor = self._create_actor_network().to(self.device)
        # Critic Network
        self.critic = self._create_critic_network().to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)
        
        self.episode_data = []

    def _create_actor_network(self): # Policy network
        return nn.Sequential(
            nn.Linear(self.state_size, 128), nn.ReLU(),
            nn.Linear(128, self.action_size), nn.Softmax(dim=-1))

    def _create_critic_network(self): # Value network
        return nn.Sequential(
            nn.Linear(self.state_size, 128), nn.ReLU(),
            nn.Linear(128, 1))

    def act(self, state):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.actor(state_tensor)
        m = Categorical(probs)
        action_index = m.sample()
        
        # 存储本回合数据
        self.episode_data.append({
            "log_prob": m.log_prob(action_index),
            "value": self.critic(state_tensor)
        })
        return action_index.item()

    def learn(self, final_reward):
        actor_loss = []
        critic_loss = []
        
        # 对于稀疏奖励，我们将最终奖励作为所有时间步的Return(G)
        for step_data in self.episode_data:
            advantage = final_reward - step_data["value"]
            
            # Critic loss: (G - V(s))^2
            critic_loss.append(advantage.pow(2))
            # Actor loss: -log_prob * advantage (detach advantage to stop gradients flowing to critic)
            actor_loss.append(-step_data["log_prob"] * advantage.detach())

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss = torch.cat(critic_loss).sum()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss = torch.cat(actor_loss).sum()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.episode_data = [] # Clear episode data

### --- 2.4 DDPG Agent (for Continuous Actions) ---
class DDPGAgent:
    # 注意: DDPG需要一个连续动作空间
    def __init__(self, state_size, max_action_value, seed=0, device='cuda'):
        self.state_size = state_size
        self.max_action = max_action_value
        self.device = torch.device(device)

        # Actor
        self.actor_local = self._create_actor().to(self.device)
        self.actor_target = self._create_actor().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)
        # Critic
        self.critic_local = self._create_critic().to(self.device)
        self.critic_target = self._create_critic().to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-3, weight_decay=1e-2)

        self.memory = deque(maxlen=20000)
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 1e-3
        
        # 复制参数
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
    def _create_actor(self): # State -> Action
        return nn.Sequential(
            nn.Linear(self.state_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh() # 输出在[-1, 1]之间
        )
    def _create_critic(self): # (State, Action) -> Q-Value
        # Critic网络需要接收 state 和 action
        class CriticNet(nn.Module):
            def __init__(self, state_size):
                super().__init__()
                self.fc1 = nn.Linear(state_size, 256)
                self.fc2 = nn.Linear(256 + 1, 128) # +1 for action
                self.fc3 = nn.Linear(128, 1)
            def forward(self, state, action):
                xs = F.relu(self.fc1(state))
                x = torch.cat([xs, action], dim=1)
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        return CriticNet(self.state_size)

    def act(self, state, noise=0.1):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += noise * np.random.randn(1) # 添加高斯噪声进行探索
        action = np.clip(action, -1, 1) # 裁剪
        # 将[-1, 1]的动作映射到[0, max_action]
        return (action + 1) / 2 * self.max_action

    def step(self, state, action, reward, next_state, done):
        # 将[0, max_action]的动作重新缩放回[-1, 1]进行存储
        action_scaled = (action / self.max_action) * 2 - 1
        self.memory.append((state, action_scaled, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            self.learn(random.sample(self.memory, self.batch_size))
            
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # --- 更新 Critic ---
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 更新 Actor ---
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 更新 Target Networks ---
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local, target):
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(self.tau*l_param.data + (1.0-self.tau)*t_param.data)
            
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)