import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


# Making the Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []

        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (np.array(batch_states), np.array(batch_actions), np.array(batch_rewards), 
                np.array(batch_next_states), np.array(batch_dones))


# Defining a class for PID control
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def control(self, error, error_sum, prev_error):
        return self.Kp * error + self.Ki * error_sum + self.Kd * (error - prev_error)


# Class Implementing Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.actor(state)


# Class Implementing Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.critic(torch.cat([state, action], 1))
    

# Class Implementing DDPG Algorithm
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        self.discount = 0.99
        self.tau = 0.01

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=128):
        # Sampling batch of transitions from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).view(batch_size,1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(1 - done).view(batch_size,1).to(device)

        # Critic loss
        target_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, target_action)
        target_q = reward + done * self.discount * target_q 
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))


# Implementing PID control over DDPG Agent
class DDPG_PID:
    def __init__(self, pid_controller, ddpg_agent):
        self.pid_controller = pid_controller
        self.ddpg_agent = ddpg_agent

    def update_pid(self, state):
        # Adjustmenting from DDPG Agent
        action = self.ddpg_agent.select_action(state)
        self.pid_controller.Kp = action[0]
        self.pid_controller.Ki = action[1]
        self.pid_controller.Kd = action[2]
        return action
        
    def control_system(self, error, error_sum, prev_error, state):
        return [self.update_pid(state), self.pid_controller.control(error, error_sum, prev_error)]


def plot_rewards(episode_rewards, window=25):
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(episode_rewards, label='Reward per Episode')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(episode_rewards)), moving_avg, color='orange', label=f'Mean Reward (last {window})')
    plt.axhline(y=0, color='r', linestyle='--', label='Semi Target Threshold')
    plt.pause(0.001)


# Setting Up The Environment 
env = gym.make('Pendulum-v1',render_mode = None) 
state_dim = env.observation_space.shape[0]
action_dim = 3  # Kp, Ki, Kd
max_action = 10.0
lambda_effort = 0.01
target_update = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pid = PID(Kp=1.0, Ki=0.001, Kd=0.01)
ddpg_agent = DDPG(state_dim, action_dim, max_action)
ddpg_pid = DDPG_PID(pid,ddpg_agent)

episodes = 5000
batch_size = 128
episode_rewards = []

for episode in range(episodes):
    state, info = env.reset() 
    done = False
    total_reward = 0
    desired_setpoint = 0
    error_sum = 0
    prev_error = 0
    duration = 0

    while not done:

        error = - state[1]
        error_sum += error

        [action, control_signal] = ddpg_pid.control_system(error, error_sum , prev_error,state)
        control_signal = np.clip(control_signal, env.action_space.low[0], env.action_space.high[0])
        prev_error = error
        next_state, reward, terminated, truncated, info = env.step([control_signal])
        done = terminated or truncated
        ddpg_pid.ddpg_agent.add_experience(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        duration += 1

        if len(ddpg_pid.ddpg_agent.replay_buffer.storage) > batch_size:
            ddpg_pid.ddpg_agent.train(batch_size)
    
    episode_rewards.append(total_reward)
    plot_rewards(episode_rewards)

env.close()
