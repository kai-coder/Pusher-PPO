import gymnasium as gym
import numpy as np
from torch import nn
import torch
import seaborn as sb
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

class ArmNet(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        div = 2
        self.actor = nn.Sequential(
            nn.Linear(30, int(128//div)),
            nn.ReLU(),

            nn.Linear(int(128//div), int(128//div)),
            nn.ReLU(),

            nn.Linear(int(128//div), int(128//div)),
            nn.ReLU(),
            
            nn.Linear(int(128//div), action_space)
        )
        self.critic = nn.Sequential(
            nn.Linear(30, int(128//div)),
            nn.ReLU(),

            nn.Linear(int(128//div), int(128//div)),
            nn.ReLU(),
            
            nn.Linear(int(128//div), 1)
        )
        self.log_std = nn.Parameter(torch.zeros((1, action_space)), requires_grad=True)

    def forward(self, x):
        return self.actor(x), self.log_std, self.critic(x)


def transform(obs):
    obs2 = np.empty((32, 30))
    obs2[:, 0:14:2] = np.cos(obs[:,:7])
    obs2[:, 1:14:2] = np.sin(obs[:,:7])
    obs2[:, 14:] = obs[:,7:]
    obs2[:, 27:30] = obs[:,14:17] - obs[:,20:23]
    obs2[:, 24:27] = obs[:,14:17] - obs[:,17:20]
    return obs2

%%time
num_envs = 32
num_timesteps = 2048
num_iter = 100
beta = 0.99
alpha = 0.95

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
env = gym.make_vec('Pusher-v5', num_envs=num_envs, max_episode_steps=num_timesteps, render_mode=None)
env = gym.wrappers.vector.TransformObservation(env, transform)
env = gym.wrappers.vector.TransformAction(env, lambda i: np.clip(i, -2, 2))
env.observation_space = gym.spaces.Box(-np.inf, np.inf, (num_envs, 30), np.float64)
env = gym.wrappers.vector.NormalizeObservation(env)
env.obs_rms = gym.wrappers.utils.RunningMeanStd(
            shape=(num_envs, 30), dtype=np.float32
        )
env = gym.wrappers.vector.TransformObservation(env, lambda i: np.clip(i, -5, 5))
env = gym.wrappers.vector.NormalizeReward(env)
env = gym.wrappers.vector.TransformReward(env, lambda i: np.clip(i, -10, 10))

model = ArmNet(7).to(DEVICE)


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

for iteration in range(num_iter):
    observation, info = env.reset()
    
    saved=False
    rewards_t = torch.zeros((num_timesteps, num_envs), device=DEVICE)
    rewards_t2 = torch.zeros((num_timesteps, num_envs), device=DEVICE)
    value_t = torch.zeros((num_timesteps + 1, num_envs), device=DEVICE)
    terminated_t = torch.zeros((num_timesteps, num_envs), dtype=bool, device=DEVICE)
    log_probs_t = torch.zeros((num_timesteps, num_envs), device=DEVICE)
    actions_t = torch.zeros((num_timesteps, num_envs, 7), device=DEVICE)
    states_t = torch.zeros((num_timesteps, num_envs, 30), device=DEVICE)
    advantages_t = torch.zeros((num_timesteps, num_envs), device=DEVICE)
    
    for t in range(num_timesteps):
        model.eval()
        observation = torch.from_numpy(observation).float().to(DEVICE)
        
        with torch.no_grad():
            mean, std, values = model(observation)
            m = torch.distributions.normal.Normal(mean, torch.exp(std.expand_as(mean)))
            action = m.sample()
            
        states_t[t] = observation
        
        observation, reward, terminated, truncated, info = env.step(np.array(action.cpu()))
        
        terminate = torch.from_numpy(terminated | truncated)
        value_t[t] = values.squeeze(1)
        rewards_t[t] = torch.from_numpy(reward)
        rewards_t2[t] = torch.from_numpy(reward) * np.sqrt(env.env.return_rms.var + env.env.epsilon)
        terminated_t[t] = terminate
        log_probs_t[t] = m.log_prob(action).sum(axis=1)
        actions_t[t] = action
    print(iteration, rewards_t2.mean())
    observation = torch.from_numpy(observation).float().to(DEVICE)
    
    with torch.no_grad():
        mean, std, values = model(observation)
    
    value_t[-1] = values.squeeze(1)
    terminated_t[-1] = 1

    for i in range(num_timesteps-1, -1, -1):
        delta_t = rewards_t[i] + beta * value_t[i + 1] - value_t[i]
        advantages_t[i, terminated_t[i]] = delta_t[terminated_t[i]]
        if i == num_timesteps - 1:
            continue
        advantages_t[i, ~terminated_t[i]] = delta_t[~terminated_t[i]] + alpha * beta * advantages_t[i+1, ~terminated_t[i]]
    terminated_t[0] = 1
    
    for i in range(50):
        terminated_t[1:] = terminated_t[1:] | terminated_t[:-1]
    for i in range(500):
        terminated_t[:-1] = terminated_t[1:] | terminated_t[:-1]


    terminated_t = terminated_t.transpose(1, 0).flatten()
    value_t = value_t[:-1].transpose(1, 0).flatten()[~terminated_t]
    rewards_t = rewards_t.transpose(1, 0).flatten()[~terminated_t]
    log_probs_t = log_probs_t.transpose(1, 0).flatten()[~terminated_t]
    actions_t = actions_t.transpose(1, 0).reshape(-1, 7)[~terminated_t]
    states_t = states_t.transpose(1, 0).reshape(-1, 30)[~terminated_t]
    advantages_t = advantages_t.transpose(1, 0).flatten()[~terminated_t]

    if (len(value_t) < 128):
        continue
    
    
    advantages_data_loader = DataLoader(
                    TensorDataset(value_t, rewards_t, log_probs_t, 
                                  actions_t, states_t, advantages_t),
                    batch_size=128, shuffle=True,drop_last=True)

    model.train()
    for epoch in range(10):
        losses = np.zeros(3, dtype=np.float64)
        t = 0
        for i in advantages_data_loader:
            
            v, r, p, a, s, ad = i
    
            mean, std, values = model(s)
            
            m = torch.distributions.normal.Normal(mean, torch.exp(std.expand_as(mean)))
            ratio = torch.exp(m.log_prob(a).sum(axis=1) - p)
            val = ad + v
    
            policy_loss_1 = ad * ratio
            alpha = 0.2
            policy_loss_2 = ad * torch.clamp(ratio, 1 - alpha, 1 + alpha)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    
            policy_loss += 0.5 * torch.square(values - val).mean()
            #policy_loss -= 0.0 * m.entropy().sum(1).mean()
            
            optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                losses[0] += -torch.min(policy_loss_1, policy_loss_2).mean().item()
                losses[1] += 0.5 * torch.square(values - val).mean().item()
                losses[2] += m.entropy().sum(1).mean().item()
                t += 1
        if epoch %3 ==0:
            print(losses/t)
        scheduler.step()
    print("")
        
    torch.save(model.state_dict(), "epoch_" + str(iteration + 1) + ".pt")