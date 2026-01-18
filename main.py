import gymnasium as gym
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from network import ArmNet

def transform(obs):
    obs2 = np.empty((2048, 30))
    obs2[:, 0:14:2] = np.cos(obs[:, :7])
    obs2[:, 1:14:2] = np.sin(obs[:, :7])
    obs2[:, 14:] = obs[:, 7:]
    obs2[:, 27:30] = obs[:, 14:17] - obs[:, 20:23]
    obs2[:, 24:27] = obs[:, 14:17] - obs[:, 17:20]
    return obs2


num_envs = 2048
num_timesteps = 256
num_iter = 100
discount_factor = 0.99
GAE_param = 0.95
clip_factor = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
env = gym.make_vec('Pusher-v5', num_envs=num_envs, max_episode_steps=num_timesteps, render_mode=None, reward_near_weight=0.2)
env = gym.wrappers.vector.TransformObservation(env, transform)
env = gym.wrappers.vector.TransformAction(env, lambda i: np.clip(i, -2, 2))
env = gym.wrappers.vector.TransformObservation(env, lambda i: np.clip(i, -5, 5))
env = gym.wrappers.vector.NormalizeReward(env)
env = gym.wrappers.vector.TransformReward(env, lambda i: np.clip(i, -10, 10))

observation, info = env.reset()
num_actions = env.action_space.shape[1]
num_observations = observation.shape[1]

model = ArmNet(num_observations, num_actions).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for iteration in range(num_iter):
    data = {
        "State": torch.zeros((num_timesteps, num_envs, num_observations), device=DEVICE),
        "Value": torch.zeros((num_timesteps + 1, num_envs), device=DEVICE),
        "Reward": torch.zeros((num_timesteps, num_envs), device=DEVICE),
        "Log_Probs": torch.zeros((num_timesteps, num_envs), device=DEVICE),
        "Terminated": torch.zeros((num_timesteps, num_envs), dtype=torch.bool, device=DEVICE),
        "Actions": torch.zeros((num_timesteps, num_envs, num_actions), device=DEVICE),
        "Advantage": torch.zeros((num_timesteps, num_envs), device=DEVICE)
    }
    track_rewards = torch.zeros((num_timesteps, num_envs), device=DEVICE)

    observation, info = env.reset()
    observation = torch.from_numpy(observation).float().to(DEVICE)

    saved = False

    model.eval()
    for t in range(num_timesteps + 1):
        with torch.no_grad():
            actions, log_probs, values = model(observation)

        if t == num_timesteps:
            data["Value"][t] = values.squeeze(1)
            data["Terminated"][-1] = True
            continue

        next_observation, reward, terminated, truncated, info = env.step(np.array(actions.cpu()))

        data["State"][t] = observation
        data["Value"][t] = values.squeeze(1)
        data["Reward"][t] = torch.from_numpy(reward)
        data["Log_Probs"][t] = log_probs
        data["Terminated"][t] = torch.from_numpy(terminated | truncated)
        data["Actions"][t] = actions

        track_rewards[t] = torch.from_numpy(reward) * np.sqrt(env.env.return_rms.var + env.env.epsilon)

        observation = torch.from_numpy(next_observation).float().to(DEVICE)
    print(iteration, track_rewards.mean())

    for i in range(num_timesteps - 1, -1, -1):
        delta = data["Reward"][i] + discount_factor * data["Value"][i + 1] - data["Value"][i]

        data["Advantage"][i] = delta

        if i == num_timesteps - 1:
            continue

        data["Advantage"][i, ~data["Terminated"][i]] += (GAE_param * discount_factor *
                                                         data["Advantage"][i + 1, ~data["Terminated"][i]])

    data["Terminated"][0] = True

    for _ in range(1):
        data["Terminated"][1:] = data["Terminated"][1:] | data["Terminated"][:-1]
    for _ in range(100):
        data["Terminated"][:-1] = data["Terminated"][1:] | data["Terminated"][:-1]

    data["Terminated"] = data["Terminated"].transpose(1, 0).flatten()

    for key, val in data.items():
        skip = True
        match key:
            case "Actions":
                data[key] = val.transpose(1, 0).reshape(-1, num_actions)[~data["Terminated"]]
            case "State":
                data[key] = val.transpose(1, 0).reshape(-1, num_observations)[~data["Terminated"]]
            case "Terminated":
                pass
            case _:
                skip = False

        if skip:
            continue

        data[key] = val[:num_timesteps].transpose(1, 0).flatten()[~data["Terminated"]]

    del data["Terminated"]

    data_loader = DataLoader(
        TensorDataset(*data.values()),
        batch_size=1024, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(10):
        for batch in data_loader:
            state, values, reward, log_probs, actions, advantages = batch

            curr_log_probs, curr_entropy, curr_values = model(state, actions)

            ratio = torch.exp(curr_log_probs - log_probs)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_factor, 1 + clip_factor)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            target_values = advantages + values
            policy_loss += 0.5 * torch.square(curr_values - target_values).mean()

            policy_loss -= 0.0 * curr_entropy.mean()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

    print("")

    torch.save(model.state_dict(), "epoch_" + str(iteration + 1) + ".pt")
