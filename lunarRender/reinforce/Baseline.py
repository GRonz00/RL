import json

from BaseNet import PolicyNet,ValueNet
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

def train_reinforce_baseline(n_ep=5000, gamma=0.99, lr_policy=1e-4, lr_value=5e-4):
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)

    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=lr_policy)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=lr_value)

    eps = 1e-8
    rewards_per_ep = []

    for episode in range(n_ep):
        state, _ = env.reset()
        done = False

        log_probs = []
        states = []
        rewards = []

        while not done:
            state_t = torch.FloatTensor(state)
            probs = policy_net(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(dist.log_prob(action))
            states.append(state_t)
            rewards.append(reward)

            state = next_state


        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # normalizzazione
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # aggiornamento
        policy_losses = []
        value_losses = []

        for log_prob, state_t, Gt in zip(log_probs, states, returns):
            value = value_net(state_t)
            delta = Gt - value.detach()  # vantaggio
            policy_losses.append(-log_prob * delta)
            value_losses.append(F.mse_loss(value,Gt))

        optimizer_policy.zero_grad()
        policy_loss = torch.stack(policy_losses).sum()
        policy_loss.backward()
        optimizer_policy.step()


        optimizer_value.zero_grad()
        value_loss = torch.stack(value_losses).sum()
        value_loss.backward()
        optimizer_value.step()

        total_reward = sum(rewards)
        rewards_per_ep.append(total_reward)

        if (episode+1) % 100 == 0:
            avg_r = np.mean(rewards_per_ep[-100:])
            print(f"Episode {episode+1}: avg reward (last 100) = {avg_r:.1f}")
        if (episode+1) % 1000 == 0:
            torch.save(policy_net.state_dict(), "policy.pth")
            torch.save(value_net.state_dict(), "value.pth")


    with open('rei_basel_train.json', 'w') as f:
        json.dump(rewards_per_ep, f)
    print("Training completato.")
    return policy_net, value_net
