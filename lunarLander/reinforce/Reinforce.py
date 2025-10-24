import json

from Net import Network
import gymnasium as gym
import numpy as np
import torch
def train_reinforce(n_ep=2000, restart=True):
    env = gym.make("LunarLander-v3")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    net = Network(state_dim, action_dim)
    if not restart:
        net.load_state_dict(torch.load("reinforce.pth"))
    learning_rate = 1e-4
    gamma = 0.99
    eps = 1e-6  # small number for mathematical stability


    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    r=[]

    for episode in range(n_ep):
        state, _ = env.reset()
        end_ep,total_reward = False,0
        probs = []  # Stores probability values of the sampled action
        rewards = []  # Stores the corresponding rewards

        while not end_ep:
            state_tensor = torch.FloatTensor(state)

            p = net(state_tensor)
            dist = torch.distributions.Categorical(p)
            action = dist.sample()

            next_state, reward, done, truncated, info = env.step(action.item())

            end_ep = done or truncated
            total_reward += reward
            rewards.append(reward)

            log_prob = dist.log_prob(action)
            probs.append(log_prob)

            state = next_state
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in rewards[::-1]:
            running_g = R + gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        log_probs = torch.stack(probs).squeeze()

        # Update the loss with the mean log probability and deltas
        # Now, we compute the correct total loss by taking the sum of the element-wise products.
        loss = -torch.sum(log_probs * deltas)

        # Update the policy network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        r.append(total_reward)
        if (episode+1)%100==0:
            print(f"Episodio {episode+1}: avg reward last 100 ep = {np.mean(r[-100:]):.1f}")


        if (episode+1)%1000==0:
            torch.save(net.state_dict(), "reinforce.pth")
            print("\nModelli salvati")

    with open('reinforce_train.json', 'w') as f:
        json.dump(r, f)
    # --- Salvataggio dei pesi ---
    torch.save(net.state_dict(), "reinforce.pth")
    print("Modelli salvati")
