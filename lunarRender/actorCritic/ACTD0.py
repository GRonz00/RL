import json

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from NN import Actor, Critic


def train_actd0(n_ep=2000,restart = True,ew=False):
    env = gym.make("LunarLander-v3",enable_wind=ew)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    if not restart:
        actor.load_state_dict(torch.load(f"actor_td0_ew={ew}.pth"))
        critic.load_state_dict(torch.load(f"critic_td0_ew={ew}.pth"))

    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-4)
    optimizer_critic = optim.Adam(critic.parameters(), lr=5e-4)
    gamma = 0.99

    rewards = []
    for episode in range(n_ep):
        state, _ = env.reset()
        end_ep,total_reward = False,0

        while not end_ep:
            state_tensor = torch.FloatTensor(state)

            probs = actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, truncated, info = env.step(action.item())
            end_ep = done or truncated
            total_reward += reward

            value = critic(state_tensor)
            next_value = critic(torch.FloatTensor(next_state))

            td_target = reward + gamma * next_value * (1 - int(done))
            delta = td_target - value


            critic_loss = delta.pow(2).mean()
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()


            actor_loss = -dist.log_prob(action) * delta.detach()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            state = next_state
        rewards.append(total_reward)
        if (episode+1)%100==0:
            print(f"Episodio {episode+1}: avg reward last 100 ep = {np.mean(rewards[-100:]):.1f}")

        if (episode+1)%1000==0:
            torch.save(actor.state_dict(), f"actor_td0_ew={ew}.pth")
            torch.save(critic.state_dict(), f"critic_td0_ew={ew}.pth")
            print("\nModelli salvati")


    # --- Salvataggio dei pesi ---
    with open(f'td0_train_ew={ew}.json', 'w') as f:
        json.dump(rewards, f)
    torch.save(actor.state_dict(), f"actor_td0_ew={ew}.pth")
    torch.save(critic.state_dict(), f"critic_td0_ew={ew}.pth")
    print("Modelli salvati")

