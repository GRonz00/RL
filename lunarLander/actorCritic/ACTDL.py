import json

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from NN import Actor, Critic


def train_actd_lambda(n_ep=2000, restart=True, lambda_theta=0.9, lambda_w=0.9):
    env = gym.make("LunarLander-v3")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    if not restart:
        actor.load_state_dict(torch.load("actor_td_lambda.pth"))
        critic.load_state_dict(torch.load("critic_td_lambda.pth"))

    optimizer_actor = optim.Adam(actor.parameters(), lr=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-4)
    gamma = 0.99

    rewards = []

    for episode in range(n_ep):
        state, _ = env.reset()
        end_ep, total_reward = False, 0

        # tracce inizializzate a 0
        z_theta = [torch.zeros_like(p) for p in actor.parameters()]
        z_w = [torch.zeros_like(p) for p in critic.parameters()]

        while not end_ep:
            state_tensor = torch.FloatTensor(state)

            # policy con stabilità numerica
            probs = actor(state_tensor)
            probs = torch.clamp(probs, 1e-8, 1.0)
            probs = probs / probs.sum()
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            end_ep = done or truncated
            total_reward += reward

            value = critic(state_tensor)
            next_value = critic(torch.FloatTensor(next_state)) if not done else torch.tensor(0.0)
            delta = reward + gamma * next_value - value
            delta = torch.clamp(delta, -10, 10)


            optimizer_actor.zero_grad()
            log_prob = dist.log_prob(action)
            log_prob.backward(retain_graph=True)
            actor_grads = [p.grad.detach().clone() for p in actor.parameters()]


            optimizer_critic.zero_grad()
            value.backward()
            critic_grads = [p.grad.detach().clone() for p in critic.parameters()]

            # aggiornamento traccie
            for i in range(len(z_theta)):
                z_theta[i] = gamma * lambda_theta * z_theta[i] + actor_grads[i]

            for i in range(len(z_w)):
                z_w[i] = gamma * lambda_w * z_w[i] + critic_grads[i]

            # si azzerano i gradienti prima di accumulare
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            # accumula i gradienti
            for p, z in zip(actor.parameters(), z_theta):
                p.grad = -delta.detach() * z  # segno negativo perché max J
            for p, z in zip(critic.parameters(), z_w):
                p.grad = -delta.detach() * z  # min L = (δ)^2/2

            # ottimizzazione
            optimizer_actor.step()
            optimizer_critic.step()

            state = next_state

        rewards.append(total_reward)
        if (episode+1) % 100 == 0:
            print(f"Episodio {episode+1}: avg reward last 100 ep = {np.mean(rewards[-100:]):.1f}")


        if (episode+1) % 1000 == 0:
            torch.save(actor.state_dict(), "actor_td_lambda.pth")
            torch.save(critic.state_dict(), "critic_td_lambda.pth")
            print("\nModelli salvati")
    with open('tdl_train.json', 'w') as f:
        json.dump(rewards, f)
    torch.save(actor.state_dict(), "actor_td_lambda.pth")
    torch.save(critic.state_dict(), "critic_td_lambda.pth")
    print("Modelli salvati (finali)")
