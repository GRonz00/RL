import json

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from NN import QNetwork


state_dim = 8
action_dim = 4
gamma = 0.99
lr = 1e-3
batch_size = 64
memory_size = 100000
epsilon_start = 0.01#1.0
epsilon_end = 0.01
epsilon_decay = 0.995
num_episodes = 4000
target_update = 10

# Replay buffer
memory = deque(maxlen=memory_size)

def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def sample_batch():
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)
    actions = np.array(actions)
    return (torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones))


def train_dqn(reset=True,ew=False):

    env = gym.make("LunarLander-v3",enable_wind=ew)

    q_net = QNetwork(state_dim, action_dim)
    if not reset:
        q_net.load_state_dict(torch.load(f"dqn.pth"))
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)

    epsilon = epsilon_start

    trs = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            #Epsilon greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.FloatTensor(state))
                    action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            store_transition(state, action, reward, next_state, done)
            state = next_state

            #Aggiorna rete Q
            if len(memory) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = sample_batch()
                q_values = q_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_net(next_states_b).max(1)[0]
                    td_target = rewards_b + gamma * max_next_q * (1 - dones_b)
                loss = nn.MSELoss()(q_values, td_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Aggiorna rete target
        if (episode + 1) % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        trs.append(total_reward)
        if (episode+1)%100==0:
            print(f"Episode {episode+1}: total reward = {np.mean(trs[-100:]):.1f}")
        if (episode+1)%1000==0:
            with open(f'dqn_train_ew={ew}.json', 'w') as f:
                json.dump(trs, f)
            torch.save(q_net.state_dict(), f"dqn_ew={ew}.pth")
            print("Modello salvato")


    with open(f'dqn_train_ew={ew}.json', 'w') as f:
        json.dump(trs, f)
    torch.save(q_net.state_dict(), f"dqn_ew={ew}.pth")
    print("Modello salvato")

def run_dqn(ew=False,human= False):
    if human:
        test_env = gym.make("LunarLander-v3", enable_wind = ew,render_mode="human")
    else:
        test_env = gym.make("LunarLander-v3", enable_wind = ew)
    num_test_episodes = 100
    q_net = QNetwork(state_dim, action_dim)
    q_net.load_state_dict(torch.load(f"dqn_ew={ew}.pth"))

    q_net.eval()
    r = []
    for episode in range(num_test_episodes):
        state, _ = test_env.reset()
        done, truncated = False, False
        total_reward = 0
        timesteps = 0

        while not (done or truncated):
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                action = torch.argmax(q_net(state_tensor)).item()
            state, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            timesteps += 1
        r.append(total_reward)
        #print(f"Test episode {episode+1}: reward = {total_reward:.1f}, timesteps = {timesteps}")
    print(np.mean(r))

    test_env.close()
if __name__ == "__main__":
    #train_dqn(ew=False,reset=False)
    run_dqn(ew=True,human=False)