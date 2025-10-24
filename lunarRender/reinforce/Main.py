import gymnasium as gym
import numpy as np
import torch

from Baseline import train_reinforce_baseline
from Net import Network


def run(actor_path,n_run=10,human=False):
    if human:
        env = gym.make("LunarLander-v3", render_mode="human")
    else:
        env = gym.make("LunarLander-v3", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    net = Network(state_dim, action_dim)
    net.load_state_dict(torch.load(actor_path))
    r = []
    for _ in range(n_run):
        state, _ = env.reset()
        end_ep = False
        total_reward = 0

        while not end_ep :
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                probs = net(state_tensor)
                action = torch.argmax(probs).item()
            state, reward, done, truncated, info = env.step(action)
            end_ep = done or truncated
            total_reward += reward
        r.append(total_reward)
        print(f"\nReward totale = {total_reward:.1f}")
    print(f"Reward inferenza medio: {np.mean(r)}")
    env.close()

if __name__ == "__main__":
    train_reinforce_baseline(n_ep=5000)
    #run("policy.pth")