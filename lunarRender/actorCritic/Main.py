import gymnasium as gym
import numpy as np
import torch

from NN import Actor
from ACTD0 import train_actd0

def run(actor_path,n_run=100,human=False,ew=True):
    if human:
        env = gym.make("LunarLander-v3", render_mode="human",enable_wind=ew)
    else:
        env = gym.make("LunarLander-v3",enable_wind=ew)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor = Actor(state_dim, action_dim)
    actor.load_state_dict(torch.load(actor_path))
    r= []
    for _ in range(n_run):
        state, _ = env.reset()
        end_ep = False
        total_reward = 0


        while not end_ep :
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                probs = actor(state_tensor)
                action = torch.argmax(probs).item()
            state, reward, done, truncated, info = env.step(action)
            end_ep = done or truncated
            total_reward += reward

        #print(f"Reward totale = {total_reward:.1f}")
        r.append(total_reward)
    print(f"Reward episodio medio: {np.mean(r)}")

    env.close()

if __name__ == "__main__":
    #train_actd0(n_ep=4000)
    run("actor_td0_ew=True.pth")