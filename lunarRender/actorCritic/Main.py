import gymnasium as gym
import torch

from NN import Actor
from ACTD0 import train_actd0

def run(actor_path,n_run=5):
    env = gym.make("LunarLander-v3", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    actor = Actor(state_dim, action_dim)
    actor.load_state_dict(torch.load(actor_path))
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

        print(f"\nReward totale = {total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    train_actd0()
    #run("actor_td_lambda.pth")