import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from NN import Actor
from ACTD0 import train_actd0

def run(actor_path,n_run=4,human=True,ew=True):
    if human:
        env = gym.make("LunarLander-v3",enable_wind=ew,render_mode="rgb_array")
        # Add video recording for every episode
        env = RecordVideo(
            env,
            video_folder=f"ac-agent_ew={ew}",    # Folder to save videos
            name_prefix="eval",               # Prefix for video filenames
            episode_trigger=lambda x: True    # Record every episode
        )

        # Add episode statistics tracking
        env = RecordEpisodeStatistics(env, buffer_length=n_run)
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

    # Print summary statistics
    print(f'\nEvaluation Summary:')
    print(f'Episode durations: {list(env.time_queue)}')
    print(f'Episode rewards: {list(env.return_queue)}')
    print(f'Episode lengths: {list(env.length_queue)}')

    # Calculate some useful metrics
    avg_reward = np.sum(env.return_queue)
    std_reward = np.std(env.return_queue)

    print(f'\nAverage reward: {avg_reward:.2f} ± {std_reward:.2f}')
    print(f'Success rate: {sum(1 for r in env.return_queue if r >= 200) / len(env.return_queue):.1%}')

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    #train_actd0(n_ep=4000)
    run("actor_td0_ew=True.pth")