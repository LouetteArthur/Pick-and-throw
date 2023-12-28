import argparse
import pybullet as p
from environment import TossingFlexpicker
import agent as a
from agent import UnhandledAgentException
import time
from tqdm import tqdm
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd

def agent_control(GUI=False, agent_name='goToBucket', model=None, episodes=10, verbose=False, reward_name='success', seed=None, save_data=None):
    # fix random seed
    if seed is not None:
        env = TossingFlexpicker(GUI=GUI, reward_name=reward_name, seed=seed)
    else:
        env = TossingFlexpicker(GUI=GUI,  reward_name=reward_name)

    # load the agent
    success = []
    rewards = []
    success_reward = []
    action_time = []
    distance_ratio = []
    distance_impact = []
    distance_time_dict = dict()
    if agent_name in a.implemented_agents:
        agent = getattr(a, agent_name + 'Agent')()

        #check if the agent is trainable
        if agent.trainable and model is not None:
            agent.load_model(model)
    else:
        raise UnhandledAgentException("Agent {} is unknown.".format(agent_name))

    if not verbose:
        pbar = tqdm(total=episodes)
    for i in range(1, episodes + 1):
        if verbose:
            print(f"\nEpisode {i} out of {episodes}")

        # run the agent
        terminated = False
        while not terminated:
            if agent.trainable:
                action, _ = agent.act(env)
            else:
                action = agent.act(env)
            observation, reward, terminated, truncated, info = env.step(action)
            success.append(int(info['is_success']))
            rewards.append(reward)
            action_time.append(info['action_time'])
            distance_ratio.append(info['distance_ratio'])
            distance_impact.append(info['distance_impact'])
            if info['is_success']:
                success_reward.append(reward)
            if verbose:
                print("Action: ", action)
                print("Observation: {}\nReward: {}\nTerminated: {}\nInfo: {}".format(observation, reward, terminated, info))
            else:
                pbar.update(1)
        env.reset()
        if GUI:
            time.sleep(1)

    print(f"Success rate: {np.mean(success)} ({np.std(success)})")
    print(f"Mean reward: {np.round(np.mean(rewards),3)} ({np.round(np.std(rewards),3)})")
    if success_reward:
        print(f"Success reward: {np.mean(success_reward)}")
    print(f"Action time: {np.round(np.mean(action_time),3)} ({np.round(np.std(action_time), 3)})")
    print(f"Distance ratio: {np.round(np.mean(distance_ratio),3)} ({np.round(np.std(distance_ratio),3)})")
    print(f"Distance impact: {np.round(np.mean(distance_impact),3)} ({np.round(np.std(distance_impact),3)})")
    #print median of distance impact
    print(f"Median distance impact: {np.median(distance_impact)}")
    if save_data:
        # Create a DataFrame
        df = pd.DataFrame({
            'Success': success,
            'Reward': rewards,
            'Time': action_time,
            'Distance Ratio': distance_ratio,
            'Distance Impact': distance_impact,
        })

        # Save DataFrame to a CSV file
        df.to_csv(f'{save_data}.csv', index=False)

if __name__ == '__main__':
    np.seterr(all='raise')  # define before your code.
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Mode: user (the user control the robot) or agent (an intelligent agent control the robot)
    parser.add_argument('-a', '--agent', type=str, default='goToBucket', help='the agent to control the gripper')
    # Number of episodes
    parser.add_argument('-e', '--episodes', type=int, default=1, help='number of episodes')
    # GUI or direct mode
    parser.add_argument('-g', '--gui', type=int, default=True, help='GUI or direct mode')
    # Verbosity
    parser.add_argument('-v', '--verbose', type=int, default=1, help='verbosity')
    # Model pretrained
    parser.add_argument('-m', '--model', type=str, default=None, help='model to use')
    # reward function
    parser.add_argument('-r', '--reward', type=str, default="success", help='reward function')
    #seed
    parser.add_argument('-s', '--seed', type=int, default=None, help='seed')
    #plot distance vs time
    parser.add_argument('-d', '--save_data', type=str, default=None, help='save the results to a csv')

    # Argument values
    args = parser.parse_args()
    agent = args.agent
    episodes = args.episodes
    gui = args.gui
    verbose = args.verbose
    model = args.model
    reward_name = args.reward
    seed = args.seed
    save_data = args.save_data

    start_time = time.time()
    agent_control(gui, agent, model, episodes, verbose, reward_name, seed, save_data)
    print("--- %s seconds ---" % (time.time() - start_time))

