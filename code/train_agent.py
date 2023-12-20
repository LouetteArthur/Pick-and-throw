import yaml
import argparse
import time
import utils
import wandb
import agent as a
import torch as th
import numpy as np
from environment import TossingFlexpicker
from agent import UnhandledAgentException
from wandb.integration.sb3 import WandbCallback
from yaml.loader import SafeLoader
    

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Mode: user (the user control the robot) or agent (an intelligent agent control the robot)
    parser.add_argument('-a', '--agent', type=str, default='user', help='user or an agent')
    # Number of episodes
    parser.add_argument('-e', '--episodes', type=int, default=1000, help='number of episodes')
    # Model pretrained
    parser.add_argument('-m', '--pretrained_model', type=str, default=None, help='model to use')
    # reward function
    parser.add_argument('-r', '--reward', type=str, default="success", help='reward function')
    # save path
    parser.add_argument('-s', '--save_path', type=str, default=None, help='path to save the model')
    # hyperparameters
    parser.add_argument('-hp', '--hyperparams', type=str, default=None, help='path to the hyperparameters')

    th.autograd.set_detect_anomaly(True)
    np.seterr(all="raise") 
    # Argument values
    args = parser.parse_args()
    agent_name = args.agent
    episodes = args.episodes
    model_path = args.pretrained_model
    reward_name = args.reward
    save_path = args.save_path
    hyperparams_name = args.hyperparams

    # start training
    start_time = time.time()
    # load the environment
    env = TossingFlexpicker(GUI=False,reward_name=reward_name)

    # load the agent
    if agent_name in a.implemented_agents:
        agent = getattr(a, agent_name + 'Agent')()
    else:
        raise UnhandledAgentException("Agent {} is unknown.".format(agent_name))

    #check if the agent is trainable
    if not agent.trainable:
        raise UnhandledAgentException("Agent {} is not trainable.".format(agent_name))

    # load the model
    if model_path is not None:
        agent.load_model(model_path)
    
    config = {
    "total_timesteps": episodes,
    "algo_name": agent_name,
    }
    
    # initialize wandb
    run = wandb.init(
    project="Tossing Flexpicker",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_{agent_name}_{hyperparams_name}",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

    # train the agent
    if save_path is None:
        save_path = f"models/{agent_name}_{hyperparams_name}"

    #read the parameters from the config file
    path = f"hyperparams/{agent_name}_{hyperparams_name}.yaml"
    try:
        with open(path, 'r') as f:
            kwargs = yaml.load(f, Loader=SafeLoader)
    except:
        raise Exception("Error while loading config file")

    print(kwargs)
    hyperparams = kwargs['hyperparams']
    agent.train(env, episodes, save_path=save_path, callback=WandbCallback(), **hyperparams)
    
    print("--- %s seconds ---" % (time.time() - start_time))
 
