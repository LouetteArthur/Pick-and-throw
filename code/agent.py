import numpy as np
from stable_baselines3 import SAC, DDPG, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import random
import torch.nn as nn
from related_work.analytic_throw_2d import main

bucket_LENGTH = 0.72
class UnhandledAgentException(Exception):
    pass

implemented_agents = {"goToBucket", "sac", "ddpg", "td3", "ppo"}

# Define the default agent class
class Agent(object):
    def __init__(self, trainable=False):
        self.trainable = trainable

    def act(self):
        pass
    
    def train(self):
        if not self.trainable:
            raise NotImplementedError("This agent is not trainable")
        else:
            pass

## Trivial agent ##
class goToBucketAgent(Agent):
    """
    The agent will go to the bucket and release the object
    """
    def __init__(self):
        super().__init__()

    def act(self, env):
        y_release = env.bucket_place_position[1]
        z_release = env.bucket_place_position[2]
        y_target = y_release
        action = np.array([y_release, z_release, 3, y_target])
        return action
    
# class optimAgent(Agent):
#     def __init__(self, trainable=False):
#         super().__init__(trainable)

#     def act(self, env):
#         error = 0.01
#         pos_obj = env._p.getBasePositionAndOrientation(env.object_id)[0]
#         pr_and_v = main(pos_obj, env.bucket_pos[:2] + (0, ), error)
#         action = np.append(pr_and_v[1:], pr_and_v[1] + abs(pr_and_v[1] - pos_obj[1]))
#         return np.float32(action)

## RL agents ##
class ddpgAgent(Agent):
    def __init__(self):
        super().__init__(trainable=True)
        self.model = None

    def load_model(self, model_path):
        if model_path is not None:
            self.model = DDPG.load(model_path)
            
    def act(self, env):
        if self.model is None:
            raise UnhandledAgentException("No model loaded")
        action = self.model.predict(env.get_observation())
        return action

    def train(self, env, episodes, callback=None, save_path="models/ddpg", **kwargs):
        # Handle action noise
        action_noise_type = kwargs.pop("action_noise")
        action_noise_std = kwargs.pop("action_noise_std")
        if action_noise_type == "NormalActionNoise":
            action_noise = NormalActionNoise(mean=0, sigma=action_noise_std * env.action_space.shape[0])
        elif action_noise_type == "OrnsteinUhlenbeckActionNoise":
            action_noise = OrnsteinUhlenbeckActionNoise(mean=0, sigma=action_noise_std * env.action_space.shape[0])
        else:
            action_noise = None
        kwargs["action_noise"] = action_noise

        policy_kwargs = kwargs.pop("policy_kwargs")
        kwargs["policy_kwargs"] = policy_kwargs

        self.model = DDPG('MlpPolicy', env, tensorboard_log="logs/DDPG", **kwargs)
        self.model.learn(total_timesteps=episodes, callback=callback)
        self.model.save(save_path)

class sacAgent(Agent):
    def __init__(self):
        super().__init__(trainable=True)
        self.model = None

    def load_model(self, model_path):
        if model_path is not None:
            self.model = SAC.load(model_path)
            
    def act(self, env):
        if self.model is None:
            raise UnhandledAgentException("No model loaded")
        random.seed(env.seed)
        action = self.model.predict(env.get_observation())
        return action

    def train(self, env, episodes, callback=None, save_path="models/sac_flexpicker", **kwargs):
        policy_kwargs = kwargs.pop("policy_kwargs")
        kwargs["policy_kwargs"] = policy_kwargs
        self.model = SAC('MlpPolicy', env, verbose=0, **kwargs)
        self.model.learn(total_timesteps=episodes, callback=callback)
        self.model.save(save_path)

class td3Agent(Agent):
    def __init__(self):
        super().__init__(trainable=True)
        self.model = None
    
    def load_model(self, model_path):
        if model_path is not None:
            self.model = TD3.load(model_path)

    def act(self, env):
        if self.model is None:
            raise UnhandledAgentException("No model loaded")
        action = self.model.predict(env.get_observation())
        return action

    def train(self, env, episodes, callback=None, save_path="models/td3_flexpicker", **kwargs):
        # Handle action noise
        action_noise_type = kwargs.pop("action_noise")
        action_noise_std = kwargs.pop("action_noise_std")
        if action_noise_type == "NormalActionNoise":
            action_noise = NormalActionNoise(mean=0, sigma=action_noise_std * env.action_space.shape[0])
        elif action_noise_type == "OrnsteinUhlenbeckActionNoise":
            action_noise = OrnsteinUhlenbeckActionNoise(mean=0, sigma=action_noise_std * env.action_space.shape[0])
        else:
            action_noise = None
        kwargs["action_noise"] = action_noise

        policy_kwargs = kwargs.pop("policy_kwargs")
        kwargs["policy_kwargs"] = policy_kwargs
        self.model = TD3('MlpPolicy', env, tensorboard_log="logs/TD3" , **kwargs)
        self.model.learn(total_timesteps=episodes, callback=callback)
        self.model.save(save_path)

class ppoAgent(Agent):
    def __init__(self):
        super().__init__(trainable=True)
        self.model = None

    def load_model(self, model_path):
        if model_path is not None:
            self.model = PPO.load(model_path)
        
    def act(self, env):
        if self.model is None:
            raise UnhandledAgentException("No model loaded")
        action = self.model.predict(env.get_observation())
        return action

    def train(self, env, episodes, callback=None, save_path="models/ppo_flexpicker", **kwargs):
        policy_kwargs = kwargs.pop("policy_kwargs")
        activation_fn = policy_kwargs.pop("activation_fn")
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]
        policy_kwargs["activation_fn"] = activation_fn
        kwargs["policy_kwargs"] = policy_kwargs
        self.model = PPO('MlpPolicy', env, verbose=1, **kwargs)
        self.model.learn(total_timesteps=episodes, callback=callback)
        self.model.save(save_path)