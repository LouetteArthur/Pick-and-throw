import numpy as np
from stable_baselines3 import SAC, DDPG, TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import random
import torch.nn as nn
from related_work.time_opt_PaT import main

bucket_LENGTH = 0.72
class UnhandledAgentException(Exception):
    pass

implemented_agents = {"pap", "timeOpt", "sac", "ddpg", "td3", "ppo"}

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
class papAgent(Agent):
    """
    The agent will go to the bucket and release the object
    """
    def __init__(self):
        super().__init__()

    def act(self, env):
        y_target = env.bucket_place_position[1]
        z_target = env.bucket_place_position[2]
        y_release = y_target
        action = np.array([y_release, 10, z_target, y_target])
        action = env.action_normalizer.normalize(action)
        return action
    
class timeOptAgent(Agent):
    def __init__(self, trainable=False):
        super().__init__(trainable)

    def act(self, env):
        error = 0.001
        pos_obj = tuple(env.init_obs[:2]) + (0.08,)
        x_r, y_r, z_r, v = main(pos_obj, env.bucket_place_position[:2] + (0.08, ), error)
        action = [y_r, v, 2*z_r-pos_obj[2], 2*y_r - pos_obj[1]]
        action = env.action_normalizer.normalize(action)
        return np.float32(action)

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
        action_noise_type = kwargs.get("action_noise", None)
        action_noise_std = kwargs.get("action_noise_std", None)
        #remove action noise std from kwargs
        kwargs.pop("action_noise_std", None)
        if action_noise_type and action_noise_std:
            if action_noise_type == "NormalActionNoise":
                action_noise = NormalActionNoise(mean=0, sigma=action_noise_std * env.action_space.shape[0])
            elif action_noise_type == "OrnsteinUhlenbeckActionNoise":
                action_noise = OrnsteinUhlenbeckActionNoise(mean=0, sigma=action_noise_std * env.action_space.shape[0])
            else:
                action_noise = None
            kwargs["action_noise"] = action_noise

        policy_kwargs = kwargs.get("policy_kwargs", None)
        if policy_kwargs is not None:
            kwargs["policy_kwargs"] = policy_kwargs

        self.model = DDPG('MlpPolicy', env, tensorboard_log="logs/DDPG", verbose=1, **kwargs)
        self.model.learn(total_timesteps=episodes, log_interval=1000, callback=callback)
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
        policy_kwargs = kwargs.get("policy_kwargs", None)
        if policy_kwargs is not None:
            kwargs["policy_kwargs"] = policy_kwargs 
        self.model = SAC('MlpPolicy', env, verbose=1, **kwargs)
        self.model.learn(total_timesteps=episodes, log_interval=1000, callback=callback)
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
        action_noise_type = kwargs.get("action_noise", None)
        action_noise_std = kwargs.get("action_noise_std", None)
        #remove action noise std from kwargs
        kwargs.pop("action_noise_std", None)
        if action_noise_type and action_noise_std:
            if action_noise_type == "NormalActionNoise":
                action_noise = NormalActionNoise(mean=0, sigma=action_noise_std * env.action_space.shape[0])
            elif action_noise_type == "OrnsteinUhlenbeckActionNoise":
                action_noise = OrnsteinUhlenbeckActionNoise(mean=0, sigma=action_noise_std * env.action_space.shape[0])
            else:
                action_noise = None
            kwargs["action_noise"] = action_noise

        policy_kwargs = kwargs.get("policy_kwargs", None)
        if policy_kwargs is not None:
            kwargs["policy_kwargs"] = policy_kwargs 
        self.model = TD3('MlpPolicy', env, verbose=1, **kwargs)
        self.model.learn(total_timesteps=episodes, log_interval=1000, callback=callback)
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
        policy_kwargs = kwargs.get("policy_kwargs", None)
        if policy_kwargs is not None:
            activation_fn = policy_kwargs.pop("activation_fn", None)
            if activation_fn:
                activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}.get(activation_fn, None)
                if activation_fn:
                    policy_kwargs["activation_fn"] = activation_fn
            kwargs["policy_kwargs"] = policy_kwargs
        self.model = PPO('MlpPolicy', env, verbose=1, **kwargs)
        self.model.learn(total_timesteps=episodes, callback=callback)
        self.model.save(save_path)