import numpy as np
import pybullet as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent import papAgent
from matplotlib import pyplot as plt
from tqdm import tqdm
from PyKDL import get_lin_trajectory

def ctraj_pilz_KDL(pos0, orn0,
               pos1, orn1,
               max_trans_vel,
               max_trans_acc,
               vel_scaling_factor,
               acc_scaling_factor,
               max_rot_vel,
               sampling_time,
               settle_steps = 0):
   """

      Args:
         pos0: the starting position [x, y, z]
         orn0: the starting orientation in quaternion [x, y, z, w]
         pos1: the goal position [x, y, z]
         orn1: the goal orientation in quaternion [x, y, z, w]
         max_trans_vel: the maximum translational velocity of the robot in m/s
         max_trans_acc: the maximum translational acceleration of the robot in (m^2)/s
         vel_scaling_factor: the proportion of max_trans_vel we want the robot to reach during the LIN move
         acc_scaling_factor: the proportion of max_trans_vel we want the robot to reach during the LIN move
         max_rot_vel: the maximum rotational acceleration of the robot in (rad^2)/s
         sampling_time: the time interval at which sample a point of the trajectory
         settle_steps: the number of point that should be added at the end of the trajectory. This is a trick to
                       allow any dynamic reaction of the environment after the LIN move to be simulated, as a
                       simulation step will be run for each point of the trajectory.

      Returns:
         linear_pos: a 3xM array containing the x, y and z coordinates of the sampled LIN trajectory
         orns: a 3xM array containing the roll, pitch and yaw of the sampled LIN trajectory
         velocities: a 6xM array containing the x,y,z,roll,pitch,yaw velocities of the sampled LIN trajectory

   """
   start_pose = pos0+p.getEulerFromQuaternion(orn0)
   goal_pose = pos1+p.getEulerFromQuaternion(orn1)
   traj = get_lin_trajectory(start_pose,
                             goal_pose,
                             max_trans_vel,
                             max_trans_acc,
                             vel_scaling_factor,
                             acc_scaling_factor,
                             max_rot_vel,
                             sampling_time,
                             settle_steps)

   linear_pos = traj[1:4,:].T
   linear_vel = traj[7:10,:].T
   orns = traj[4:7,:].T
   angular_vel = traj[10:13,:].T
   velocities = np.concatenate([linear_vel, angular_vel], axis=1)

   return linear_pos, orns, velocities

def _compute_target_joint_velocities(self, target_velocities):
    # Compute the position and velocities of the joints
    cur_joint_states = p.getJointStates(self.robot_id,
                                            self.get_robot_main_joints_ids() + self.get_finger_ids(),
                                            physicsClientId=self.cid)
    cur_joint_positions = list(np.asarray(cur_joint_states, dtype=object)[:, 0])
    cur_joint_velocities = list(np.asarray(cur_joint_states, dtype=object)[:, 1])
    appliedJointMotorTorque = list(np.asarray(cur_joint_states, dtype=object)[:, 3])

    # Compute the jacobian
    lin_jac, ang_jac = p.calculateJacobian(self.robot_id,
                                                self.get_robot_endeffector_id(),
                                                localPosition=[0,0,0],
                                                objPositions=cur_joint_positions,
                                                objVelocities=cur_joint_velocities,
                                                objAccelerations=appliedJointMotorTorque,
                                                physicsClientId=self.cid)
    jacobian = np.array(lin_jac + ang_jac)[:, 0:len(self.get_robot_main_joints_ids())]
    end_effector_vel = np.array([target_velocities]).T
    target_joint_velocities = np.matmul(np.linalg.pinv(jacobian), end_effector_vel).T[0]
    return target_joint_velocities

def spawn_cube_on_conveyor(conveyor_pos, workspace, seed=None, physicsClient=None):
    """
    Source: https://github.com/erwincoumans/bullet3/blob/master/examples/pybullet/examples/shiftCenterOfMass.py
    Spawns a cube on the conveyor
    """
    if seed is not None:
        np.random.seed(seed)

    cube_size = np.random.uniform(0.03, 0.06)
    meshScale = [cube_size, cube_size, cube_size]
    visualShift = [0, 0, 0]
    collisionShift = [0, 0, 0]

    inertiaShiftX = np.random.uniform(-cube_size/4, cube_size/4)
    inertiaShiftY = np.random.uniform(-cube_size/4, cube_size/4)
    inertiaShiftZ = np.random.uniform(-cube_size/4, cube_size/4)
    inertiaShift = [inertiaShiftX, inertiaShiftY, inertiaShiftZ]

    # spawn the cube at a random location in the workspace of the robot (radius = 0.8m) 
    # and on the conveyor (width = 1.4m and length = 2m) but not too close to the edge (5cm from the edge max) 
    x = np.random.uniform(workspace['x_min'] + cube_size/2, workspace['x_max'] - cube_size/2)
    y = np.random.uniform(workspace['y_min'] + cube_size/2 + conveyor_pos, workspace['y_max'] - 0.1 - cube_size/2 + conveyor_pos)
    z = cube_size/2
    
    mass = np.random.uniform(0.01, 0.5)

    visualShapeId = physicsClient.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName="cube.obj",
                                        rgbaColor=np.concatenate((np.random.uniform(0, 1, 3), [1])),
                                        specularColor=[0, 0, 0],
                                        visualFramePosition=visualShift,
                                        meshScale=meshScale)
    collisionShapeId = physicsClient.createCollisionShape(shapeType=p.GEOM_MESH,
                                            fileName="cube.obj",
                                            collisionFramePosition=collisionShift,
                                            meshScale=meshScale)

    object_id = physicsClient.createMultiBody(baseMass=mass,
                                baseInertialFramePosition=inertiaShift,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=[x, y, z],
                                useMaximalCoordinates=False)
    return object_id
    
class ActionSpaceNormalizer:
    """
    Normalizes the action space to [-1, 1]
    """
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def normalize(self, action):
        return 2 * (action - self.low) / (self.high - self.low) - 1
    
    def denormalize(self, action):
        return (action + 1) * (self.high - self.low) / 2 + self.low

class PickAndPlaceReward(nn.Module):
    '''
    This is a simple neural network that takes in the observation 
    and outputs the time it takes to reach the bucket
    '''
    def __init__(self):
        super(PickAndPlaceReward, self).__init__()
        self.fc1 = nn.Linear(4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def train(self, env, episodes, save_path=None, logs=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(device))
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        loss_fn = nn.MSELoss()
        agent = papAgent()
        losses = []
        for episode in range(episodes):
            env.reset()
            done = False
            while not done:
                observation = np.array(env.get_observation())
                observation = torch.tensor(np.array(observation), dtype=torch.float32).to(device)
                action = agent.act(env)
                _, _, done, _, info = env.step(action)
                optimizer.zero_grad()
                action_time = torch.tensor(np.array(info['action_time'], dtype=np.float32)).unsqueeze(0).to(device)
                pred = self.forward(observation)
                loss = loss_fn(pred, action_time)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            if episode % 100 == 0:
                #if logs:
                    # write losses to csv
                    # with open("learning_curves/PickAndPlaceReward_losses.csv", "a") as f: # change "w" to "a"
                    #     f.write(str(losses[-1]) + ", ")
                print("Episode: {}, Loss: {}".format(episode, loss.item()))
            if episode % 1000 == 0 and save_path is not None:
                self.save(save_path)
        if save_path is not None:
            self.save(save_path)

    def evaluation(self, env, episodes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(device))
        self.to(device)
        agent = papAgent()
        errors = []
        pbar = tqdm(total=episodes)
        for episode in range(episodes):
            env.reset()
            done = False
            while not done:
                observation = np.array(env.init_obs)
                observation = torch.tensor(np.array(observation), dtype=torch.float32).to(device)
                action = agent.act(env)
                _, _, done,_, info = env.step(action)
                action_time = torch.tensor(np.array(info['action_time'], dtype=np.float32)).unsqueeze(0).to(device)
                pred = self.forward(observation)
                error = (abs(pred - action_time)).item()
                errors.append(error)
            pbar.update(1)
        
        plt.hist(errors, bins=100)
        print("Mean error: {}".format(np.mean(errors)))
        print("Std error: {}".format(np.std(errors)))
        print("Max error: {}".format(np.max(errors)))
                       
class UnhandledRewardException(Exception):
    pass

class Rewardfunction():
    def __init__(self,reward_name, env) -> None:
        self.implemented_rewards = {"success": self.success_reward,
                                    "success_time_and_distance": self.success_time_and_distance_reward,
                       "success_and_time": self.success_and_time_reward}
        self.env = env
        self.reward_name = reward_name

        if reward_name not in self.implemented_rewards:
            raise UnhandledRewardException
        
        if reward_name == "success_and_time" or reward_name == "success_time_and_distance":
            self.pickAndPlaceReward = PickAndPlaceReward()
            self.pickAndPlaceReward.load_state_dict(torch.load("models/PaP_reward.pt"))

    def get_reward(self, success):
        reward_func = self.implemented_rewards[self.reward_name]
        return reward_func(success)

    def success_and_time_reward(self, success):
        if success:
            pred = self.pickAndPlaceReward(torch.tensor(self.env.init_obs))
            reward = pred - self.env.action_time
            reward = reward.detach().numpy()[0]
            return reward
        else:
            return -self.env.action_time
        
    def success_reward(self, success):
        if success:
            return 1
        else:
            return 0
        
    def success_time_and_distance_reward(self, success):
        if success:
            pred = self.pickAndPlaceReward(torch.tensor(self.env.init_obs))
            reward = pred - self.env.action_time
            reward = reward.detach().numpy()[0]
            return reward
        else:
            reward = - self.env.action_time -  self.env.distance_impact_to_bucket
            return reward

