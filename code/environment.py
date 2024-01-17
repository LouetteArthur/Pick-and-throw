import utils
import pybullet as p
import time
import pybullet_data
import numpy as np
from flexpicker import Flexpicker
import random
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
from pybullet_utils import bullet_client as bc
import matplotlib.pyplot as plt
import yaml

#Robot parameters
MAX_SPEED_FLEXPICKER = 10
MAX_ROT_SPEED= 2880*np.pi/180
MIN_SPEED_FOR_THROW = 0.25
MAX_ACCELERATION_FLEXPICKER = 100

#Simulation parameters
TIME_STEP = 1./240.
MAX_SEED = 100000000
BUCKET_WIDTH = 0.21
CONVEYOR_WIDTH = 1.4
BUCKET_LENGTH = 0.72
CONVEYOR_HEIGHT = 0.17 # used to put the conveyor at z=0
MAX_STEP_SIMULATION = 5/TIME_STEP # 5 seconds max for one throw

class TossingFlexpicker(Env):
    def __init__(self, GUI=True, domain_randomization=True, reward_name="success", seed=None):
        """
        GUI: True for GUI, False for headless
        user_control: True for user control, False for automatic control
        """
        super(TossingFlexpicker, self).__init__()

        # read the boundaries of the workspace in the config file (yaml)
        with open("config/workspace.yaml", 'r') as workspace_file:
            workspace = yaml.safe_load(workspace_file)

        self.workspace = workspace['workspace']
        self.x_min = self.workspace['x_min']
        self.x_max = self.workspace['x_max']
        self.y_min = self.workspace['y_min']
        self.y_max = self.workspace['y_max']
        self.z_min = self.workspace['z_min']
        self.z_max = self.workspace['z_max']
        self.gripper_opening_reaction_time = self.workspace['gripper_opening_reaction_time']
        self.gripper_opening_time = self.workspace['gripper_opening_time']

        # max 5 seconds
        self.max_step_simulation = MAX_STEP_SIMULATION
        
        # set random seed if provided
        if seed is not None: 
            self.seed = seed
        else:
            self.seed = np.random.randint(0, MAX_SEED)
        random.seed(self.seed)

        # Connect to the physics server
        self.GUI = GUI
        self._p = bc.BulletClient(connection_mode=p.GUI if self.GUI else p.DIRECT, options="")#or self._p.DIRECT for non-graphical version
        self.physicsClientId = -1
        self.ownsPhysicsClient = 0

        # Set the context space (x_object, y_object, x_bucket, y_bucket)
        self.observation_space = Box(low=np.float32(np.array([self.x_min, self.y_min, self.x_min, 0])),
                                     high=np.float32(np.array([self.x_max, self.y_max, self.x_max, self.y_max])),
                                     shape=(4,), dtype=np.float32)

        # Set the action space (y_release, speed, z_target, y_target)
        low = np.float32(np.array([self.y_min, MIN_SPEED_FOR_THROW, self.z_min, self.y_min]))
        high = np.float32(np.array([self.y_max, MAX_SPEED_FLEXPICKER, self.z_max, self.y_max]))
        self.action_normalizer = utils.ActionSpaceNormalizer(low, high)
        low_n = self.action_normalizer.normalize(low)
        high_n = self.action_normalizer.normalize(high)
        self.action_space = Box(low=low_n, high=high_n, shape=(4,), dtype=np.float32)

        # Set the reward function
        self.reward_func = utils.Rewardfunction(reward_name, self)

        # Set the gravity
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        self._p.setGravity(0,0,-9.81)

        # Configure debug visualizer flags to remove grid
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_PLANAR_REFLECTION, 0)
        self.planeId = self._p.loadURDF("plane.urdf", [0,0,-CONVEYOR_HEIGHT])
        self._p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-0, cameraPitch=-35, cameraTargetPosition=[0,0,0])

        # Randomized Environment
        self.domain_randomization = domain_randomization

        # Load the conveyor and the buckets
        self.load_conveyor_and_bucket()

        # Load the object and the robot
        self.load_object_and_robot()

        # create the variables for the toss and the reward associated urdf
        self.throw_and_reward_variable()

    def load_conveyor_and_bucket(self):
        if self.domain_randomization:
            place_position_x = random.uniform(self.x_min, self.x_max)
            self.random_y_conveyor = random.uniform(-self.y_max, 0)
            place_position_y = self.y_max + self.random_y_conveyor
            place_position_z = 0.15
            self.bucket_place_position = (place_position_x, place_position_y, place_position_z)
            self.conv_id = self._p.loadURDF("urdf/convoyer.urdf", [0,-CONVEYOR_WIDTH/2 + self.y_max - 0.1 + self.random_y_conveyor,-CONVEYOR_HEIGHT])
        else:
            place_position_x = random.uniform(self.x_min, self.x_max)
            place_position_y = self.y_max
            place_position_z = 0.15
            self.bucket_place_position = (place_position_x, place_position_y, place_position_z) 
            self.conv_id = self._p.loadURDF("urdf/convoyer.urdf", [0,-CONVEYOR_WIDTH/2 + self.y_max - 0.1,-CONVEYOR_HEIGHT])
        
        if self.domain_randomization:
            bucket_offset_y = self.y_max - 0.04 + BUCKET_LENGTH/2 + self.random_y_conveyor# 0.05 is used to create a gap between the conveyor and the bin to avoid sliding objects
            bucket_orientation = self._p.getQuaternionFromEuler([0,0,-np.pi/2])
            bucket_pos = [place_position_x, bucket_offset_y, -CONVEYOR_HEIGHT]
            self.bucket_id = self._p.loadURDF("urdf/bucket.urdf", bucket_pos, bucket_orientation)
        else:
            bucket_offset_y = self.y_max + BUCKET_LENGTH/2 - 0.04 # 0.08 is used to create a gap between the conveyor and the bin to avoid sliding objects
            bucket_orientation = self._p.getQuaternionFromEuler([0,0,-np.pi/2])
            bucket_pos = [place_position_x, bucket_offset_y, -CONVEYOR_HEIGHT]
            self.bucket_id = self._p.loadURDF("urdf/bucket.urdf", bucket_pos, bucket_orientation)
        
    def load_object_and_robot(self):
        # Load the object
        if self.domain_randomization:
            self.object_id = utils.spawn_cube_on_conveyor(conveyor_pos=self.random_y_conveyor, workspace=self.workspace, seed=self.seed, physicsClient=self._p)
        else:
            self.object_id = utils.spawn_cube_on_conveyor(conveyor_pos=0, workspace=self.workspace, seed=self.seed, physicsClient=self._p)
        self.bucket_pos, _ = self._p.getBasePositionAndOrientation(self.bucket_id)
        self.cube_init_position, cube_orientation = self._p.getBasePositionAndOrientation(self.object_id)
        
        #gripper variable
        if self.domain_randomization:
            gripper_opening_reaction_time_noise = 0.002
            gripper_opening_time_noise = 0.005
            #make it always positive
            self.noisy_gripper_opening_reaction_time = np.clip(random.normalvariate(self.gripper_opening_reaction_time, gripper_opening_reaction_time_noise), 0, None)
            self.noisy_gripper_opening_time = np.clip(random.normalvariate(self.gripper_opening_time, gripper_opening_time_noise), 0, None)
        else:
            self.noisy_gripper_opening_reaction_time = self.gripper_opening_reaction_time
            self.noisy_gripper_opening_time = self.gripper_opening_time

        # Load the robot
        initial_height_flexpicker = 0.6
        self.robot =  Flexpicker(gripper_opening_time=self.noisy_gripper_opening_time, position=self.cube_init_position[:2] +(initial_height_flexpicker,), orientation=p.getQuaternionFromEuler([0,np.pi,cube_orientation[2]-np.pi/2]), GUI=self.GUI, physicsClient=self._p)
        self.robot.grasp(self.object_id, self.bucket_place_position)

    def throw_and_reward_variable(self):
        # reset the variables for the toss and the reward associated
        self.has_thrown = False
        self.init_obs = self.get_observation()
        self.release_position = self.cube_init_position
        self.distance_cube_bucket = np.round(np.linalg.norm(np.array(self.init_obs[:2]) - np.array(self.bucket_place_position[:2])), 3)
        self.action_time = 0
        self.distance_ratio = 0
        self.distance_impact_to_bucket = 0

    def legal_action(self, action):
        """
        Check if the action is legal and maps it to the legal action space if not
        """
        epsilon = 0.001
        if action[0] < self._p.getLinkState(self.robot.id, self.robot.end_effector_id)[0][1]:
            action[0] = self._p.getLinkState(self.robot.id, self.robot.end_effector_id)[0][1]+0.01

        if action[0] > self.bucket_place_position[1]:
            action[0] =  self.bucket_place_position[1]-epsilon
        
        if action[2] > self.z_max:
            action[2] = self.z_max-epsilon
        
        if action[2] < self.z_min:
            action[2] = self.z_min+epsilon

        if action[3] > self.bucket_place_position[1]:
            action[3] =  self.bucket_place_position[1]-epsilon

        if action[3] < action[0]:
            action[3] = action[0]
        return action
        
    def step_simulation(self):
        self.max_step_simulation -= 1
        if not self.has_thrown and not self.max_step_simulation:
            if self.GUI:
                print("out of time")
            self.robot.open_gripper()

        self._p.stepSimulation()
        if self.GUI:
            time.sleep(1./240.)

    def step(self, action):
        """
        action: (x_release, speed, z_target, x_target) for End Effector Position Control
        """
        action = self.action_normalizer.denormalize(action)
        self.action = self.legal_action(action)

        # compute the release position
        init_pos, init_orn = self._p.getLinkState(self.robot.id, self.robot.end_effector_id)[0:2]
        z_target = self.action[2]
        y_target = self.action[3]
        m_x = (self.bucket_place_position[0] - init_pos[0]) / (self.bucket_place_position[1] - init_pos[1])
        offset_x = self.bucket_place_position[0] - m_x * self.bucket_place_position[1]
        x_target = m_x * y_target + offset_x
        if x_target > self.x_max:
            x_target = self.x_max
            #change z and y to be the intersection between the plane x=self.x_max and the line between the initial position and the target position
            z_target = init_pos[2] + (z_target - init_pos[2])/(x_target - init_pos[0])*(self.x_max - init_pos[0])
            y_target = init_pos[1] + (y_target - init_pos[1])/(x_target - init_pos[0])*(self.x_max - init_pos[0])
        if x_target < self.x_min:
            x_target = self.x_min
            #change z and y to be the intersection between the plane x=self.x_min and the line between the initial position and the target position
            z_target = init_pos[2] + (z_target - init_pos[2])/(x_target - init_pos[0])*(self.x_min - init_pos[0])
            y_target = init_pos[1] + (y_target - init_pos[1])/(x_target - init_pos[0])*(self.x_min - init_pos[0])
        target_pos = (x_target, y_target, z_target)

        # compute the target position on the line between the initial position and the release position
        y_release = self.action[0]
        if y_release > y_target:
            y_release = y_target - 0.01

        x_release = m_x*y_release + offset_x
        m_z = (target_pos[2] - init_pos[2])/(target_pos[1] - init_pos[1])
        offset_z = target_pos[2] - m_z*target_pos[1]
        z_release = m_z*y_release + offset_z
        release_pos = (x_release, y_release, z_release)
        #check that release and target position are in workspace
        assert release_pos[0] <= self.x_max and release_pos[0] >= self.x_min, "release position x is not in workspace"
        assert release_pos[1] <= self.y_max and release_pos[1] >= self.y_min, "release position y is not in workspace"
        assert release_pos[2] <= self.z_max and release_pos[2] >= self.z_min, "release position z is not in workspace"
        assert target_pos[0] <= self.x_max and target_pos[0] >= self.x_min, "target position x is not in workspace"
        assert target_pos[1] <= self.y_max and target_pos[1] >= self.y_min, "target position y is not in workspace"
        assert target_pos[2] <= self.z_max and target_pos[2] >= self.z_min, "target position z is not in workspace"

        # compute yaw
        yaw = np.arcsin(np.dot(np.array(self.bucket_place_position[:2]) - np.array(init_pos[:2]), np.array([0, 1])) / np.linalg.norm(np.array(self.bucket_place_position[:2]) - np.array(init_pos[:2])))
        if self.bucket_place_position[0] < init_pos[0]:
            yaw = -yaw
        else:
            yaw += np.pi

        # compute linear trajectory
        target_orn = self._p.getQuaternionFromEuler([0,0,yaw])
        self.speed = self.action[1]
        scaling_factor = self.action[1]/MAX_SPEED_FLEXPICKER
        lin_pos, orn, velocities = utils.ctraj_pilz_KDL(init_pos, init_orn, target_pos, target_orn, MAX_SPEED_FLEXPICKER, MAX_ACCELERATION_FLEXPICKER, scaling_factor, 1, MAX_ROT_SPEED, TIME_STEP)
        # plt.plot(np.sqrt(velocities[:, 0]**2+velocities[:, 1]**2+velocities[:, 2]**2))
        # #plot a an horizontal line at the desired action speed of the robot
        # plt.plot(np.ones(velocities.shape[0])*self.action[2])
        # plt.show()
        # #plot in 3D the trajectory
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(lin_pos[:, 0], lin_pos[:, 1], lin_pos[:, 2])
        # #plot the target position
        # ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='r', marker='o')
        # #plot the release position
        # ax.scatter(release_pos[0], release_pos[1], release_pos[2], c='g', marker='o')
        # #plot the initial position
        # ax.scatter(init_pos[0], init_pos[1], init_pos[2], c='b', marker='o')
        # #plot the bucket position
        # ax.scatter(self.bucket_place_position[0], self.bucket_place_position[1], 0.15, c='y', marker='o')
        # #plot the trajectory of the object from release position following a ballistic motion until it hits the ground
        # t = np.linspace(0, 1, 100)
        # x = release_pos[0] + velocities[int((velocities.shape[0])/2)][0]*t
        # y = release_pos[1] + velocities[int((velocities.shape[0])/2)][1]*t
        # z = release_pos[2] + velocities[int((velocities.shape[0])/2)][2]*t - 0.5*9.81*t**2
        # ax.plot(x, y, z)
        # print("desired action after orocos calculation", lin_pos[-1] + (np.linalg.norm(velocities[-1][:3]),))
        # plt.show()
        delay = round(self.noisy_gripper_opening_reaction_time/TIME_STEP)
        self.max_time_step = len(lin_pos)
        for i in range(self.max_time_step):
            speed = np.concatenate((velocities[i][:3], [velocities[i][5]]))
            action = tuple(np.concatenate((lin_pos[i], [orn[i][2]]))) + (speed,)
            self.robot.move(action, control_method='position')
            self.step_simulation()
            # throw the object when the position is reached + The gripper openning delay
            pos = self._p.getBasePositionAndOrientation(self.object_id)[0]
            if not self.has_thrown and pos[1] > release_pos[1]:
                delay -= 1
                if not delay:
                    self.action_time = i*TIME_STEP
                    self.robot.release()
                    self.has_thrown = True
                    self.release_position, _ = self._p.getBasePositionAndOrientation(self.object_id)
                    self.distance_release = np.round(np.linalg.norm(np.array(self.release_position[:2]) - np.array(self.init_obs[:2])), 3)

            if self.has_thrown:
                self.distance_ratio = np.clip(self.distance_release/self.distance_cube_bucket, 0, 1)
                reward, terminated = self.get_reward_and_is_terminated()
                if terminated:
                    return self.get_observation(), reward, terminated, False, {"is_success": self.success(), "action_time": self.action_time, "distance_ratio": self.distance_ratio, "distance_impact": self.distance_impact_to_bucket}

        if not self.has_thrown:
            self.action_time = self.max_time_step*TIME_STEP
            action = tuple(np.concatenate((lin_pos[-1], [orn[-1][2]])))+ (speed,)
            self.robot.move(action, control_method='position')
            while delay>0:
                delay -= 1
                self.action_time += TIME_STEP
                self.step_simulation()
            self.robot.release()
            self.has_thrown = True
            self.release_position, _ = self._p.getBasePositionAndOrientation(self.object_id)
            self.distance_release = np.round(np.linalg.norm(np.array(self.release_position[:2]) - np.array(self.init_obs[:2])), 3)
        # wait for the object to fall
        terminated = False
        while not terminated:
            for _ in range(10):
                self.step_simulation()
            self.distance_ratio = np.clip(self.distance_release/self.distance_cube_bucket, 0, 1)
            reward, terminated = self.get_reward_and_is_terminated()
        return self.get_observation(), reward, terminated, False, {"is_success": self.success(), "action_time": np.round(self.action_time, 3), "distance_ratio": self.distance_ratio, "distance_impact": self.distance_impact_to_bucket}
    

    def success(self):
        """
        IF THE OBJECT HAS BEEN THROWN, checks if the object has fallen in the target bucket
        """
        object_aabb = self._p.getAABB(self.object_id)
        bucket_aabb = self._p.getAABB(self.bucket_id)
        in_bucket=False
        if object_aabb[0][0] > bucket_aabb[0][0] and object_aabb[1][0] < bucket_aabb[1][0]:
            if object_aabb[0][1] > bucket_aabb[0][1] and object_aabb[1][1] < bucket_aabb[1][1]:
                in_bucket = True
                self.distance_impact_to_bucket = 0
        return True if self._p.getContactPoints(self.object_id, self.bucket_id, linkIndexB=-1) and in_bucket else False

    def missed(self):
        """
        IF THE OBJECT HAS BEEN THROWN, checks if the object has not fallen in the target bucket
        """
        lin_velocity, _ = self._p.getBaseVelocity(self.object_id)
        stuck_on_target_bucket = self._p.getContactPoints(self.object_id, self.bucket_id) and np.linalg.norm(lin_velocity) < 0.01
        if self._p.getContactPoints(self.object_id) and not self._p.getContactPoints(self.object_id, self.bucket_id) and not self._p.getContactPoints(self.object_id, self.robot.id):
            #extract the distance between the object and the bucket
            self.distance_impact_to_bucket = np.round(np.linalg.norm(np.array(self._p.getContactPoints(self.object_id)[0][5][:2]) - np.array(self.bucket_place_position[:2])), 3)
            return True
        elif stuck_on_target_bucket:
            self.distance_impact_to_bucket = np.round(np.linalg.norm(np.array(self._p.getContactPoints(self.object_id)[0][5][:2]) - np.array(self.bucket_place_position[:2])), 3)
            return True

    def get_reward_and_is_terminated(self):
        """
        return:
            reward: float value
            terminated: bool (True if the episode is terminated)
        """
        if self.success():
            return self.reward_func.get_reward(success=True), True
        
        if self.missed():
            return self.reward_func.get_reward(success=False), True

        if self.max_step_simulation <= 0:
            return self.reward_func.get_reward(success=False), True
        
        return 0, False

    def get_observation(self):
        """
        observation: (x_object, y_object, x_bucket, y_bucket)
        """
        position, _ = self._p.getBasePositionAndOrientation(self.object_id)
        return np.float32(np.array((position[:2] + (self.bucket_place_position[0],) + (self.bucket_place_position[1],))))

    def reset(self, seed=None, options=None):
        if (self.physicsClientId < 0):
            self.ownsPhysicsClient = True

        self.physicsClientId = self._p._client
        if self.GUI:
            # remove all urdf
            self._p.removeBody(self.object_id)
            self._p.removeBody(int(self.bucket_id))
            self._p.removeBody(self.conv_id)
            self._p.removeBody(self.robot.id)
        else :
            self._p.resetSimulation()
            self._p.setGravity(0,0,-9.81)
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
            self.planeId = self._p.loadURDF("plane.urdf", [0,0,-CONVEYOR_HEIGHT])

        # reset the max time step
        self.max_step_simulation = MAX_STEP_SIMULATION

        # select a random seed
        if seed is not None: 
            self.seed = seed
        else:
            self.seed = np.random.randint(0, MAX_SEED)
        random.seed(self.seed)

        # Configure debug visualizer flags to remove grid
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_PLANAR_REFLECTION, 0)
        self.planeId = self._p.loadURDF("plane.urdf", [0,0,-CONVEYOR_HEIGHT])

        self.load_conveyor_and_bucket()
        self.load_object_and_robot()
        self.throw_and_reward_variable()
            
        return self.get_observation(), {}

    def close(self):
        if (self.ownsPhysicsClient):
            if (self.physicsClientId >= 0):
                self._p.disconnect()
        self.physicsClientId = -1
        

