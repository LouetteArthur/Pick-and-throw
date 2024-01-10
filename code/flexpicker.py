import pybullet as p
import time
import numpy as np
from collections import namedtuple

class Flexpicker:
    def __init__(self, gripper_opening_time, position=[0,0,1], orientation=p.getQuaternionFromEuler([0,3.141,0]),GUI=True, physicsClient=None):
        # The flexpicker is only represented by a gripper for now
        self._p = physicsClient
        self.id = self._p.loadURDF("urdf/flexpicker.urdf", position, orientation)
        self.pos = position
        self.ori = orientation
        self.__parse_joint_info__()
        self.gripper_range = [0, 0.04]
        self.end_effector_id = 6
        self.GUI = GUI
        self.gripper_opening_time = gripper_opening_time

    def __parse_joint_info__(self):
        numJoints = self._p.getNumJoints(self.id)
        jointInfo = namedtuple('jointInfo',
            ['id','name','type','damping','friction','lowerLimit','upperLimit','maxForce','maxVelocity','controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = self._p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != self._p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                self._p.setJointMotorControl2(self.id, jointID, self._p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID,jointName,jointType,jointDamping,jointFriction,jointLowerLimit,
                            jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
            self.joints.append(info)

        self.lower_limits = [info.lowerLimit for info in self.joints if info.controllable]
        self.upper_limits = [info.upperLimit for info in self.joints if info.controllable]
        self.joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable]


    def grasp(self, object_id, bucket_place_position):
        # grasp an object
        self.open_gripper()
        while not self.is_gripper_open():
            self._p.stepSimulation()
            if self.GUI:
                time.sleep(1./1000.)
        init_pos, _ = self._p.getBasePositionAndOrientation(object_id)
        position_to_grasp = tuple(list(init_pos) + np.array([0, 0, 0.1]))
        self.move(position_to_grasp + (self._p.getBasePositionAndOrientation(object_id)[0][2]-np.pi/2,), "position")
        self.grasp_cons_id = self._p.createConstraint(self.id, self.end_effector_id, object_id, -1, self._p.JOINT_FIXED, [0,0,0], [0,0,0], [0,0,0])
        self.close_gripper(object_id)
        while not self.is_gripper_closed(object_id):
            self._p.stepSimulation()
            if self.GUI:
                time.sleep(1./1000.)

        # compute yaw
        yaw = np.arcsin(np.dot(np.array(bucket_place_position[:2]) - np.array(init_pos[:2]), np.array([0, 1])) / np.linalg.norm(np.array(bucket_place_position[:2]) - np.array(init_pos[:2])))
        if bucket_place_position[0] < init_pos[0]:
            yaw = -yaw
        else:
            yaw += np.pi
        # raise the object
        object_approach_position = 0.08
        self.move(position_to_grasp[:2] +(object_approach_position,) + (yaw,), "position")
        # check if the object is raised and well oriented
        while abs(self._p.getBasePositionAndOrientation(object_id)[0][2] - object_approach_position) > 0.005 and abs(self._p.getBasePositionAndOrientation(object_id)[1][2] - yaw) > 0.005:
            self._p.stepSimulation()
            if self.GUI:
                time.sleep(1./1000.)
        self.grasp = True

    def release(self):
        # release the object
        self._p.removeConstraint(self.grasp_cons_id)
        self.open_gripper()
        self.grasp = False


    def move(self, action, control_method):
        # Move the flexpicker to a given position and orientation
        speed = None
        if control_method == "position":
            if len(action) == 4:
                x, y, z, yaw = action
            elif len(action) == 5:
                x, y, z, yaw, speed = action
            else:
                raise ValueError("Invalid action length")
            position = [x, y, z]
            orientation = self._p.getQuaternionFromEuler([0, 0, yaw])
            joint_poses = self._p.calculateInverseKinematics(self.id, 6, position, orientation, self.lower_limits, self.upper_limits, self.joint_ranges,
            maxNumIterations=100, residualThreshold=1e-6)
        elif control_method == "joint":
            joint_poses = action
        else:
            print("Error: control method not recognized")
            pass

        if speed is None:
            for i, joint_id in enumerate(self.controllable_joints):
                self._p.setJointMotorControl2(self.id, joint_id, self._p.POSITION_CONTROL, joint_poses[i], force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)
        else:
            for i, joint_id in enumerate(self.controllable_joints):
                if joint_id in [0, 1, 2, 3]:
                    self._p.setJointMotorControl2(self.id, joint_id, self._p.POSITION_CONTROL, joint_poses[i], speed[joint_id], force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)
                else:
                    self._p.setJointMotorControl2(self.id, joint_id, self._p.POSITION_CONTROL, joint_poses[i], force=self.joints[joint_id].maxForce, maxVelocity=self.joints[joint_id].maxVelocity)

    def move_gripper(self, open_length):
        # open the gripper to a given length
        max_velocity = self.gripper_range[1] / self.gripper_opening_time
        self._p.setJointMotorControl2(self.id, 7, self._p.POSITION_CONTROL, open_length, force=self.joints[7].maxForce, maxVelocity=max_velocity)
        self._p.setJointMotorControl2(self.id, 8, self._p.POSITION_CONTROL, -open_length, force=self.joints[8].maxForce, maxVelocity=max_velocity)

    def is_gripper_open(self):
        # check if the gripper is open
        error = 1e-4
        return self._p.getJointState(self.id, 7)[0] > self.gripper_range[1] - error and self._p.getJointState(self.id, 8)[0] < -self.gripper_range[1] + error
        
    def open_gripper(self):
        # open the gripper to the max length
        self.move_gripper(self.gripper_range[1])

    def is_gripper_closed(self, object_id = None):
        # check if the gripper is closed
        error = 1e-4
        totally_closed = self._p.getJointState(self.id, 7)[0] < self.gripper_range[0] + error and self._p.getJointState(self.id, 8)[0] > -self.gripper_range[0] - error
        has_grasped = False
        if object_id is not None:
            has_grasped = True if self._p.getContactPoints(self.id, object_id, linkIndexA=7) and self._p.getContactPoints(self.id, object_id, linkIndexA=8) else False
        return totally_closed or has_grasped

    def close_gripper(self, object_id = None):
        # close the gripper to the min length
        self.move_gripper(self.gripper_range[0])

    def get_gripper_opening_length(self):
        # return the current gripper opening length
        return self._p.getJointState(self.id, 7)[0]*2

    def getJointInfo(self, joint_id):
        return self._p.getJointInfo(self.id, joint_id)
