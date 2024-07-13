import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces

class RobotEnv(gym.Env): # Class inherits from gym.Env, so we can use with OpenAI gym.
    def __init__(self):
        super(RobotEnv, self).__init__ # Calls the constructor of OpenAI gym, to ensure proper intialization.
        self.action_space = spaces.Box(-1, 1, (3,))
        self.observation_space = spaces.Box(-np.inf, np.inf, (9,))
        self.physics = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Access PyBullet URDF files.
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)