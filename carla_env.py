import json
import random
import time
import math
import numpy as np
from gym import spaces
import gym

import carla
from car import Car
from waypoints import WaypointManager
from reward_function import reward_function
import train


class CarlaGym(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        self.spec = None
        self.metadata = None
        self.num_envs = 1

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(60.0)
        self.world = self.client.get_world()
        self.world = self.client.load_world(train.WORLD)
        self.map = self.world.get_map()

        # used to run  car creation code once
        self.has_created_env = False

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / train.STEPS_PER_SECOND
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 100
        self.world.apply_settings(settings)

    def __create(self):
        '''Sets up the training environment'''

        self.actor_list = []

        self.vehicle = Car(self.world)
        self.waypoints = WaypointManager(self.world)

        while self.vehicle.front_camera is None:
            self.world_tick()
            time.sleep(0.1)

        self.spectator = self.world.get_spectator()

        self.vehicle.reset_location()
        self.waypoints.generate(self.vehicle.get_location())
        self.__reset_training_vars()

    def __reset_training_vars(self):
        self.last_distance = 0.0
        self.waypoint_steps = 0

    def reset(self):
        if not self.has_created_env:
            self.__create()
            self.has_created_env = True
        else:
            self.vehicle.reset_location()
            self.waypoints.generate(self.vehicle.get_location())
            self.__reset_training_vars()

        self.waypoints.update_location(self.vehicle)

        return self.waypoints.location

    def render():
        # needed for training script, but carla handles rendering for us
        pass

    def update_camera(self):
        '''Moves the spectator camera to follow the car'''
        self.spectator.set_transform(carla.Transform(self.vehicle.get_transform().location + carla.Location(z=15),
                                                     carla.Rotation(pitch=-90)))

    def world_tick(self):
        self.world.tick()

    def step(self, action):
        self.world_tick()

        self.update_camera()

        self.vehicle.apply_input(action)
        reward, done = reward_function(self)

        # end after the time limit is up
        self.waypoint_steps += 1
        if self.waypoint_steps >= train.WAYPOINT_STEPS_MAX:
            done = True
            reward = -20.0

        self.waypoints.update_location(self.vehicle)

        info = {}

        return self.waypoints.location, reward, done, info

    def close(self):
        pass
