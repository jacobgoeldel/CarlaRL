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

WORLD = "town03"
WAYPOINT_STEPS_MAX = 80
IM_WIDTH = 160
IM_HEIGHT = 120
STEER_AMT = 0.3
SPEED = 0.3
WAYPOINT_COMPLETED_RANGE = 1.0
WAYPOINT_DISTANCE_BETWEEN = 5.0
WAYPOINT_PATH_LENGTH = 200.0
STEPS_PER_SECOND = 10.0


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
        self.world = self.client.load_world(WORLD)
        self.map = self.world.get_map()

        # used to run  car creation code once
        self.has_created_env = False

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / STEPS_PER_SECOND
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

    def reward_function(self, action):
        if len(self.vehicle.collision_hist) != 0:
            reward = -50.0
            done = True
        else:
            done = False

            current_position = self.vehicle.get_location()
            distance = self.waypoints.distance_to_point(current_position)

            # give a reward if the car went through a waypoint and remove that waypoint
            if distance < WAYPOINT_COMPLETED_RANGE:
                reward = 20
                self.waypoints.pop(0)
                self.waypoint_steps = 0  # reset the time
                # end the episode if all of the waypoints have been followed
                if not self.waypoints:
                    done = True
            else:  # reward based on how close it is to the next waypoint
                reward = (WAYPOINT_DISTANCE_BETWEEN -
                          distance) * 0.2 + 0.1

        return reward, done

    def step(self, action):
        self.world_tick()

        self.update_camera()

        self.vehicle.apply_input(action)
        reward, done = self.reward_function(action)

        # end after the time limit is up
        self.waypoint_steps += 1
        if self.waypoint_steps >= WAYPOINT_STEPS_MAX:
            done = True
            reward = -20.0

        self.waypoints.update_location(self.vehicle)

        info = {}

        return self.waypoints.location, reward, done, info

    def close(self):
        pass
