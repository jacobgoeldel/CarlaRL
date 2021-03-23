import carla
import json
import random
import time
import math
import numpy as np
import cv2
from gym import spaces
import gym

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
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        # used to run  car creation code once
        self.has_created_env = False

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / STEPS_PER_SECOND
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 100
        self.world.apply_settings(settings)

    def __create_ego_vehicle(self):
        self.vehicle = self.world.spawn_actor(
            self.model_3, carla.Transform(carla.Location(z=1000.0)))
        self.actor_list.append(self.vehicle)

        # create the camera sensor
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"90")
        self.front_camera = None

        transform = carla.Transform(carla.Location(x=2.5, z=0.9))
        self.sensor = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.__process_img(data))

        # add the sensor that will detect collisions
        self.colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(
            self.colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.__collision_data(event))

        # add the senors that will detect when the car has crossed a road line
        self.linesensor = self.blueprint_library.find(
            "sensor.other.lane_invasion")
        self.linesensor = self.world.spawn_actor(
            self.linesensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.linesensor)
        self.linesensor.listen(lambda event: self.__line_invasion_data(event))

        # create the sensor that will return the cars gps location
        # used for getting relative position to waypoints
        self.gpssensor = self.blueprint_library.find("sensor.other.gnss")
        self.gpssensor = self.world.spawn_actor(
            self.gpssensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.gpssensor)
        self.gpssensor.listen(lambda event: self.__gps_data(event))

        # create the sensor that is an accelerometer, gyroscope and compass.
        # mainly uses the compass to get waypoint position relative to the car.
        self.imusensor = self.blueprint_library.find("sensor.other.imu")
        self.imusensor = self.world.spawn_actor(
            self.imusensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.imusensor)
        self.imusensor.listen(lambda event: self.__imu_data(event))

    def __create(self):
        '''Sets up the training environment'''

        self.actor_list = []

        self.__create_ego_vehicle()

        while self.front_camera is None:
            self.world_tick()
            time.sleep(0.1)

        self.spectator = self.world.get_spectator()

        self.__reset_car_location()
        self.__reset_training_vars()

    def __reset_car_location(self):
        spawn = random.choice(self.map.get_spawn_points())

        self.vehicle.apply_control(carla.VehicleControl(
            throttle=0.0, brake=0.0))  # stop any inputs
        self.vehicle.set_target_velocity(
            carla.Vector3D())  # stop the car from moving
        self.vehicle.set_target_angular_velocity(carla.Vector3D())
        # move to new location with a slight rotation
        # rotation is meant to force it to learn to turn
        new_transform = carla.Transform(spawn.location, carla.Rotation(
            pitch=spawn.rotation.pitch, yaw=spawn.rotation.yaw + random.randrange(-5, 5), roll=spawn.rotation.roll))
        self.vehicle.set_transform(new_transform)

        self.__generate_waypoints(spawn.location)

    def __reset_training_vars(self):
        self.collision_hist = []
        self.line_hist = []
        self.latitude = 0
        self.longitude = 0
        self.orientation = 0
        self.last_distance = 0.0

        self.waypoint_steps = 0

    def reset(self):
        if not self.has_created_env:
            self.__create()
            self.has_created_env = True
        else:
            self.__reset_car_location()
            self.__reset_training_vars()

        self.__update_waypoint_location()

        return self.waypoint_location

    def render():
        pass

    def __generate_waypoints(self, start_location):
        '''Creates a list of waypoints based on the cars current position'''
        self.waypoints = []

        # get closest waypoint to the location given
        waypoint = self.map.get_waypoint(
            start_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # get a series of waypoints based on the first one and add it to a list.
        # the car will be rewarded for following these waypoints
        for i in range(int(WAYPOINT_PATH_LENGTH // WAYPOINT_DISTANCE_BETWEEN)):
            self.waypoints.append(waypoint)
            waypoint = waypoint.next(WAYPOINT_DISTANCE_BETWEEN).pop()

        # remove the first waypoint, as it is at the vehicles spawn
        self.waypoints.pop(0)

    def __collision_data(self, event):
        self.collision_hist.append(event)

    def __line_invasion_data(self, event):
        self.line_hist.append(event)

    def __gps_data(self, event):
        self.latitude = event.latitude
        self.longitude = event.longitude

    def __imu_data(self, event):
        if event.compass != math.nan:
            self.orientation = event.compass

    def __process_img(self, image):
        '''Gets image from camera sensor and transforms it into a grayscale array between 0 and 1'''
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        gray = cv2.cvtColor(i3, cv2.COLOR_RGB2GRAY)
        gray = gray.astype('uint8')
        self.front_camera = gray.reshape(
            (IM_HEIGHT, IM_WIDTH, 1))

    def __update_waypoint_location(self):
        '''Updates the relative x and y to the next waypoint'''
        if len(self.waypoints) != 0:
            # first get the lat and long local to the car
            waypoint_loc = self.map.transform_to_geolocation(
                self.waypoints[0].transform.location)
            local_x = self.longitude - waypoint_loc.longitude
            local_y = self.latitude - waypoint_loc.latitude

            # scale to be larger but still roughly below 1
            # as the model will have an easier time with it then
            local_x *= 10000
            local_y *= 10000

            # rotate based on the orientation to make the position relative
            # a waypoint in front of the car should always on the same axis, no matter what direction the car is facing
            rel_x = local_x * (math.cos(self.orientation)) - \
                local_y * (math.sin(self.orientation))
            rel_y = local_x * (math.sin(self.orientation)) - \
                local_y * (math.cos(self.orientation))

            rel_x = min(1.0, max(rel_x, -1.0))
            rel_y = min(1.0, max(rel_y, -1.0))

            location_list = np.array([rel_x, rel_y])

            self.waypoint_location = location_list

    def update_camera(self):
        '''Moves the spectator camera to follow the car'''
        self.spectator.set_transform(carla.Transform(self.vehicle.get_transform().location + carla.Location(z=15),
                                                     carla.Rotation(pitch=-90)))

    def world_tick(self):
        self.world.tick()

    def reward_function(self, action):
        if len(self.collision_hist) != 0:
            reward = -50.0
            done = True
        else:
            done = False

            def distance_between_transforms(trans1, trans2):
                return math.sqrt((trans1.x - trans2.x)**2 + (trans1.y - trans2.y)**2 + (trans1.z - trans2.z)**2)
            current_position = self.vehicle.get_location()

            distance = distance_between_transforms(
                current_position, self.waypoints[0].transform.location)

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

    def apply_vehicle_input(self, action):
        control = carla.VehicleControl(throttle=SPEED)

        if action == 0:  # left
            control.steer = -STEER_AMT
        elif action == 1:  # straight
            control.steer = 0
        elif action == 2:  # right
            control.steer = STEER_AMT
        else:
            assert False, "Vehicle was given input action outside of it's range"

        self.vehicle.apply_control(control)

    def step(self, action):
        self.world_tick()

        self.update_camera()

        self.apply_vehicle_input(action)
        reward, done = self.reward_function(action)

        # end after the time limit is up
        self.waypoint_steps += 1
        if self.waypoint_steps >= WAYPOINT_STEPS_MAX:
            done = True
            reward = -20.0

        self.__update_waypoint_location()

        info = {}

        return self.waypoint_location, reward, done, info

    def close(self):
        pass
