import numpy as np
import random
import math
import cv2

import carla_env
import carla


class Car:
    def __init__(self, world):
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        model_3 = self.blueprint_library.filter("model3")[0]
        self.vehicle = world.spawn_actor(
            model_3, carla.Transform(carla.Location(z=1000.0)))

        # create the camera sensor
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{carla_env.IM_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{carla_env.IM_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"90")

        self.rgb_cam = self.world.spawn_actor(
            self.rgb_cam, carla.Transform(carla.Location(x=2.5, z=0.9)), attach_to=self.vehicle)
        self.rgb_cam.listen(lambda data: self.__process_img(data))
        self.front_camera = None

        # create the collision sensor
        self.colsensor = self.__create_sensor(
            "sensor.other.collision", lambda event: self.__collision_data(event))
        self.collision_hist = []

        # add the senors that will detect when the car has crossed a road line
        self.linesensor = self.__create_sensor(
            "sensor.other.lane_invasion", lambda event: self.__line_invasion_data(event))
        self.line_hist = []

        # create the sensor that will return the cars gps location
        # used for getting relative position to waypoints
        self.gpssensor = self.__create_sensor(
            "sensor.other.gnss", lambda event: self.__gps_data(event))
        self.latitude = 0
        self.longitude = 0

        # create the sensor that is an accelerometer, gyroscope and compass.
        # mainly uses the compass to get waypoint position relative to the car.
        self.imusensor = self.__create_sensor(
            "sensor.other.imu", lambda event: self.__imu_data(event))
        self.orientation = 0

    def __create_sensor(self, sensor_name, sensor_function, sensor_location=carla.Location()):
        sensor = self.blueprint_library.find(sensor_name)
        sensor = self.world.spawn_actor(
            sensor, carla.Transform(sensor_location), attach_to=self.vehicle)
        sensor.listen(sensor_function)
        return sensor

    def reset_location(self):
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

    def apply_input(self, action):
        control = carla.VehicleControl(throttle=carla_env.SPEED)

        if action == 0:  # left
            control.steer = -carla_env.STEER_AMT
        elif action == 1:  # straight
            control.steer = 0
        elif action == 2:  # right
            control.steer = carla_env.STEER_AMT
        else:
            assert False, "Vehicle was given input action outside of it's range"

        self.vehicle.apply_control(control)

    def get_location(self):
        return self.vehicle.get_location()

    def get_transform(self):
        return self.vehicle.get_transform()

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
        i2 = i.reshape((carla_env.IM_HEIGHT, carla_env.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        gray = cv2.cvtColor(i3, cv2.COLOR_RGB2GRAY)
        gray = gray.astype('uint8')
        self.front_camera = gray.reshape(
            (carla_env.IM_HEIGHT, carla_env.IM_WIDTH, 1))
