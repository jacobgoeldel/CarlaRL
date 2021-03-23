import math
import numpy as np

import carla
import carla_env


class WaypointManager:
    def __init__(self, world):
        self.world = world
        self.map = self.world.get_map()
        self.waypoints = []

    def generate(self, start_location):
        '''Creates a list of waypoints based on the cars current position'''
        self.waypoints = []

        # get closest waypoint to the location given
        waypoint = self.map.get_waypoint(
            start_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # get a series of waypoints based on the first one and add it to a list.
        # the car will be rewarded for following these waypoints
        for i in range(int(carla_env.WAYPOINT_PATH_LENGTH // carla_env.WAYPOINT_DISTANCE_BETWEEN)):
            self.waypoints.append(waypoint)
            waypoint = waypoint.next(carla_env.WAYPOINT_DISTANCE_BETWEEN).pop()

        # remove the first waypoint, as it is at the vehicles spawn
        self.waypoints.pop(0)

    def update_location(self, vehicle):
        '''Updates the relative x and y to the next waypoint'''
        if len(self.waypoints) != 0:
            # first get the lat and long local to the car
            waypoint_loc = self.map.transform_to_geolocation(
                self.waypoints[0].transform.location)
            local_x = vehicle.latitude - waypoint_loc.longitude
            local_y = vehicle.longitude - waypoint_loc.latitude

            # scale to be larger but still roughly below 1
            # as the model will have an easier time with it then
            local_x *= 10000
            local_y *= 10000

            # rotate based on the orientation to make the position relative
            # a waypoint in front of the car should always on the same axis, no matter what direction the car is facing
            rel_x = local_x * (math.cos(vehicle.orientation)) - \
                local_y * (math.sin(vehicle.orientation))
            rel_y = local_x * (math.sin(vehicle.orientation)) - \
                local_y * (math.cos(vehicle.orientation))

            rel_x = min(1.0, max(rel_x, -1.0))
            rel_y = min(1.0, max(rel_y, -1.0))

            location_list = np.array([rel_x, rel_y])

            self.location = location_list

    def distance_to_point(self, location):
        waypoint_loc = self.waypoints[0].transform.location
        return math.sqrt((waypoint_loc.x - location.x)**2 + (waypoint_loc.y - location.y)**2 + (waypoint_loc.z - location.z)**2)
