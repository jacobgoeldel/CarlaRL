import train


def reward_function(env):
    if len(env.vehicle.collision_hist) != 0:
        reward = -50.0
        done = True
    else:
        done = False

        current_position = env.vehicle.get_location()
        distance = env.waypoints.distance_to_point(current_position)

        # give a reward if the car went through a waypoint and remove that waypoint
        if distance < train.WAYPOINT_COMPLETED_RANGE:
            reward = 20
            env.waypoints.pop(0)
            env.waypoint_steps = 0  # reset the time
            # end the episode if all of the waypoints have been followed
            if not env.waypoints:
                done = True
        else:  # reward based on how close it is to the next waypoint
            reward = (train.WAYPOINT_DISTANCE_BETWEEN -
                      distance) * 0.2 + 0.1

    return reward, done
