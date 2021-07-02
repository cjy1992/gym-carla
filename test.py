#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import time

import cv2
import gym
import sys


def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 100,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.025,  # time interval between two frames
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'reduction_at_intersection': 0.75,
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    }
    from gym_carla.envs.carla_env import CarlaEnv

    # Set gym-carla environment
    # env = gym.make('carla-v0', params=params)
    env = CarlaEnv(params)
    obs = env.reset()

    try:

        while True:
            action = [2.0, 0.0]
            start = time.time()
            obs, r, done, info = env.step(action)

            fps = 1 / (time.time() - start)
            sys.stdout.write("\r")
            sys.stdout.write(f"[{fps:.1f} fps]")
            sys.stdout.flush()

            cv2.imshow('camera', obs['camera'])
            cv2.waitKey(1)

            if done:
                obs = env.reset()
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
