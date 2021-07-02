#!/usr/bin/env python
import time

import cv2
import gym
import sys


def main():
    from gym_carla.envs.carla_env import CarlaEnv

    # parameters for the gym_carla environment
    params = {
        # carla connection parameters+
        'host': 'localhost',
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate

        # simulation parameters
        'verbose': False,
        'vehicles': 100,  # number of vehicles in the simulation
        'walkers': 0,     # number of walkers in the simulation
        'obs_size': 288,  # sensor width and height
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.025,  # time interval between two frames
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 6,  # desired speed (m/s)
        'speed_reduction_at_intersection': 0.75,
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    }

    env = CarlaEnv(params)
    obs = env.reset()

    try:

        while True:
            action = [2.0, 0.0]
            start = time.time()
            obs, r, done, info = env.step(action)

            fps = 1 / (time.time() - start)
            sys.stdout.write("\r")
            sys.stdout.write(f"[{fps:.1f} fps] rew={r:.2f}")
            sys.stdout.flush()

            cv2.imshow('camera', obs['camera'])
            cv2.waitKey(1)

            if done:
                obs = env.reset()
                done = False
    finally:
        env.reset()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
