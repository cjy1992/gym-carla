#!/usr/bin/env python
import time
import sys
import numpy as np


def main():
    from gym_carla.envs.carla_pid_env import CarlaPidEnv

    # parameters for the gym_carla environment
    params = {
        # carla connection parameters+
        'host': 'localhost',
        'port': 2000,  # connection port
        'town': 'Town01',  # which town to simulate
        'traffic_manager_port': 8000,

        # simulation parameters
        'verbose': False,
        'vehicles': 100,  # number of vehicles in the simulation
        'walkers': 0,     # number of walkers in the simulation
        'obs_size': 288,  # sensor width and height
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.025,  # time interval between two frames
        'reward_weights': [0.3, 0.3, 0.3],
        'continuous_speed_range': [0.0, 6.0],  # continuous acceleration range
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

    env = CarlaPidEnv(params)
    env.reset()

    try:
        while True:
            # target speed, steer
            action = [6.0, 0.0]
            start = time.time()
            obs, r, done, info = env.step(action)
            speed_mag = np.linalg.norm(obs["speed"])

            fps = 1 / (time.time() - start)
            sys.stdout.write("\r")
            sys.stdout.write(f"[{fps:.1f} fps] speed={speed_mag:.2f} rew={r:.2f}")
            sys.stdout.flush()

            if done:
                obs = env.reset()
    finally:
        env.reset()


if __name__ == '__main__':
    main()
