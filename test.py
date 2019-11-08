#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla

import scipy
import os
import numpy as np

def main():
	# parameters for the gym_carla environment
	params = {
		'number_of_vehicles': 100,
		'number_of_walkers': 0,
		'display_size': 256,  # screen size of bird-eye render
		'max_past_step': 1,  # the number of past steps to draw
		'dt': 0.1,  # time interval between two frames
		'discrete': False,  # whether to use discrete control space
		'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
		'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
		'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
		'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
		'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
		'port': 2000,  # connection port
		'town': 'Town03',  # which town to simulate
		'task_mode': 'roundabout',  # mode of the task, [random, roundabout (only for Town03)]
		'max_time_episode': 1000,  # maximum timesteps per episode
		'max_waypt': 12,  # maximum number of waypoints
		'obs_range': 32,  # observation range (meter)
		'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
		'd_behind': 12,  # distance behind the ego vehicle (meter)
		'out_lane_thres': 2.0,  # threshold for out of lane
		'desired_speed': 8,  # desired speed (m/s)
	}

	# Set gym-carla environment
	env = gym.make('carla-v0', params=params)
	obs = env.reset()

	img_dir = '/media/jianyu/DATA/Codes/imgs'
	if not os.path.exists(img_dir):
		os.makedirs(img_dir)

	i=0

	while True:
		action = [1.0, 0.0]
		obs,r,done,info = env.step(action)

		img = np.concatenate((obs['camera'],obs['lidar'],obs['birdeye']), axis=1)
		scipy.misc.imsave(os.path.join(img_dir, str(i) + '.png'), img)

		i=i+1

		if done:
			obs = env.reset()


if __name__ == '__main__':
	main()