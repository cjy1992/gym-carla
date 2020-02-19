#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from collections import deque
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner


class CarlaEnv(gym.Env):
	"""An OpenAI gym wrapper for CARLA simulator."""

	def __init__(self, params):
		# parameters
		self.display_size = params['display_size']  # rendering screen size
		self.max_past_step = params['max_past_step']
		self.number_of_vehicles = params['number_of_vehicles']
		self.number_of_walkers = params['number_of_walkers']
		self.dt = params['dt']
		self.task_mode = params['task_mode']
		self.max_time_episode = params['max_time_episode']
		self.max_waypt = params['max_waypt']
		self.obs_range = params['obs_range']
		self.lidar_bin = params['lidar_bin']
		self.d_behind = params['d_behind']
		self.obs_size = int(self.obs_range/self.lidar_bin)
		self.out_lane_thres = params['out_lane_thres']
		self.desired_speed = params['desired_speed']
		self.max_ego_spawn_times = params['max_ego_spawn_times']
		self.target_waypt_index = params['target_waypt_index']

		# Destination
		if params['task_mode'] == 'roundabout':
			self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
		else:
			self.dests = None

		# action and observation spaces
		self.discrete = params['discrete']
		self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
		self.n_acc = len(self.discrete_act[0])
		self.n_steer = len(self.discrete_act[1])
		if self.discrete:
			self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
		else:
			self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
			params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
			params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
		self.observation_space = spaces.Dict({'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
			'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
			'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)})

		# Connect to carla server and get world object
		print('connecting to Carla server...')
		client = carla.Client('localhost', params['port'])
		client.set_timeout(10.0)
		self.world = client.load_world(params['town'])
		print('Carla server connected!')

		# Set weather
		self.world.set_weather(carla.WeatherParameters.ClearNoon)

		# Get spawn points
		self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
		self.walker_spawn_points = []
		for i in range(self.number_of_walkers):
			spawn_point = carla.Transform()
			loc = self.world.get_random_location_from_navigation()
			if (loc != None):
				spawn_point.location = loc
				self.walker_spawn_points.append(spawn_point)

		# Create the ego vehicle blueprint
		self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

		# Collision sensor
		self.collision_hist = [] # The collision history
		self.collision_hist_l = 1 # collision history length
		self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

		# Lidar sensor
		self.lidar_data = None
		self.lidar_height = 2.1
		self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
		self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
		self.lidar_bp.set_attribute('channels', '32')
		self.lidar_bp.set_attribute('range', '5000')

		# Camera sensor
		self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
		self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
		self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
		# Modify the attributes of the blueprint to set image resolution and field of view.
		self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
		self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
		self.camera_bp.set_attribute('fov', '110')
		# Set the time in seconds between sensor captures
		self.camera_bp.set_attribute('sensor_tick', '0.02')

		# Set fixed simulation step for synchronous mode
		self.settings = self.world.get_settings()
		self.settings.fixed_delta_seconds = self.dt

		# Record the time of total steps and resetting steps
		self.reset_step = 0
		self.total_step = 0
		
		# Initialize the renderer
		self._init_renderer()

	def reset(self):
		# Clear sensor objects	
		self.collision_sensor = None
		self.lidar_sensor = None
		self.camera_sensor = None

		# Delete sensors, vehicles and walkers
		self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

		# Disable sync mode
		self._set_synchronous_mode(False)

		# Spawn surrounding vehicles
		random.shuffle(self.vehicle_spawn_points)
		count = self.number_of_vehicles
		if count > 0:
			for spawn_point in self.vehicle_spawn_points:
				if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
					count -= 1
				if count <= 0:
					break
		while count > 0:
			if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
				count -= 1

		# Spawn pedestrians
		random.shuffle(self.walker_spawn_points)
		count = self.number_of_walkers
		if count > 0:
			for spawn_point in self.walker_spawn_points:
				if self._try_spawn_random_walker_at(spawn_point):
					count -= 1
				if count <= 0:
					break
		while count > 0:
			if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
				count -= 1

		# Get actors polygon list
		self.vehicle_polygons = []
		vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
		self.vehicle_polygons.append(vehicle_poly_dict)
		self.walker_polygons = []
		walker_poly_dict = self._get_actor_polygons('walker.*')
		self.walker_polygons.append(walker_poly_dict)

		# Spawn the ego vehicle
		ego_spawn_times = 0
		while True:
			if ego_spawn_times > self.max_ego_spawn_times:
				self.reset()

			if self.task_mode == 'random':
				transform = random.choice(self.vehicle_spawn_points)
			if self.task_mode == 'roundabout':
				# self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
				self.start=[52.1,-4.2, 178.66] # static
				transform = self._set_carla_transform(self.start)
			if self._try_spawn_ego_vehicle_at(transform):
				break
			else:
				ego_spawn_times += 1
				time.sleep(0.1)

		# Add collision sensor
		self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
		self.collision_sensor.listen(lambda event: get_collision_hist(event))
		def get_collision_hist(event):
			impulse = event.normal_impulse
			intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
			self.collision_hist.append(intensity)
			if len(self.collision_hist)>self.collision_hist_l:
				self.collision_hist.pop(0)
		self.collision_hist = []

		# Add lidar sensor
		self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
		self.lidar_sensor.listen(lambda data: get_lidar_data(data))
		def get_lidar_data(data):
			self.lidar_data = data

		# Add camera sensor
		self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
		self.camera_sensor.listen(lambda data: get_camera_img(data))
		def get_camera_img(data):
			array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
			array = np.reshape(array, (data.height, data.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]
			self.camera_img = array

		# Update timesteps
		self.time_step=0
		self.reset_step+=1

		# Enable sync mode
		self.settings.synchronous_mode = True
		self.world.apply_settings(self.settings)

		self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
		self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

		# Set ego information for render
		self.birdeye_render.set_hero(self.ego, self.ego.id)

		return self._get_obs()
	
	def step(self, action):
		# Calculate acceleration and steering
		if self.discrete:
			acc = self.discrete_act[0][action//self.n_steer]
			steer = self.discrete_act[1][action%self.n_steer]
		else:
			acc = action[0]
			steer = action[1]

		# Convert acceleration to throttle and brake
		if acc > 0:
			throttle = np.clip(acc/3,0,1)
			brake = 0
		else:
			throttle = 0
			brake = np.clip(-acc/8,0,1)

		# Apply control
		act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
		self.ego.apply_control(act)

		self.world.tick()

		# Append actors polygon list
		vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
		self.vehicle_polygons.append(vehicle_poly_dict)
		while len(self.vehicle_polygons) > self.max_past_step:
			self.vehicle_polygons.pop(0)
		walker_poly_dict = self._get_actor_polygons('walker.*')
		self.walker_polygons.append(walker_poly_dict)
		while len(self.walker_polygons) > self.max_past_step:
			self.walker_polygons.pop(0)

		# route planner
		self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

		# state information
		info = {
			'waypoints': self.waypoints,
			'vehicle_front': self.vehicle_front
		}
		
		# Update timesteps
		self.time_step += 1
		self.total_step += 1

		return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def render(self, mode):
		pass

	def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
		"""Create the blueprint for a specific actor type.

		Args:
			actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

		Returns:
			bp: the blueprint object of carla.
		"""
		blueprints = self.world.get_blueprint_library().filter(actor_filter)
		blueprint_library = []
		for nw in number_of_wheels:
			blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
		bp = random.choice(blueprint_library)
		if bp.has_attribute('color'):
			if not color:
				color = random.choice(bp.get_attribute('color').recommended_values)
			bp.set_attribute('color', color)
		return bp

	def _init_renderer(self):
		"""Initialize the birdeye view renderer.
		"""
		pygame.init()
		self.display = pygame.display.set_mode(
		(self.display_size * 4, self.display_size),
		pygame.HWSURFACE | pygame.DOUBLEBUF)

		pixels_per_meter = self.display_size / self.obs_range
		pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
		birdeye_params = {
			'screen_size': [self.display_size, self.display_size],
			'pixels_per_meter': pixels_per_meter,
			'pixels_ahead_vehicle': pixels_ahead_vehicle
		}
		self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

	def _set_synchronous_mode(self, synchronous = True):
		"""Set whether to use the synchronous mode.
		"""
		self.settings.synchronous_mode = synchronous
		self.world.apply_settings(self.settings)

	def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
		"""Try to spawn a surrounding vehicle at specific transform with random bluprint.

		Args:
			transform: the carla transform object.

		Returns:
			Bool indicating whether the spawn is successful.
		"""
		blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
		blueprint.set_attribute('role_name', 'autopilot')
		vehicle = self.world.try_spawn_actor(blueprint, transform)
		if vehicle is not None:
			vehicle.set_autopilot()
			return True
		return False

	def _try_spawn_random_walker_at(self, transform):
		"""Try to spawn a walker at specific transform with random bluprint.

		Args:
			transform: the carla transform object.

		Returns:
			Bool indicating whether the spawn is successful.
		"""
		walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
		# set as not invencible
		if walker_bp.has_attribute('is_invincible'):
			walker_bp.set_attribute('is_invincible', 'false')
		walker_actor = self.world.try_spawn_actor(walker_bp, transform)

		if walker_actor is not None:
			walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
			walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
			# start walker
			walker_controller_actor.start()
			# set walk to random point
			walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
			# random max speed
			walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
			return True
		return False

	def _try_spawn_ego_vehicle_at(self, transform):
		"""Try to spawn the ego vehicle at specific transform.

		Args:
			transform: the carla transform object.

		Returns:
			Bool indicating whether the spawn is successful.
		"""
		vehicle = None
		# Check if ego position overlaps with surrounding vehicles
		overlap = False
		for idx, poly in self.vehicle_polygons[0].items():
			poly_center = np.mean(poly, axis=0)
			ego_center = np.array([transform.location.x, transform.location.y])
			dis = np.linalg.norm(poly_center - ego_center)
			if dis > 8:
				continue
			else:
				overlap = True
				break

		if not overlap:
			vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

		if vehicle is not None:
			self.ego=vehicle
			return True
			
		return False

	def _set_carla_transform(self, pose):
		"""Get a carla tranform object given pose.

		Args:
			pose: [x, y, yaw].

		Returns:
			transform: the carla transform object
		"""
		transform = carla.Transform()
		transform.location.x = pose[0]
		transform.location.y = pose[1]
		transform.rotation.yaw = pose[2]
		return transform

	def _get_actor_polygons(self, filt):
		"""Get the bounding box polygon of actors.

		Args:
			filt: the filter indicating what type of actors we'll look at.

		Returns:
			actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
		"""
		actor_poly_dict={}
		for actor in self.world.get_actors().filter(filt):
			# Get x, y and yaw of the actor
			trans=actor.get_transform()
			x=trans.location.x
			y=trans.location.y
			yaw=trans.rotation.yaw/180*np.pi
			# Get length and width
			bb=actor.bounding_box
			l=bb.extent.x
			w=bb.extent.y
			# Get bounding box polygon in the actor's local coordinate
			poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
			# Get rotation matrix to transform to global coordinate
			R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
			# Get global bounding box polygon
			poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
			actor_poly_dict[actor.id]=poly
		return actor_poly_dict

	def _get_ego(self):
		""" Get the ego vehicle object
		"""
		return self.ego

	def _get_obs(self):
		"""Get the observations."""
		## Birdeye rendering
		self.birdeye_render.vehicle_polygons = self.vehicle_polygons
		self.birdeye_render.walker_polygons = self.walker_polygons
		self.birdeye_render.waypoints = self.waypoints

		self.birdeye_render.render(self.display)
		# pygame.display.flip()
		birdeye = pygame.surfarray.array3d(self.display)
		birdeye = birdeye[0:self.display_size, :, :]
		birdeye = np.fliplr(np.rot90(birdeye, 3))  # flip to regular view
		birdeye = resize(birdeye, (self.obs_size, self.obs_size))  # resize
		birdeye = birdeye * 255
		birdeye.astype(np.uint8)

		## Lidar image generation
		point_cloud = []
		# Get point cloud data
		for location in self.lidar_data:
			point_cloud.append([location.x, location.y, -location.z])
		point_cloud = np.array(point_cloud)
		# Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
		# and z is set to be two bins.
		y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind+self.lidar_bin, self.lidar_bin)
		x_bins = np.arange(-self.obs_range/2, self.obs_range/2+self.lidar_bin, self.lidar_bin)
		z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]
		# Get lidar image according to the bins
		lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
		lidar[:,:,0] = np.array(lidar[:,:,0]>0, dtype=np.uint8)
		lidar[:,:,1] = np.array(lidar[:,:,1]>0, dtype=np.uint8)
		# Add the waypoints to lidar image
		wayptimg = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240)
		wayptimg = np.expand_dims(wayptimg, axis=2)
		wayptimg = np.fliplr(np.rot90(wayptimg, 3))
		wayptimg.astype(np.uint8)
		# Get the final lidar image
		lidar = np.concatenate((lidar, wayptimg), axis=2)
		lidar = np.flip(lidar, axis=1)
		lidar = np.rot90(lidar, 1)
		lidar = lidar * 255
		# Display lidar image
		lidar_surface = pygame.Surface((self.display_size, self.display_size)).convert()
		lidar_display = resize(lidar, (self.display_size, self.display_size))
		lidar_display = np.flip(lidar_display, axis=1)
		lidar_display = np.rot90(lidar_display, 1)
		pygame.surfarray.blit_array(lidar_surface, lidar_display)
		self.display.blit(lidar_surface, (self.display_size, 0))

		# Display camera image
		camera_surface = pygame.Surface((self.display_size, self.display_size)).convert()
		camera_display = resize(self.camera_img, (self.display_size, self.display_size))
		camera_display = np.flip(camera_display, axis=1)
		camera_display = np.rot90(camera_display, 1)
		camera_display = camera_display * 255
		pygame.surfarray.blit_array(camera_surface, camera_display)
		self.display.blit(camera_surface, (self.display_size * 2, 0))

		camera = pygame.surfarray.array3d(self.display)
		camera = camera[2*self.display_size:3*self.display_size, :, :]
		camera = np.fliplr(np.rot90(camera, 3))  # flip to regular view
		camera = resize(camera, (self.obs_size, self.obs_size))  # resize
		camera = camera * 255
		camera.astype(np.uint8)

		# Vehicle classification and regression maps (requires further normalization)
		vh_clas = np.zeros((self.obs_size, self.obs_size))
		vh_regr = np.zeros((self.obs_size, self.obs_size, 6))

		# Filter out vehicles that are far away from ego
		vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
		keys_to_remove = []
		ego_trans = self.ego.get_transform()
		ego_x = ego_trans.location.x
		ego_y = ego_trans.location.y
		ego_yaw = ego_trans.rotation.yaw/180*np.pi
		for key in vehicle_poly_dict:
			poly = vehicle_poly_dict[key]
			for i in range(poly.shape[0]):
				dx = poly[i, 0] - ego_x
				dy = poly[i, 1] - ego_y
				poly[i, 0] = (dy-dx*np.tan(ego_yaw))*np.cos(ego_yaw)
				poly[i, 1] = dx/np.cos(ego_yaw) + (dy-dx*np.tan(ego_yaw))*np.sin(ego_yaw)
			if poly[0, 0]**2 + poly[0, 1]**2 > 3 * self.obs_range**2:
				keys_to_remove.append(key)
			else:
				vehicle_poly_dict[key] = poly
		for key in keys_to_remove:
			del vehicle_poly_dict[key]

		for i in range(self.obs_size):
			for j in range(self.obs_size):
				if abs(birdeye[i, j, 0] - 0)<20 and abs(birdeye[i, j, 1] - 255)<20 and abs(birdeye[i, j, 2] - 0)<20:
					x = (j - self.obs_size/2) * self.obs_range / self.obs_size
					y = (self.obs_size - i) * self.obs_range / self.obs_size - self.d_behind
					vh_clas[i, j] = 1
					xc = (poly[0, 0] + poly[2, 0]) / 2
					yc = (poly[0, 1] + poly[2, 1]) / 2
					dx = xc - x
					dy = yc - y
					dr = np.sqrt(dx ** 2 + dy ** 2)
					cos = 1 if dr == 0 else dy / dr
					sin = 0 if dr == 0 else dx / dr
					w = np.sqrt((poly[0, 0] - poly[1, 0]) ** 2 + (poly[0, 1] - poly[1, 1]) ** 2)
					l = np.sqrt((poly[2, 0] - poly[1, 0]) ** 2 + (poly[2, 1] - poly[1, 1]) ** 2)
					vh_regr[i, j, :] = np.array([cos, sin, dx, dy, np.log(w), np.log(l)])

		vh_clas_surface = pygame.Surface((self.display_size, self.display_size)).convert()
		vh_clas_display = np.stack([vh_clas, vh_clas, vh_clas], axis=2)
		vh_clas_display = resize(vh_clas_display, (self.display_size, self.display_size))
		vh_clas_display = np.flip(vh_clas_display, axis=1)
		vh_clas_display = np.rot90(vh_clas_display, 1)
		vh_clas_display = vh_clas_display * 255
		pygame.surfarray.blit_array(vh_clas_surface, vh_clas_display)
		self.display.blit(vh_clas_surface, (self.display_size * 3, 0))

		# Display on pygame
		pygame.display.flip()

		# ## State observation,  [waypt_x, waypt_y, speed_ego], where waypt_x and waypt_y 
		# # is the xy position of target waypoint in ego's local coordinate (right-handed), where
		# # ego vehicle is at the origin and heading to the positive x axis
		# target_waypt = self.waypoints[self.target_waypt_index][0:2]
		# d_target_waypt = target_waypt - np.array([ego_x, ego_y])
		# R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],[-np.sin(ego_yaw), np.cos(ego_yaw)]])
		# local_target_waypt = np.matmul(R, np.expand_dims(d_target_waypt, 1))  # [2,1]
		# local_target_waypt = np.squeeze(local_target_waypt)  # [2,]
		# v = self.ego.get_velocity()
		# speed = np.sqrt(v.x**2 + v.y**2)
		# state = np.array([local_target_waypt[0], -local_target_waypt[1], speed]) 

		lateral_dis, w = self._get_lane_dis(self.waypoints, ego_x, ego_y)
		delta_yaw = np.arcsin(np.cross(w, 
			np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
		v = self.ego.get_velocity()
		speed = np.sqrt(v.x**2 + v.y**2)
		state = np.array([lateral_dis, - delta_yaw, speed]) 
		
		obs = {'birdeye': birdeye, 'lidar': lidar, 'camera': camera, 
			'vh_clas': vh_clas, 'vh_reg_map': vh_regr, 'state': state}
		
		return obs

	def _get_reward(self):
		"""Calculate the step reward."""
		# reward for speed tracking
		v = self.ego.get_velocity()
		speed = np.sqrt(v.x**2 + v.y**2)
		r_speed = -abs(speed - self.desired_speed)
		
		# reward for collision
		r_collision = 0
		if len(self.collision_hist) > 0:
			r_collision = -1

		# reward for steering:
		r_steer = -self.ego.get_control().steer**2

		# reward for out of lane
		ego_x, ego_y = self._get_ego_pos()
		dis, w = self._get_lane_dis(self.waypoints, ego_x, ego_y)
		r_out = 0
		if abs(dis) > self.out_lane_thres:
			r_out = -1

		# longitudinal speed
		lspeed = np.array([v.x, v.y])
		lspeed_lon = np.dot(lspeed, w)

		# cost for too fast
		r_fast = 0
		if lspeed_lon > self.desired_speed:
			r_fast = -1

		# cost for lateral acceleration
		r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

		r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

		return r

	def _get_ego_pos(self):
		"""Get the ego vehicle pose (x, y)."""
		ego_trans = self.ego.get_transform()
		ego_x = ego_trans.location.x
		ego_y = ego_trans.location.y
		return ego_x, ego_y

	def _get_lane_dis(self, waypoints, x, y):
		"""Calculate distance from (x, y) to waypoints."""
		dis_min = 1000
		for pt in waypoints:
			d = np.sqrt((x-pt[0])**2 + (y-pt[1])**2)
			if d < dis_min:
				dis_min = d
				waypt=pt
		vec = np.array([x - waypt[0],y - waypt[1]])
		lv = np.linalg.norm(np.array(vec))
		w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
		cross = np.cross(w, vec/lv)
		dis = - lv * cross
		return dis, w

	def _terminal(self):
		"""Calculate whether to terminate the current episode."""
		# Get ego state
		ego_x, ego_y = self._get_ego_pos()

		# If collides
		if len(self.collision_hist)>0: 
			return True

		# If reach maximum timestep
		if self.time_step>self.max_time_episode:
			return True

		# If at destination
		if self.dests is not None: # If at destination
			for dest in self.dests:
				if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
					return True

		# If out of lane
		dis, _ = self._get_lane_dis(self.waypoints, ego_x, ego_y)
		if abs(dis) > self.out_lane_thres:
			return True

		return False

	def _clear_all_actors(self, actor_filters):
		"""Clear specific actors."""
		for actor_filter in actor_filters:
			for actor in self.world.get_actors().filter(actor_filter):
				if actor.is_alive:
					if actor.type_id == 'controller.ai.walker':
						actor.stop()
					actor.destroy()
