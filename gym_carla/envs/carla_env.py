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
from matplotlib.path import Path
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
    self.display_route = params['display_route']
    self.pixor_size = params['pixor_size']

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
      'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
      'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
      'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32),
      'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32),
      'costmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 1), dtype=np.uint8)}) #costmap should be a 2d array

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

    # Get pixel grid points
    x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    self.pixel_grid = np.vstack((x,y)).T 

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
        #transform = random.choice(self.vehicle_spawn_points)
        transform = self.vehicle_spawn_points[0]
        #transform.rotation.yaw = 0
        tup = (transform.location.x, transform.location.y, transform.rotation.yaw)
        print("Transform: " + str(tup))
      if self.task_mode == 'roundabout':
        self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
        # self.start=[52.1,-4.2, 178.66] # static
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

  def _display_to_rgb(self, display):
    """ Transform image grabbed from pygame display to an rgb image uint8 matrix
    """
    rgb = np.fliplr(np.rot90(display, 3))  # flip to regular view
    rgb = resize(rgb, (self.obs_size, self.obs_size))  # resize
    rgb = rgb * 255
    return rgb

  def _rgb_to_display_surface(self, rgb):
    """ Generate pygame surface given an rgb image uint8 matrix
    """
    surface = pygame.Surface((self.display_size, self.display_size)).convert()
    display = resize(rgb, (self.display_size, self.display_size))
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    pygame.surfarray.blit_array(surface, display)
    return surface

  def _get_obs(self):
    """Get the observations."""
    ## Birdeye rendering
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    # birdeye view with roadmap and actors
    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = self._display_to_rgb(birdeye)

    # Roadmap
    roadmap_render_types = ['roadmap']
    if self.display_route:
      roadmap_render_types.append('waypoints')
    self.birdeye_render.render(self.display, roadmap_render_types)
    roadmap = pygame.surfarray.array3d(self.display)
    roadmap = roadmap[0:self.display_size, :, :]
    roadmap = self._display_to_rgb(roadmap)
    # Add ego vehicle
    for i in range(self.obs_size):
      for j in range(self.obs_size):
        if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
          roadmap[i, j, :] = birdeye[i, j, :]

    # Display birdeye image
    birdeye_surface = self._rgb_to_display_surface(birdeye)
    self.display.blit(birdeye_surface, (0, 0))

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
    if self.display_route:
      wayptimg = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240)
    else:
      wayptimg = birdeye[:,:,0] < 0  # Equal to a zero matrix
    wayptimg = np.expand_dims(wayptimg, axis=2)
    wayptimg = np.fliplr(np.rot90(wayptimg, 3))

    # Get the final lidar image
    lidar = np.concatenate((lidar, wayptimg), axis=2)
    lidar = np.flip(lidar, axis=1)
    lidar = np.rot90(lidar, 1)
    lidar = lidar * 255

    # Display lidar image
    lidar_surface = self._rgb_to_display_surface(lidar)
    self.display.blit(lidar_surface, (self.display_size, 0))

    ## Display camera image
    camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
    camera_surface = self._rgb_to_display_surface(camera)
    self.display.blit(camera_surface, (self.display_size * 2, 0))

    ## Display roadmap image
    # roadmap_surface = self._rgb_to_display_surface(roadmap)
    # self.display.blit(roadmap_surface, (self.display_size * 3, 0))

    ## Vehicle classification and regression maps (requires further normalization)
    vh_clas = np.zeros((self.pixor_size, self.pixor_size))
    vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

    # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
    def get_actor_info(actor):
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x  # half the length
      w=bb.extent.y  # half the width
      return (x, y, yaw, l, w)

    def global_to_local_pose(pose, ego_pose):
      x, y, yaw = pose
      ego_x, ego_y, ego_yaw = ego_pose
      R = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)], 
        [-np.sin(ego_yaw), np.cos(ego_yaw)]])
      vec_local = R.dot(np.array([x - ego_x, y - ego_y]))
      yaw_local = yaw - ego_yaw
      return (vec_local[0], vec_local[1], yaw_local)

    def local_to_pixel_info(info):
      """Here the ego local coordinate is left-handed, the pixel
      coordinate is also left-handed, with its origin at the left bottom.
      """
      x, y, yaw, l, w = info
      x_pixel = (x + self.d_behind)/self.obs_range*self.pixor_size 
      y_pixel = y/self.obs_range*self.pixor_size + self.pixor_size/2
      yaw_pixel = yaw
      l_pixel = l/self.obs_range*self.pixor_size
      w_pixel = w/self.obs_range*self.pixor_size
      return (x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel)

    def get_pixels_from_info(info):
      """Get pixels from information in pixel coordinate, 
      which the origin is at the left bottom.
      """
      poly = get_poly_from_info(info)     
      p = Path(poly) # make a polygon
      grid = p.contains_points(self.pixel_grid)
      isinPoly = np.where(grid==True)
      pixels = np.take(self.pixel_grid, isinPoly, axis=0)[0]
      return pixels

    def get_poly_from_info(info):
      x, y, yaw, l, w = info
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to the coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      return poly

    # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
    # for convenience
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi
    for actor in self.world.get_actors().filter('vehicle.*'):
      x, y, yaw, l, w = get_actor_info(actor)
      x_local, y_local, yaw_local = global_to_local_pose(
        (x, y, yaw), (ego_x, ego_y, ego_yaw))
      if actor.id != self.ego.id and np.sqrt(x_local**2 + y_local**2) < self.obs_range**1.5:
        x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = local_to_pixel_info(
          (x_local, y_local, yaw_local, l, w))
        cos_t = np.cos(yaw_pixel)
        sin_t = np.sin(yaw_pixel)
        logw = np.log(w_pixel)
        logl = np.log(l_pixel)
        pixels = get_pixels_from_info((x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel))
        for pixel in pixels:
          vh_clas[pixel[0], pixel[1]] = 1
          dx = x_pixel - pixel[0]
          dy = y_pixel - pixel[1]
          # dx = (x_pixel - pixel[0])/self.obs_size*self.obs_range
          # dy = (y_pixel - pixel[1])/self.obs_size*self.obs_range
          vh_regr[pixel[0], pixel[1], :] = np.array(
            [cos_t, sin_t, dx, dy, logw, logl])

    # Flip the image matrix so that the origin is at the left-bottom
    vh_clas = np.flip(vh_clas, axis=0)
    vh_regr = np.flip(vh_regr, axis=0)

    ## Display pixor images
    # vh_clas_display = np.stack([vh_clas, vh_clas, vh_clas], axis=2) * 255
    # vh_clas_surface = self._rgb_to_display_surface(vh_clas_display)
    # self.display.blit(vh_clas_surface, (self.display_size * 4, 0))
    # vh_regr1 = vh_regr[:, :, 0:3]
    # vh_regr2 = vh_regr[:, :, 3:6]
    # vh_regr1_surface = self._rgb_to_display_surface(vh_regr1)
    # self.display.blit(vh_regr1_surface, (self.display_size * 5, 0))
    # vh_regr2_surface = self._rgb_to_display_surface(vh_regr2)
    # self.display.blit(vh_regr2_surface, (self.display_size * 6, 0))

    # Display on pygame
    pygame.display.flip()

    # State observation
    lateral_dis, w = self._get_preview_lane_dis(self.waypoints, ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

    # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
    pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]


    def _get_perp_dis(x1, y1, x2, y2, x3, y3):
      x = np.array([x3, y3])
      p = np.array([x1, y1])
      q = np.array([x2, y2])
      #compute whether point x is even in the range of the line
      lamb = np.dot((x - p), (q - p)) / np.dot((q - p), (q - p))
      if lamb <= 1 and lamb >= 0:
        s = p + (lamb * (q - p))
        return np.linalg.norm(x - s)#abs( ((y2 - y1) * x3) - ((x2 - x1) * y3) + (x2 * y1) - (y2 * x1)) / np.sqrt((y2 - y1) **2 + (x2 - x1) ** 2)
      return float('inf')

    """_get_costmap generates a costmap for a current waypoint and its preceding waypoint.
    I refer a lot to global vs local frame. Global means the xy coordinate in the Carla coordinates
    Local is the coordinate in the costmap matrix.
    Also the letters x and y are swapped when referring to the local frame. I have to fix this later because
    it's confusing to read but it works """

    def _get_costmap(pWaypoint, cWaypoint, cost):
      single_costmap = np.zeros((self.obs_size, self.obs_size))

      #Definitions for the waypoints' x and y coordinates in the global frame 
      laneWidth = pWaypoint.lane_width
      cX = cWaypoint.transform.location.x
      cY = cWaypoint.transform.location.y
      pX = pWaypoint.transform.location.x
      pY = pWaypoint.transform.location.y

      #If we draw a square around the center of the ego vehicle (length is determined by range), this is the bottom left corner in global coords
      corner_x = ego_x - (self.obs_range / 2)
      corner_y = ego_y - (self.obs_range / 2)

      #Here we create two matrices with the same dimensions as the costmap. One represents the x coordinate and one represents the y coordinate in the local frame.
      y_array, x_array = np.meshgrid(np.arange(0, self.obs_size), np.arange(0, self.obs_size))
      #y_array is [[0 1 2 ... 255] [0 1 2 ... 255] ...] 
      #x_array is [[0 0 0 .... 0] [1 1 1 .... 1]... [255 255 ... 255]]

      rotated_x_array = (2 * ego_x) - ((x_array * self.lidar_bin) + corner_x)
      rotated_y_array = (y_array * self.lidar_bin) + corner_y
      c = np.cos(ego_yaw)
      s = np.sin(ego_yaw)
      global_x_array = (c * (rotated_x_array - ego_x)) - (s * (rotated_y_array - ego_y)) + ego_x #for each point in our matrix, we have their global coordinates 
      global_y_array = (s * (rotated_x_array - ego_x)) + (c * (rotated_y_array - ego_y)) + ego_y

      p = np.array([pX, pY])
      q = np.array([cX, cY])
      q_dif = q - p
      lamb_array= (((global_x_array - pX) * (cX - pX)) + ((global_y_array - pY) * (cY - pY ))) / np.dot((q_dif), (q_dif))
      sX = pX + (lamb_array * (cX - pX))
      sY = pY + (lamb_array * (cY - pY))

      takeNormX = global_x_array - sX
      takeNormY = global_y_array - sY
      distanceMap = np.sqrt(np.square(takeNormX) + np.square(takeNormY))
      penal = (laneWidth / 2) * (-cost) / abs(cost)      
      perpDis = np.where((lamb_array <=1) & (lamb_array >= 0) & (distanceMap <= laneWidth / 2), distanceMap, penal) #will have perpDistance in the spot if its within the lane
      single_costmap = perpDis * (abs(cost / (laneWidth / 2))) + cost

      return single_costmap


    """
    Explanation of how my costmap implementation works:
      First we get a list of all of the waypoints from the current position. We iterate through this list in pairs so that there is a current 
      waypoint and a previous waypoint. These along with parameter cost are passed into _get_costmap which returns a costmap only relevant to the
      lane defined by the line between the two points. This costmap is summed with the global costmap. This profess is repeated for the left and right
      lanes of the current waypoint if they exist and are in the same direction.  
    """

    cost = -10
    costmap = np.zeros((self.obs_size, self.obs_size))
    if len(self.routeplanner._actualWaypoints) < 1:
      print("Not enough waypoints to form costmap")
      costmap = None

    else:
      pWaypoint = self.routeplanner._actualWaypoints[0]
      for cWaypoint in self.routeplanner._actualWaypoints[1:]:

        currentDirection = cWaypoint.lane_id #positive or negative integer depending on which direction the lane is going in 
        costmap = costmap + _get_costmap(pWaypoint, cWaypoint, cost)

        #The current implementation of left and right lanes is contingent on whether the current lane has a left/right lane AND the previous lane has a left/right lane
        pleftWaypoint = pWaypoint.get_left_lane()
        prightWaypoint = pWaypoint.get_right_lane()
        cleftWaypoint = cWaypoint.get_left_lane()
        crightWaypoint = cWaypoint.get_right_lane()
        pWaypoint = cWaypoint

        if pleftWaypoint and (pleftWaypoint.lane_id * currentDirection >= 0): #check if left waypoint exists for the previous waypoint and it goes in the same direction 
          if cleftWaypoint and (cleftWaypoint.lane_id * currentDirection >= 0): #check if the left waypoint exists for the current waypoint and it goes in the same direction
            costmap = costmap + _get_costmap(pleftWaypoint, cleftWaypoint, cost)

        if prightWaypoint and (prightWaypoint.lane_id * currentDirection >= 0):
          if crightWaypoint and (crightWaypoint.lane_id * currentDirection >= 0):
            costmap = costmap + _get_costmap(prightWaypoint, crightWaypoint, cost)


    #Here we convert the cost map which ranges from -cost to 0 (low cost to high cost) to a displayable costmap that has values from 0 to 255
    costmap = np.clip(costmap, cost, 0)
    costmap = (costmap - cost) * 255 / abs(cost)

    costmap_surface = self._rgb_to_display_surface(np.moveaxis(np.array([costmap, costmap, costmap]), 0, -1))
    self.display.blit(costmap_surface, (self.display_size * 3, 0))

    obs = {}
    obs.update({
      'birdeye':birdeye.astype(np.uint8),
      'lidar':lidar.astype(np.uint8),
      'camera':camera.astype(np.uint8),
      'roadmap':roadmap.astype(np.uint8),
      'vh_clas':np.expand_dims(vh_clas, -1).astype(np.float32),
      'vh_regr':vh_regr.astype(np.float32),
      'state': state,
      'pixor_state': pixor_state,
      'costmap' : costmap
    })
    
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

  def _get_preview_lane_dis(self, waypoints, x, y, idx=2):
    """Calculate distance from (x, y) to waypoints."""
    dis_min = 1000
    waypt = waypoints[idx]
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
