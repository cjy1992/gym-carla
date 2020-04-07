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


    """TODO:
    1. Make costmap such that it reflects the pictures 
    2. Figure out why some lanes are shown in the cost map even though they have no waypoint there
    3. Vectorize 
    """

    #returns a quadratic function given 3 unique points
    def _get_quadratic(x1, y1, x2, y2, x3, y3):
      denom = (x1-x2) * (x1-x3) * (x2-x3);
      a     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom;
      b     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom;
      c     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom;
      
      #we can calculate cost for any x between x2 and x3
      return lambda x: a*(x**2) + b*x + c  

    # Generates a costmap for a single waypoint
    def _get_costmap(waypoint, cost):
      single_costmap = np.zeros((self.obs_size, self.obs_size))

      # #these are the corner points. we have to first translate to make ego_x, ego_y the origin, rotate, and then translate back
      # temp_x = -(self.obs_size / 2)
      # temp_y = -(self.obs_size / 2) 

      # #now we rotate the points by the yaw and translate back
      # corner_x = (temp_x * np.cos(ego_yaw) - temp_y * np.sin(ego_yaw)) + ego_x
      # corner_y = (temp_x * np.sin(ego_yaw) + temp_y * np.cos(ego_yaw)) + ego_y
      corner_x = ego_x - (self.obs_range / 2)
      corner_y = ego_y - (self.obs_range / 2)

      #center waypoint
      laneWidth = waypoint.lane_width
      x_center = waypoint.transform.location.x
      y_center = waypoint.transform.location.y
      quad_center = _get_quadratic(0, cost, laneWidth / 2, 0, -laneWidth / 2, 0) #this quadratic takes in distance from waypoint and outputs a cost

      #now we have to iterate through a circle centered at the waypoint, calculate the cost at each point and put it into the costmap
      #problem: right now it is a square instead of a circle
      for x in np.arange(x_center - (laneWidth / 2), x_center + (laneWidth / 2), self.lidar_bin):
        for y in np.arange(y_center - (laneWidth / 2), y_center + (laneWidth / 2), self.lidar_bin):
          #we rotate x and y in the oppopsite direction to find the relative coordinates of the point to the new corner
          # rotated_x = ((x - ego_x) * np.cos(-ego_yaw) - (y - ego_y) * np.sin(-ego_yaw)) + ego_x
          # rotated_y = ((x - ego_x) * np.sin(-ego_yaw) + (y - ego_y) * np.cos(-ego_yaw)) + ego_y
          rotated_x = ( ((x - ego_x) * np.cos(-ego_yaw)) - ((y - ego_y) * np.sin(-ego_yaw)) ) + ego_x
          rotated_y = ( ((x - ego_x) * np.sin(-ego_yaw)) + ((y - ego_y) * np.cos(-ego_yaw)) ) + ego_y

          rotated_x = rotated_x - (2 * (rotated_x - ego_x))
          #we have to transform x and y so that they are relative to the corner
          transformedX = rotated_x - corner_x
          transformedY = rotated_y - corner_y

          if transformedX < self.obs_range and transformedX >= 0 and transformedY < self.obs_range and transformedY >= 0:
            print("Transformed " + str((transformedX, transformedY)))
            transformedX = int((rotated_x - corner_x) / self.lidar_bin)
            transformedY = int((rotated_y - corner_y) / self.lidar_bin)
            distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
            if distance < laneWidth / 2:
              single_costmap[transformedX][transformedY] = quad_center(distance)
              #print("Sample: " + str(single_costmap[transformedX][transformedY]) + " at " + str(transformedX) + ", " + str(transformedY))
              #print(single_costmap[transformedX][transformedY])

      return single_costmap

    cost = -50
    costmap = np.zeros((self.obs_size, self.obs_size))

    prevWaypoint = None
    for waypoint in self.routeplanner._actualWaypoints:
      costmap = np.add(costmap, _get_costmap(waypoint, cost))
      leftWaypoint = waypoint.get_left_lane()
      rightWaypoint = waypoint.get_right_lane()
      #check if there actually are waypoints 
      if leftWaypoint:
        costmap = np.add(costmap, _get_costmap(leftWaypoint, cost))
      if rightWaypoint:
        costmap = np.add(costmap, _get_costmap(rightWaypoint, cost))


    #costmap is a 2d ndarray that goes from cost to 0 so we have to scale it from 0 to 255
    costmap = np.clip(costmap, cost, 0)
    costmap = (costmap - cost) * 255 / abs(cost)
    #print("Costmap: " + str(costmap))
    # Display costmap
    costmap_surface = self._rgb_to_display_surface(np.moveaxis(np.array([costmap, costmap, costmap]), 0, -1))
    self.display.blit(costmap_surface, (self.display_size * 3, 0))

    print("ego x: " + str(ego_x) + ", ego y: " + str(ego_y))

    # actualWaypointsList = []
    # for (waypoint, _) in self.routeplanner._actualWaypoints:
    #   tup = (waypoint.transform.location.x, waypoint.transform.location.y)
    #   actualWaypointsList.append(tup)
    # print("Actual Waypoints: " + str(actualWaypointsList))
    # print("Waypoints List: " + str(self.waypoints))


    """implementing costmap:
    Costmap is a 2d array that has costs at each point. The center of the cost map is the location of 
    the ego vehicle. First work on creating the 2d array. Then create the visual (surface) in pygame.
    Finally, update the obs dictionary with costmap
    """

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
