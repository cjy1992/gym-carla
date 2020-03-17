#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This file is modified from <https://github.com/carla-simulator/carla>:
# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from enum import Enum
from collections import deque
import random
import numpy as np
import carla

from gym_carla.envs.misc import distance_vehicle, is_within_distance_ahead, compute_magnitude_angle

class RoadOption(Enum):
  """
  RoadOption represents the possible topological configurations when moving from a segment of lane to other.
  """
  VOID = -1
  LEFT = 1
  RIGHT = 2
  STRAIGHT = 3
  LANEFOLLOW = 4

class RoutePlanner():
  def __init__(self, vehicle, buffer_size):
    self._vehicle = vehicle
    self._world = self._vehicle.get_world()
    self._map = self._world.get_map()

    self._sampling_radius = 5
    self._min_distance = 4

    self._target_waypoint = None
    self._buffer_size = buffer_size
    self._waypoint_buffer = deque(maxlen=self._buffer_size)

    self._waypoints_queue = deque(maxlen=600)
    self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
    self._waypoints_queue.append( (self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))
    self._target_road_option = RoadOption.LANEFOLLOW

    self._last_traffic_light = None
    self._proximity_threshold = 15.0

    self._compute_next_waypoints(k=200)

  def _compute_next_waypoints(self, k=1):
    """
    Add new waypoints to the trajectory queue.

    :param k: how many waypoints to compute
    :return:
    """
    # check we do not overflow the queue
    available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
    k = min(available_entries, k)

    for _ in range(k):
      last_waypoint = self._waypoints_queue[-1][0]
      next_waypoints = list(last_waypoint.next(self._sampling_radius))

      if len(next_waypoints) == 1:
        # only one option available ==> lanefollowing
        next_waypoint = next_waypoints[0]
        road_option = RoadOption.LANEFOLLOW
      else:
        # random choice between the possible options
        road_options_list = retrieve_options(
          next_waypoints, last_waypoint)

        road_option = road_options_list[1]
        # road_option = random.choice(road_options_list)
        
        next_waypoint = next_waypoints[road_options_list.index(
          road_option)]

      self._waypoints_queue.append((next_waypoint, road_option))

  def run_step(self):
    waypoints = self._get_waypoints()
    red_light, vehicle_front = self._get_hazard()
    # red_light = False
    return waypoints, red_light, vehicle_front

  def _get_waypoints(self):
    """
    Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
    follow the waypoints trajectory.

    :param debug: boolean flag to activate waypoints debugging
    :return:
    """

    # not enough waypoints in the horizon? => add more!
    if len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
      self._compute_next_waypoints(k=100)

    #   Buffering the waypoints
    while len(self._waypoint_buffer)<self._buffer_size:
      if self._waypoints_queue:
        self._waypoint_buffer.append(
          self._waypoints_queue.popleft())
      else:
        break

    waypoints=[]

    for i, (waypoint, _) in enumerate(self._waypoint_buffer):
      waypoints.append([waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.rotation.yaw])

    # current vehicle waypoint
    self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
    # target waypoint
    self._target_waypoint, self._target_road_option = self._waypoint_buffer[0]

    # purge the queue of obsolete waypoints
    vehicle_transform = self._vehicle.get_transform()
    max_index = -1

    for i, (waypoint, _) in enumerate(self._waypoint_buffer):
      if distance_vehicle(
          waypoint, vehicle_transform) < self._min_distance:
        max_index = i
    if max_index >= 0:
      for i in range(max_index - 1):
        self._waypoint_buffer.popleft()

    return waypoints

  def _get_hazard(self):
    # retrieve relevant elements for safe navigation, i.e.: traffic lights
    # and other vehicles
    actor_list = self._world.get_actors()
    vehicle_list = actor_list.filter("*vehicle*")
    lights_list = actor_list.filter("*traffic_light*")

    # check possible obstacles
    vehicle_state = self._is_vehicle_hazard(vehicle_list)

    # check for the state of the traffic lights
    light_state = self._is_light_red_us_style(lights_list)

    return light_state, vehicle_state

  def _is_vehicle_hazard(self, vehicle_list):
    """
    Check if a given vehicle is an obstacle in our way. To this end we take
    into account the road and lane the target vehicle is on and run a
    geometry test to check if the target vehicle is under a certain distance
    in front of our ego vehicle.

    WARNING: This method is an approximation that could fail for very large
     vehicles, which center is actually on a different lane but their
     extension falls within the ego vehicle lane.

    :param vehicle_list: list of potential obstacle to check
    :return: a tuple given by (bool_flag, vehicle), where
         - bool_flag is True if there is a vehicle ahead blocking us
           and False otherwise
         - vehicle is the blocker object itself
    """

    ego_vehicle_location = self._vehicle.get_location()
    ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    for target_vehicle in vehicle_list:
      # do not account for the ego vehicle
      if target_vehicle.id == self._vehicle.id:
        continue

      # if the object is not in our lane it's not an obstacle
      target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
      if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
              target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
        continue

      loc = target_vehicle.get_location()
      if is_within_distance_ahead(loc, ego_vehicle_location,
                    self._vehicle.get_transform().rotation.yaw,
                    self._proximity_threshold):
        return True

    return False

  def _is_light_red_us_style(self, lights_list):
    """
    This method is specialized to check US style traffic lights.

    :param lights_list: list containing TrafficLight objects
    :return: a tuple given by (bool_flag, traffic_light), where
         - bool_flag is True if there is a traffic light in RED
           affecting us and False otherwise
         - traffic_light is the object itself or None if there is no
           red traffic light affecting us
    """
    ego_vehicle_location = self._vehicle.get_location()
    ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

    if ego_vehicle_waypoint.is_intersection:
      # It is too late. Do not block the intersection! Keep going!
      return False

    if self._target_waypoint is not None:
      if self._target_waypoint.is_intersection:
        potential_lights = []
        min_angle = 180.0
        sel_magnitude = 0.0
        sel_traffic_light = None
        for traffic_light in lights_list:
          loc = traffic_light.get_location()
          magnitude, angle = compute_magnitude_angle(loc,
                                 ego_vehicle_location,
                                 self._vehicle.get_transform().rotation.yaw)
          if magnitude < 80.0 and angle < min(25.0, min_angle):
            sel_magnitude = magnitude
            sel_traffic_light = traffic_light
            min_angle = angle

        if sel_traffic_light is not None:
          if self._last_traffic_light is None:
            self._last_traffic_light = sel_traffic_light

          if self._last_traffic_light.state == carla.libcarla.TrafficLightState.Red:
            return True
        else:
          self._last_traffic_light = None

    return False

def retrieve_options(list_waypoints, current_waypoint):
  """
  Compute the type of connection between the current active waypoint and the multiple waypoints present in
  list_waypoints. The result is encoded as a list of RoadOption enums.

  :param list_waypoints: list with the possible target waypoints in case of multiple options
  :param current_waypoint: current active waypoint
  :return: list of RoadOption enums representing the type of connection from the active waypoint to each
       candidate in list_waypoints
  """
  options = []
  for next_waypoint in list_waypoints:
    # this is needed because something we are linking to
    # the beggining of an intersection, therefore the
    # variation in angle is small
    next_next_waypoint = next_waypoint.next(3.0)[0]
    link = compute_connection(current_waypoint, next_next_waypoint)
    options.append(link)

  return options


def compute_connection(current_waypoint, next_waypoint):
  """
  Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
  (next_waypoint).

  :param current_waypoint: active waypoint
  :param next_waypoint: target waypoint
  :return: the type of topological connection encoded as a RoadOption enum:
       RoadOption.STRAIGHT
       RoadOption.LEFT
       RoadOption.RIGHT
  """
  n = next_waypoint.transform.rotation.yaw
  n = n % 360.0

  c = current_waypoint.transform.rotation.yaw
  c = c % 360.0

  diff_angle = (n - c) % 180.0
  if diff_angle < 1.0:
    return RoadOption.STRAIGHT
  elif diff_angle > 90.0:
    return RoadOption.LEFT
  else:
    return RoadOption.RIGHT
