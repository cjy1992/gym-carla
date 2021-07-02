#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import random
import sys
import time
from gym_carla.envs.vehicle_position import get_vehicle_position, get_vehicle_orientation

import gym
from gym import spaces
from gym.utils import seeding

from gym_carla.envs.misc import *
from gym_carla.envs.route_planner import RoutePlanner


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        # parameters
        self.max_past_step = params['max_past_step']
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']
        self.max_time_episode = params['max_time_episode']
        self.max_waypt = params['max_waypt']
        self.d_behind = params['d_behind']
        self.obs_size = 288
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.speed_reduction_at_intersection = params['reduction_at_intersection']
        self.max_ego_spawn_times = params['max_ego_spawn_times']

        # Destination
        self.dests = None

        # action and observation spaces
        self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
                                                 params['continuous_steer_range'][0]]),
                                       np.array([params['continuous_accel_range'][1],
                                                 params['continuous_steer_range'][1]]),
                                       dtype=np.float32)  # acc, steer
        observation_space_dict = {
            'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=1000, shape=(self.obs_size, self.obs_size, 3), dtype=np.float32),
            'state': spaces.Box(np.array([-50, -50, 0]), np.array([50, 50, 4]), dtype=np.float32)
        }

        self.observation_space = spaces.Dict(observation_space_dict)

        # Connect to carla server and get world object
        print('Connecting to Carla server...')
        self.town = params['town']
        self.client = carla.Client('localhost', params['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(params['town'])
        self.map = self.world.get_map()
        self.tm = self.client.get_trafficmanager()
        self.tm_port = self.tm.get_port()
        print(f'Carla server connected at localhost:{params["port"]}!')

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_blueprint(params['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=1, z=2))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # depth sensor
        self.depth_array = np.zeros((self.obs_size, self.obs_size, 1), dtype=np.uint8)
        self.depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        self.depth_bp.set_attribute('image_size_x', str(self.obs_size))
        self.depth_bp.set_attribute('image_size_y', str(self.obs_size))
        self.depth_bp.set_attribute('fov', '110')
        self.depth_bp.set_attribute('sensor_tick', '0.02')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

    def reset(self):
        # Clear sensor objects
        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        # hard reset
        # self.world = self.client.load_world(self.town)

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision',
                                'sensor.lidar.ray_cast',
                                'sensor.camera.rgb',
                                'vehicle.*',
                                'controller.ai.walker',
                                'walker.*'])

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

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:

            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()

            transform = random.choice(self.vehicle_spawn_points)
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

            sys.stdout.write("\r")
            sys.stdout.write(f"Trying to spawn ego vehicle: {ego_spawn_times}/{self.max_ego_spawn_times}")
            sys.stdout.flush()

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            self.collision_info = {
                "frame": event.frame,
                "other_actor": event.other_actor.type_id,
            }
            self.collision_hist.append(event)

        self.collision_info = None

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            self.camera_img = array

        self.depth_sensor = self.world.spawn_actor(self.depth_bp, self.camera_trans, attach_to=self.ego)
        self.depth_sensor.listen(lambda data: get_depth_img(data))

        def get_depth_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array.astype(np.float32)
            # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
            normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
            normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
            depth_meters = normalized_depth * 1000
            self.depth_array = depth_meters

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.tm.set_synchronous_mode(True)
        self.world.apply_settings(self.settings)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front, self.road_option = self.routeplanner.run_step()

        return self._get_obs()

    def step(self, action: list):

        # Get acceleration and steering
        acc = action[0]
        steer = action[1]

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc, 0, 1)

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
        self.ego.apply_control(act)
        self.world.tick()

        # route planner
        self.waypoints, _, self.vehicle_front, self.road_option = self.routeplanner.run_step()

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front,
            'road_option': self.road_option
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        return self._get_obs(), self._get_reward(act), self._terminal(), copy.deepcopy(info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def _create_vehicle_blueprint(self, actor_filter, color=None, number_of_wheels=None):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        if number_of_wheels is None:
            number_of_wheels = [4]
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=None):
        """Try to spawn a surrounding vehicle at specific transform with random blueprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        if number_of_wheels is None:
            number_of_wheels = [4]
        blueprint = self._create_vehicle_blueprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)

        if vehicle:
            vehicle.set_autopilot(True, self.tm_port)
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random blueprint.

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
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_obs(self):
        """Get the observations."""
        # State observation
        v = self.ego.get_velocity()
        state = np.array([v.x, v.y, int(self.road_option.value)])

        obs = {
            'camera': self.camera_img,
            'depth': self.depth_array,
            'state': state
        }

        return obs

    def _get_reward(self, control):
        """Calculate the step reward."""

        steer = control.steer
        command = self.road_option.name
        v = self.ego.get_velocity()
        collision = self.collision_info
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        distance = get_vehicle_position(self.map, self.ego)
        speed_red = self.speed_reduction_at_intersection

        # speed and steer behavior
        if command in ['RIGHT', 'LEFT']:
            r_a = 2 - np.abs(speed_red * self.desired_speed - speed) / speed_red * self.desired_speed
            is_opposite = steer > 0 and command == 'LEFT' or steer < 0 and command == 'RIGHT'
            r_a -= steer ** 2 if is_opposite else 0
        elif command == 'STRAIGHT':
            r_a = 1 - np.abs(speed_red * self.desired_speed - speed) / speed_red * self.desired_speed
        # follow lane
        else:
            r_a = 2 - np.abs(self.desired_speed - speed) / self.desired_speed

        # collision
        r_c = 0
        if collision:
            r_c = -5
            if str(collision['other_actor']).startswith('vehicle'):
                r_c = -10

        # distance to center
        r_dist = - distance / 2
        return r_a + r_c + r_dist

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    print("reached destination")
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            print("out of lane")
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
