#!/usr/bin/env python

from __future__ import division

import copy
import random
import sys
import time
from gym_carla.envs.vehicle_position import get_vehicle_position, get_vehicle_orientation
from termcolor import colored

import gym
from gym import spaces
from gym.utils import seeding
from .listeners import get_collision_hist, get_camera_img, get_depth_img
from .utils import unnormalize_pid_action

from gym_carla.envs.misc import *
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.controller import PIDController
import weakref
from collections import namedtuple

speed_proto = namedtuple('speed', 'x y z')


class CarlaPidEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):

        self.config = params

        # Destination
        self.dests = None

        # action and observation spaces
        assert "continuous_speed_range" in params.keys(), "You need to specify the continuous_speed_range"
        assert "continuous_steer_range" in params.keys(), "You need to specify the continuous_steer_range"
        self.action_space = spaces.Box(np.array([params['continuous_speed_range'][0],
                                                 params['continuous_steer_range'][0]]),
                                       np.array([params['continuous_speed_range'][1],
                                                 params['continuous_steer_range'][1]]),
                                       dtype=np.float32)  # acc, steer
        observation_space_dict = {
            'camera': spaces.Box(low=0, high=255, shape=(self.config['obs_size'],
                                                         self.config['obs_size'], 3), dtype=np.uint8),
            'depth': spaces.Box(low=0, high=1000, shape=(self.config['obs_size'],
                                                         self.config['obs_size'], 3), dtype=np.float32),
            'state': spaces.Box(np.array([-50, -50, 0]), np.array([50, 50, 4]), dtype=np.float32)
        }

        self.observation_space = spaces.Dict(observation_space_dict)
        tm_port = params['traffic_manager_port']
        print(colored("Connecting to CARLA...", "white"))
        self.client = carla.Client(self.config['host'], self.config['port'])
        self.client.set_timeout(20.0)
        self.client.load_world(self.config['town'])
        self.town = self.config['town']
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.settings = self.world.get_settings()
        self.map = self.world.get_map()
        self.tm = self.client.get_trafficmanager(port=tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm_port = self.tm.get_port()
        self.speed_control = PIDController(K_P=1.0)
        print(colored(f"Successfully connected to CARLA at {self.config['host']}:{self.config['port']}", "green"))

        # parameters that come from the config dictionary
        self.sensor_width, self.sensor_height = self.config['obs_size'], self.config['obs_size']
        self.fps = int(1 / self.config['dt'])
        self.max_steps = self.config['max_time_episode']
        self.reward_weights = self.config['reward_weights']
        self.target_step_distance = self.config['desired_speed'] * self.config['dt']
        self._proximity_threshold = 15

        # local state vars
        self.ego = None
        self.route_planner = None
        self.waypoints = None
        self.vehicle_front = None
        self.red_light = None
        self.road_option = None
        self.camera_img = None
        self.last_position = None
        self.distance_to_traffic_light = None
        self.is_vehicle_hazard = None
        self.last_steer = None

        self.time_step = 0
        self.total_step = 0
        self.to_clean = {}
        self.collision_info = {}
        self.collision_hist = []

    def set_weather(self, weather_option):
        weather = carla.WeatherParameters(*weather_option)
        self.world.set_weather(weather)

    def reset(self):

        self._destroy_actors()
        self.collision_info = {}
        self.collision_hist = []
        self.time_step = 0
        self._set_synchronous_mode(False)

        if self.config['verbose']:
            print(colored("setting actors", "white"))
        # set actors
        ego_vehicle, _ = self.set_ego()
        self.ego = ego_vehicle
        self.last_position = get_pos(ego_vehicle)
        self.last_steer = 0
        rgb_camera = self.set_camera(ego_vehicle,
                                     sensor_width=self.sensor_width,
                                     sensor_height=self.sensor_height,
                                     fov=110)
        collision_sensor = self.set_collision_sensor(ego_vehicle)
        sp_vehicles, sp_walkers, sp_walker_controllers = self.set_actors(vehicles=self.config['vehicles'],
                                                                         walkers=self.config['walkers'])
        # we need to store the pointers to avoid "out of scope" errors
        self.to_clean = dict(vehicles=[ego_vehicle, *sp_vehicles, *sp_walkers],
                             sensors=[collision_sensor, rgb_camera],
                             controllers=sp_walker_controllers)
        if self.config['verbose']:
            print(colored(f"spawned {len(sp_vehicles)} vehicles and {len(sp_walkers)} walkers", "green"))

        # attaching handlers
        weak_self = weakref.ref(self)
        rgb_camera.listen(lambda data: get_camera_img(weak_self, data))
        collision_sensor.listen(lambda event: get_collision_hist(weak_self, event))

        self._set_synchronous_mode(True)

        self.route_planner = RoutePlanner(ego_vehicle, self.config['max_waypt'])
        self.waypoints, self.red_light, self.distance_to_traffic_light, \
            self.is_vehicle_hazard, self.vehicle_front, self.road_option = self.route_planner.run_step()
        return self._step([0, 0, 0])[0]

    def render(self, mode='human'):
        pass

    def _step(self, action: list):
        """
        Performs a simulation step.
        @param action: List with control signals: [target_speed, steer].
        """
        # Action UnNormalization
        action = unnormalize_pid_action(action)

        # Apply control
        speed = self.ego.get_velocity()
        speed = np.linalg.norm([speed.x, speed.y])
        delta_speed = action[0] - speed
        throttle = self.speed_control.step(delta_speed)

        act = carla.VehicleControl(throttle=throttle, brake=0, steer=float(action[1]))
        self.ego.apply_control(act)
        self.update_spectator(self.ego)
        self.world.tick()

        # route planner
        self.waypoints, self.red_light, self.distance_to_traffic_light,\
            self.is_vehicle_hazard, self.vehicle_front, self.road_option = self.route_planner.run_step()

        # state information
        info = {
            'waypoints': self.waypoints,
            'road_option': self.road_option
        }

        step_reward = self._get_reward(act)
        obs = self._get_obs()
        self.last_steer = float(action[1])
        self.last_position = get_pos(self.ego)

        return obs, step_reward, self._terminal(), copy.deepcopy(info)

    def _get_obs(self):
        """Get the observations."""
        # State observation
        ego_trans = self.ego.get_transform()
        ego_v = self.ego.get_velocity()
        ego_loc = self.ego.get_location()
        ego_control = self.ego.get_control()

        traffic_light_state = self.red_light
        distance_to_traffic_light = self.distance_to_traffic_light
        front_vehicle_distance = 15
        front_vehicle_velocity = speed_proto(x=10, y=10, z=10)
        if self.vehicle_front is not None:
            front_vehicle_location = self.vehicle_front.get_location()
            front_vehicle_distance = np.array([ego_loc.x - front_vehicle_location.x, ego_loc.y - front_vehicle_location.y,
                                               ego_loc.z - front_vehicle_location.z])
            front_vehicle_distance = np.linalg.norm(front_vehicle_distance)
            front_vehicle_velocity = self.vehicle_front.get_velocity()

        # calculate distance and orientation
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        lateral_dis, w = get_lane_dis(self.waypoints, ego_loc.x, ego_loc.y)
        delta_yaw = -np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        average_delta_yaw = get_average_delta_yaw(self.waypoints, ego_yaw)

        # affordance vector construction
        velocity_norm_factor = 15.0
        affordances = np.array([
            1 if traffic_light_state else 0,
            distance_to_traffic_light / 80.0,
            1 if self.is_vehicle_hazard else 0,
            front_vehicle_distance / 15.0,
            front_vehicle_velocity.x / velocity_norm_factor,   # x
            front_vehicle_velocity.y / velocity_norm_factor,   # y
            front_vehicle_velocity.z / velocity_norm_factor,   # z
            lateral_dis / 2.0,
            delta_yaw,
            ego_v.x / velocity_norm_factor,
            ego_v.y / velocity_norm_factor,
            ego_v.z / velocity_norm_factor,
            ego_control.steer,
            self.last_steer,
            average_delta_yaw
        ])

        obs = {
            'camera': self.camera_img,
            'affordances': affordances,
            'speed': np.array([ego_v.x, ego_v.y, ego_v.z]),
            'hlc': int(self.road_option.value) - 1
        }

        return obs

    def step(self, action):
        for _ in range(3):
            self._step(action)
        self.time_step += 1
        return self._step(action)

    def _get_reward(self, control):
        """Calculate the step reward."""

        steer = control.steer
        command = self.road_option.name
        v = self.ego.get_velocity()
        ego_loc = self.ego.get_location()
        collision = self.collision_info
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        distance, _ = get_lane_dis(self.waypoints, ego_loc.x, ego_loc.y)
        speed_red = self.config['speed_reduction_at_intersection']

        # speed and steer behavior
        if command in ['RIGHT', 'LEFT']:
            r_a = -np.abs(speed_red * self.config['desired_speed'] - speed) / speed_red * self.config[
                'desired_speed']
            is_opposite = steer > 0 and command == 'LEFT' or steer < 0 and command == 'RIGHT'
            r_a -= steer ** 2 if is_opposite else 0
        elif command == 'STRAIGHT':
            r_a = - np.abs(speed_red * self.config['desired_speed'] - speed) / speed_red * self.config[
                'desired_speed']
        # follow lane
        else:
            r_a = -np.abs(self.config['desired_speed'] - speed) / self.config['desired_speed']

        # collision
        r_c = 0
        if collision:
            r_c = -50
            if str(collision['other_actor']).startswith('vehicle'):
                r_c = -100

        # distance to center
        r_dist = - np.abs(distance / 2)

        # distance traveled
        ego_pos = get_pos(self.ego)
        step_distance_traveled = (ego_pos[0] - self.last_position[0]) ** 2 + (ego_pos[1] - self.last_position[1]) ** 2
        step_distance_traveled = np.sqrt(step_distance_traveled)
        r_dist_traveled = -np.abs(self.target_step_distance - step_distance_traveled) / self.target_step_distance

        return r_a + r_c + r_dist + r_dist_traveled

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            return True

        # If reach maximum timestep
        if self.time_step > self.max_steps:
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    if self.config['verbose']:
                        print(colored("reached destination", "red"))
                    return True

        # If out of lane
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.config['out_lane_thres']:
            if self.config['verbose']:
                print(colored("out of lane", "red"))
            return True

        return False

    def _destroy_actors(self):
        # destroy sensors, ego vehicle and social actors

        if 'sensors' in self.to_clean.keys():
            if self.config['verbose']:
                print(colored("destroying sensors", "white"))
            for sensor in self.to_clean['sensors']:
                if sensor.is_listening:
                    sensor.stop()
                if sensor.is_alive:
                    sensor.destroy()

        if 'controllers' in self.to_clean.keys():
            if self.config['verbose']:
                print(colored("destroying controllers", "white"))
            for controller in self.to_clean['controllers']:
                controller.stop()
                if controller.is_alive:
                    controller.destroy()

        if 'vehicles' in self.to_clean.keys():
            if self.config['verbose']:
                print(colored("destroying vehicles and walkers", "white"))
            for actor in self.to_clean['vehicles']:
                if actor.is_alive:
                    actor.destroy()

    def _set_synchronous_mode(self, state: bool):
        self.settings.fixed_delta_seconds = None
        if state:
            self.settings.fixed_delta_seconds = self.config['dt']
        self.settings.synchronous_mode = state
        self.world.apply_settings(self.settings)

    def _create_vehicle_blueprint(self, actor_filter, color=None, number_of_wheels=None):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        if number_of_wheels is None:
            number_of_wheels = [4]
        blueprints = self.blueprint_library.filter(actor_filter)
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

    def _try_spawn_random_vehicle_at(self, transform):
        """Try to spawn a surrounding vehicle at specific transform with random blueprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_blueprint('vehicle.*', number_of_wheels=[4])
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        return vehicle

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

        walker_controller_actor = None
        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
        return walker_actor, walker_controller_actor

    def set_actors(self, vehicles: int, walkers: int):

        spawned_vehicles, spawned_walkers, spawned_walker_controllers = [], [], []
        vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        if self.config['verbose']:
            print(colored("spawning vehicles", "white"))
        # Spawn surrounding vehicles
        count = vehicles
        max_tries = count * 2
        while max_tries > 0:
            random_vehicle = self._try_spawn_random_vehicle_at(random.choice(vehicle_spawn_points))
            if random_vehicle:
                spawned_vehicles.append(random_vehicle)
                count -= 1
            if count <= 0:
                break
            max_tries -= 1

        if self.config['verbose']:
            print(colored("spawning walkers", "white"))
        # Spawn pedestrians
        walker_spawn_points = []
        for i in range(walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)

        count = walkers
        max_tries = count * 2
        while max_tries > 0:
            random_walker, random_walker_controller = self._try_spawn_random_walker_at(
                random.choice(walker_spawn_points))
            if random_walker and random_walker_controller:
                spawned_walkers.append(random_walker)
                spawned_walker_controllers.append(random_walker_controller)
                count -= 1
            if count <= 0:
                break
            max_tries -= 1

        if self.config['verbose']:
            print(colored("activating autopilots", "white"))
        # set autopilot for all the vehicles
        for v in spawned_vehicles:
            v.set_autopilot(True, self.tm_port)

        return spawned_vehicles, spawned_walkers, spawned_walker_controllers

    def set_camera(self, vehicle, sensor_width: int, sensor_height: int, fov: int):
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=1.0, z=2.0))
        rgb_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        rgb_camera.blur_amount = 0.0
        rgb_camera.motion_blur_intensity = 0
        rgb_camera.motion_max_distortion = 0

        # Camera calibration
        calibration = np.identity(3)
        calibration[0, 2] = sensor_width / 2.0
        calibration[1, 2] = sensor_height / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
        return rgb_camera

    def set_collision_sensor(self, vehicle):
        """
        In case of collision, this sensor will update the 'collision_info' attribute with a dictionary that contains
        the following keys: ["frame", "actor_id", "other_actor"].
        """
        bp = self.blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        return collision_sensor

    def update_spectator(self, vehicle):
        """
        The following code would move the spectator actor, to point the view towards a desired vehicle.
        """
        spectator = self.world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))

    def set_ego(self):
        """
        Add ego agent to the simulation. Return an Carla.Vehicle object.
        :return: The ego agent and ego vehicle if it was added successfully. Otherwise returns None.
        """
        # These vehicles are not considered because
        # the cameras get occluded without changing their absolute position
        info = {}
        available_vehicle_bps = [bp for bp in self.blueprint_library.filter("vehicle.*")]
        ego_vehicle_bp = random.choice([x for x in available_vehicle_bps if x.id not in
                                        ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck',
                                         'vehicle.volkswagen.t2', 'vehicle.bh.crossbike']])

        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        ego_vehicle = self.try_spawn_ego(ego_vehicle_bp, spawn_points)
        if ego_vehicle is None:
            print(colored("Couldn't spawn ego vehicle", "red"))
            return None
        info['vehicle'] = ego_vehicle.type_id
        info['id'] = ego_vehicle.id
        self.update_spectator(vehicle=ego_vehicle)

        for v in self.world.get_actors().filter("vehicle.*"):
            if v.id != ego_vehicle.id:
                v.set_autopilot(True)

        return ego_vehicle, info

    def try_spawn_ego(self, ego_vehicle_bp, spawn_points):
        ego_vehicle = None
        for p in spawn_points:
            ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, p)
            if ego_vehicle:
                ego_vehicle.set_autopilot(False)
                return ego_vehicle
        return ego_vehicle
