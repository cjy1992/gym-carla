# gym-carla
A Gym wrapper for CARLA simulator

## System Requirements
- Ubuntu 16.04

## Installation
1. Setup conda environment
```
$ conda create -n env_name python=3.6
$ conda activate env_name
$ pip install -r requirements
```

2. Clone this git repo
```
$ git clone https://github.com/cjy1992/gym-carla.git
```

3. Enter the repo root folder and install the package:
```
$ pip install -e .
```

4. Download [CARLA_0.9.6](https://github.com/carla-simulator/carla/releases/tag/0.9.6), extract it to some folder, and add CARLA to ```PYTHONPATH``` environment variable:
```
$ export PYTHONPATH=$PYTHONPATH:$YourFolder$/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```

5. Enter the CARLA root folder and launch the CARLA server by:
```
$ ./CarlaUE4.sh -windowed
```
You can use ```Alt+F1``` to get back your mouse control.

Or you can run in non-display mode by:
```
$ DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000
```

## Usage
1. See ```test.py``` in the repo's root folder about how to use the CARLA gym wrapper.

2.  We provide a dictionary observation including front view camera (obs['camera']), birdeye view lidar point cloud (obs['lidar']) and birdeye view semantic representation (obs['birdeye']).
<div align="center">
  <img src="obs.png" width=75%>
</div>

3. The termination condition is either the ego vehicle collides, runs out of lane, reaches a destination, or reaches the maximum episode timesteps. Users may modify function _terminal in carla_env.py to enable customized termination condition.

4. The reward is a weighted combination of longitudinal speed and penalties for collision, exceeding maximum speed, out of lane, large steering and large lateral accleration.  Users may modify function _get_reward in carla_env.py to enable customized termination condition.
