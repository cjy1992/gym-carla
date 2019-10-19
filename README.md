# gym-carla
A Gym wrapper for CARLA simulator

## Installation

1. Clone this git repo
```
$ git clone https://github.com/cjy1992/gym-carla.git
```

2. Enter the repo root folder and install the package:
```
$ pip install -e .
```

3. Download [CARLA_0.9.6](https://github.com/carla-simulator/carla/releases/tag/0.9.6), extract it to some folder, and add CARLA to ```PYTHONPATH``` environment variable:
```
$ export PYTHONPATH=$PYTHONPATH:$YourFolder$/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```

4. Enter the CARLA root folder and launch the CARLA server by:
```
$ ./CarlaUE4.sh -windowed
```
You can use ```Alt+F1``` to get back your mouse control.


5. Run ```test.py``` in the repo root folder
