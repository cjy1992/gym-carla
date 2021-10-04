import numpy as np

STEER_RANGE = (-40, 40)
MAX_THROTTLE = 20
MAX_BRAKE = 20

def range_normalization(value: float, maximum: float, minimum: float, factor: float=1):
    """Variable linear normalization to range [-factor, factor]

    Args:
        value (float): Unnormalized variable
        maximum (float): Maximum value
        minimum (float): Minimum value
        factor (float, optional): Range Scale. Defaults to 1.

    Returns:
        float: Normalized variable
    """    
    value_traslation = (minimum + maximum) / 2
    value_factor = np.abs(maximum - value_traslation)
    value = np.clip(value, minimum, maximum)
    value = (value - value_traslation) / value_factor
    return value * factor

def range_unnormalization(value: float, maximum: float, minimum: float, factor: float=1):
    """Variable linear unnormalization from range [-factor, factor]

    Args:
        value (float): Normalized variable
        maximum (float): Maximum value
        minimum (float): Minimum value
        factor (float, optional): Range Scale. Defaults to 1.

    Returns:
        float: Unnormalized variable
    """    
    value_traslation = (minimum + maximum) / 2
    value_factor = np.abs(maximum - value_traslation)
    value = np.clip(value, -1, 1)
    value = value_factor * value / factor + value_traslation
    return value

def normalize_action(action: list):
    throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])
    # Steer Normalization to [-0.5, 0.5]
    steer = range_normalization(steer, STEER_RANGE[1], STEER_RANGE[0], 0.5)
    
    # Throttle and Brake Normalization, allways positive
    throttle = np.abs(throttle) / MAX_THROTTLE
    brake = np.abs(brake) / MAX_BRAKE

    return throttle, brake, steer

def unnormalize_action(action: list):
    throttle, brake, steer = float(action[0]), float(action[1]), float(action[2])

    steer = range_unnormalization(steer, STEER_RANGE[1], STEER_RANGE[0], 0.5)

    # Throttle and Brake UnNormalization
    throttle = np.abs(throttle) * MAX_THROTTLE
    brake = np.abs(brake) * MAX_BRAKE

    return throttle, brake, steer

TARGET_SPEED_RANGE = (0, 20)
def normalize_pid_action(action: list):
    target_speed, steer = float(action[0]), float(action[1])
    # Steer Normalization from [-40, 40] to [-0.5, 0.5]
    steer = range_normalization(steer, STEER_RANGE[1], STEER_RANGE[0], 0.5)
    # Target Speed normalization from [0, 20] to [-1, 1]
    target_speed = range_normalization(target_speed, TARGET_SPEED_RANGE[1], TARGET_SPEED_RANGE[0], 1)
    return steer, target_speed

def unnormalize_pid_action(action: list):
    target_speed, steer = float(action[0]), float(action[1])
    steer = range_unnormalization(steer, STEER_RANGE[1], STEER_RANGE[0], 0.5)
    target_speed = range_unnormalization(target_speed, TARGET_SPEED_RANGE[1], TARGET_SPEED_RANGE[0], 1)
    return steer, target_speed