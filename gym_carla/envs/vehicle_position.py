import numpy as np


def vec3d_dot(v1, v2):
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z


def vec3d_mag(v):
    return np.sqrt(vec3d_dot(v, v))


def vec3d_angle(v1, v2, deg=False):
    cos_theta = vec3d_dot(v1, v2) / (vec3d_mag(v1) * vec3d_mag(v2))
    if deg:
        return np.rad2deg(np.arccos(cos_theta))
    return np.arccos(cos_theta)


def get_vehicle_orientation(world_map, vehicle):
    """
    Returns the vehicle's orientation w.r.t to lane's tangent
    """

    # get lane tangent vector
    a1 = world_map.get_waypoint(vehicle.get_location(), project_to_road=True)
    b1 = a1.next(2.0)[0]
    vector_lane = b1.transform.location - a1.transform.location

    # get vehicle vector
    vertices = vehicle.bounding_box.get_world_vertices(vehicle.get_transform())
    a2 = vertices[0]
    b2 = vertices[4]
    vector_vehicle = b2 - a2

    return vec3d_angle(vector_vehicle, vector_lane, deg=True)


def get_vehicle_position(world_map, vehicle):
    """
    Returns the distance of the vehicle to lane's center
    """
    vehicle_center = vehicle.get_transform()
    lane_center = world_map.get_waypoint(vehicle.get_location(), project_to_road=True)
    d_x = vehicle_center.location.x - lane_center.transform.location.x
    d_y = vehicle_center.location.y - lane_center.transform.location.y
    d = np.sqrt(d_x * d_x + d_y * d_y)
    return d
