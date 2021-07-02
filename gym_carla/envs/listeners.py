import numpy as np


# create listeners
def get_camera_img(weak_self, data):
    self = weak_self()
    if not self:
        return

    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (data.height, data.width, 4))
    array = array[:, :, :3]
    self.camera_img = array


def get_depth_img(weak_self, data):
    self = weak_self()
    if not self:
        return

    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (data.height, data.width, 4))
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000
    self.depth_array = depth_meters


def get_collision_hist(weak_self, event):
    self = weak_self()
    if not self:
        return

    self.collision_info = {
        "frame": event.frame,
        "other_actor": event.other_actor.type_id,
    }
    self.collision_hist.append(event)
