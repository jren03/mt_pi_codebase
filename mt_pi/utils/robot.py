import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def get_waypoint(
    start_pose, target_pose, max_linear_delta=0.005, max_angular_delta=0.01
):
    """
    Generate waypoints for both position and orientation interpolation.

    :param start_pose: tuple (start_position, start_orientation) where each is a numpy array
    :param target_pose: tuple (target_position, target_orientation) where each is a numpy array
    :param max_linear_delta: maximum linear step size
    :param max_angular_delta: maximum angular step size in radians
    :return: generator function for waypoints, number of steps
    """
    start_pos, start_ori = start_pose
    target_pos, target_ori = target_pose

    # Calculate linear interpolation
    total_linear_delta = target_pos - start_pos
    linear_distance = np.linalg.norm(total_linear_delta)
    num_linear_steps = max(1, int(np.ceil(linear_distance / max_linear_delta)))
    linear_delta = total_linear_delta / num_linear_steps

    # Calculate angular interpolation
    # Step size is the maximum of the two step sizes
    total_angular_delta = target_ori - start_ori
    angular_distance = np.linalg.norm(total_angular_delta)
    num_angular_steps = max(1, int(np.ceil(angular_distance / max_angular_delta)))
    ori_chg = R.from_euler("xyz", [start_ori, target_ori], degrees=False)
    slerp = Slerp([0, max(num_angular_steps, num_linear_steps)], ori_chg)

    def gen_waypoint(i):
        step = min(i, num_linear_steps)

        # Calculate position using linear_delta and the current step
        pos = start_pos + linear_delta * step

        # Calculate orientation using slerp and the original step size
        ori = slerp(step).as_euler("xyz")

        return np.concatenate([pos, ori])

    return gen_waypoint, num_linear_steps


# rotation interpolation
def get_ori(initial_euler, final_euler, num_steps):
    diff = np.linalg.norm(final_euler - initial_euler)
    ori_chg = R.from_euler(
        "xyz", [initial_euler.copy(), final_euler.copy()], degrees=False
    )
    if diff < 0.02 or num_steps < 2:

        def gen_ori(i):
            return initial_euler

    else:
        slerp = Slerp([1, num_steps], ori_chg)

        def gen_ori(i):
            interp_euler = slerp(i).as_euler("xyz")
            return interp_euler

    return gen_ori


# positional interpolation
def get_waypoint_interpolation(start_pt, target_pt, max_delta=0.005):
    total_delta = target_pt - start_pt
    num_steps = (np.linalg.norm(total_delta) // max_delta) + 1
    remainder = np.linalg.norm(total_delta) % max_delta
    if remainder > 1e-3:
        num_steps += 1
    delta = total_delta / num_steps

    def gen_waypoint(i):
        return start_pt + delta * min(i, num_steps)

    return gen_waypoint, int(num_steps)
