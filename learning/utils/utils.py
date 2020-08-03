import numpy as np
def compute_dist(env):
    closest_point, closest_tangent = env.closest_curve_point(env.cur_pos, env.cur_angle)
    closest_point_vec = closest_point - env.cur_pos
    error = np.linalg.norm(closest_point_vec)
    return error