import numpy as np
from datetime import datetime
def compute_dist(env):
    d_env = env.unwrapped
    closest_point, closest_tangent = d_env.closest_curve_point(d_env.cur_pos, d_env.cur_angle)
    closest_point_vec = closest_point - d_env.cur_pos
    error = np.linalg.norm(closest_point_vec)
    return error

def log_metrics(dists, filename='mean_distance.txt'):
    with open(filename, 'a') as f:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        string = f"Time: {current_time} | Mean distance: {np.mean(dists)} | Std: {np.std(dists)}\n"
        f.write(string)