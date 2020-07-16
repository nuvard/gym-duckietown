# Author: Mikita Sazanovich + Nastya Nikolskaya

import numpy as np
import scipy.stats as stats

# parameters for the pure pursuit controller
PEAK_VELOCITY = 0.3
GAIN = 1.7
FOLLOWING_DISTANCE = 1

MAX = -100000
COF_LANE = 1000
COF_ALIGN = 100000
COF_DIST = 10000
COF_SPEED = 500
 
class PurePursuitExpert:
    def __init__(self, env, following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env.unwrapped
        self.vel = PEAK_VELOCITY
        self.following_distance = following_distance
        self.prev_vec = None
        self.Kp = 0#0.3
        self.min_vel = PEAK_VELOCITY
        self.target_vel = 1
        self.gain = GAIN
        self.max_iterations = max_iterations

    def predict(self, observation):  # we don't really care about the observation for this implementation
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
        if closest_point is None:
            return 0.0, 0.0  # Should return done in the environment

        iterations = 0
        lookup_distance = self.following_distance * self.vel
        curve_point = None
        #self.dream_action()
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_tangent = self.env.closest_curve_point(follow_point, self.env.cur_angle)
 
            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)
        
        dot = np.dot(curve_tangent, self.env.get_dir_vec())
        dot /=  np.linalg.norm(curve_tangent)*np.linalg.norm(self.env.get_dir_vec())
        angle = np.arccos(dot)
        self.max_steering = 2
        if angle < 0.01:
            self.vel += 0.03 # += self.pid_control()
            print(self.vel)
            self.gain *=0.9
            self.max_steering = 0.2

        if angle > 0.2:
            self.vel = PEAK_VELOCITY
            self.gain = GAIN
        
        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = self.gain * -dot
        steering = np.clip(steering, -1*self.max_steering, self.max_steering)
        if self.prev_vec is not None:
          #print(point_vec)
          #print(prev_point_vec)
          coeff = np.dot(point_vec, prev_vec)
          coeff /= np.linalg.norm(point_vec)*np.linalg.norm(self.prev_vec) 
          #print(coeff)
          prev_vec = 0.9 * self.prev_vec + 0.1 * point_vec
        else:
          prev_vec = point_vec

        velocity = (PurePursuitExpert.__get_speed_density_at(steering)
                    / PurePursuitExpert.__get_speed_density_at(0.0)
                    * self.vel)
        return (velocity, steering)
    
    def dream_action(self):
        i, j = self.env.get_grid_coords(self.env.cur_pos)
        #curve = self.env._get_curve(i,j)
        #print(curve)
        dist = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle).dist
        try:
          curve = self.env._get_tile(i,j)['kind'].startswith('curve')
          print(curve)
        except ValueError:
            curve = False
        if not curve:
            if abs(dist) < 0.05:
                self.cof_speed = COF_SPEED * 5

    def pid_control(self):
        """
        Proportional control for the speed.
        :param target: (float)
        :param current: (float)
        :return: (float)
        """
        return self.Kp * (self.target_vel - self.vel)

    @staticmethod
    def __get_speed_density_at(x):
        return stats.norm.pdf(x, 0.0, 2.0)

    

class StanleyExpert:
    def __init__(self, env, following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env.unwrapped
        self.peak_vel = 0.3
        self.delta_gain = 6
        self.psi_gain = 1
        self.k_coeff = 0.2
        self.following_distance = following_distance
        self.max_iterations = max_iterations

    def predict(self, observation, prev_point_vec, prev_vel):  # we don't really care about the observation for this implementation
        
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
        closest_point_vec = closest_point - self.env.cur_pos
        error = np.linalg.norm(closest_point_vec)
        error_vec = closest_point_vec
        closest_point_vec /= error

        if closest_point is None:
            return 0.0, 0.0  # Should return done in the environment

               
        curve_vec = (closest_point_vec - self.env.cur_pos)
        vel = self.peak_vel
        dot = np.dot(closest_tangent, self.env.get_dir_vec())
        dot /=  np.linalg.norm(closest_tangent)*np.linalg.norm(self.env.get_dir_vec())
        psi_angle = np.arccos(dot)
        print(psi_angle)
        delta = np.arctan(self.k_coeff * error / (vel+1e-8))
        steering =  - (self.psi_gain * psi_angle + self.delta_gain * delta) 
        
        velocity = (StanleyExpert.__get_speed_density_at(steering)
                    / StanleyExpert.__get_speed_density_at(0.0)
                    * vel)
        return (velocity, steering), prev_point_vec

    def dream_action(self):
        i, j = self.env.get_grid_coords(self.env.cur_pos)
        #curve = self.env._get_curve(i,j)
        #print(curve)
        dist = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle).dist
        try:
          curve = self.env._get_tile(i,j)['kind'].startswith('curve')
        except ValueError:
            curve = False
        if not curve:
            if abs(dist) < 0.05:
                self.cof_speed = COF_SPEED * 5

    def pid_control(target, current):
        """
        Proportional control for the speed.
        :param target: (float)
        :param current: (float)
        :return: (float)
        """
        return Kp * (target - current)

                
    @staticmethod
    def __get_speed_density_at(x):
        return stats.norm.pdf(x, 0.0, 2.0)