import numpy as np


class MinJerkTrajectoryPlanner:
    def __init__(self, initial_positions, final_positions, duration):
        self.q_i = np.array(initial_positions)  # Initial joint positions (6 joints)
        self.q_f = np.array(final_positions)    # Final joint positions (6 joints)
        self.D = duration                       # Duration of the trajectory
        self.delta_q = self.q_f - self.q_i      # Change in joint positions

    def get_desired_position(self, t):
        if t >= self.D:
            return self.q_f
        tau = t / self.D
        scaling_factor = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        q_d = self.q_i + self.delta_q * scaling_factor
        return q_d
    
    def get_desired_velocity(self, t):
        if t >= self.D:
            return np.zeros(6)
        tau = t / self.D
        scaling_factor = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
        dq_d = self.delta_q * scaling_factor / self.D
        return dq_d
