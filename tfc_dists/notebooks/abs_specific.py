import random

import numpy as np
import scipy.stats as ss
from numpy.core.umath_tests import inner1d

random.seed(12345)

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def normalize(v):
    return v / np.sqrt(inner1d(v, v))


def bearing_to_angle(bearing, is_rad=False):
    if is_rad:
        return (2 * np.pi - (bearing - (0.5 * np.pi))) % (2 * np.pi)
    else:
        return (360 - (bearing - 90)) % 360


class Traffic:

    def __init__(self, bounds, density, velocities, vert_rate, tracks, timestep=1):
        self.target_density = density
        assert len(bounds) == 6
        self.x_bounds = bounds[:2]
        self.y_bounds = bounds[2:4]
        self.z_bounds = bounds[4:]
        assert self.x_bounds[0] < self.x_bounds[1]
        assert self.y_bounds[0] < self.y_bounds[1]
        assert self.z_bounds[0] < self.z_bounds[1]
        self.x_size = abs(self.x_bounds[0] - self.x_bounds[1])
        self.y_size = abs(self.y_bounds[0] - self.y_bounds[1])
        self.z_size = abs(self.z_bounds[0] - self.z_bounds[1])
        self.centre_coord = [
            self.x_bounds[0] + (self.x_size // 2),
            self.y_bounds[0] + (self.y_size // 2),
            self.z_bounds[0] + (self.z_size // 2),
        ]

        self.total_vol = self.x_size * self.y_size * self.z_size
        # experiment volume is ellipse with axes 80% of xy sizes
        self.exp_vol = self.x_size * self.y_size * 0.8 * np.pi * self.z_size
        self.target_agents = int(self.target_density * self.total_vol) + 1
        self.timestep = timestep

        self.velocity_distr = velocities
        self.track_distr = tracks
        self.vert_rate_distr = vert_rate
        self.velocities = np.empty((self.target_agents, 3))
        self.positions = np.empty((self.target_agents, 3))

    def setup(self, **kwargs):
        self.add_agents(num=self.target_agents, init=True)

    def step(self, **kwargs):
        self.positions = self.positions + self.velocities * self.timestep
        # self.filter_oob_agents()

        oob_mask = ((self.x_bounds[0] > self.positions[:, 0])
                    | (self.positions[:, 0] > self.x_bounds[1])
                    | (self.y_bounds[0] > self.positions[:, 1])
                    | (self.positions[:, 1] > self.y_bounds[1])
                    | (self.z_bounds[0] > self.positions[:, 2])
                    | (self.positions[:, 2] > self.z_bounds[1]))
        self.positions = self.positions[oob_mask]
        self.velocities = self.velocities[oob_mask]

        # Collect stats here

        n_agent_diff = self.target_agents - self.positions.shape[0]
        if n_agent_diff > 0:
            self.add_agents(n_agent_diff+5)

    def end(self, **kwargs):
        pass

    def add_agents(self, num=1, init=False):
        if hasattr(self.velocity_distr, 'rvs'):
            vels = abs(np.array(self.velocity_distr.rvs(num)))
        elif hasattr(self.velocity_distr, 'resample'):
            vels = abs(np.array(self.velocity_distr.resample(num)).flatten())
        else:
            vels = np.ones(num) * abs(self.velocity_distr)

        if hasattr(self.track_distr, 'rvs'):
            tracks = np.array(self.track_distr.rvs(num)) % 360 + 1
        elif hasattr(self.track_distr, 'resample'):
            tracks = np.array(self.track_distr.resample(num).flatten()) % 360 + 1
        else:
            tracks = np.ones(num) * self.track_distr
        angles = np.radians(bearing_to_angle(tracks))
        xy_vels = np.vstack((np.cos(angles), np.sin(angles))).T * vels[:, None]

        if hasattr(self.vert_rate_distr, 'rvs'):
            vert_rates = np.array(self.vert_rate_distr.rvs(num))
        elif hasattr(self.vert_rate_distr, 'resample'):
            vert_rates = np.array(self.vert_rate_distr.resample(num)).flatten()
        else:
            vert_rates = np.ones(num) * self.vert_rate_distr
        new_velocities = np.hstack((xy_vels, vert_rates[:, None]))

        new_positions = np.vstack((
            (np.random.rand(num) * self.x_bounds[1]) + self.x_bounds[0],
            (np.random.rand(num) * self.y_bounds[1]) + self.y_bounds[0],
            (np.random.rand(num) * self.z_bounds[1]) + self.z_bounds[0],
        )).T

        if init:
            self.positions = new_positions
            self.velocities = new_velocities
        else:
            self.positions = np.vstack((self.positions, new_positions))
            self.velocities = np.vstack((self.velocities, new_velocities))

    def filter_oob_agents(self):
        # oob_indices = np.where(
        #     (self.x_bounds[0] > self.positions[:, 0])
        #     | (self.positions[:, 0] > self.x_bounds[1])
        #     | (self.y_bounds[0] > self.positions[:, 1])
        #     | (self.positions[:, 1] > self.y_bounds[1])
        #     | (self.z_bounds[0] > self.positions[:, 2])
        #     | (self.positions[:, 2] > self.z_bounds[1])
        # )
        # self.positions = np.delete(self.positions, oob_indices, axis=0)
        # self.velocities = np.delete(self.velocities, oob_indices, axis=0)

        oob_mask = ((self.x_bounds[0] > self.positions[:, 0])
                    | (self.positions[:, 0] > self.x_bounds[1])
                    | (self.y_bounds[0] > self.positions[:, 1])
                    | (self.positions[:, 1] > self.y_bounds[1])
                    | (self.z_bounds[0] > self.positions[:, 2])
                    | (self.positions[:, 2] > self.z_bounds[1]))
        self.positions = self.positions[oob_mask]
        self.velocities = self.velocities[oob_mask]

    def get_exp_agents(self):
        ooexp_indices = np.where(
            (
                    (np.power((self.positions[:, 0] - self.centre_coord[0]), 2) / np.power(self.x_size, 2)) +
                    (np.power((self.positions[:, 1] - self.centre_coord[1]), 2) / np.power(self.y_size, 2))
            ) >= 1
        )
        return np.delete(self.positions, ooexp_indices, axis=0)


class OwnshipAgent:

    def __init__(self, path, velocity):
        self.path = np.array(path)
        self.position = self.path[0, :]
        self.path_idx = 1
        self.velocity = velocity

        assert self.path.shape[1] == 3

    def setup(self, **kwargs):
        pass

    def step(self, **kwargs):
        sub_goal = self.path[self.path_idx, :]
        vec_to_goal = sub_goal - self.position
        step_to_goal = normalize(vec_to_goal) * self.velocity
        if np.sqrt(inner1d(step_to_goal, step_to_goal)) > np.sqrt(inner1d(vec_to_goal, vec_to_goal)):
            self.path_idx += 1
        new_pos = self.position + step_to_goal
        self.position = new_pos

    def end(self, **kwargs):
        pass


class Simulation:

    def __init__(self, traffic: Traffic, ownship: OwnshipAgent, steps=5000, conflict_dists=(15, 10)):
        self.steps = steps
        self.traffic = traffic
        self.ownship = ownship
        self.conflict_xy_dist = conflict_dists[0]
        self.conflict_z_dist = conflict_dists[1]
        self.setup_data_logging()

    def setup(self):
        pass

    def run(self):
        self.traffic.setup()
        self.ownship.setup()

        for t in range(self.steps):
            if self.ownship.path_idx >= self.ownship.path.shape[0]:
                print(f'\nOwnship reached end of path in {t} steps')
                self.end(early_end_t=t)
                break

            self.traffic.step()
            self.ownship.step()

            print(f'\rCompleted {t} steps', end="")

            traffic_dists = self.traffic.positions - self.ownship.position
            xy_dists = np.linalg.norm(traffic_dists[:, :2], axis=1)
            z_dists = np.abs(traffic_dists[:, 2])
            # conflict_indices = np.where((xy_dists < self.conflict_xy_dist) & (z_dists < self.conflict_z_dist))[0]
            # n_conflicts = ((xy_dists < self.conflict_xy_dist) & (z_dists < self.conflict_z_dist)).sum()

            self.conflict_log += int(((xy_dists < self.conflict_xy_dist) & (z_dists < self.conflict_z_dist)).sum())

        self.traffic.end()
        self.ownship.end()

    def end(self, early_end_t=None):
        self.end_timestep = early_end_t if not None else self.steps

    def setup_data_logging(self):
        self.conflict_log = 0


class BatchSimulation:
    pass


if __name__ == '__main__':
    for _ in range(200):
        tfc = Traffic([0, 1e4, 0, 1e4, 0, 1524], 1e-8, ss.norm(110, 20), ss.norm(0, 2), ss.uniform(1, 360))
        own = OwnshipAgent([[1, 1, 20], [300, 600, 800], [2000, 5000, 900], [3000, 6000, 20]], 10)
        sim = Simulation(tfc, own, conflict_dists=(20, 15))
        sim.run()
        sim.end()
        print(sim.conflict_log)
