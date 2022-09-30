import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, cdist
import scipy.stats as ss
import scipy.optimize as sio
import agentpy as ap
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import IPython
import random
import pytransform3d.plot_utils as p3d
from pytransform3d.transformations import translate_transform
import pandas as pd

rng = np.random.default_rng(12345)
random.seed(12345)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class AircraftAgent(ap.Agent):
    def setup(self):
        self.max_vel = 205
        self.path = []

    def setup_fail(self, fail_prob):
        self.fail_prob = fail_prob
        self.failed = False

    def setup_pos(self, space: ap.Space):
        self.space = space
        self.neighbors = space.neighbors
        self.position = space.positions[self]

    def setup_goal(self, space: ap.Space, goal=np.array([0, 0, 0])):
        self.goal = goal
        self.path.append(goal)
        self.path_idx = 0

    def update(self):
        if random.random() < self.fail_prob and not self.failed:
            self.failed = True
            self.position[2] = 100
            for p in self.path:
                p[2] = 100
            self.space.move_to(self, self.position)
            return
        sub_goal = self.path[self.path_idx]
        vec_to_goal = sub_goal - self.position
        step_to_goal = normalize(vec_to_goal) * self.max_vel
        if (np.linalg.norm(step_to_goal) > np.linalg.norm(vec_to_goal)):
            try:
                self.space.remove_agents([self])
            except KeyError:
                pass
            return
        new_pos = self.position + step_to_goal
        self.position = new_pos
        self.space.move_to(self, new_pos)


class LimaModel(ap.Model):
    def setup(self):
        self.shape = np.array([self.p.xy_size, self.p.xy_size, self.p.z_size])
        self.space = ap.Space(self, shape=self.shape)
        volume = self.p.xy_size * self.p.xy_size * self.p.z_size
        self.start_pop = int(np.ceil(volume * self.p.target_density))
        print(self.start_pop)
        self.agents = ap.AgentList(self, self.start_pop, AircraftAgent)
        #         lanes = [self.get_random_lane() for agent in self.agents]
        start_pos = [rng.random(3) * self.shape for _ in self.agents]
        self.space.add_agents(self.agents, positions=start_pos)
        self.agents.setup_pos(self.space)
        self.agents.setup_fail(self.p.fail_prob)
        for agent, start in zip(self.agents, start_pos):
            goal = self.make_goal_pos(start)
            agent.setup_goal(self.space, goal)

        self.xy_sep_maxs = []
        self.z_sep_maxs = []
        self.xy_sep_means = []
        self.z_sep_means = []
        self.xy_sep_mins = []
        self.z_sep_mins = []
        self.tfc_dens = []
        self.conflict_counts = []
        self.pop = []

    def step(self):
        self.agents.update()
        if len(self.agents) < (self.start_pop + ss.norm(0, 4).rvs()):
            new_agent = AircraftAgent(self)
            self.agents.append(new_agent)
            self.space.add_agents([new_agent], [rng.random(3) * self.shape])
            new_agent.setup_pos(self.space)
            new_agent.setup_goal(self.space, rng.random(3) * self.shape)
            new_agent.setup_fail(self.p.fail_prob)

    def update(self):
        exp_positions = self.filter_exp_positions(self.agents.position)
        n_agents = len(exp_positions)

        if len(self.space.agents) < 1:
            self.end()
            return

        # Separations
        xy_seps = pdist(exp_positions[:, :2], 'euclidean')
        z_seps = pdist(np.vstack((exp_positions[:, 2], np.zeros(exp_positions.shape[0]))).T, 'euclidean')
        # assert xy_seps.shape[0] == z_seps.shape[0]
        # xy_seps = xy_seps[xy_seps > 0.0]  # Remove distance from self
        # z_seps = z_seps[z_seps > 0.0]  # Remove distance from self
        # assert xy_seps.shape[0] == z_seps.shape[0]
        self.record('Mean XY Sep', xy_seps.mean())
        self.record('Mean Z Sep', z_seps.mean())
        self.record('Min XY Sep', xy_seps.min())
        self.record('Min Z Sep', z_seps.min())
        self.record('Max XY Sep', xy_seps.max())
        self.record('Max Z Sep', z_seps.max())
        xy_conflicts = xy_seps < self.p.conflict_xy_dist
        z_conflicts = z_seps < self.p.conflict_z_dist
        conflicts = np.logical_and(xy_conflicts, z_conflicts)
        self.record('Instant Conflict Counts', conflicts.sum())
        self.record('Instant Population', n_agents)

        self.xy_sep_maxs += [xy_seps.max()]
        self.z_sep_maxs += [z_seps.max()]
        self.xy_sep_mins += [xy_seps.min()]
        self.z_sep_mins += [z_seps.min()]
        self.xy_sep_means += [xy_seps.mean()]
        self.z_sep_means += [z_seps.mean()]
        self.tfc_dens += [n_agents / (self.p.xy_size * self.p.xy_size * self.p.z_size)]
        self.pop += [n_agents]

        # Traffic Densities
        self.record('Traffic Density', n_agents / (self.p.xy_size * self.p.xy_size * self.p.z_size))

    def end(self):
        self.report('Min XY Sep', np.array(self.xy_sep_mins).min())
        self.report('Min Z Sep', np.array(self.z_sep_mins).min())
        self.report('Mean XY Sep', np.array(self.xy_sep_means).mean())
        self.report('Mean Z Sep', np.array(self.z_sep_means).mean())
        self.report('Max XY Sep', np.array(self.xy_sep_maxs).max())
        self.report('Max Z Sep', np.array(self.z_sep_maxs).max())
        self.report('Max Traffic Density', np.array(self.tfc_dens).max())
        self.report('Min Traffic Density', np.array(self.tfc_dens).min())
        self.report('Mean Traffic Density', np.array(self.tfc_dens).mean())
        self.report('Mean Instant Conflict Count', np.array(self.conflict_counts).mean())
        self.report('Mean Population', np.array(self.pop).mean())

    def get_random_lane(self):
        return random.choice(self.p.lanes)

    def make_goal_pos(self, start):
        xy_goal = np.array(rng.random(2) * self.p.xy_size)
        dist = np.linalg.norm(xy_goal - start[:2])
        climb_or_descend = rng.random() > 0.5
        vert_dist = dist * np.sin(np.radians(self.p.vert_angle))
        z_goal = start[2] + vert_dist if climb_or_descend else start[2] - vert_dist
        return np.hstack([xy_goal, z_goal])

    def filter_exp_positions(self, positions):
        positions = np.array(positions)
        midpoint = np.array([self.p.xy_size // 2, self.p.xy_size // 2, self.p.z_size // 2])
        xy_distances = np.array([np.linalg.norm(pos[:2] - midpoint[:2]) for pos in positions])
        return positions[xy_distances < self.p.xy_size * 0.8]


if __name__ == '__main__':
    params = {
        'xy_size': 10000,
        'z_size': 914,
        'target_density': 5e-9,
        # 'population': 200,
        'steps': 20000,
        'vert_angle': 5.6,
        'conflict_xy_dist': 15,
        'conflict_z_dist': 5,
        # 'spawn_rate': 10000,  #steps per spawn
        'fail_prob': 0.0,  # Failure probability of each agent at each timestep
    }
    model = LimaModel(params)
    results = model.run()

    df = results.variables.LimaModel

    df.plot('Instant Population', 'Instant Conflict Counts', kind='scatter')
    df.reset_index().plot('t', 'Traffic Density', kind='scatter')
    # exit()
# param_ranges = {
#     'seed': 12345,
#     'ndim': 3,
#     'size': 5000,
#     'population': ap.IntRange(2, 5000),
#     'steps': 1000,
#     'n_layers': 10,
#     'spawn_rate': 2000,  # steps per spawn
#     'fail_prob': 0.0,  # Failure probability of each agent at each timestep
# }
# sample = ap.Sample(param_ranges, n=32, method='saltelli')
# exp = ap.Experiment(LimaModel, sample, record=True)
# exp_results = exp.run(n_jobs=-1)
#
# res = exp_results.arrange_reporters()
#
# exp_df = exp_results.reporters
# exp_df.plot('Mean Traffic Density', 'Min 3D Sep', kind='scatter')
#
#
# def count_diffs(a, b, to_min, to_max):
#     n_better = 0
#     n_worse = 0
#
#     for f in to_min:
#         n_better += a[f] < b[f]
#         n_worse += a[f] > b[f]
#
#     for f in to_max:
#         n_better += a[f] > b[f]
#         n_worse += a[f] < b[f]
#
#     return n_better, n_worse
#
#
# def find_skyline_bnl(df, to_min, to_max):
#     """Finds the skyline using a block-nested loop."""
#
#     rows = df.to_dict(orient='index')
#
#     # Use the first row to initialize the skyline
#     skyline = {df.index[0]}
#
#     # Loop through the rest of the rows
#     for i in df.index[1:]:
#
#         to_drop = set()
#         is_dominated = False
#
#         for j in skyline:
#
#             n_better, n_worse = count_diffs(rows[i], rows[j], to_min, to_max)
#
#             # Case 1
#             if n_worse > 0 and n_better == 0:
#                 is_dominated = True
#                 break
#
#             # Case 3
#             if n_better > 0 and n_worse == 0:
#                 to_drop.add(j)
#
#         if is_dominated:
#             continue
#
#         skyline = skyline.difference(to_drop)
#         skyline.add(i)
#
#     return pd.Series(df.index.isin(skyline), index=df.index)
#
#
# pareto_front = find_skyline_bnl(exp_df, to_min=[], to_max=['Mean Traffic Density', 'Min 3D Sep'])
# # exp_df[pareto_front].sort_values('Mean Traffic Density').plot('Mean Traffic Density')
# exp_df[pareto_front].sort_values('Mean Traffic Density').plot('Max Traffic Density', 'Min 3D Sep', c='r')
# exp_df[pareto_front].sort_values('Mean Traffic Density').plot('Max Traffic Density', 'Min 3D Sep', kind='scatter',
#                                                               c='r')
# print(exp_df['Traffic Density'][exp_df['Traffic Density'] < exp_df['Traffic Density'].quantile(0.9)])
# print(exp_df['Traffic Density'])
#
# exp_df.plot(x='Max Traffic Density', y='Min 3D Sep', kind='scatter', logx=True)
