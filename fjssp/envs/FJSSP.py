from collections import defaultdict
from typing import Optional, Type, Union

import gym
import numpy as np
import plotly.graph_objects as go
from gym import spaces
from gym.spaces.space import Space
from gym.utils import seeding


class FJSSPObservationSpace(Space[np.ndarray]):
    """FJSSP observation space."""

    def __init__(
        self,
        num_jobs: int,
        max_ops: int,
        num_machines: int,
        max_time_per_operation: int,
        mask_p: float,
        dtype: Type = np.int32,
        seed: Optional[Union[int, seeding.RandomNumberGenerator]] = None,
    ) -> None:

        self.num_jobs = num_jobs
        self.max_ops = max_ops
        self.num_machines = num_machines
        self.max_time_per_operation = max_time_per_operation
        self.mask_p = mask_p

        super().__init__(
            shape=(2, self.num_jobs, self.max_ops, self.num_machines),
            dtype=dtype,
            seed=seed,
        )

    def sample(self):
        # sample num_jobs x max_ops x num_machines tensor
        J = self.np_random.integers(
            1,
            self.max_time_per_operation + 1,
            size=(self.num_jobs, self.max_ops, self.num_machines),
        )

        # sample mask
        s = self.num_jobs * self.max_ops * self.num_machines
        M = np.zeros(s, dtype=np.int32)
        M[: int(self.mask_p * s)] = 1
        self.np_random.shuffle(M)
        M = M.reshape(self.num_jobs, self.max_ops, self.num_machines)

        # mask
        J *= M

        # ensure the operation precedence
        # i.e., row i should not be all zeros
        # if there is at least one positive element
        # at rows i+1, i+2, ...
        # sort by number of non zero rows
        for j in range(J.shape[0]):
            J[j] = J[j][(J[j] == 0).sum(axis=1).ravel().argsort()]

        # generate processing tensor
        P = np.zeros_like(J)

        return np.stack((J, P))

    def contains(self, x) -> bool:
        J, P = x[0], x[1]
        return bool(
            (x.shape == self.shape)
            and ((J.min() >= 0) and (J.max() <= self.max_time_per_operation))
            and np.array_equal(P, P.astype(bool))
        )
        # probably I should also check for jobs-operation-machine status
        # but will do it later...

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self) -> str:
        return (
            f"num_jobs: {self.num_jobs} "
            f"max_ops: {self.max_ops} "
            f"num_machines: {self.num_machines} "
            f"max_time_per_operation: {self.max_time_per_operation}"
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FJSSPObservationSpace)
            and self.shape == other.shape
            and self.num_jobs == other.num_jobs
            and self.max_ops == other.max_ops
            and self.num_machines == other.num_machines
            and self.max_time_per_operation == other.max_time_per_operation
        )


class FJSSPEnv(gym.Env):
    metadata = {"render_modes": [None]}

    def __init__(
        self,
        data_path=None,
        num_jobs=5,
        num_machines=5,
        max_ops=5,
        max_time_per_operation=10,
        mask_p=0.5,
    ) -> None:
        if data_path is not None:
            # read jobs from file
            self.jobs, self.num_jobs, self.num_machines, self.max_ops = self.read_jobs(
                data_path
            )

        else:
            self.jobs = None
            self.num_jobs = num_jobs
            self.num_machines = num_machines
            self.max_ops = max_ops
            self.max_time_per_operation = max_time_per_operation

        assert 0 <= mask_p <= 1, "mask_p should be between 0 and 1"
        self.mask_p = mask_p

        self.observation_space = FJSSPObservationSpace(
            num_jobs=self.num_jobs,
            max_ops=self.max_ops,
            num_machines=self.num_machines,
            max_time_per_operation=self.max_time_per_operation,
            mask_p=self.mask_p,
        )

        self.action_space = spaces.Dict(
            {
                # the last job is "do-nothing" job
                "selected_job": spaces.Discrete(self.num_jobs + 1),
                "selected_machine": spaces.Discrete(self.num_machines),
            }
        )

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # reset time
        self.total_time = 0

        # reset schedule
        self.schedule: defaultdict = defaultdict(lambda: defaultdict(list))

        # reset J and P tensors
        if self.jobs is not None:
            # provided from file
            self.J = np.zeros((self.num_jobs, self.max_ops, self.num_machines))
            for job_idx, job in enumerate(self.jobs):
                for op_idx, op in enumerate(job):
                    for m, t in op:
                        self.J[job_idx, op_idx, m - 1] = t
            self.P = np.zeros_like(self.J)
        else:
            self.observation_space = FJSSPObservationSpace(
                num_jobs=self.num_jobs,
                max_ops=self.max_ops,
                num_machines=self.num_machines,
                max_time_per_operation=self.max_time_per_operation,
                mask_p=self.mask_p,
                seed=seed,
            )
            JP = self.observation_space.sample()
            self.J, self.P = JP

        # get trivial upper bound
        self.get_upper_bound()

        # info message
        self.info_message = "good luck"

        observation = self.__get_obs()
        info = self.__get_info()

        return (observation, info) if return_info else observation

    def step(self, action):
        j, m = action["selected_job"], action["selected_machine"]
        reward = 0

        # do nothing job
        if j == self.num_jobs:
            self.__make_one_time_step()
            self.info_message = "Iterating to the next time step!"
            reward = -1
        else:
            # check if job is already being processed
            if np.sum(self.P[j]) > 0:
                self.info_message = (
                    f"Invalid action: job {j} is currently being processed!"
                )
                reward = -2
            # check the machine availability
            elif np.sum(self.P[:, :, m]) != 0:
                self.info_message = f"Invalid action: machine {m} is busy!"
                reward = -2
            else:
                # find the first non-processed operation i of job j
                idxs = np.argwhere(np.sum(self.J[j], axis=1) > 0)

                if len(idxs) > 0:
                    i = idxs[0][0]

                    # check if selected machine can perform the job
                    if self.J[j, i, m]:
                        # make updates
                        self.P[j, i, m] = 1
                        tmp = self.J[j, i, m]
                        self.J[j, i, :] = 0
                        self.J[j, i, m] = tmp
                        self.info_message = f"Action performed successfully!"
                        reward = 0
                    else:
                        self.info_message = f"Invalid action: the next operation of job {j} cannot be done by machine {m}!"
                        reward = -2
                else:
                    self.info_message = (
                        f"Invalid action: all operations for job {j} has finished!"
                    )
                    reward = -2

        if (np.sum(self.J) == 0) and (np.sum(self.P) == 0):
            done = True
            self.info_message = "All tasks are finished!"
            self.__update_schedule()
        elif self.total_time > self.upper_bound:
            done = True
            self.info_message = "Upper bound is reached!"
            self.__update_schedule()
        else:
            done = False

        observation = self.__get_obs()
        info = self.__get_info()

        return observation, reward, done, info

    def render(self, mode=None):
        pass

    def close(self):
        pass

    def __get_obs(self):
        return np.stack((self.J, self.P))

    def __get_info(self):
        return {
            "current_time": self.total_time,
            "info_message": self.info_message,
            "upper_bound": self.upper_bound,
        }

    def __update_schedule(self):
        """Updates schedule."""
        # job, operation, machine
        joms = np.array(np.where(self.P == 1)).T
        for jom in joms:
            j, o, m = jom
            self.schedule[j]["time"].append(self.total_time)
            self.schedule[j]["machine"].append(m)
            self.schedule[j]["operation"].append(o)

    def __make_one_time_step(self):
        """Makes one time step in environment."""
        # update schedule
        self.__update_schedule()

        # update time
        self.total_time += 1

        # update jobs tensor
        self.J = self.J - self.P

        # update processing tensor
        self.P = (self.J > 0) * self.P

    def get_upper_bound(self):
        """Returns trivial upper bound.
        Eeach job j is completely finished before job j+1.
        """
        self.upper_bound = 0
        for j in range(self.num_jobs):
            for i in range(self.max_ops):
                if np.sum(self.J[j, i, :]) == 0:
                    continue
                else:
                    best_t = np.inf
                    for m in range(self.num_machines):
                        if self.J[j, i, m] != 0:
                            best_t = np.min([best_t, self.J[j, i, m]])
                self.upper_bound += best_t

    def read_jobs(self, data_path):
        with open(data_path, "r") as f:
            lines = f.readlines()

            max_ops = 0

            # number of jobs, number of machines
            num_jobs, num_machines = [int(x) for x in lines[0].strip().split("\t")[:2]]

            jobs = []
            for line in lines[1:]:
                line = [int(x) for x in line.strip().split(" ") if x != ""]
                if len(line) == 0:
                    continue

                job = []
                n_ops = line[0]
                max_ops = np.max([n_ops, max_ops])
                i = 1
                while i < len(line) - 1:
                    ops = []
                    n_machines = line[i]
                    j = i + 1
                    while j < i + 1 + 2 * n_machines - 1:
                        # (machine, processing_time)
                        ops.append((line[j], line[j + 1]))
                        j += 2
                    job.append(ops)

                    i = j
                jobs.append(job)
        return jobs, num_jobs, num_machines, max_ops

    def plot_schedule(self):
        """Plots schedule."""
        fig = go.Figure()

        for k in self.schedule.keys():
            fig.add_trace(
                go.Scatter(
                    x=self.schedule[k]["time"],
                    y=self.schedule[k]["machine"],
                    mode="markers",
                    name=f"J{k}",
                    marker_symbol="square",
                )
            )

        fig.update_layout(
            {
                "title": "FJSSP",
                "xaxis": dict(title="Time", range=[-1, self.total_time + 1]),
                "yaxis": dict(
                    title="Machines",
                    range=[-1, self.num_machines],
                    tickvals=list(range(self.num_machines)),
                    ticktext=[f"M-{m}" for m in range(self.num_machines)],
                ),
            }
        )
        fig.show()
