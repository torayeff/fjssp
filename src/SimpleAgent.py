import numpy as np


class SimpleAgent:
    def __init__(self):
        self.rewards = 0

    def act(self, observation):
        J, P = observation[0], observation[1]

        num_jobs = J.shape[0]
        num_machines = J.shape[2]
        do_nothing = True

        for j in range(num_jobs):

            selected_j = j
            selected_m = None

            if np.sum(P[j]) > 0:
                # one of the operations of this job is being processed
                continue

            # find the first non-processed operation i of job j
            idxs = np.argwhere(np.sum(J[j], axis=1) > 0)
            if len(idxs) == 0:
                # all operations for this job has been processed
                continue

            i = idxs[0][0]
            # find available machine
            for m in range(num_machines):
                if np.sum(P[:, :, m]) != 0:
                    # this machine is busy
                    continue

                if J[j, i, m] == 0:
                    # this machine cannot process this operation
                    continue

                selected_m = m
                do_nothing = False
                break

            if selected_m is not None:
                return {"selected_job": selected_j, "selected_machine": selected_m}

        if do_nothing:
            return {"selected_job": J.shape[0], "selected_machine": 0}
        else:
            print("Something is wrong! This should not happen!")