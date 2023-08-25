import numpy as np


class BasicPolicy():
    def __init__(self):
        pass

    def action_np(self, conditions):
        action = np.zeros(4)
        x = conditions[0][0]
        y = conditions[0][1]
        if x > 0:
            if np.random.rand(1) < 0.5:
                action[2] = 1
            else:
                action[0] = -1
        else:
            if np.random.rand(1) < 0.5:
                action[2] = -1
            else:
                action[0] = 1

        if y > 0:
            if np.random.rand(1) < 0.5:
                action[1] = 1
            else:
                action[3] = -1
        else:
            if np.random.rand(1) < 0.5:
                action[1] = -1
            else:
                action[3] = 1

        return action
