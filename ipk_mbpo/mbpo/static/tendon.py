import sys
import numpy as np
import pdb

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        # TODO: termination func need to debug
        z = next_obs[:, 0]
        # pdb.set_trace()
        # print('next_obs: {}'.format(next_obs))
        print('z: {}'.format(z))
        done = (z >= -10.0) & (z <= 10.0)
        '''
        if z is None:
            done = True
        elif (np.abs(z) >= 500).any():
            done = True
        elif (np.abs(z) <= 10).all():
            done = True
        else:
            done = False
        '''
        done = done[:, None]
        return done
