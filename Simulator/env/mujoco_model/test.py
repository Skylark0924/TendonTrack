import numpy as np


def get_mbpo_next_observation(obs, real_next_obs, ipk_act, mbpo_act):
    last_x_rate = np.array([0.1])
    last_y_rate = np.array([0.1])
    if isinstance(obs, list):
        obs_tmp = obs[0]
    else:
        obs_tmp = obs
    assert type(obs_tmp) == np.ndarray
    batch_size = obs_tmp.shape[0]

    if batch_size == 1:
        real_next_obs_tmp = real_next_obs.reshape(1, 3)
        ipk_act_tmp = ipk_act.reshape(1, 4)
        mbpo_act_tmp = mbpo_act.reshape(1, 4)
    else:
        real_next_obs_tmp = real_next_obs
        ipk_act_tmp = ipk_act
        mbpo_act_tmp = mbpo_act

    assert obs_tmp.shape == (batch_size, 3)
    assert real_next_obs_tmp.shape == (batch_size, 3)
    assert ipk_act_tmp.shape == (batch_size, 4)
    assert mbpo_act_tmp.shape == (batch_size, 4)

    ipk_trans = real_next_obs_tmp - obs_tmp
    # transform the four dim action into 2-dim x/y action
    # x_action = 3-1
    # y_action = 2-4
    # TODO: The action waits to be redesigned
    ipk_2d_act = np.array([ipk_act_tmp[:, 2] - ipk_act_tmp[:, 0], ipk_act_tmp[:, 1] - ipk_act_tmp[:, 3]]).T
    mbpo_2d_act = np.array([mbpo_act_tmp[:, 2] - mbpo_act_tmp[:, 0], mbpo_act_tmp[:, 1] - mbpo_act_tmp[:, 3]]).T

    '''
    We assume that in a single action the result of actions are linear,
          believe that every actions effect equally and the change of distance is ignored.
    So we can use the inflence of ipk_action as a reference.
    '''
    # if ipk_2d_act[:, 0].any() == 0 or ipk_2d_act[:, 1].any() == 0:
    #     x_rate = last_x_rate
    #     y_rate = last_y_rate
    # else:
    #     x_rate = ipk_trans[:, 0] / ipk_2d_act[:, 0]
    #     y_rate = ipk_trans[:, 1] / ipk_2d_act[:, 1]
    #
    #     last_x_rate = x_rate
    #     last_y_rate = y_rate
    ipk_2d_act[np.where(ipk_2d_act == 0)] = 1
    x_rate = ipk_trans[:, 0] / ipk_2d_act[:, 0]
    y_rate = ipk_trans[:, 1] / ipk_2d_act[:, 1]
    # x_rate[np.isinf(x_rate)] = self.last_x_rate
    # y_rate[np.isinf(y_rate)] = self.last_y_rate
    assert x_rate.shape == y_rate.shape
    # if x_rate.shape == (1,):
    #     self.last_x_rate = x_rate
    #     self.last_y_rate = y_rate
    print(' x_rate:{}, y_rate:{}'.format(x_rate, y_rate))

    mbpo_trans = np.array([x_rate * mbpo_2d_act[:, 0], y_rate * mbpo_2d_act[:, 1], ipk_trans[:, 2]]).T

    try:
        obs_tmp.shape == mbpo_trans.shape
    except:
        pdb.set_trace()
    batch_mbpo_next_obs = obs_tmp + mbpo_trans

    return batch_mbpo_next_obs


def accurancy_calcu_from_pool(timestep):
    observations = np.array([[-35., 11., 26.04262], [-30., 7., 26.04665], [-23., 3., 26.05189],
                             [-8., 7., 26.05825], [-4., 19., 26.06528]])
    real_next_observations = np.array([[-30., 7., 26.04665], [-23., 3., 26.05189], [-15., 0., 26.05812],
                                       [0., 5., 26.06514], [4., 17., 26.07254]])
    # actions = np.array([[-0.97826, -0.00397, -0.32509, -0.86647], [-0.29164, 0.993, -0.30658, 0.18074],
    #                     [0.63059, 0.33216, 0.48467, -0.7593]])
    ipk_actions = np.array([[0., -0.23913, -0.76087, 0., ], [-0.81081, 0., 0., -0.18919], [0., -0.11538, -0.88462, 0.],
                            [0., -0.46667, -0.53333, 0.], [-0.17391, -0.82609, 0., 0.]])
    # rewards = np.array([[-75.20425], [-12.47889], [-6.44079]])
    # real_rewards = np.array([[-68.], [0.99997], [1.99977]])

    index = 1
    basic_accurancy = np.zeros(4)
    direction_deviation_coeff = []
    for i in range(len(timestep)):
        observations_this_path = observations[index:index + timestep[i] + 1]
        real_next_observations_this_path = real_next_observations[index:index + timestep[i] + 1]
        # actions_this_path = actions[index:index + timestep[i] + 1]
        # ipk_actions_this_path = ipk_actions[index:index + timestep[i] + 1]
        # rewards_this_path = rewards[index:index + timestep[i] + 1]
        # real_rewards_this_path = real_rewards[index:index + timestep[i] + 1]

        goal_this_path = np.array(observations_this_path[0])

        for j in range(len(observations_this_path) - 1):
            direction = np.array([(observations_this_path[j][0] - real_next_observations_this_path[j][0]), (
                    observations_this_path[j][1] - real_next_observations_this_path[j][1])])
            if (direction == np.array([0.0, 0.0])).all():
                direction = 0.01 * np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
            # if (real_next_observations_this_path[j] != observations_this_path[j + 1]).any():
            #    goal_this_path = np.array(observations_this_path[j + 1])
            goal_this_path = np.array(observations_this_path[j])
            direction_deviation_coeff.append(deviation_coeff_calcu(direction, goal_this_path[0:2]))
            print('')

        index = index + timestep[i] + 1

    # pdb.set_trace()
    ipk_actions_all = ipk_actions[:(np.sum(timestep))]
    assert len(ipk_actions_all) == len(direction_deviation_coeff)
    ipk_actions_abs = np.abs(ipk_actions_all)
    ipk_actions_res = np.zeros((np.sum(timestep), 4))
    for i in range(len(ipk_actions_all)):
        tmp = direction_deviation_coeff[i]
        ipk_actions_res[i] = ipk_actions_abs[i] * np.array([tmp[0], tmp[1], tmp[0], tmp[1]])
        print('')
    ipk_actions_mean = [np.nanmean(ipk_actions_res[:, 0]), np.nanmean(ipk_actions_res[:, 1]),
                        np.nanmean(ipk_actions_res[:, 2]), np.nanmean(ipk_actions_res[:, 3])]
    ipk_actions_var = [np.nanvar(ipk_actions_res[:, 0]), np.nanvar(ipk_actions_res[:, 1]),
                       np.nanvar(ipk_actions_res[:, 2]), np.nanvar(ipk_actions_res[:, 3])]

    ipk_basic_accurancy = {'ipk_basic_mean': ipk_actions_mean, 'ipk_basic_variance': ipk_actions_var}

    return ipk_basic_accurancy


def deviation_coeff_calcu(a, b):
    # print('a,b:{},{},{},{}'.format(a, b, type(a), type(b)))
    c = (a.dot(b) / b.dot(b)) * b
    d = (a - c) / (np.linalg.norm(a))
    return (a - c) / (np.linalg.norm(a))


def BSpline_Calcu(n=5):
    tmp_rand = np.random.random([2, n])-0.5
    tmp_zero = np.array([[0, 0, 0], [0, 0, 0]])
    ctrl_points = np.hstack((tmp_zero, tmp_rand, tmp_zero))
    n_spline = n + 3  # Number of splines
    N = 4  # acc
    dummy_time = np.linspace(0, 1, num=N, endpoint=False)
    A = np.array([[-1, 3, -3, 1],
                  [3, -6, 3, 0],
                  [-3, 0, 3, 0],
                  [1, 4, 1, 0]])
    dummy_T = np.array([dummy_time ** 3, dummy_time ** 2, dummy_time, np.ones(N)]).T
    coeff = 1 / 6 * np.matmul(dummy_T, A)
    xy_t = np.zeros([N * n_spline, 2])
    for i in range(n_spline):
        a = ctrl_points[:, i:i + 4].T
        xy_t[i * N:(i + 1) * N] = np.matmul(coeff, ctrl_points[:, i:i + 4].T)

    return xy_t


if __name__ == '__main__':
    xy_t = BSpline_Calcu()
    print(xy_t.shape)
    # timstp = np.array([3])
    # accurancy_calcu_from_pool(timstp)
    # a = np.array([-4., 101.])
    # b = np.array([0., 1.])
    # print(cos_angle_calcu(a, b))

    # obs = [np.array([[-25., -265., 4254.]])]
    # real_next_obs = np.array([[-26., -258., 4246.]])
    # ipk_act = np.array([[-1., 0.81488, -1., -1., ]])
    # mbpo_act = np.array([[-1., 0.81488, 1., -1.]])
    # print('obs:{},\r\nreal_next_obs:{},\r\nipk_act:{},\r\nmbpo_act:{}'.format(obs, real_next_obs, ipk_act, mbpo_act))
    # # sampler = IPKSampler(max_path_length=10, min_pool_size=10, batch_size=10)
    # mbpo_next_obs = get_mbpo_next_observation(obs=obs,
    #                                           real_next_obs=real_next_obs,
    #                                           ipk_act=ipk_act,
    #                                           mbpo_act=mbpo_act)
    # print(mbpo_next_obs)
