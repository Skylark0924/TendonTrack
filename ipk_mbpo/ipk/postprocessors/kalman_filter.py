from math import *
import tensorflow as tf
import pdb
import numpy as np


def accurancy_calcu_from_pool(timestep, pool):
    observations = pool.fields['observations']
    real_next_observations = pool.fields['real_next_observations']
    actions = pool.fields['actions']
    ipk_actions = pool.fields['ipk_actions']
    rewards = pool.fields['rewards']
    real_rewards = pool.fields['real_rewards']
    # print('pool_fields: {}'.format(pool.fields))
    # infos = pool.fields['infos']
    # print('accurancy_calcu_info: {},{}'.format(infos, type(infos)))

    index = 1
    direction_deviation_coeff = []
    for i in range(len(timestep)):
        observations_this_path = observations[index:index + timestep[i] + 1]
        real_next_observations_this_path = real_next_observations[index:index + timestep[i] + 1]
        # actions_this_path = actions[index:index + timestep[i] + 1]
        # ipk_actions_this_path = ipk_actions[index:index + timestep[i] + 1]
        # rewards_this_path = rewards[index:index + timestep[i] + 1]
        # real_rewards_this_path = real_rewards[index:index + timestep[i] + 1]
        #
        # goal_this_path = np.array(observations_this_path[0])

        for j in range(len(observations_this_path) - 1):
            direction = np.array([(observations_this_path[j][0] - real_next_observations_this_path[j][0]), (
                    observations_this_path[j][1] - real_next_observations_this_path[j][1])])
            if (direction == np.array([0.0, 0.0])).all():
                direction = 0.01 * np.array([np.random.rand() - 0.5, np.random.rand() - 0.5])
            goal_this_path = np.array(observations_this_path[j])
            direction_deviation_coeff.append(deviation_coeff_calcu(direction, goal_this_path[0:2]))

        index = index + timestep[i] + 1

    ipk_actions_all = ipk_actions[:(np.sum(timestep))]
    assert len(ipk_actions_all) == len(direction_deviation_coeff)
    # 避开0取倒数
    ipk_actions_abs = np.where(ipk_actions_all != 0, 1 / ipk_actions_all, 0)
    ipk_actions_res = np.zeros((np.sum(timestep), 4))
    for i in range(len(ipk_actions_all)):
        tmp = direction_deviation_coeff[i]
        # 对应电机单位动作的偏差 (a=1)
        ipk_actions_res[i] = ipk_actions_abs[i] * np.array([tmp[0], tmp[1], tmp[0], tmp[1]])
    ipk_actions_mean = [np.nanmean(ipk_actions_res[:, 0]), np.nanmean(ipk_actions_res[:, 1]),
                        np.nanmean(ipk_actions_res[:, 2]), np.nanmean(ipk_actions_res[:, 3])]
    ipk_actions_var = [np.nanvar(ipk_actions_res[:, 0]), np.nanvar(ipk_actions_res[:, 1]),
                       np.nanvar(ipk_actions_res[:, 2]), np.nanvar(ipk_actions_res[:, 3])]

    ipk_basic_accurancy = {'ipk_basic_mean': ipk_actions_mean, 'ipk_basic_variance': ipk_actions_var}
    return ipk_basic_accurancy


def deviation_coeff_calcu(a, b):
    # a: direction; b: goal
    # print('a,b:{},{},{},{}'.format(a, b, type(a), type(b)))
    if (np.linalg.norm(a) == 0) or (np.linalg.norm(b) == 0):
        return np.array([0.0, 0.0])
    c = (a.dot(b) / b.dot(b)) * b
    # d = (a - c) / (np.linalg.norm(a))
    return (a - c) / (np.linalg.norm(a))


class KalmanFilter():
    def __init__(self):
        self.mu = tf.keras.backend.variable(np.ones((4,)) * 0.1, dtype=tf.float32,
                                            name="mu")
        self.log_sigma = tf.keras.backend.variable(np.ones((4,)) * 1, dtype=tf.float32,
                                                   name="log_sigma")
        self.zeta_basic = tf.keras.backend.variable(0.9 * np.ones((1,)), dtype=tf.float32, name="zeta_basic")
        self.zeta_real = tf.keras.backend.variable(0.1 * np.ones((1,)), dtype=tf.float32, name="zeta_real")
        self.mu_np = np.zeros((4,))
        self.log_sigma_np = np.zeros((4,))
        self.lim_val = 50

    def get_ipk_basic_norm_info(self, info):
        # mu = tf.convert_to_tensor(info['ipk_basic_mean'], dtype='float32')
        # log_sigma_square = tf.convert_to_tensor(tf.log(info['ipk_basic_variance']), dtype='float32')
        print('mu_type: {}'.format(info['ipk_basic_mean']))
        tf.keras.backend.set_value(
            self.mu,
            info['ipk_basic_mean']
        )
        tf.keras.backend.set_value(
            self.log_sigma,
            np.log(np.sqrt(info['ipk_basic_variance']))
        )
        self.mu_np = tf.keras.backend.get_value(self.mu)
        self.log_sigma_np = tf.keras.backend.get_value(self.log_sigma)

        return {
            'mu': self.mu_np,
            'log_sigma': self.log_sigma_np,
        }

    def update_zeta(self, zeta_basic):
        zeta_basic_np = zeta_basic * np.ones((1,))
        zeta_real_np = (1 - zeta_basic) * np.ones((1,))
        # print('zeta_basic_np_type: {}'.format(zeta_basic_np.dtype))
        tf.keras.backend.set_value(
            self.zeta_basic,
            zeta_basic_np
        )
        tf.keras.backend.set_value(
            self.zeta_real,
            zeta_real_np
        )
        print('zeta_basic: {}, zeta_real: {}'.format(tf.keras.backend.get_value(self.zeta_basic),
                                                     tf.keras.backend.get_value(self.zeta_real)))
        # self.zeta_basic = zeta_basic
        # self.zeta_real = 1 - self.zeta_basic

    def kalman_filter_1d(self, input):
        """
        Attention: 输入的是标准差的对数
        """
        # nu: mbpo_gau mean; log_r: mbpo_gau log scale
        nu, log_r, ipk_basic_action = input
        # print('nu_shape: {}'.format(nu.shape))

        # mu = a(1+mu)
        # sigma^2 = a^2 * sigma^2 -> log_sigma = 0.5 * log(a^2) + log_sigma
        # ipk_basic_action = tf.Print(ipk_basic_action, [ipk_basic_action, tf.shape(ipk_basic_action)],
        #                             '[kal]: ipk_basic_action:')

        mu, log_sigma = tf.multiply(ipk_basic_action, tf.add(tf.ones_like(self.mu), self.mu)), tf.add(
            0.5 * tf.log(tf.square(ipk_basic_action)), self.log_sigma)
        # When action is 0, log a^2 will be -inf and lead to an nan result of the fusion
        # So we need to transform -inf into 0
        log_sigma = tf.where(tf.is_inf(log_sigma), tf.zeros_like(log_sigma) * 0,
                              log_sigma)
        # if log_sigma is inf use 0 else use element in log_sigma

        # mu = tf.Print(mu, [mu, tf.shape(mu)], '[kal]: mu: ')
        # log_sigma = tf.Print(log_sigma, [log_sigma, tf.shape(log_sigma)], '[kal]: log_sigma: ')

        log_sigma_sqaure = tf.minimum(log_sigma * 2, self.lim_val)
        log_r_sqaure = tf.minimum(log_r * 2, self.lim_val)

        r_square = tf.exp(log_sigma_sqaure)
        sigma_square = tf.exp(log_r_sqaure)

        # print(' [ zeta_basic ]: {}, [ zeta_real ]: {}'.format(self.zeta_basic, self.zeta_real))

        # one-dimension Kalman Fusion
        new_mu = tf.truediv(tf.add(tf.multiply(self.zeta_basic, tf.multiply(r_square, mu)),
                                   tf.multiply(self.zeta_real, tf.multiply(sigma_square, nu))),
                            tf.add(tf.multiply(self.zeta_basic, r_square), tf.multiply(self.zeta_real, sigma_square)))
        new_sigma_square = tf.reciprocal(
            tf.add(tf.multiply(self.zeta_real, tf.reciprocal(r_square)),
                   tf.multiply(self.zeta_basic, tf.reciprocal(sigma_square))))
        return [new_mu, tf.log(tf.sqrt(new_sigma_square))]


if __name__ == '__main__':
    info = {'ipk_basic_mean': [[-0.01, -0.1, 0.1, 1], [-0.01, -0.1, 0.1, 1]],
            'ipk_basic_variance': [[1., 50., 3., 4.], [1., 2., 3., 4.]]}

    mbpo_info = {'shift': [[100, 23, 345, 232], [100, 23, 345, 232]],
                 'log_scale_diag': [[30, 50, 20, 20], [20, 20, 20, 20]]}
    mbpo_info = [tf.convert_to_tensor(mbpo_info['shift'], dtype='float32'),
                 tf.convert_to_tensor(mbpo_info['log_scale_diag'], dtype='float32')]
    kalman = KalmanFilter()
    basic_distribution = kalman.get_ipk_basic_norm_info(info)
    fusion_info = kalman.kalman_filter_1d(input=mbpo_info)
    print(fusion_info)
