from collections import defaultdict

import numpy as np
import scipy.stats
from scipy.stats import norm

from softlearning.samplers.simple_sampler import SimpleSampler
from ipk.policies.ipk_gaussian_policy import FeedforwardGaussianPolicy
from ipk.policies.ipk_uniform_policy import UniformPolicy
from softlearning.samplers.base_sampler import BaseSampler
from ipk.policies.ipk_basic_policy import IPKBasicPolicy
import pdb
import tensorflow as tf

IPK = True
use_fake_env = False


class IPKSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(IPKSampler, self).__init__(**kwargs)

        self._path_length = 0
        self.path_lengths = []
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

        self.zeta_basic = 0.90
        self.zeta_real = 1 - self.zeta_basic

        self.last_x_rate = 0.1
        self.last_y_rate = 0.1

        self.mbpo_rew_scope_rim = -10  # ipk在范围内而 mbpo 不在范围内, mbpo_reward_core 预估值 TODO: tune
        self.mbpo_rew_np_coeff = 1  # mbpo_uniform reward 估计中, np差的放缩比例  TODO: tune
        self.mbpo_rew_kl_coeff = 0.5  # mbpo_gaussian reward 估计中, kl散度的放缩比例  TODO: tune

        self.KL_diver_list = []
        self.KL_diver_mean = 0
        self.KL_diver_max = 0
        self.KL_diver_min = 0

        self.zeta_real_list = []
        self.zeta_real_mean = 0
        self.zeta_real_max = 0
        self.zeta_real_min = 0

        self.target_distri = -2

        self.ipk_basic_policy = IPKBasicPolicy()

    def _process_observations(self,
                              observation,
                              action,
                              ipk_action,
                              reward,
                              real_reward,
                              terminal,
                              next_observation,
                              real_next_observation,
                              info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'ipk_actions': ipk_action,
            'rewards': [reward],
            'real_rewards': [real_reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'real_next_observations': real_next_observation,
            'infos': info,
        }

        return processed_observation

    # @property
    def sample(self, fake_env):
        """
        uniform_policy: reward -> R_{KL} between basic controller and uniform random action
                        action -> follow ipk_basic_policy
        gaussian_policy: reward -> - yita_basic * [R_{KL} between basic controller and
                                mbpo N(shift, scale_diag)] + yita_real * real_reward
                         action -> kalman fusion between ipk_basic_policy & mbpo policy
        fake_env: for estimating the next_state and reward of mbpo
        :return:
        """
        # print('\r\n[ Episode: {} Start Sampling ] -------{} '.format(self._n_episodes + 1, self.policy))
        if self._current_observation is None:
            self._current_observation = self.env.reset()
            print(self._current_observation)

        '''
        mbpo_action: uniform_action or gaussian_action
        ipk_action: ipk_basic_action or ipk_fusion_action
        uniform_action & ipk_basic_action in ipk_uniform_policy.py
        gaussian_action & ipk_fusion_action in ipk_gaussian_policy.py
        '''
        observation = [
            self.env.convert_to_active_observation(
                self._current_observation)[None]
        ]
        # print(' [ Observation ]: {}'.format(observation))

        if isinstance(self.policy, UniformPolicy):
            mbpo_action, ipk_action = self.policy.actions_np(observation)

            # print(' [ mbpo_action ]: {}, type:{} \r\n [ ipk_action ]: {}, type:{}'
            #       .format(mbpo_action, type(mbpo_action), ipk_action, type(ipk_action)))
            if ipk_action.shape[0] > 1:
                pdb.set_trace()

            mbpo_action = np.squeeze(mbpo_action)
            ipk_action = np.squeeze(ipk_action)

            # Interaction
            real_next_observation, real_reward, terminal, info = self.env.step(ipk_action)

            # print(' [ real_next_obs ]: {}, type:{}'
            #       .format(real_next_observation, type(real_next_observation)))

            if use_fake_env:
                mbpo_next_observation, mbpo_reward = real_next_observation, real_reward
            else:
                '''MBPO next obs estimation'''
                mbpo_next_observation = self.get_mbpo_next_observation(obs=observation,
                                                                       real_next_obs=real_next_observation,
                                                                       ipk_act=ipk_action,
                                                                       mbpo_act=mbpo_action)
                # print(' [ mbpo_next_obs ]: {}, type:{}'
                #       .format(mbpo_next_observation, type(real_next_observation)))

                '''MBPO uniform reward estimation'''
                try:
                    if info['flg'] is True and abs(mbpo_next_observation[0][0]) <= 20 and abs(
                            mbpo_next_observation[0][1]) <= 20:  # inside the success scope
                        mbpo_reward = 10
                    else:  # outside the scope
                        # mbpo_reward = 10
                        mbpo_xy = np.array([mbpo_action[2] + mbpo_action[0], mbpo_action[3] + mbpo_action[1]])
                        basic_xy = np.array([ipk_action[2] + ipk_action[0], ipk_action[3] + ipk_action[1]])
                        # mbpo_reward = np.mean((np.true_divide(mbpo_xy, basic_xy)) * real_reward)
                        # print(' [ mbpo_reward ]: {}, mbpo_xy: {}, basic_xy: {}'.format(mbpo_reward, mbpo_xy, basic_xy))
                        mbpo_reward = - self.mbpo_rew_np_coeff * (np.sum(
                            abs(mbpo_xy - basic_xy)) + self.target_distri) + real_reward
                    # print(' [ mbpo_reward ]: {}; real_reward: {};'.format(mbpo_reward, real_reward))
                except:
                    pdb.set_trace()

            # info
            # if info is not None:
            # print(' [ info ]: {}, type: {}'.format(info, type(info)))
            # print('[ Stop Sampling ] ----------------------------------------------------------------------------')

        elif self.policy._deterministic:
            '''Evalution Interaction'''
            mbpo_action, ipk_fusion_action = self.policy.actions_np(observation)
            # print(' [ mbpo_action ]: {}, type:{}'.format(mbpo_action, type(mbpo_action)))
            # print(' [ ipk_fusion_action ]: {}, type:{}'.format(ipk_fusion_action, type(ipk_fusion_action)))

            # log_pis = self.policy.log_pis_np(observation, mbpo_action)
            # kal_log_pis = self.policy.kal_bas_log_pis_np(observation, self.policy.ipk_basic_action)
            # print('KL-zeta: {}, log_pis: {}, kal_log_pis: {}'.format((kal_log_pis - log_pis), log_pis, kal_log_pis))

            mbpo_action = np.squeeze(mbpo_action)
            ipk_action = np.squeeze(ipk_fusion_action)
            print(' [ Eval / mbpo_action ]: {}'.format(mbpo_action))

            mbpo_next_observation, mbpo_reward, terminal, info = self.env.step(mbpo_action)
            real_next_observation, real_reward = mbpo_next_observation, mbpo_reward


        elif isinstance(self.policy, FeedforwardGaussianPolicy):
            if IPK:
                '''Update zeta coefficient'''
                # ipk_fusion_gaussian = self.policy.get_kal_diagnostics(observation)
                mbpo_gaussian = self.policy.get_distribution(observation)
                # print(' [ mbpo_gaussian ]: {};\r\n [ ipk_fusion_gaussian ]: {}'.format(mbpo_gaussian,
                #                                                                        ipk_fusion_gaussian))
                # KL_diver = self.KL_divergence(ipk_fusion_gaussian, mbpo_gaussian)
                # ipk_basic_gaussian = self.policy.get_basic_diagnostics(observation)
                ipk_basic_gaussian = {
                    'shifts': self.ipk_basic_policy.action_np(observation) + self.policy.kalman_filter.mu_np,
                    'log_scale_diags': self.policy.kalman_filter.log_sigma_np.reshape((1, 4))}
                # print(' [ ipk_basic_gaussian ]: {}'.format(ipk_basic_gaussian))
                try:
                    mbpo_xy = {'shifts': np.array([mbpo_gaussian['shifts'][0][2] + mbpo_gaussian['shifts'][0][0],
                                                   mbpo_gaussian['shifts'][0][3] + mbpo_gaussian['shifts'][0][1]]),
                               'log_scale_diags': np.array(
                                   [(mbpo_gaussian['log_scale_diags'][0][2] + mbpo_gaussian['log_scale_diags'][0][
                                       0]) / 2.0,
                                    (mbpo_gaussian['log_scale_diags'][0][3] + mbpo_gaussian['log_scale_diags'][0][
                                        1]) / 2.0])}
                    basic_xy = {
                        'shifts': np.array([ipk_basic_gaussian['shifts'][0][2] + ipk_basic_gaussian['shifts'][0][0],
                                            ipk_basic_gaussian['shifts'][0][3] + ipk_basic_gaussian['shifts'][0][1]]),
                        'log_scale_diags': np.array(
                            [(ipk_basic_gaussian['log_scale_diags'][0][2] + ipk_basic_gaussian['log_scale_diags'][0][
                                0]) / 2.0,
                             (ipk_basic_gaussian['log_scale_diags'][0][3] + ipk_basic_gaussian['log_scale_diags'][0][
                                 1]) / 2.0])}
                except:
                    pdb.set_trace()
                KL_diver = np.clip(self.multivar_continue_KL_divergence(basic_xy, mbpo_xy), 0, 50)
                print("KL_diver: {}".format(KL_diver))
                self.KL_diver_list.append(KL_diver)

                self.update_zeta(KL_diver)
                self.zeta_real_list.append(self.zeta_real)

                '''Interaction'''
                mbpo_action, ipk_fusion_action = self.policy.actions_np(observation)
                # print(' [ mbpo_action ]: {}, type:{}'.format(mbpo_action, type(mbpo_action)))
                # print(' [ ipk_fusion_action ]: {}, type:{}'.format(ipk_fusion_action, type(ipk_fusion_action)))

                # log_pis = self.policy.log_pis_np(observation, mbpo_action)
                # kal_log_pis = self.policy.kal_bas_log_pis_np(observation, self.policy.ipk_basic_action)
                # print('KL-zeta: {}, log_pis: {}, kal_log_pis: {}'.format((kal_log_pis - log_pis), log_pis, kal_log_pis))

                mbpo_action = np.squeeze(mbpo_action)
                ipk_action = np.squeeze(ipk_fusion_action)
                print(' [ mbpo_action ]: {}'.format(mbpo_action))

                real_next_observation, real_reward, terminal, info = self.env.step(ipk_action)
                if use_fake_env:
                    mbpo_next_observation, mbpo_reward, mbpo_term, mbpo_info = fake_env.step(
                        np.array(observation[0][0]), np.array(mbpo_action))

                else:
                    '''MBPO next obs estimation'''
                    mbpo_next_observation = self.get_mbpo_next_observation(obs=observation,
                                                                           real_next_obs=real_next_observation,
                                                                           ipk_act=ipk_action,
                                                                           mbpo_act=mbpo_action)
                    # print(
                    #     ' [ mbpo_next_observation ]: {}, shape: {}'.format(mbpo_next_observation,
                    #                                                        mbpo_next_observation.shape))
                    # print(
                    #     ' [ real_next_observation ]: {}, shape: {}'.format(real_next_observation,
                    #                                                        real_next_observation.shape))

                    '''MBPO reward estimation'''
                    # # ipk_fusion_gaussian = self.policy.get_kal_diagnostics(observation)
                    # mbpo_gaussian = self.policy.get_distribution(observation)
                    # # print(' [ mbpo_gaussian ]: {};\r\n [ ipk_fusion_gaussian ]: {}'.format(mbpo_gaussian,
                    # #                                                                        ipk_fusion_gaussian))
                    # # KL_diver = self.KL_divergence(ipk_fusion_gaussian, mbpo_gaussian)
                    # # ipk_basic_gaussian = self.policy.get_basic_diagnostics(observation)
                    # ipk_basic_gaussian = {'shifts': self.policy.ipk_basic_action + self.policy.kalman_filter.mu_np,
                    #                       'log_scale_diags': self.policy.kalman_filter.log_sigma_square_np.reshape((1, 4))}
                    # # print(' [ ipk_basic_gaussian ]: {}'.format(ipk_basic_gaussian))
                    # try:
                    #     mbpo_xy = {'shifts': np.array([mbpo_gaussian['shifts'][0][2] + mbpo_gaussian['shifts'][0][0],
                    #                                    mbpo_gaussian['shifts'][0][3] + mbpo_gaussian['shifts'][0][1]]),
                    #                'log_scale_diags': np.array(
                    #                    [(mbpo_gaussian['log_scale_diags'][0][2] + mbpo_gaussian['log_scale_diags'][0][
                    #                        0]) / 2.0,
                    #                     (mbpo_gaussian['log_scale_diags'][0][3] + mbpo_gaussian['log_scale_diags'][0][
                    #                         1]) / 2.0])}
                    #     basic_xy = {
                    #         'shifts': np.array([ipk_basic_gaussian['shifts'][0][2] + ipk_basic_gaussian['shifts'][0][0],
                    #                             ipk_basic_gaussian['shifts'][0][3] + ipk_basic_gaussian['shifts'][0][1]]),
                    #         'log_scale_diags': np.array(
                    #             [(ipk_basic_gaussian['log_scale_diags'][0][2] + ipk_basic_gaussian['log_scale_diags'][0][
                    #                 0]) / 2.0,
                    #              (ipk_basic_gaussian['log_scale_diags'][0][3] + ipk_basic_gaussian['log_scale_diags'][0][
                    #                  1]) / 2.0])}
                    # except:
                    #     pdb.set_trace()
                    # KL_diver = self.KL_divergence(basic_xy, mbpo_xy)
                    # print("KL_diver: {}".format(KL_diver))
                    # self.KL_diver_list.append(sum(KL_diver))
                    #
                    # if len(self.KL_diver_list) + 1 >= self._max_path_length:
                    #     self.KL_diver_mean = np.mean(self.KL_diver_list)
                    #     self.KL_diver_max = np.max(self.KL_diver_list)
                    #     self.KL_diver_min = np.min(self.KL_diver_list)
                    #     self.KL_diver_list = []
                    # self.update_zeta(KL_diver)

                    # actions = self.policy.actions([observation])
                    # kal_actions = self.policy.kal_actions([observation], self.policy.ipk_basic_action)

                    try:
                        if info['flg'] == 1 and abs(mbpo_next_observation[0][0]) <= 20 and abs(
                                mbpo_next_observation[0][1]) <= 20:
                            mbpo_reward_core = 10
                        else:
                            mbpo_reward_core = - self.mbpo_rew_kl_coeff * (
                                    np.sum(abs(KL_diver)) + self.target_distri) + real_reward
                    except:
                        pdb.set_trace()
                    mbpo_reward = self.zeta_basic * mbpo_reward_core + self.zeta_real * real_reward
                    print(
                        ' [ mbpo_reward ]: {}; real_reward: {}; [ KL ]: {}'.format(mbpo_reward, real_reward, KL_diver))

                # info
                # if info is not None:
                # print(' [ info ]: {}, type: {}'.format(info, type(info)))
                # print('[ Stop Sampling ] --------------------------------------------------------------------------------')

            else:  # MBPO
                mbpo_action, ipk_action = self.policy.actions_np(observation)
                print(' [ mbpo_action ]: {}, type:{}'.format(mbpo_action, type(mbpo_action)))

                mbpo_action = np.squeeze(mbpo_action)
                ipk_action = mbpo_action

                # Interaction
                real_next_observation, real_reward, terminal, info = self.env.step(ipk_action)

                mbpo_gaussian = self.policy.get_distribution(observation)
                print(' [ mbpo_gaussian ]: {}'.format(mbpo_gaussian))
                mbpo_reward = real_reward
                mbpo_next_observation = real_next_observation
                print(
                    ' [ mbpo_next_observation ]: {}, shape: {}'.format(mbpo_next_observation,
                                                                       mbpo_next_observation.shape))
                print(
                    ' [ real_next_observation ]: {}, shape: {}'.format(real_next_observation,
                                                                       real_next_observation.shape))
                # print('[ Stop Sampling ] --------------------------------------------------------------------------------')

        else:
            print('wuhan jiayou!')
            raise ValueError('The class of policy is neither FeedforwardGaussianPolicy nor UniformPolicy')



        self._path_length += 1
        self._path_return += real_reward  # mbpo_reward
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=mbpo_action,
            ipk_action=ipk_action,
            reward=mbpo_reward,
            real_reward=real_reward,
            terminal=terminal,
            next_observation=mbpo_next_observation if mbpo_next_observation.shape != (1, 3)
            else mbpo_next_observation.reshape(3, ),
            real_next_observation=real_next_observation,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        # sample 终止
        if terminal or self._path_length >= self._max_path_length:
            print('[ Stop Sampling ]------------------------------------')
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            # print('sample_path: {}'.format(last_path))
            # pdb.set_trace()
            self.pool.add_path(last_path)
            self.path_lengths.append(self._path_length)

            # pdb.set_trace()
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            # pdb.set_trace()

            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            if info['flg'] == 0:
                self._current_observation = real_next_observation
            else:
                self._current_observation = np.concatenate((info['obs'], real_next_observation[3:]))

        return real_next_observation, mbpo_reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        # raw_batch_dic = self.pool.random_batch(batch_size, observation_keys=observation_keys, **kwargs)
        # batch_dic = {
        #     'observations': np.concatenate((raw_batch_dic['observations'], raw_batch_dic['observations']), axis=0),
        #     'next_observations': np.concatenate(
        #         (raw_batch_dic['next_observations'], raw_batch_dic['real_next_observations']), axis=0),
        #     'actions': np.concatenate((raw_batch_dic['actions'], raw_batch_dic['ipk_actions']), axis=0),
        #     'rewards': np.concatenate((raw_batch_dic['rewards'], raw_batch_dic['real_rewards']), axis=0),
        #     'terminals': np.concatenate((raw_batch_dic['terminals'], raw_batch_dic['terminals']), axis=0)}
        #
        # return batch_dic

        # raw_batch_dic = self.pool.random_batch(batch_size, observation_keys=observation_keys, **kwargs)
        # batch_dic = {
        #     'observations': raw_batch_dic['observations'],
        #     'next_observations': raw_batch_dic['next_observations'],
        #     'actions': raw_batch_dic['actions'],
        #     'ipk_actions': raw_batch_dic['ipk_actions'],
        #     'rewards': raw_batch_dic['rewards'],
        #     'terminals': raw_batch_dic['terminals'],
        # }
        #
        # return batch_dic

        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(IPKSampler, self).get_diagnostics()

        # if len(self.KL_diver_list) + 1 >= self._max_path_length:
        self.KL_diver_mean = np.mean(self.KL_diver_list)
        self.KL_diver_max = np.max(self.KL_diver_list)
        self.KL_diver_min = np.min(self.KL_diver_list)
        self.KL_diver_list = []

        self.zeta_real_mean = np.mean(self.zeta_real_list)
        self.zeta_real_max = np.max(self.zeta_real_list)
        self.zeta_real_min = np.min(self.zeta_real_min)
        self.zeta_real_list = []

        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
            'KL-mean': self.KL_diver_mean,
            'KL-max': self.KL_diver_max,
            'KL-min': self.KL_diver_min,
            'zeta-mean': self.zeta_real_mean,
            'zeta-max': self.zeta_real_max,
            'zeta-min': self.zeta_real_min,
        })

        return diagnostics

    def update_target_dis(self, target_distri):
        self.target_distri = target_distri

    def update_zeta(self, KL_diver):
        if isinstance(self.policy, UniformPolicy):
            pass
        elif isinstance(self.policy, FeedforwardGaussianPolicy):
            # a, b, c = self.calcu_zeta_coeff(0.0, 1.0, 0.0, 10)
            # # print('a,b,c: {} {} {}'.format(a, b, c))
            # zeta = 1 / (1 + np.exp(-(np.sum(abs(self.KL_diver)) - target_distri) / c))    # self.zeta_basic *
            # self.zeta_basic = a * (zeta - b)

            self.zeta_basic = np.clip(np.tanh((np.sum(abs(KL_diver)) + self.target_distri) * 0.5), 0, 1)
            print('ipk_sampler/zeta_basic: {}'.format(self.zeta_basic))
            # try:
            #     self.zeta_basic = tf.keras.backend.get_value(
            #         zeta
            #     )
            # except:
            #     pdb.set_trace()
            self.zeta_real = 1 - self.zeta_basic
            print('zeta_basic: {}, zeta_real: {}\r\n'.format(self.zeta_basic, self.zeta_real))
            self.policy.kalman_filter.update_zeta(self.zeta_basic)
        else:
            pass

    # def update_zeta(self, zeta):
    #     if isinstance(self.policy, UniformPolicy):
    #         pass
    #     elif isinstance(self.policy, FeedforwardGaussianPolicy):
    #         # a, b, c = self.calcu_zeta_coeff(0.0, 1.0, 0.0, 10)
    #         # # print('a,b,c: {} {} {}'.format(a, b, c))
    #         # zeta = 1 / (1 + np.exp(-(np.sum(abs(self.KL_diver)) - target_distri) / c))    # self.zeta_basic *
    #         # self.zeta_basic = a * (zeta - b)
    #
    #         # self.zeta_basic = np.clip(np.tanh((np.sum(abs(self.KL_diver)) + target_distri) * 0.2), 0, 1)
    #
    #         try:
    #             self.zeta_basic = tf.keras.backend.get_value(
    #                 zeta
    #             )
    #         except:
    #             pdb.set_trace()
    #         self.zeta_real = 1 - self.zeta_basic
    #         print('zeta_basic: {}, zeta_real: {}\r\n'.format(self.zeta_basic, self.zeta_real))
    #         self.policy.kalman_filter.update_zeta(self.zeta_basic)
    #     else:
    #         pass

    def KL_divergence(self, p_norm, q_norm):  # TODO: TEMP D_KL use only shifts and scale_diags
        try:
            assert p_norm['shifts'].shape == q_norm['shifts'].shape
            assert p_norm['log_scale_diags'].shape == q_norm['log_scale_diags'].shape
        except:
            pdb.set_trace()
        batch_size = p_norm['shifts'].shape[0]
        # batch_D_KL = np.zeros((batch_size, 4))
        # print('batch_size : {}'.format(p_norm['shifts'].shape))
        # DEBUG_flg = True
        # if DEBUG_flg:
        p_mean = p_norm['shifts']
        p_mean = self.nan_inf_proc(p_mean)
        p_variance = np.exp(p_norm['log_scale_diags'])

        q_mean = q_norm['shifts']
        q_variance = np.exp(q_norm['log_scale_diags'])
        # print('....p_mean:{},p_variance:{},q_mean:{},q_variance:{}'.format(p_mean, p_variance, q_mean, q_variance))
        D_KL = -0.5 * (2 * np.log(p_variance / q_variance) - (p_variance / q_variance) ** 2
                       - (p_mean - q_mean) ** 2 / (q_variance ** 2) + 1)
        # print('D_KL:{}'.format(D_KL))
        batch_D_KL = D_KL
        # else:
        #     for i in range(batch_size):
        #         p_mean = p_norm['shifts'][i]
        #         p_mean = self.nan_inf_proc(p_mean)
        #         p_variance = np.exp(p_norm['log_scale_diags'])[i]
        #
        #         q_mean = q_norm['shifts'][i]
        #         q_variance = np.exp(q_norm['log_scale_diags'])[i]
        #         print('{}....p_mean:{},p_variance:{},q_mean:{},q_variance:{}'.format(i, p_mean, p_variance, q_mean,
        #                                                                              q_variance))
        #         D_KL = np.zeros(4)
        #         for j in range(4):
        #             x = np.arange(-1, 1, 0.01)
        #             # pdb.set_trace()
        #
        #             # p_norm_1 = norm.pdf(x, loc=p_mean[j], scale=p_variance[j])
        #             # q_norm_1 = norm.pdf(x, loc=q_mean[j], scale=q_variance[j])
        #             # D_KL[j] = scipy.stats.entropy(p_norm_1, q_norm_1)
        #             D_KL[j] = -0.5 * (2 * np.log(p_variance[j] / q_variance[j]) - (p_variance[j] / q_variance[j]) ** 2
        #                               - (p_mean[j] - q_mean[j]) ** 2 / (q_variance[j] ** 2) + 1)
        #
        #         print('D_KL:{}'.format(D_KL))
        #         batch_D_KL[i] = D_KL

        return batch_D_KL
        # return D_KL

    def multivar_continue_KL_divergence(self, p, q):
        p, q = [np.array(np.transpose(p['shifts'])), np.diag(p['log_scale_diags']**2)], \
               [np.array(np.transpose(q['shifts'])), np.diag(q['log_scale_diags']**2)]

        a = np.log(np.linalg.det(q[1]) / (np.linalg.det(p[1]) + 1e-5))
        b = np.trace(np.dot(np.linalg.inv(q[1]), p[1]))
        c = np.dot(np.dot(np.transpose(q[0] - p[0]), np.linalg.inv(q[1])), (q[0] - p[0]))
        n = p[1].shape[0]
        return 0.5 * (a - n + b + c)

    def get_mbpo_next_observation(self, obs, real_next_obs, ipk_act, mbpo_act):
        if isinstance(obs, list):
            obs_tmp = obs[0][:, :3]
            obs_tmp_2 = obs[0][:, 3:]
        else:
            obs_tmp = obs[:, :3]
            obs_tmp_2 = obs[:, 3:]
        assert type(obs_tmp) == np.ndarray
        batch_size = obs_tmp.shape[0]

        # TODO: accum_actions + mbpo_act?
        if batch_size == 1:
            # real_next_obs_tmp = real_next_obs.reshape(1, 7)[:, :3]
            real_next_obs_tmp = real_next_obs.reshape(1, 3)
            ipk_act_tmp = ipk_act.reshape(1, 4)
            mbpo_act_tmp = mbpo_act.reshape(1, 4)
        else:
            real_next_obs_tmp = real_next_obs[:, :3]
            ipk_act_tmp = ipk_act
            mbpo_act_tmp = mbpo_act

        mbpo_accum_actions = obs_tmp_2

        # print('obs_tmp:{},{},{}'.format(obs_tmp, type(obs_tmp), obs_tmp.shape))
        # print('next_obs_tmp:{},{},{},{}'.format(real_next_obs_tmp, type(real_next_obs_tmp), real_next_obs_tmp.shape,
        #                                         batch_size))
        # print('ipk_act_tmp:{},{},{},{}'.format(ipk_act_tmp, type(ipk_act_tmp), ipk_act_tmp.shape,
        #                                        batch_size))
        assert obs_tmp.shape == (batch_size, 3)
        assert real_next_obs_tmp.shape == (batch_size, 3)
        assert ipk_act_tmp.shape == (batch_size, 4)
        assert mbpo_act_tmp.shape == (batch_size, 4)
        # print('get_mbpo_xxx:\r\nobs:{}\r\nreal_next_obs:{}\r\nipk_act:{}\r\nmbpo_act:{}'
        #       .format(obs_tmp, real_next_obs_tmp, ipk_act_tmp, mbpo_act_tmp))
        # batch_mbpo_next_obs = np.zeros((batch_size, 3))
        # DEBUG_flg = True
        # if DEBUG_flg:
        ipk_trans = real_next_obs_tmp - obs_tmp
        # transform the four dim action into 2-dim x/y action
        # x_action = 3+1
        # y_action = 2+4
        ipk_2d_act = np.array([ipk_act_tmp[:, 2] + ipk_act_tmp[:, 0], -(ipk_act_tmp[:, 1] + ipk_act_tmp[:, 3])]).T
        mbpo_2d_act = np.array([mbpo_act_tmp[:, 2] + mbpo_act_tmp[:, 0], -(mbpo_act_tmp[:, 1] + mbpo_act_tmp[:, 3])]).T

        '''
        We assume that in a single action the result of actions are linear,
              believe that every actions effect equally and the change of distance is ignored.
        So we can use the inflence of ipk_action as a reference.
        '''
        ipk_2d_act[np.where(ipk_2d_act < 0.05)] = 100
        x_rate = ipk_trans[:, 0] / ipk_2d_act[:, 0]
        y_rate = ipk_trans[:, 1] / ipk_2d_act[:, 1]
        # x_rate[np.isinf(x_rate)] = self.last_x_rate
        # y_rate[np.isinf(y_rate)] = self.last_y_rate
        assert x_rate.shape == y_rate.shape
        # if x_rate.shape == (1,):
        #     self.last_x_rate = x_rate
        #     self.last_y_rate = y_rate
        # print(' x_rate:{}, y_rate:{}'.format(x_rate, y_rate))

        mbpo_trans = np.array([x_rate * mbpo_2d_act[:, 0], y_rate * mbpo_2d_act[:, 1], ipk_trans[:, 2]]).T

        try:
            obs_tmp.shape == mbpo_trans.shape
        except:
            pdb.set_trace()
        batch_mbpo_next_obs = obs_tmp + mbpo_trans
        # print('batch_mbpo_next_obs:{},{}'.format(batch_mbpo_next_obs, batch_mbpo_next_obs.shape))
        # if DEBUG_flg:
        #     for i in range(batch_size):
        #         ipk_trans = real_next_obs_tmp[i] - obs_tmp[i]
        #         ipk_2d_act = np.array([ipk_act_tmp[i][2] - ipk_act_tmp[i][0], ipk_act_tmp[i][1] - ipk_act_tmp[i][3]])
        #         mbpo_2d_act = np.array(
        #             [mbpo_act_tmp[i][2] - mbpo_act_tmp[i][0], mbpo_act_tmp[i][1] - mbpo_act_tmp[i][3]])
        #
        #         x_rate = ipk_trans[0] / ipk_2d_act[0] if ipk_2d_act[0] != 0 else 0
        #         y_rate = ipk_trans[1] / ipk_2d_act[1] if ipk_2d_act[1] != 0 else 0
        #
        #         mbpo_trans = np.array([x_rate * mbpo_2d_act[0], y_rate * mbpo_2d_act[1], ipk_trans[2]]).reshape((3,))
        #
        #         mbpo_next_obs = obs_tmp[i] + mbpo_trans
        #         batch_mbpo_next_obs[i] = mbpo_next_obs
        #         print('mbpo_next_obs:{}'.format(mbpo_next_obs))
        try:
            type(batch_mbpo_next_obs[0][0]) == np.float64
        except:
            pdb.set_trace()

        # 使用real的accum_actions作为mbpo的基础
        # batch_mbpo_next_obs = np.concatenate([batch_mbpo_next_obs, mbpo_accum_actions], axis=1)
        # assert batch_mbpo_next_obs.shape == (batch_size, 7)
        return batch_mbpo_next_obs

    def nan_inf_proc(self, a):
        where_are_nan = np.isnan(a)
        where_are_inf = np.isnan(a)
        a[where_are_nan] = 0
        a[where_are_inf] = 0
        return a

    def calcu_zeta_coeff(self, lower, upper, range_L, range_H):
        k = lower / upper
        b = (.5 - k) / (1 - k)
        a = upper / (1 - b)
        c = range_H / (np.log((9 + b) / (1 - b)))
        return a, b, c


if __name__ == '__main__':
    obs = [np.array([[67, -268, 4259.3]])]
    real_next_obs = np.array([71, -272, 4287])
    ipk_act = np.array([-1., 0.63463, - 1., 1.])
    mbpo_act = np.array([[-1, -1, -1, 1]])
    print('obs:{},\r\nreal_next_obs:{},\r\nipk_act:{},\r\nmbpo_act:{}'.format(obs, real_next_obs, ipk_act, mbpo_act))
    # sampler = IPKSampler(max_path_length=10, min_pool_size=10, batch_size=10)
    mbpo_next_obs = IPKSampler.get_mbpo_next_observation(obs=obs,
                                                         real_next_obs=real_next_obs,
                                                         ipk_act=ipk_act,
                                                         mbpo_act=mbpo_act)
    print(mbpo_next_obs)
