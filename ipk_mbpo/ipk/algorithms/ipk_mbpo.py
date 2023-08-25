import math
import os
import pickle
from collections import OrderedDict
from itertools import count
from numbers import Number
import pdb

import gtimer as gt
import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util

import mbpo.utils.filesystem as filesystem
from ipk.models.constructor import construct_model, format_samples_for_training
# from mbpo.models.fake_env import FakeEnv
from ipk.models.fake_env import FakeEnv
from mbpo.utils.logging import Progress
from mbpo.utils.visualization import visualize_policy
from mbpo.utils.writer import Writer

from mbpo.algorithms.mbpo import MBPO
from ipk.replay_pools.ipk_simple_replay_pool import IPKSimpleReplayPool
from ipk.algorithms.rl_algorithm import RLAlgorithm
from ipk.postprocessors import kalman_filter
from ipk.policies.ipk_basic_policy import IPKBasicPolicy

IPK = True
use_fake_env = False


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class IPK(RLAlgorithm):
    """Model-Based Reinforcement Learning based on Imexplicit Priori Knowledge (IPK)

    On the base of the previous MBPO, we change the policy output into a prob form.
    And use an action representation technique for infinite DOFs.

    References
    ----------
        Junjia Liu, Jiaying Shou
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            static_fns,
            plotter=None,
            tf_summaries=False,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            target_distribution='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,

            deterministic=False,
            model_train_freq=250,
            num_networks=7,
            num_elites=5,
            model_retain_epochs=20,
            rollout_batch_size=100e3,  # 100e3
            real_ratio=0.1,
            rollout_schedule=[20, 100, 1, 1],
            hidden_dim=200,
            max_model_t=None,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(IPK, self).__init__(**kwargs)

        obs_dim = np.prod(training_environment.observation_space.shape)
        act_dim = np.prod(training_environment.action_space.shape)
        self._model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim,
                                      num_networks=num_networks, num_elites=num_elites)
        self._static_fns = static_fns
        self.fake_env = FakeEnv(self._model, self._static_fns)

        self._rollout_schedule = rollout_schedule
        self._max_model_t = max_model_t

        # self._model_pool_size = model_pool_size
        # print('[ MBPO ] Model pool size: {:.2E}'.format(self._model_pool_size))
        # self._model_pool = SimpleReplayPool(pool._observation_space, pool._action_space, self._model_pool_size)

        self._model_retain_epochs = model_retain_epochs

        self._model_train_freq = model_train_freq
        self._rollout_batch_size = int(rollout_batch_size)
        self._deterministic = deterministic
        self._real_ratio = real_ratio

        self._log_dir = os.getcwd()
        self._writer = Writer(self._log_dir)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter
        self._tf_summaries = tf_summaries

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)
        print('[ MBPO ] Target entropy: {}'.format(self._target_entropy))
        self._target_distribution = (
            -2  # 5.0 TODO: Need to tune
            if target_distribution == 'auto'
            else target_distribution)
        print('[ IPK ] Target Gaussian Distribution: {}'.format(self._target_distribution))
        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self.ipk_basic_policy = IPKBasicPolicy()

        self._build()

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()

        # variable_names = [v.name for v in tf.trainable_variables()]
        # print(variable_names)

    def _train(self):

        """Return a generator that performs RL training.

        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        training_environment = self._training_environment
        evaluation_environment = self._evaluation_environment
        # policy = self._policy
        pool = self._pool
        model_metrics = {}

        if not self._training_started:
            self._init_training()

            # 要求 pool 中的总样本数超过 n_initial_exploration_steps, 使用初始化探索策略, 其中包含初始化 sampler
            self._initial_exploration_hook(
                training_environment, self._initial_exploration_policy, pool, self.fake_env)

            # 返回basic_policy 的正态分布信息
            self.basic_info = kalman_filter.accurancy_calcu_from_pool(
                self.sampler.path_lengths, pool)

        print('ipk_basic_mean: {}, ipk_basic_variance: {}'.format(self.basic_info['ipk_basic_mean'],
                                                                  self.basic_info['ipk_basic_variance']))
        # 向 gaussian_policy 传递 basic_policy 的正态分布信息
        self.basic_gaussian = self._policy.kalman_filter.get_ipk_basic_norm_info(info=self.basic_info)
        print('basic_gaussian: {}'.format(self.basic_gaussian))
        policy = self._policy

        # 初始化 sampler
        self.sampler.initialize(training_environment, policy, pool)

        gt.reset_root()
        gt.rename_root('RLAlgorithm')
        gt.set_def_unique(False)

        # training 前钩 (update target distribution difference)
        self._training_before_hook()

        # 开始 epoch 循环
        for self._epoch in gt.timed_for(range(self._epoch, self._n_epochs)):
            print('Epoch Loop')
            # epoch 前钩 (self._train_steps_this_epoch = 0)
            self._epoch_before_hook()
            gt.stamp('epoch_before_hook')

            self._training_progress = Progress(self._epoch_length * self._n_train_repeat)
            start_samples = self.sampler._total_samples

            # 开始 timestep 循环 (无限循环)
            for i in count():
                print('Timestep Loop')
                samples_now = self.sampler._total_samples
                self._timestep = samples_now - start_samples

                # 如果超过 start_samples + self._epoch_length 并且总样本数多于起始训练数量, 才会停止 sample
                if (samples_now >= start_samples + self._epoch_length
                        and self.ready_to_train):
                    break
                print('     sample_now:{}, start_samples:{}, epoch_length:{}'
                      .format(samples_now, start_samples, self._epoch_length))

                # timestep 前钩 (无操作)
                self._timestep_before_hook()
                gt.stamp('timestep_before_hook')

                # 每 self._model_train_freq 个 timesteps 训练一次 model
                if self._timestep % self._model_train_freq == 0 and self._real_ratio < 1.0:
                    self._training_progress.pause()
                    print('[ MBPO ] log_dir: {} | ratio: {}'.format(self._log_dir, self._real_ratio))
                    print(
                        '[ MBPO ] Training model at epoch {} | freq {} | timestep {} (total: {}) | epoch train steps: {} (total: {})'.format(
                            self._epoch, self._model_train_freq, self._timestep, self._total_timestep,
                            self._train_steps_this_epoch, self._num_train_steps)
                    )

                    model_train_metrics = self._train_model(batch_size=256, max_epochs=None, holdout_ratio=0.2,
                                                            max_t=self._max_model_t)
                    model_metrics.update(model_train_metrics)
                    gt.stamp('epoch_train_model')

                    self._set_rollout_length()
                    self._reallocate_model_pool()
                    model_rollout_metrics = self._rollout_model(rollout_batch_size=self._rollout_batch_size,
                                                                deterministic=self._deterministic)
                    model_metrics.update(model_rollout_metrics)

                    gt.stamp('epoch_rollout_model')
                    # self._visualize_model(self._evaluation_environment, self._total_timestep)
                    self._training_progress.resume()

                # model-free 交互采样
                print('_do_sampling')
                self._do_sampling(timestep=self._total_timestep, fake_env=self.fake_env)
                gt.stamp('sample')

                # model-free algorithm start learning 通过判断 pool 中的 samples是否达到最小样本要求
                if self.ready_to_train:
                    print('_do_training_repeats')
                    self._do_training_repeats(timestep=self._total_timestep)
                gt.stamp('train')

                # timestep 后钩 (无操作)
                self._timestep_after_hook()
                gt.stamp('timestep_after_hook')

            training_paths = self.sampler.get_last_n_paths(
                math.ceil(self._epoch_length / self.sampler._max_path_length))
            gt.stamp('training_paths')
            evaluation_paths = self._evaluation_paths(
                policy, evaluation_environment, self.fake_env)
            gt.stamp('evaluation_paths')

            training_metrics = self._evaluate_rollouts(
                training_paths, training_environment)
            gt.stamp('training_metrics')
            if evaluation_paths:
                evaluation_metrics = self._evaluate_rollouts(
                    evaluation_paths, evaluation_environment)
                gt.stamp('evaluation_metrics')
            else:
                evaluation_metrics = {}

            # epoch 后钩 (未用)
            self._epoch_after_hook(training_paths)
            gt.stamp('epoch_after_hook')

            sampler_diagnostics = self.sampler.get_diagnostics()

            diagnostics = self.get_diagnostics(
                iteration=self._total_timestep,
                batch=self._evaluation_batch(),
                training_paths=training_paths,
                evaluation_paths=evaluation_paths)

            time_diagnostics = gt.get_times().stamps.itrs

            diagnostics.update(OrderedDict((
                *(
                    (f'evaluation/{key}', evaluation_metrics[key])
                    for key in sorted(evaluation_metrics.keys())
                ),
                *(
                    (f'training/{key}', training_metrics[key])
                    for key in sorted(training_metrics.keys())
                ),
                *(
                    (f'times/{key}', time_diagnostics[key][-1])
                    for key in sorted(time_diagnostics.keys())
                ),
                *(
                    (f'sampler/{key}', sampler_diagnostics[key])
                    for key in sorted(sampler_diagnostics.keys())
                ),
                *(
                    (f'model/{key}', model_metrics[key])
                    for key in sorted(model_metrics.keys())
                ),
                ('epoch', self._epoch),
                ('timestep', self._timestep),
                ('timesteps_total', self._total_timestep),
                ('train-steps', self._num_train_steps),
            )))

            if self._eval_render_mode is not None and hasattr(
                    evaluation_environment, 'render_rollouts'):
                training_environment.render_rollouts(evaluation_paths)

            yield diagnostics

        # 结束 sample -> 关闭环境交互 (self.env.close())
        self.sampler.terminate()

        # training 后钩 (无操作)
        self._training_after_hook()

        self._training_progress.close()

        yield {'done': True, **diagnostics}

    def train(self, *args, **kwargs):
        return self._train(*args, **kwargs)

    def _log_policy(self):
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        weights = self._policy.get_weights()
        data = {'policy_weights': weights}
        full_path = os.path.join(save_path, 'policy_{}.pkl'.format(self._total_timestep))
        print('Saving policy to: {}'.format(full_path))
        pickle.dump(data, open(full_path, 'wb'))

    def _log_model(self):
        save_path = os.path.join(self._log_dir, 'models')
        filesystem.mkdir(save_path)
        print('Saving model to: {}'.format(save_path))
        self._model.save(save_path, self._total_timestep)

    def _set_rollout_length(self):
        min_epoch, max_epoch, min_length, max_length = self._rollout_schedule
        if self._epoch <= min_epoch:
            y = min_length
        else:
            dx = (self._epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self._rollout_length = int(y)
        print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
            self._epoch, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        ))

    def _reallocate_model_pool(self):
        obs_space = self._pool._observation_space
        act_space = self._pool._action_space

        rollouts_per_epoch = self._rollout_batch_size * self._epoch_length / self._model_train_freq
        model_steps_per_epoch = int(self._rollout_length * rollouts_per_epoch)
        new_pool_size = self._model_retain_epochs * model_steps_per_epoch

        if not hasattr(self, '_model_pool'):
            print('[ MBPO ] Initializing new model pool with size {:.2e}'.format(
                new_pool_size
            ))
            self._model_pool = IPKSimpleReplayPool(obs_space, act_space, new_pool_size)

        elif self._model_pool._max_size != new_pool_size:
            print('[ MBPO ] Updating model pool | {:.2e} --> {:.2e}'.format(
                self._model_pool._max_size, new_pool_size
            ))
            samples = self._model_pool.return_all_samples()
            new_pool = IPKSimpleReplayPool(obs_space, act_space, new_pool_size)
            new_pool.add_samples(samples)
            assert self._model_pool.size == new_pool.size
            self._model_pool = new_pool

    def _train_model(self, **kwargs):
        env_samples = self._pool.return_all_samples()
        train_inputs, train_outputs = format_samples_for_training(env_samples)
        model_metrics = self._model.train(train_inputs, train_outputs, **kwargs)
        return model_metrics

    def _rollout_model(self, rollout_batch_size, **kwargs):

        print('[ Model Rollout ] Starting | Epoch: {} | Rollout length: {} | Batch size: {}'.format(
            self._epoch, self._rollout_length, rollout_batch_size
        ))
        batch = self.sampler.random_batch(rollout_batch_size)
        obs = batch['observations']
        # print('rollout_obs:{},{}'.format(obs, type(obs)))
        steps_added = []
        # KL_diver_mean = np.zeros(self._rollout_length)
        for i in range(self._rollout_length):
            if IPK:
                if use_fake_env:
                    mbpo_next_obs, mbpo_reward, mbpo_term, mbpo_info = self.fake_env.step(obs, mbpo_act, **kwargs)
                else:
                    '''Update zeta coefficient'''
                    try:
                        # ipk_fusion_gaussian = self._policy.get_kal_diagnostics([obs])
                        mbpo_gaussian = self._policy.get_distribution([obs])
                        # print('obs_shape:{},guassian_shape:{}'.format(obs.shape, mbpo_gaussian['shifts'].shape))
                        # print('\r\nmbpo_action : {}\r\n ipk_fusion_action : {}\r\n ipk_fusion_gaussian : {}\r\n'
                        #       ' mbpo_gaussian : {}'
                        #       .format(mbpo_act, ipk_fusion_act, ipk_fusion_gaussian, mbpo_gaussian))

                        # mu = a(1+mu)
                        # sigma^2 = a^2 * sigma^2 -> log_sigma = 0.5 * log(a^2) + log_sigma
                        ipk_basic_gaussian = {'shifts': self.ipk_basic_policy.action_np(obs) * (np.ones((1,4)) + self._policy.kalman_filter.mu_np),
                                              'log_scale_diags': 0.5 * np.log(np.square(self.ipk_basic_policy.action_np(obs))) + self._policy.kalman_filter.log_sigma_np.reshape(
                                                  (1, 4))}

                        mbpo_xy = {'shifts': np.array([mbpo_gaussian['shifts'][0][2] + mbpo_gaussian['shifts'][0][0],
                                                       mbpo_gaussian['shifts'][0][3] + mbpo_gaussian['shifts'][0][1]]),
                                   'log_scale_diags': np.array(
                                       [max(mbpo_gaussian['log_scale_diags'][0][2], mbpo_gaussian['log_scale_diags'][0][
                                           0]),
                                        max(mbpo_gaussian['log_scale_diags'][0][3], mbpo_gaussian['log_scale_diags'][0][
                                            1])])}
                        basic_xy = {
                            'shifts': np.array([ipk_basic_gaussian['shifts'][0][2] + ipk_basic_gaussian['shifts'][0][0],
                                                ipk_basic_gaussian['shifts'][0][3] + ipk_basic_gaussian['shifts'][0][1]]),
                            'log_scale_diags': np.array(
                                [max(ipk_basic_gaussian['log_scale_diags'][0][2], ipk_basic_gaussian['log_scale_diags'][0][
                                    0]),
                                 max(ipk_basic_gaussian['log_scale_diags'][0][3], ipk_basic_gaussian['log_scale_diags'][0][
                                     1])])}
                    except:
                        pdb.set_trace()
                    KL_diver = self.sampler.multivar_continue_KL_divergence(basic_xy, mbpo_xy)
                    self.sampler.update_zeta(KL_diver)

                    '''Interaction'''
                    mbpo_act, ipk_fusion_act = self._policy.actions_np([obs])
                    real_next_obs, real_rew, term, info = self.fake_env.step(obs, ipk_fusion_act, **kwargs)
                    steps_added.append(len(obs))

                    '''MBPO rollout rew estimation'''
                    mbpo_reward_core = - self.sampler.mbpo_rew_kl_coeff * (
                                        np.sum(abs(KL_diver)) + self.sampler.target_distri) + real_rew
                    # mbpo_reward_core = - self.sampler.mbpo_rew_kl_coeff * np.sum(abs(KL_diver)) + real_rew
                    mbpo_reward = self.sampler.zeta_basic * mbpo_reward_core + self.sampler.zeta_real * real_rew

                    print('mbpo_reward:{}; real_reward:{}; KL:{}'.format(mbpo_reward, real_rew,
                                                                         KL_diver))

                    '''MBPO rollout next obs estimation'''
                    mbpo_next_obs = self.sampler.get_mbpo_next_observation(obs=obs,
                                                                           real_next_obs=real_next_obs,
                                                                           ipk_act=ipk_fusion_act,
                                                                           mbpo_act=mbpo_act)

                    print('obs_rollout: {},{}'.format(obs, obs.shape))
                    print('mbpo_next_observation: {},{}'.format(mbpo_next_obs, mbpo_next_obs.shape))
                    # ------------------------------------------------------
            else:
                mbpo_act, ipk_fusion_act = self._policy.actions_np([obs])
                ipk_fusion_act = mbpo_act

                real_next_obs, real_rew, term, info = self.fake_env.step(obs, ipk_fusion_act, **kwargs)
                steps_added.append(len(obs))

                # ------------- Reward engineering ------------------
                mbpo_reward = real_rew
                mbpo_next_obs = real_next_obs
                print('obs_rollout: {},{}'.format(obs, obs.shape))
                print('mbpo_next_observation: {},{}'.format(mbpo_next_obs, mbpo_next_obs.shape))

            samples = {'observations': obs, 'actions': mbpo_act, 'ipk_actions': ipk_fusion_act,
                       'next_observations': mbpo_next_obs, 'real_next_observations': real_next_obs,
                       'rewards': mbpo_reward, 'real_rewards': real_rew,
                       'terminals': term}
            self._model_pool.add_samples(samples)

            nonterm_mask = ~term.squeeze(-1)
            if nonterm_mask.sum() == 0:
                print(
                    '[ Model Rollout ] Breaking early: {} | {} / {}'.format(i, nonterm_mask.sum(), nonterm_mask.shape))
                break

            obs = real_next_obs[nonterm_mask]
        # self.KL_diver = np.mean(KL_diver_mean)
        mean_rollout_length = sum(steps_added) / rollout_batch_size
        rollout_stats = {'mean_rollout_length': mean_rollout_length}
        print('[ Model Rollout ] Added: {:.1e} | Model pool: {:.1e} (max {:.1e}) | Length: {} | Train rep: {}'.format(
            sum(steps_added), self._model_pool.size, self._model_pool._max_size, mean_rollout_length,
            self._n_train_repeat
        ))
        return rollout_stats

    def _visualize_model(self, env, timestep):
        ## save env state
        state = env.unwrapped.state_vector()
        qpos_dim = len(env.unwrapped.sim.data.qpos)
        qpos = state[:qpos_dim]
        qvel = state[qpos_dim:]

        print('[ Visualization ] Starting | Epoch {} | Log dir: {}\n'.format(self._epoch, self._log_dir))
        visualize_policy(env, self.fake_env, self._policy, self._writer, timestep)
        print('[ Visualization ] Done')
        ## set env state
        env.unwrapped.set_state(qpos, qvel)

    def _training_batch(self, batch_size=None):
        batch_size = batch_size or self.sampler._batch_size
        batch_size = int(batch_size / 2)  # For concat data !!!
        env_batch_size = int(batch_size * self._real_ratio)
        model_batch_size = batch_size - env_batch_size

        ## can sample from the env pool even if env_batch_size == 0
        # pdb.set_trace()
        # Remember to sample and make sure the datch size is equal to 256 which is defined in ipk_development
        env_batch = self._pool.random_batch(env_batch_size)
        env_batch = {
            'observations': np.concatenate((env_batch['observations'], env_batch['observations']), axis=0),
            'next_observations': np.concatenate(
                (env_batch['next_observations'], env_batch['real_next_observations']), axis=0),
            'actions': np.concatenate((env_batch['actions'], env_batch['ipk_actions']), axis=0),
            # 'ipk_actions': np.concatenate((env_batch['ipk_actions'], env_batch['ipk_actions']), axis=0),
            'rewards': np.concatenate((env_batch['rewards'], env_batch['real_rewards']), axis=0),
            'terminals': np.concatenate((env_batch['terminals'], env_batch['terminals']), axis=0)}

        if model_batch_size > 0:
            model_batch = self._model_pool.random_batch(model_batch_size)
            model_batch = {
                'observations': np.concatenate((model_batch['observations'], model_batch['observations']), axis=0),
                'next_observations': np.concatenate(
                    (model_batch['next_observations'], model_batch['real_next_observations']), axis=0),
                'actions': np.concatenate((model_batch['actions'], model_batch['ipk_actions']), axis=0),
                # 'ipk_actions': np.concatenate((env_batch['ipk_actions'], env_batch['ipk_actions']), axis=0),
                'rewards': np.concatenate((model_batch['rewards'], model_batch['real_rewards']), axis=0),
                'terminals': np.concatenate((model_batch['terminals'], model_batch['terminals']), axis=0)}

            keys = env_batch.keys()
            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}



        else:
            ## if real_ratio == 1.0, no model pool was ever allocated,
            ## so skip the model pool sampling
            batch = env_batch
        return batch

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        # self._ipk_actions_ph = tf.placeholder(
        #     tf.float32,
        #     shape=(None, *self._action_shape),
        #     name='ipk_basic_actions',
        # )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions(
            [self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))
        Q_training_ops = tuple(
            tf.contrib.layers.optimize_loss(
                Q_loss,
                self.global_step,
                learning_rate=self._Q_lr,
                optimizer=Q_optimizer,
                variables=Q.trainable_variables,
                increment_global_step=False,
                summaries=((
                               "loss", "gradients", "gradient_norm", "global_gradient_norm"
                           ) if self._tf_summaries else ()))
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.
        """

        actions = self._policy.actions([self._observations_ph])
        # kal_actions = self._policy.kal_actions([self._observations_ph], self._ipk_actions_ph)
        log_pis = self._policy.log_pis([self._observations_ph], actions)
        # kal_log_pis = self._policy.kal_log_pis([self._observations_ph], kal_actions, self._ipk_actions_ph)

        try:
            assert log_pis.shape.as_list() == [None, 1]
            # assert kal_log_pis.shape.as_list() == [None, 1]
        except:
            pdb.set_trace()

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            # H_0: self._target_entropy, log\pi_t(a_t|\pi_t): log_pis
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        # log_zeta = self._log_zeta = tf.get_variable(
        #     'log_zeta',
        #     dtype=tf.float32,
        #     initializer=0.0
        # )
        # zeta = tf.exp(log_zeta)
        # kal_log_pis = tf.clip_by_value(kal_log_pis, -20, 20)
        # log_pis = tf.clip_by_value(log_pis, -20, 20)

        # if isinstance(self._target_distribution, Number):
        #     # D_0: self._target_distribution, log\pi_t(a_t|\pi_t): log_pis
        #     zeta_loss = -tf.reduce_mean(
        #         log_zeta * tf.stop_gradient(kal_log_pis - log_pis + self._target_distribution))
        #
        #     self._zeta_optimizer = tf.train.AdamOptimizer(
        #         self._policy_lr, name='zeta_optimizer')
        #     self._zeta_train_op = self._zeta_optimizer.minimize(
        #         loss=zeta_loss, var_list=[log_zeta])
        #
        #     self._training_ops.update({
        #         'zeta': self._zeta_train_op
        #     })

        # self._zeta = tf.tanh(zeta / 0.6)

        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                    # zeta * (kal_log_pis - log_pis)
                    alpha * log_pis
                    - min_Q_log_target  # Q_
                    - policy_prior_log_probs)  # logZ(s_t)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")
        policy_train_op = tf.contrib.layers.optimize_loss(
            policy_loss,
            self.global_step,
            learning_rate=self._policy_lr,
            optimizer=self._policy_optimizer,
            variables=self._policy.trainable_variables,
            increment_global_step=False,
            summaries=(
                "loss", "gradients", "gradient_norm", "global_gradient_norm"
            ) if self._tf_summaries else ())

        self._training_ops.update({'policy_train_op': policy_train_op})

    # def _init_IPK_eta_update(self, basic_distribution):
    #     """
    #     Creates a `tf.optimizer.minimize` operations for updating
    #     ipk_eta_basic with gradient descent, and adds it to
    #     `self._training_ops` attribute.
    #     """
    #     actions = self._policy.actions([self._observations_ph])
    #     mbpo_gaussian = self.sampler.policy.get_distribution([
    #         self.sampler.env.convert_to_active_observation(
    #             self._observations_ph)[None]])
    #     basic_gaussian = OrderedDict({
    #         'shifts': basic_distribution['mu'],
    #         'log_scale_diags': basic_distribution['log_sigma_square'],
    #     })
    #     policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
    #         loc=tf.zeros(self._action_shape),
    #         scale_diag=tf.ones(self._action_shape))
    #     policy_prior_log_probs = policy_prior.log_prob(actions)
    #     distribution_KL = self.sampler.KL_divergence(basic_gaussian, mbpo_gaussian)
    #
    #     # assert log_pis.shape.as_list() == [None, 1]
    #
    #     eta_basic = self._eta_basic = tf.get_variable(
    #         'eta_basic',
    #         dtype=tf.float32,
    #         initializer=1.0)
    #
    #     if isinstance(self._target_distribution, Number):
    #         # H_0: self._target_entropy, log\pi_t(a_t|\pi_t): log_pis
    #         eta_loss = -tf.reduce_mean(
    #             eta_basic * tf.stop_gradient(distribution_KL + self._target_distribution))
    #
    #         self._eta_optimizer = tf.train.AdamOptimizer(
    #             self._policy_lr, name='eta_optimizer')
    #         self._eta_train_op = self._eta_optimizer.minimize(
    #             loss=eta_loss, var_list=[eta_basic])
    #
    #         self._training_ops.update({
    #             'ada_eta': self._eta_train_op
    #         })
    #
    #     self.sampler.eta_basic = eta_basic

    def _init_training(self):
        self._update_target(tau=1.0)

    def _training_before_hook(self, *args, **kwargs):
        # pass
        # self._training_progress.update()
        # self._training_progress.set_description()
        #
        # feed_dict = self._get_feed_dict(iteration, batch)
        #
        # self._session.run(self._training_ops, feed_dict)
        self.sampler.update_target_dis(self._target_distribution)
        # self.sampler.update_zeta(self._zeta)

    def _timestep_after_hook(self, *args, **kwargs):
        # print('After hook KL_diver : {}'.format(self.KL_diver))
        print('zeta_basic: {}, zeta_real: {}\r\n'.format(self.sampler.zeta_basic, self.sampler.zeta_real))

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        # print('training_iteration:{}\r\ntraining_batch:{}'.format(iteration, batch))
        self._training_progress.update()
        self._training_progress.set_description()
        # pdb.set_trace()

        feed_dict = self._get_feed_dict(iteration, batch)

        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            # self._ipk_actions_ph: batch['ipk_actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)

        (Q_values, Q_losses, alpha, global_step) = self._session.run(
            (self._Q_values,
             self._Q_losses,
             self._alpha,
             # self._zeta,
             self.global_step),
            feed_dict)

        diagnostics = OrderedDict({
            'Q-avg': np.mean(Q_values),
            'Q-std': np.std(Q_values),
            'Q_loss': np.mean(Q_losses),
            'alpha': alpha,
            # 'zeta': zeta,
        })

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
            # '_log_zeta': self._log_zeta,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        # if hasattr(self, '_zeta_optimizer'):
        #     saveables['_zeta_optimizer'] = self._zeta_optimizer

        return saveables
