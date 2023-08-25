"""GaussianPolicy."""

from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from softlearning.distributions.squash_bijector import SquashBijector
from softlearning.models.feedforward import feedforward_model

from ipk.policies.ipk_base_policy import LatentSpacePolicy

from ipk.postprocessors.kalman_filter import KalmanFilter

import pdb

SCALE_DIAG_MIN_MAX = (-20, 2)


# SHIFT_MIN_MAX = (-5, 5)


class GaussianPolicy(LatentSpacePolicy):
    def __init__(self,
                 input_shapes,
                 output_shape,
                 squash=True,
                 preprocessor=None,
                 name=None,
                 *args,
                 **kwargs):
        self._Serializable__initialize(locals())

        self._input_shapes = input_shapes
        self._output_shape = output_shape
        self._squash = squash
        self._name = name
        self._preprocessor = preprocessor

        super(GaussianPolicy, self).__init__(*args, **kwargs)

        self.kalman_filter = KalmanFilter()

        self.condition_inputs = [
            tf.keras.layers.Input(shape=input_shape)
            for input_shape in input_shapes
        ]

        self.ipk_basic_act = tf.keras.layers.Input(shape=output_shape)
        self.convert_2d_matrix = tf.keras.backend.constant(np.array([[1, 0, 1, 0], [0, 1, 0, 1]]).T, dtype=tf.float32)

        conditions = tf.keras.layers.Lambda(
            lambda x: tf.concat(x, axis=-1)
        )(self.condition_inputs)

        if preprocessor is not None:
            conditions = preprocessor(conditions)

        # small_conditions = tf.keras.layers.Lambda(
        #     lambda conditions: conditions / 5
        # )(conditions)
        shift_and_log_scale_diag = self._shift_and_log_scale_diag_net(
            input_shapes=(conditions.shape[1:],),
            output_size=output_shape[0] * 2,
        )(conditions)

        shift, log_scale_diag = tf.keras.layers.Lambda(
            lambda shift_and_log_scale_diag: tf.split(
                shift_and_log_scale_diag,
                num_or_size_splits=2,
                axis=-1)
        )(shift_and_log_scale_diag)

        # shift = tf.keras.layers.Lambda(
        #     lambda shift: tf.clip_by_value(
        #         shift, *SHIFT_MIN_MAX)
        # )(shift)

        log_scale_diag = tf.keras.layers.Lambda(
            lambda log_scale_diag: tf.clip_by_value(
                log_scale_diag, *SCALE_DIAG_MIN_MAX)
        )(log_scale_diag)

        kal_shift, kal_log_scale_diag = tf.keras.layers.Lambda(
            lambda inputs: self.kalman_filter.kalman_filter_1d(inputs)
        )((shift, log_scale_diag, self.ipk_basic_act))

        kal_log_scale_diag = tf.keras.layers.Lambda(
            lambda kal_log_scale_diag: tf.clip_by_value(
                kal_log_scale_diag, *SCALE_DIAG_MIN_MAX)
        )(kal_log_scale_diag)

        batch_size = tf.keras.layers.Lambda(
            lambda x: tf.shape(x)[0])(conditions)

        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(output_shape),
            scale_diag=tf.ones(output_shape))

        latents = tf.keras.layers.Lambda(
            lambda batch_size: base_distribution.sample(batch_size)
        )(batch_size)

        # self.kal_info_model = tf.keras.Model((*self.condition_inputs, self.ipk_basic_act),
        #                                      (kal_shift, kal_log_scale_diag))
        self.latents_model = tf.keras.Model(self.condition_inputs, latents)
        self.latents_input = tf.keras.layers.Input(shape=output_shape)

        def raw_actions_fn(inputs):
            shift, log_scale_diag, latents = inputs
            bijector = tfp.bijectors.Affine(
                shift=shift,
                scale_diag=tf.exp(log_scale_diag))
            actions = bijector.forward(latents)
            return actions

        print('shift: {}, log_scale_diag: {}'.format(shift, log_scale_diag))

        # ---------- kalman fusion raw action layers --------
        kal_raw_actions = tf.keras.layers.Lambda(
            raw_actions_fn
        )((kal_shift, kal_log_scale_diag, latents))

        kal_raw_actions_for_fixed_latents = tf.keras.layers.Lambda(
            raw_actions_fn
        )((kal_shift, kal_log_scale_diag, self.latents_input))
        # -----------------------------------------------------

        # ---------- mbpo raw action layers -----------------
        raw_actions = tf.keras.layers.Lambda(
            raw_actions_fn
        )((shift, log_scale_diag, latents))

        raw_actions_for_fixed_latents = tf.keras.layers.Lambda(
            raw_actions_fn
        )((shift, log_scale_diag, self.latents_input))
        # -------------------------------------------------

        squash_bijector = (
            SquashBijector()
            if self._squash
            else tfp.bijectors.Identity())

        # -------- kalman fusion actions ------------------
        kal_actions = tf.keras.layers.Lambda(
            lambda kal_raw_actions: squash_bijector.forward(kal_raw_actions)
        )(kal_raw_actions)

        kal_actions_for_fixed_latents = tf.keras.layers.Lambda(
            lambda kal_raw_actions: squash_bijector.forward(kal_raw_actions)
        )(kal_raw_actions_for_fixed_latents)
        # --------------------------------------------------

        # --------- mbpo actions --------------------------
        actions = tf.keras.layers.Lambda(
            lambda raw_actions: squash_bijector.forward(raw_actions)
        )(raw_actions)

        actions_for_fixed_latents = tf.keras.layers.Lambda(
            lambda raw_actions: squash_bijector.forward(raw_actions)
        )(raw_actions_for_fixed_latents)
        # -----------------------------------------------------

        # output fusion actions after kalman filter between original mbpo and ipk
        self.kal_actions_model = tf.keras.Model(
            (*self.condition_inputs, self.ipk_basic_act), kal_actions)

        self.kal_actions_model_for_fixed_latents = tf.keras.Model(
            (*self.condition_inputs, self.latents_input, self.ipk_basic_act),
            kal_actions_for_fixed_latents)
        # ------------------------------------------------------

        # ---------- output the original mbpo actions ---------
        self.actions_model = tf.keras.Model(
            self.condition_inputs, actions)

        self.actions_model_for_fixed_latents = tf.keras.Model(
            (*self.condition_inputs, self.latents_input),
            actions_for_fixed_latents)
        # ----------------------------------------------------

        # --------- kalman fusion deterministic action -------
        kal_deterministic_actions = tf.keras.layers.Lambda(
            lambda kal_shift: squash_bijector.forward(kal_shift)
        )(kal_shift)

        self.kal_deterministic_actions_model = tf.keras.Model(
            (*self.condition_inputs, self.ipk_basic_act), kal_deterministic_actions)
        # --------------------------------------------------

        # ----------mbpo deterministic action -------------
        deterministic_actions = tf.keras.layers.Lambda(
            lambda shift: squash_bijector.forward(shift)
        )(shift)

        self.deterministic_actions_model = tf.keras.Model(
            self.condition_inputs, deterministic_actions)
        # -----------------------------------------------

        def log_pis_fn(inputs):
            shift, log_scale_diag, actions = inputs
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(output_shape),
                scale_diag=tf.ones(output_shape))
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=shift,
                    scale_diag=tf.exp(log_scale_diag)),
            ))
            distribution = (
                tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector))

            log_pis = distribution.log_prob(actions)[:, None]
            return log_pis

        # def log_pis_fn_2d(inputs):
        #     convert_2d_matrix = tf.keras.backend.constant(
        #         np.array([[1, 0, 1, 0], [0, 1, 0, 1]]).T, dtype=tf.float32)
        #     shift, log_scale_diag, actions = inputs
        #     # shift_2d = tf.convert_to_tensor([tf.add(shift[:, 0], shift[:, 2]), tf.add(shift[:, 1], shift[:, 3])])
        #     # log_scale_diag_2d = tf.convert_to_tensor(
        #     #     [tf.truediv(tf.add(log_scale_diag[:, 0], log_scale_diag[:, 2]), 2.0),
        #     #      tf.truediv(tf.add(log_scale_diag[:, 1], log_scale_diag[:, 3]), 2.0)])
        #     # actions_2d = tf.convert_to_tensor(
        #     #     [tf.add(actions[:, 0], actions[:, 2]), tf.add(actions[:, 1], actions[:, 3])])
        #     shift_2d = tf.matmul(shift, convert_2d_matrix)
        #     log_scale_diag_2d = tf.truediv(tf.matmul(log_scale_diag, convert_2d_matrix), 2.0)
        #     actions_2d = tf.matmul(actions, convert_2d_matrix)
        #     base_distribution = tfp.distributions.MultivariateNormalDiag(
        #         loc=tf.zeros((2,)),
        #         scale_diag=tf.ones((2,)))
        #     bijector = tfp.bijectors.Chain((
        #         squash_bijector,
        #         tfp.bijectors.Affine(
        #             shift=shift_2d,
        #             scale_diag=tf.exp(log_scale_diag_2d)),
        #     ))
        #     distribution = (
        #         tfp.distributions.ConditionalTransformedDistribution(
        #             distribution=base_distribution,
        #             bijector=bijector))
        #     # pdb.set_trace()
        #     log_pis = distribution.log_prob(actions_2d)[:, None]
        #     return log_pis

        self.actions_input = tf.keras.layers.Input(shape=output_shape)
        self.kal_actions_input = tf.keras.layers.Input(shape=output_shape, name='kal_actions_input')

        log_pis = tf.keras.layers.Lambda(
            log_pis_fn)([shift, log_scale_diag, actions])

        log_pis_for_action_input = tf.keras.layers.Lambda(
            log_pis_fn)([shift, log_scale_diag, self.actions_input])

        self.log_pis_model = tf.keras.Model(
            (*self.condition_inputs, self.actions_input),
            log_pis_for_action_input)

        kal_log_pis_for_action_input = tf.keras.layers.Lambda(
            log_pis_fn)([kal_shift, kal_log_scale_diag, self.kal_actions_input])

        self.kal_log_pis_model = tf.keras.Model(
            (*self.condition_inputs, self.kal_actions_input, self.ipk_basic_act),
            kal_log_pis_for_action_input)

        def add_fn(inputs):
            mu, act = inputs
            shift = tf.add(mu, act)
            return shift

        def bas_log_pis_fn(inputs):
            actions = inputs
            shift = tf.add(self.kalman_filter.mu, actions)
            log_scale_diag = self.kalman_filter.log_sigma
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(output_shape),
                scale_diag=tf.ones(output_shape))
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=shift,
                    scale_diag=tf.exp(log_scale_diag)),
            ))
            distribution = (
                tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector))

            log_pis = distribution.log_prob(actions)[:, None]
            return log_pis

        basic_shift = tf.keras.layers.Lambda(add_fn)([self.kalman_filter.mu, self.ipk_basic_act])

        kal_bas_log_pis_for_action_input = tf.keras.layers.Lambda(
            bas_log_pis_fn)(self.ipk_basic_act)

        self.kal_bas_log_pis_model = tf.keras.Model(
            (*self.condition_inputs, self.ipk_basic_act),
            kal_bas_log_pis_for_action_input)

        self.diagnostics_model = tf.keras.Model(
            self.condition_inputs,
            (shift, log_scale_diag, log_pis, raw_actions, actions))

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        raise NotImplementedError

    def get_weights(self):
        return self.actions_model.get_weights()

    def set_weights(self, *args, **kwargs):
        return self.actions_model.set_weights(*args, **kwargs)

    @property
    def trainable_variables(self):
        return self.actions_model.trainable_variables

    @property
    def non_trainable_weights(self):
        """Due to our nested model structure, we need to filter duplicates."""
        return list(set(super(GaussianPolicy, self).non_trainable_weights))

    def actions(self, conditions):
        if self._deterministic:
            return self.deterministic_actions_model(conditions)
        return self.actions_model(conditions)

    def kal_actions(self, conditions, ipk_basic_act_tf):
        # ipk_basic_action = self.ipk_basic_policy.action_np(conditions)
        # print('!!!!!!!!!!!{}'.format(ipk_basic_action))
        if self._deterministic:
            return self.kal_deterministic_actions_model([*conditions, ipk_basic_act_tf])
        return self.kal_actions_model([*conditions, ipk_basic_act_tf])

    def log_pis(self, conditions, actions):
        assert not self._deterministic, self._deterministic
        return self.log_pis_model([*conditions, actions])

    def kal_log_pis(self, conditions, actions, ipk_basic_act_tf):
        assert not self._deterministic, self._deterministic
        return self.kal_log_pis_model([*conditions, actions, ipk_basic_act_tf])

    def kal_bas_log_pis(self, conditions, ipk_basic_act_tf):
        assert not self._deterministic, self._deterministic
        return self.kal_bas_log_pis_model([*conditions, ipk_basic_act_tf])

    def actions_np(self, conditions):
        return super(GaussianPolicy, self).actions_np(conditions)

    def log_pis_np(self, conditions, actions):
        assert not self._deterministic, self._deterministic
        return self.log_pis_model.predict([*conditions, actions])

    def kal_log_pis_np(self, conditions, actions, ipk_basic_act_np):
        assert not self._deterministic, self._deterministic
        return self.kal_log_pis_model.predict([*conditions, actions, ipk_basic_act_np])

    def kal_bas_log_pis_np(self, conditions, ipk_basic_act_np):
        assert not self._deterministic, self._deterministic
        return self.kal_bas_log_pis_model.predict([*conditions, ipk_basic_act_np])

    def get_diagnostics(self, conditions):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (shifts_np,
         log_scale_diags_np,
         log_pis_np,
         raw_actions_np,
         actions_np) = self.diagnostics_model.predict(conditions)

        return OrderedDict({
            'shifts-mean': np.mean(shifts_np),
            'shifts-std': np.std(shifts_np),

            'log_scale_diags-mean': np.mean(log_scale_diags_np),
            'log_scale_diags-std': np.std(log_scale_diags_np),

            '-log-pis-mean': np.mean(-log_pis_np),
            '-log-pis-std': np.std(-log_pis_np),

            'raw-actions-mean': np.mean(raw_actions_np),
            'raw-actions-std': np.std(raw_actions_np),

            'actions-mean': np.mean(actions_np),
            'actions-std': np.std(actions_np),
        })

    # def get_kal_diagnostics(self, conditions):
    #     ipk_basic_action = self.ipk_basic_policy.action_np(conditions)
    #     (kal_shift, kal_log_scale_diag) = self.kal_info_model.predict([*conditions, ipk_basic_action])
    #     return OrderedDict({
    #         'shifts': kal_shift,
    #         'log_scale_diags': kal_log_scale_diag,
    #     })

    def get_basic_diagnostics(self, conditions):
        ipk_basic_action = self.ipk_basic_policy.action_np(conditions)
        # (kal_shift, kal_log_scale_diag) = self.kal_info_model.predict([*conditions, ipk_basic_action])
        (shifts_np,
         log_scale_diags_np,
         log_pis_np,
         raw_actions_np,
         actions_np) = self.diagnostics_model.predict(conditions)

        kal_shift = np.mean(shifts_np) + ipk_basic_action
        kal_log_scale_diag = np.mean(log_scale_diags_np)
        return OrderedDict({
            'shifts': kal_shift,
            'log_scale_diags': kal_log_scale_diag,
        })


class FeedforwardGaussianPolicy(GaussianPolicy):
    def __init__(self,
                 hidden_layer_sizes,
                 activation='sigmoid',
                 output_activation='tanh',
                 *args, **kwargs):
        self._hidden_layer_sizes = hidden_layer_sizes
        self._activation = activation
        self._output_activation = output_activation

        self._Serializable__initialize(locals())
        super(FeedforwardGaussianPolicy, self).__init__(*args, **kwargs)

    def _shift_and_log_scale_diag_net(self, input_shapes, output_size):
        shift_and_log_scale_diag_net = feedforward_model(
            input_shapes=input_shapes,
            hidden_layer_sizes=self._hidden_layer_sizes,
            output_size=output_size,
            activation=self._activation,
            output_activation=self._output_activation)

        return shift_and_log_scale_diag_net

    def get_distribution(self, conditions):
        """Return diagnostic information of the policy.

        Returns the mean, min, max, and standard deviation of means and
        covariances.
        """
        (shifts_np,
         log_scale_diags_np,
         log_pis_np,
         raw_actions_np,
         actions_np) = self.diagnostics_model.predict(conditions)

        return OrderedDict({
            'shifts': shifts_np,
            'log_scale_diags': log_scale_diags_np,
        })


if __name__ == '__main__':
    obs = [np.array([[-29., 237., 4287.5]])]
    policy = FeedforwardGaussianPolicy(input_shapes=((3,),), output_shape=(4,), hidden_layer_sizes=(256, 256))
    mbpo_act = policy.actions_model.predict(obs)
    print(mbpo_act)
    mbpo_act = policy.actions_model.predict(obs)
    print(mbpo_act)
    mbpo_act = policy.actions_model.predict(obs)
    print(mbpo_act)
    ipk_basic_act = np.array([[1, 4, 1, 0]])
    ipk_act = policy.kal_actions_model.predict([*obs, ipk_basic_act])
    print(ipk_act)
