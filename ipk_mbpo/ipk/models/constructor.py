import numpy as np
import tensorflow as tf

from mbpo.models.fc import FC
from mbpo.models.bnn import BNN


def construct_model(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):
    print('[ BNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))
    params = {'name': 'BNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
    model = BNN(params)

    model.add(FC(hidden_dim, input_dim=obs_dim + act_dim, activation="swish", weight_decay=0.000025))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
    model.add(FC(obs_dim + rew_dim, weight_decay=0.0001))
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
    return model


def format_samples_for_training(samples):
    obs = samples['observations']
    # 使用ipk+mbpo的数据来train model
    # 后期需加上混合采样 zeta_real的概率采real, zeta_mbpo的概率采mbpo
    # act = samples['actions']
    # next_obs = samples['next_observations']
    # rew = samples['rewards']
    act = samples['ipk_actions']
    next_obs = samples['real_next_observations']
    rew = samples['real_rewards']
    mbpo_act = samples['actions']
    mbpo_next_obs = samples['next_observations']
    mbpo_rew = samples['rewards']
    delta_obs = next_obs - obs
    mbpo_delta_obs = mbpo_next_obs - obs
    inputs = np.concatenate((obs, act), axis=-1)
    outputs = np.concatenate((rew, delta_obs), axis=-1)
    mbpo_inputs = np.concatenate((obs, mbpo_act), axis=-1)
    mbpo_outputs = np.concatenate((mbpo_rew, mbpo_delta_obs), axis=-1)
    inputs = np.concatenate((inputs, mbpo_inputs), axis=0)
    outputs = np.concatenate((outputs, mbpo_outputs), axis=0)
    return inputs, outputs


def reset_model(model):
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.name)
    model.sess.run(tf.initialize_vars(model_vars))


if __name__ == '__main__':
    model = construct_model()
