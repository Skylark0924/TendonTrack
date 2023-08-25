TYPE = 'MBPO'
params = {
    'type': TYPE,
    'universe': 'gym',
    'domain': 'Tendon_Tracking',
    'task': 'v1',

    'log_dir': '~/ray_{}/'.format(TYPE),
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 1000,  # timesteps of one epoch in gaussian policy part
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,  # 250
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,  # 100e3
        'deterministic': False,
        'num_networks': 7,
        'num_elites': 5,
        'real_ratio': 0.05,
        'target_entropy': -3,
        'max_model_t': None,
        'rollout_schedule': [20, 100, 1, 1],
    }
}
