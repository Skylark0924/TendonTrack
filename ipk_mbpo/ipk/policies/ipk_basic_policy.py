import numpy as np
import pdb
import tensorflow as tf

random_single = True

'''
Basic Controller
'''
class IPKBasicPolicy():
    def __init__(self):
        self.factor = 2

    def action_np(self, condition):  # array([[x, y, height]]) or [array([[x, y, height]])]
        obs = condition[0][:, :3] if isinstance(condition, list) else condition[:,:3]
        assert obs.shape[1] == 3
        try:
            batch_size = np.array(obs).shape[0]
        except:
            pdb.set_trace()
        # print(' [ batch_size ]: {}'.format(batch_size))
        # if obs.dtype == tf.float32:
        #     # batch_action = tf.zeros((batch_size, 4))
        #     const0 = tf.constant(0)
        #     for i in range(batch_size):
        #         single_action = tf.zeros(4)
        #         x = obs[i][0]
        #         y = obs[i][1]
        #
        #         abs_x = tf.abs(x)
        #         abs_y = tf.abs(y)
        #         abs_x_y = tf.add(abs_x, abs_y)
        #
        #         if tf.greater(x, const0):
        #             if np.random.rand(1) < 0.5:
        #                 single_action[0] = abs_x / abs_x_y if tf.not_equal(abs_x_y, const0) else const0  # +
        #             else:
        #                 single_action[2] = abs_x / abs_x_y if tf.not_equal(abs_x_y, const0) else const0  # -
        #         elif tf.equal(x, const0):
        #             single_action[0] = const0
        #             single_action[2] = const0
        #         else:
        #             if np.random.rand(1) < 0.5:
        #                 single_action[0] = -abs_x / abs_x_y if tf.not_equal(abs_x_y, const0) else const0  # -
        #             else:
        #                 single_action[2] = -abs_x / abs_x_y if tf.not_equal(abs_x_y, const0) else const0  # +
        #
        #         if tf.greater(y, const0):
        #             if np.random.rand(1) < 0.5:
        #                 single_action[1] = -abs_y / abs_x_y if tf.not_equal(abs_x_y, const0) else const0  # +
        #             else:
        #                 single_action[3] = -abs_y / abs_x_y if tf.not_equal(abs_x_y, const0) else const0  # -
        #         elif tf.equal(y, const0):
        #             single_action[1] = const0
        #             single_action[3] = const0
        #         else:
        #             if np.random.rand(1) < 0.5:
        #                 single_action[1] = abs_y / abs_x_y if tf.not_equal(abs_x_y, const0) else const0  # -
        #             else:
        #                 single_action[3] = abs_y / abs_x_y if tf.not_equal(abs_x_y, const0) else const0 # +
        #         # batch_action[i] = single_action
        #         pdb.set_trace()
        #     return self.factor * batch_action.reshape((batch_size, 4))

        batch_action = np.zeros((batch_size, 4))
        if random_single:
            for i in range(batch_size):
                single_action = np.zeros(4)
                x = obs[i][0]
                y = obs[i][1]

                abs_x = abs(x) if x is not None else 1
                abs_y = abs(y) if y is not None else 1
                abs_x_y = abs_x + abs_y

                if x > 0:
                    if np.random.rand(1) < 0.5:
                        single_action[0] = abs_x / abs_x_y if abs_x_y != 0 else 0  # +
                    else:
                        single_action[2] = abs_x / abs_x_y if abs_x_y != 0 else 0  # -
                elif x == 0:
                    single_action[0] = 0
                    single_action[2] = 0
                else:
                    if np.random.rand(1) < 0.5:
                        single_action[0] = -abs_x / abs_x_y if abs_x_y != 0 else 0  # -
                    else:
                        single_action[2] = -abs_x / abs_x_y if abs_x_y != 0 else 0  # +

                if y > 0:
                    if np.random.rand(1) < 0.5:
                        single_action[1] = -abs_y / abs_x_y if abs_x_y != 0 else 0  # +
                    else:
                        single_action[3] = -abs_y / abs_x_y if abs_x_y != 0 else 0  # -
                elif y == 0:
                    single_action[1] = 0
                    single_action[3] = 0
                else:
                    if np.random.rand(1) < 0.5:
                        single_action[1] = abs_y / abs_x_y if abs_x_y != 0 else 0  # -
                    else:
                        single_action[3] = abs_y / abs_x_y if abs_x_y != 0 else 0  # +
                batch_action[i] = single_action
        else:
            for i in range(batch_size):
                single_action = np.zeros(4)
                x = obs[i][0]
                y = obs[i][1]

                abs_x = abs(x)
                abs_y = abs(y)
                abs_x_y = abs_x + abs_y

                if x > 0:
                    if np.random.rand(1) < 1:  # 更可能出现S型以保持高度 TODO: tune
                        single_action[0] = abs_x / abs_x_y if abs_x_y != 0 else 0  # +
                        single_action[2] = -0.5 * single_action[0]
                    else:
                        single_action[2] = abs_x / abs_x_y if abs_x_y != 0 else 0  # -
                elif x == 0:
                    single_action[0] = 0
                    single_action[2] = 0
                else:
                    if np.random.rand(1) < 1:
                        single_action[0] = -abs_x / abs_x_y if abs_x_y != 0 else 0  # -
                        single_action[2] = -0.5 * single_action[0]
                    else:
                        single_action[2] = -abs_x / abs_x_y if abs_x_y != 0 else 0  # +

                if y > 0:
                    if np.random.rand(1) < 1:
                        single_action[1] = -abs_y / abs_x_y if abs_x_y != 0 else 0  # +
                        single_action[3] = -0.5 * single_action[1]
                    else:
                        single_action[3] = -abs_y / abs_x_y if abs_x_y != 0 else 0  # -
                elif y == 0:
                    single_action[1] = 0
                    single_action[3] = 0
                else:
                    if np.random.rand(1) < 1:
                        single_action[1] = abs_y / abs_x_y if abs_x_y != 0 else 0  # -
                        single_action[3] = -0.5 * single_action[1]
                    else:
                        single_action[3] = abs_y / abs_x_y if abs_x_y != 0 else 0  # +
                batch_action[i] = single_action

                # batch_action = np.squeeze(batch_action)
                # print('batch_action : {}'.format(batch_action))
        return self.factor * batch_action.reshape((batch_size, 4))
