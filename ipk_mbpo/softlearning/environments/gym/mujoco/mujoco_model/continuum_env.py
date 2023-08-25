from __future__ import division
from mujoco_py import load_model_from_path, MjSim, MjViewer
import random
import math
import time
import os
import numpy as np
# from scipy.misc import imsave
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import copy
import shutil
import gym
import cv2
import random
from softlearning.environments.gym.mujoco.mujoco_model.img_segementation import detect_contor as img_seg

initial_pos = np.zeros(50).tolist()
'''
joint_pos = [[29.70 / 180 * math.pi, -85 / 180 * math.pi, 115 / 180 * math.pi],  # yellow
             [31.00 / 180 * math.pi, -78 / 180 * math.pi, 105 / 180 * math.pi],
             [30.55 / 180 * math.pi, -70 / 180 * math.pi, 99 / 180 * math.pi],
             [20 / 180 * math.pi, -45 / 180 * math.pi, 60 / 180 * math.pi]]
             '''
# closed_pos = [1.12810781, -0.59798289, -0.53003607]
fig_size_1 = (214, 214)  # For the workbench camera
fig_size_2 = (214, 214)  # For the upper camera
gaussian_noise_parameters = (20, 20)
gaussian_blur_prarmeters = ((5, 5), 1.5)
safety_threshold = 0.01  # Used to determine the stablity of current joint positions
grasp_steps = 120  # The minimum steps in simulator for the gripper to close is 120
drop_steps = 60  # It takes roughly 12 steps to reach ground from a 0.3m high position, extra iterations are for the convergance of final postion
sensor_threshold = 2.0  # Used to judge whether a cube is firmly grasped
action_scale = 30

IMAGE_LOG_DIR = '/home/lab/Github/TendonTrack/Simulator/utils/image_log_IPK/'


# os.mkdir(IMAGE_LOG_DIR)
# shutil.rmtree(IMAGE_LOG_DIR)
# os.mkdir(IMAGE_LOG_DIR)


class vector():
    # 3D vector class
    def __init__(self, x=0, y=0, z=1):
        self.x, self.y, self.z = x, y, z

    def add(self, v):
        return vector(self.x + v.x, self.y + v.y, self.z + v.z)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def mul_vec(self, v):
        return vector(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)

    def mul_num(self, s):
        return vector(self.x * s, self.y * s, self.z * s)


class quaternion():
    # Quaternion class used for 3D rotation
    def __init__(self, w=0, v=vector(0, 0, 1)):
        self.v = v
        self.w = w

    def rev(self):
        N = math.sqrt(self.w * self.w + self.v.x * self.v.x + self.v.y * self.v.y + self.v.z * self.v.z)
        return quaternion(self.w / N, vector(-self.v.x / N, -self.v.y / N, -self.v.z / N))

    def mul(self, q):
        res_v = self.v.mul_vec(q.v).add(q.v.mul_num(self.w)).add(self.v.mul_num(q.w))
        res_w = self.w * q.w - self.v.dot(q.v)
        return quaternion(res_w, res_v)


class lab_env():
    def __init__(self, env, args):
        # super(lab_env, self).__init__(env)
        # The real-world simulator
        self.model = load_model_from_path \
                (
                "/home/sjy/pycharm_remote/TendonTrack/Simulator/env/mujoco_model/tendon.xml")
        self.sim = MjSim(self.model)
        self.image_num = 0
        # self.obj_mov_flg = False
        self.d_mov = 0.1  # 0.05连续运动 0.1
        self.d_mov_x = 1
        self.d_mov_y = 0
        self.REAL_HEIGHT_flg = True
        self.RANDOM_flg = True  # Random Spline / Routine
        self.xy_t = np.array([0.0, 0.0])
        self.z_t = np.array([0.0])
        self.cnt = 0
        self.prev_z = 0

        self.XY_coeff = 0.1
        # self.Rotate_Matrix = np.array([[np.cos(d_mov), -np.sin(d_mov)],
        #                               [np.sin(d_mov), np.cos(d_mov)]])

    def reset(self, task_id):
        self.task = task_id
        self.grasping = -1
        self.last_grasp = -1
        # Configure gravity

        # for i in range(8):
        self.sim.data.ctrl[:] = 0.0
        # Configure joint positions
        # for i in range(50):
        self.sim.data.qpos[:] = 0.0
        # for i in range(8):
        #     self.sim.data.ctrl[i] = 0.0

        if self.RANDOM_flg:
            # RANDOM Spline!!!!
            # self.sim.data.qpos[0] = 1 * (random.random() - 0.5)
            # self.sim.data.qpos[1] = 1 * (random.random() - 0.5)
            self.xy_t, self.z_t = BSpline_Calcu()
            self.sim.data.qpos[0] = self.xy_t[0][0]
            self.sim.data.qpos[1] = self.xy_t[0][1]
            self.sim.data.qpos[2] = self.z_t[0]
            self.prev_z = self.z_t[0]
            # self.sim.data.ctrl[0] = self.sim.data.qpos[2] - prev_z
            self.cnt = 1
        else:
            # ROUTINE
            dx = random.random() - 0.5
            dy = random.random() - 0.5
            self.d_mov_x = self.d_mov * dx / np.sqrt(dx ** 2 + dy ** 2)
            self.d_mov_y = self.d_mov * dy / np.sqrt(dx ** 2 + dy ** 2)
            self.sim.data.qpos[0] = 4 * self.d_mov_x
            self.sim.data.qpos[1] = 4 * self.d_mov_y

        # for i in range(3):
        #    self.sim.data.qpos[i] = joint_pos[task_id][i]
        # self.pos_forward()
        self.sim.forward()
        # self.obj_mov_flg = False

        # remapped_pos = [remap(self.sim.data.qpos[0], -30 / 180 * math.pi, 45 / 180 * math.pi, -1, 1),
        #                remap(self.sim.data.qpos[1], -105 / 180 * math.pi, -50 / 180 * math.pi, -1, 1),
        #                remap(self.sim.data.qpos[2], 0 / 180 * math.pi, 180 / 180 * math.pi, -1, 1), 0]

        return self.get_state()  # (remapped_pos,) + self.get_state()

    def update_obj(self):
        # x_prev = self.sim.data.qpos[0]
        # y_prev = self.sim.data.qpos[1]
        # if abs(x_prev) + abs(y_prev) < 1.2:
        #     self.sim.data.qpos[0] += 0.05
        # else:
        #     self.sim.data.qpos[0] = x_prev * np.cos(self.d_mov) - y_prev * np.sin(self.d_mov)
        #     self.sim.data.qpos[1] = x_prev * np.sin(self.d_mov) + y_prev * np.cos(self.d_mov)
        done = False
        if self.RANDOM_flg:
            # Random Spline!!!
            # d_x = random.random() - 0.5
            # d_y = random.random() - 0.5
            # self.sim.data.qpos[0] += self.d_mov * d_x / np.sqrt(d_x ** 2 + d_y ** 2)
            # self.sim.data.qpos[1] += self.d_mov * d_y / np.sqrt(d_x ** 2 + d_y ** 2)
            if self.cnt < self.xy_t.shape[0]:
                self.sim.data.qpos[0] = self.xy_t[self.cnt][0]
                self.sim.data.qpos[1] = self.xy_t[self.cnt][1]
                self.sim.data.qpos[2] = self.z_t[self.cnt]
                self.prev_z = self.z_t[self.cnt]
                # print('Height of the target: {}'.format(10 * self.prev_z))
                # self.sim.data.ctrl[0] = self.sim.data.qpos[2] - prev_z
                self.cnt += 1
            else:
                done = True
        else:
            # Routine!!
            x_prev = self.sim.data.qpos[0]
            y_prev = self.sim.data.qpos[1]
            if abs(x_prev) + abs(y_prev) < 0.6:
                self.sim.data.qpos[0] += self.d_mov_x
                self.sim.data.qpos[1] += self.d_mov_y
            else:
                self.sim.data.qpos[0] = x_prev * np.cos(self.d_mov) - y_prev * np.sin(self.d_mov)
                self.sim.data.qpos[1] = x_prev * np.sin(self.d_mov) + y_prev * np.cos(self.d_mov)

        self.sim.forward()
        # self.obj_mov_flg = True
        return done, np.array(self.get_state())

    def step(self, action):
        factor_ctrl = 0.1  # 0.005 0.1
        d_ctrl = factor_ctrl * np.asarray(action) / 100
        for cnt in range(100):
            for i in range(4):
                self.sim.data.ctrl[2 * i] = constrain(self.sim.data.ctrl[2 * i] + d_ctrl[i], -5, 5)
                self.sim.data.ctrl[2 * i + 1] = - self.sim.data.ctrl[2 * i]
            self.sim.step()
        self.sim.data.qpos[2] = self.prev_z
        self.sim.forward()

        # print(self.sim.data.qpos[10])
        # self.pos_forward()
        # self.sim.forward()

        # print(' mujoco control:{}'.format(self.sim.data.ctrl))
        return np.array(self.get_state())

    def get_state(self):
        width = 960.0  # 960
        height = 720.0  # 720
        image_1 = copy.deepcopy(self.sim.render(width=width, height=height, camera_name='front_camera'))
        # print(width, height)
        cXY, img = img_seg(image_1)

        self.image_num += 1
        cv2.imwrite(IMAGE_LOG_DIR + "{}.jpg".format(self.image_num), image_1)
        if not cXY:
            return [None, None, None], img
        else:
            if self.REAL_HEIGHT_flg:
                # 前置摄像头与目标的距离应为三维距离
                dz = self.sim.data.get_camera_xpos("front_camera")[2] - self.prev_z
                # print('Dz1: {}, Dz2: {}'.format(self.sim.data.get_camera_xpos("front_camera")[2], self.prev_z))
                dx = self.xy_t[self.cnt - 1][0]
                # print('Dx: {}'.format(dx))
                dy = self.xy_t[self.cnt - 1][1]
                # print('Dy: {}'.format(dy))
                cXY[0][2] = 100 * (np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)- 2.6)
                # print('Distance from Z direction is {}'.format(cXY[0][2]))
            # 转为三维array
            cXYZ = np.array([cXY[0][0] - width // 2, cXY[0][1] - height // 2, cXY[0][2]])
            # 前两个维度X, Y乘以缩放因子； H保持不变
            cXYZ[:2] = cXYZ[:2] * self.XY_coeff
            return cXYZ, img

    def safety_check(self):
        # return 0 if safe, otherwise 1
        backup = [self.sim.data.qpos[i] for i in range(14)]
        self.sim.step()
        s = 0
        for i in range(6):
            s += abs(backup[i] - self.sim.data.qpos[i])
            self.sim.data.qpos[i] = backup[i]
        return s > safety_threshold

    def render(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.render()


def get_x_y(fig):
    gray_fig = fig.sum(axis=2)
    x, y = np.where(gray_fig > 0)
    x_mean = x.mean() if x.shape[0] > 0 else math.nan
    y_mean = y.mean() if y.shape[0] > 0 else math.nan
    return x_mean, y_mean


def BSpline_Calcu(n=20, rho_range=1.5):
    # Calculate the spline curve parameters
    tmp_rand_rho_theta = np.vstack([np.random.random([n]) * rho_range,
                                    np.linspace(0, 4 * np.pi, n, endpoint=False)])
    tmp_rand = np.zeros([2, n])
    for i in range(n):
        rho = tmp_rand_rho_theta[0][i]
        theta = tmp_rand_rho_theta[1][i]
        tmp_rand[0][i] = rho * np.cos(theta)
        tmp_rand[1][i] = rho * np.sin(theta)
    tmp_zero = np.array([[0., 0., 0.], [0., 0., 0.]])
    ctrl_points = np.hstack((tmp_zero, tmp_rand, tmp_zero))  # Control points
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

    x, y = xy_t[:, 0], xy_t[:, 1]
    #  TODO: Workspace
    z = -np.sqrt(4 - x ** 2 - y ** 2) + np.sqrt(4)

    return xy_t, z


def crop(fig, x, y, height, width):
    Height, Width, _ = fig.shape
    x = constrain(x, height // 2, Height - height // 2)
    y = constrain(y, width // 2, Width - width // 2)
    return fig[x - height // 2: x + height // 2, y - width // 2: y + width // 2, :]


def constrain(x, lower_bound, upper_bound):
    x = (upper_bound + lower_bound) / 2 if math.isnan(x) else x
    x = upper_bound if x > upper_bound else x
    x = lower_bound if x < lower_bound else x
    return x  # int(round(x))


def create_env(env_id, args):
    env = env_id  # gym.make(env_id)
    if env == 'Goal_LfD':
        env = lab_env(env, args)
    return env


if __name__ == '__main__':
    env = create_env('Goal_LfD', None)
    state, images = env.reset(0)
    print(state)
    fig1 = env.sim.render(width=960, height=720, camera_name='front_camera')
    cv2.imwrite("tmp.jpg", fig1)
    state, images = env.step([0.2, -0.2, 0.4, -1])
    print(env.sim.data.ctrl[0])
    # imshow(images[0])
    # imshow(images[1])
    # fig1 = env.sim.render(width=960, height=720, camera_name='front_camera')
    # fig2 = env.sim.render(width=960, height=720, camera_name='global_camera')

    # plt.imshow(fig1)
    # plt.imsave("1.jpg", fig1)

    t = 0

    video = cv2.VideoWriter("sample.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10.0, (1920, 720), True)
    while True:
        state, img = env.step([0, 0, 0, 0])
        # fig1 = env.sim.render(width=960, height=720, camera_name='front_camera')
        fig2 = env.sim.render(width=960, height=720, camera_name='global_camera')
        video.write(cv2.cvtColor(np.concatenate((img, fig2), axis=1), cv2.COLOR_BGR2RGB))

        if t % 100:
            stat, [next_state, _] = env.update_obj()
            print('state:{}, nextobs:{}'.format(stat, next_state))
            # 如果到达三阶样条终点
            if stat:
                state, images = env.reset(0)
        t += 1
    video.release()
