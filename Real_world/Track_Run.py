# -*-coding:utf-8-*-
import sys
from PyQt5 import QtWidgets
from Real_world.UI_Track import Ui_Track
# import cv2
import numpy as np
# from arduino.arduino import Arduino
# import ray
# import ray.rllib.agents.dqn as dqn
# from ray.rllib.agents.dqn import DQNTrainer
# from ray.tune.logger import pretty_print
import argparse
from Prototype.utils import *

# import json

# import gym
# from gym_tendontrack.envs.tendontrack_env import TendonTrackEnv


class Track_Run(QtWidgets.QMainWindow, Ui_Track):
    def __init__(self):
        super(Track_Run, self).__init__()
        self.setupUi(self)

        self.pin11 = 3
        self.pin12 = 4
        self.pin21 = 5
        self.pin22 = 6
        self.pin31 = 7
        self.pin32 = 8
        self.pin41 = 9
        self.pin42 = 10
        # self.cap = cv2.VideoCapture(1)
        # set blue thresh 设置HSV中蓝色、天蓝色范围
        self.lower_blue = np.array([120, 80, 80])
        self.upper_blue = np.array([124, 255, 255])
        self.num = 4
        # uno.turnOff()
        self.move_init()

    def move_init(self):
        self.Up.pressed.connect(self.Up_Pressed)
        self.Up.released.connect(self.Up_Released)
        self.Down.pressed.connect(self.Down_Pressed)
        self.Down.released.connect(self.Down_Released)
        self.Right.pressed.connect(self.Right_Pressed)
        self.Right.released.connect(self.Right_Released)
        self.Left.pressed.connect(self.Left_Pressed)
        self.Left.released.connect(self.Left_Released)

        self.Up_2.pressed.connect(self.Up_2_Pressed)
        self.Up_2.released.connect(self.Up_2_Released)
        self.Down_2.pressed.connect(self.Down_2_Pressed)
        self.Down_2.released.connect(self.Down_2_Released)
        self.Right_2.pressed.connect(self.Right_2_Pressed)
        self.Right_2.released.connect(self.Right_2_Released)
        self.Left_2.pressed.connect(self.Left_2_Pressed)
        self.Left_2.released.connect(self.Left_2_Released)

    def Up_Pressed(self):
        print('1')
        send_real_action(1, 0, 0, 0)

    def Up_Released(self):
        send_real_action(0, 0, 0, 0)

    def Down_Pressed(self):
        send_real_action(2, 0, 0, 0)

    def Down_Released(self):
        send_real_action(0, 0, 0, 0)

    # def Up_Pressed(self):
    #     uno.setLow(self.pin12)
    #     uno.setHigh(self.pin11)
    #     print("Aaaaaa!")
    #
    # def Up_Released(self):
    #     uno.setLow(self.pin11)
    #
    # def Down_Pressed(self):
    #     uno.setLow(self.pin11)
    #     uno.setHigh(self.pin12)
    #
    # def Down_Released(self):
    #     uno.setLow(self.pin12)
    #
    def Right_Pressed(self):
        send_real_action(0, 1, 0, 0)

    def Right_Released(self):
        send_real_action(0, 0, 0, 0)

    def Left_Pressed(self):
        send_real_action(0, 2, 0, 0)

    def Left_Released(self):
        send_real_action(0, 0, 0, 0)
    def Up_2_Pressed(self):
        print('1')
        send_real_action(0, 0, 1, 0)

    def Up_2_Released(self):
        send_real_action(0, 0, 0, 0)

    def Down_2_Pressed(self):
        send_real_action(0, 0, 2, 0)

    def Down_2_Released(self):
        send_real_action(0, 0, 0, 0)


    def Right_2_Pressed(self):
        send_real_action(0, 0, 0, 1)

    def Right_2_Released(self):
        send_real_action(0, 0, 0, 0)

    def Left_2_Pressed(self):
        send_real_action(0, 0, 0, 2)

    def Left_2_Released(self):
        send_real_action(0, 0, 0, 0)
    #
    # def Up_2_Pressed(self):
    #     uno.setLow(self.pin31)
    #     uno.setHigh(self.pin32)
    #
    # def Up_2_Released(self):
    #     uno.setLow(self.pin32)
    #
    # def Down_2_Pressed(self):
    #     uno.setLow(self.pin32)
    #     uno.setHigh(self.pin31)
    #
    # def Down_2_Released(self):
    #     uno.setLow(self.pin31)
    #
    # def Right_2_Pressed(self):
    #     uno.setLow(self.pin41)
    #     uno.setHigh(self.pin42)
    #
    # def Right_2_Released(self):
    #     uno.setLow(self.pin42)
    #
    # def Left_2_Pressed(self):
    #     uno.setLow(self.pin42)
    #     uno.setHigh(self.pin41)
    #
    # def Left_2_Released(self):
    #     uno.setLow(self.pin41)

    def cap_show(self):
        while True:
            # get a frame and show 获取视频帧并转成HSV格式, 利用cvtColor()将BGR格式转成HSV格式，参数为cv2.COLOR_BGR2HSV。
            ret, frame = self.cap.read()
            cv2.imshow('Capture', frame)
            # change to hsv model
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # get mask 利用inRange()函数和HSV模型中蓝色范围的上下界获取mask，mask中原视频中的蓝色部分会被弄成白色，其他部分黑色。
            mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            cv2.imshow('Mask', mask)

            # detect blue 将mask于原视频帧进行按位与操作，则会把mask中的白色用真实的图像替换：
            res = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow('Result', res)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap_exit()
                break

    # def get_target(self):
    #
    # def manual_move(self):
    #     if
    #

    # def auto_move(self, args):
    #     ray.init()
    #
    #     # env = gym.make('tendontrack-v0')
    #
    #     def config_env(args):
    #         config = json.load(open(args.config))
    #         return config
    #
    #     def config_agent(env_config):
    #         config = dqn.DEFAULT_CONFIG.copy()
    #         config["num_gpus"] = 0
    #         config["num_workers"] = 1
    #         config["env"] = TendonTrackEnv
    #         config["env_config"] = env_config
    #         return config
    #
    #     env_config = config_env(args)
    #     agent_config = config_agent(env_config)
    #     trainer = DQNTrainer(
    #         env=TendonTrackEnv,
    #         config=agent_config)
    #     for i in range(1000):
    #         # Perform one iteration of training the policy with DQN
    #         result = trainer.train()
    #         print(pretty_print(result))
    #
    #         if i % 100 == 0:
    #             checkpoint = trainer.save()
    #             print("checkpoint saved at", checkpoint)

    def cap_exit(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--scenario', type=str, default='PongNoFrameskip-v4')
    # parser.add_argument('--config', type=str, default='config/global_config.json', help='config file')
    parser.add_argument('--algo', type=str, default='DQN', choices=['DQN', 'DDQN', 'DuelDQN'],
                        help='choose an algorithm')
    parser.add_argument('--inference', action="store_true", help='inference or training')
    parser.add_argument('--ckpt', type=str, help='inference or training')
    parser.add_argument('--epoch', type=int, default=10, help='number of training epochs')
    parser.add_argument('--num_step', type=int, default=10 ** 3,
                        help='number of timesteps for one episode, and for inference')
    parser.add_argument('--save_freq', type=int, default=100, help='model saving frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='model saving frequency')
    parser.add_argument('--time_span', type=int, default=30, help='time interval to collect data')

    args = parser.parse_args()

    # uno = Arduino('COM4')
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyle(QStyleFactory.create('GTK+'))
    myshow = Track_Run()
    # myshow.auto_move(args)
    myshow.show()
    # myshow.cap_show()
    sys.exit(app.exec_())
