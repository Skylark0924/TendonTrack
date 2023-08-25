from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import os

if __name__ == '__main__':
    # model = load_model_from_path("./tendon_test_usebody.xml")
    model = load_model_from_path("/home/sjy/pycharm_remote/TendonTrack/ipk_mbpo/softlearning/environments/gym/mujoco/mujoco_model/tendon.xml")#"./tendon.xml"
    sim = MjSim(model)
    viewer = MjViewer(sim)
    t = 0
    flg = 1
    while True:
        #if t <= 500:
        #    sim.data.ctrl[0] = .5 * t / 50.
        #    # sim.data.ctrl[0] = -0.1
        #elif t <= 1500:
        #    sim.data.ctrl[0] = -.5 * (t - 1000) / 50.
        if t <= 100:
            action = .5 * t / 10.
            # sim.data.ctrl[0] = -0.1
        elif t <= 300:
            action = -.5 * (t - 200) / 10.
        #elif t <= 400:
        #    action = .5 * (t - 400) / 10.
        #else:
        #    t = 0
        #    continue

        action_prev = sim.data.ctrl[0]
        t += 1
        for i in range(50):
            sim.data.ctrl[0] += (action-action_prev)/50
            sim.data.ctrl[1] = -sim.data.ctrl[0]
            sim.step()
        viewer.render()
        if os.getenv('TESTING') is not None:
            break
