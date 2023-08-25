from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import os
import io

if __name__ == '__main__':
    # model = load_model_from_path("./tendon_test_usebody.xml")
    model = load_model_from_path(
        "/home/sjy/pycharm_remote/TendonTrack/ipk_mbpo/softlearning/environments/gym/mujoco/mujoco_model/tendon.xml")  # "./tendon.xml"
    sim = MjSim(model)
    RECORD_flg = False
    if not RECORD_flg:
        viewer = MjViewer(sim)
    t = 0
    flg = 1
    cnt = 500
    # d_action = 0.1
    d_step = 1e-3
    filename = 'MonteCarlo.txt'

    for i in range(cnt):
        sim.data.qpos[:] = 0.0
        sim.data.ctrl[:] = 0.0
        sim.forward()
        # action = 10 * (np.random.rand(4) - 0.5)
        action = np.array([2.372112706, -2.416926599, -3.001663798, -2.364358833])
        # action = np.array([1.618270403, -2.501089085, -2.075275735, -2.60163815])
        # action = np.array([-2.667242369, -0.332449972, 4.171190507, -4.701133344])
        print('action: {} {}'.format(action, action.shape))
        # action_prev = sim.data.ctrl[0]
        # t += 1
        steps = int(max(abs(action)) // d_step)
        d_action = action / steps
        if RECORD_flg:
            with io.open(filename, 'a') as f:
                for stp in range(steps):
                    for j in range(4):
                        sim.data.ctrl[2 * j] += d_action[j]
                        sim.data.ctrl[2 * j + 1] = -sim.data.ctrl[2 * j]
                    sim.step()
                    if stp % 100 == 0:
                        pos = sim.data.get_camera_xpos("front_camera")
                        if stp % 1000 == 0:
                            print('{},{}: pos={},{}'.format(i, stp, pos, type(pos)))
                        ctrl = np.zeros(4)
                        for j in range(4):
                            ctrl[j] = sim.data.ctrl[2 * j]
                        meta_txt = ' '.join(str(x) for x in pos) + ' ' \
                                   + ' '.join(str(x) for x in ctrl) + '\r\n'
                        f.write(meta_txt)
        else:
            pos = sim.data.get_camera_xpos("front_camera")
            print('pos={}'.format(pos))
            for stp in range(steps):
                for j in range(4):
                    sim.data.ctrl[2 * j] += d_action[j]
                    sim.data.ctrl[2 * j + 1] = -sim.data.ctrl[2 * j]
                sim.step()
                if stp % 10 == 0:
                    pos = sim.data.get_camera_xpos("front_camera")
                    viewer.render()
                    viewer.add_marker(pos=pos,
                                      label=str(stp))
            pos = sim.data.get_camera_xpos("front_camera")
            print('pos={}'.format(pos))
        # print(sim.data.qpos[41])
        if os.getenv('TESTING') is not None:
            break
