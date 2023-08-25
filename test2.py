from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm


def BSpline_Calcu(n=20, rho_range=3):
    """
    n: number of points
    rho_range: range of the rho rand
    """
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

    return xy_t


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=45, azim=45)  # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴

    X = np.arange(-2, 2, 0.001)
    Y = np.arange(-2, 2, 0.001)
    X, Y = np.meshgrid(X, Y)
    Z1 = -np.sqrt(9 - X ** 2 - Y ** 2) + 2
    ax.plot_surface(X, Y, Z1, cmap=cm.YlOrBr, zorder=0, alpha=0.8)

    Z2 = -np.sqrt(9 - X ** 2 - Y ** 2) + 3
    ax.plot_surface(X, Y, Z2, cmap=cm.Blues, zorder=2, alpha=0.8)

    xy = BSpline_Calcu()
    x, y = xy[:, 0], xy[:, 1]
    Z3 = -np.sqrt(4 - x ** 2 - y ** 2) + 2
    ax.plot(x, y, Z3, color='black', marker='p', zorder=3, linewidth=2)

    import time

    # line.set_zorder(3)
    plt.savefig('/home/lab/Github/TendonTrack/Simulator/Image/{}.eps'.format(time.time()))
    plt.show()
