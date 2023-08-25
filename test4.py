import numpy as np


def univar_continue_KL_divergence(p, q):
    return np.log(q[1] / p[1]) + (p[1] ** 2 + (p[0] - q[0]) ** 2) / (2 * q[1] ** 2) - 0.5


p = (mu1, sigma1) = 0.1, 0.1
q = (mu2, sigma2) = 0, 0.1
q2 = (mu3, sigma3) = 3, 0.1
print(univar_continue_KL_divergence(q,p))  # 16.8094379124341
print(univar_continue_KL_divergence(p, q2))  # 402.3905620875658
print(univar_continue_KL_divergence(q, q2))  # 0.0
