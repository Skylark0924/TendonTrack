import numpy as np
from scipy.linalg import logm


def multivar_continue_KL_divergence(p, q):
    a = np.log(np.linalg.det(q[1])/np.linalg.det(p[1]))
    b = np.trace(np.dot(np.linalg.inv(q[1]), p[1]))
    c = np.dot(np.dot(np.transpose(q[0] - p[0]), np.linalg.inv(q[1])), (q[0] - p[0]))
    n = p[1].shape[0]
    return 0.5 * (a - n + b + c)


p = (mu1, sigma1) = np.transpose(np.array([0.2, 0.1, 0.5, 0.4])), np.diag([0.14, 0.52, 0.2, 0.4])
q = (mu2, sigma2) = np.transpose(np.array([0.3, 0.6, -0.5, -0.8])), np.diag([0.24, 0.02, 0.31, 0.51])
print(multivar_continue_KL_divergence(p, q))  # 16.8094379124341
print(multivar_continue_KL_divergence(q, p))  # 402.3905620875658
print(multivar_continue_KL_divergence(p, p))  # 0.0
