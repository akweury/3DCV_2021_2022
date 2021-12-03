import numpy as np
import scipy.io as io

d_path = './data/data.mat'

d = io.loadmat(d_path)
K_0, K_1 = d['K_0'].item(), d['K_1'].item()
R_1, t_1 = d['R_1'].item(), d['t_1'].item()
cornersCam0, cornersCam1 = d['cornersCam0'].item(), d['cornersCam1'].item()
