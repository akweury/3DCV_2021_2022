import numpy as np
import scipy.io as io
import cv2 as cv


def skew(a):
    return np.array([
        [0, -a[0][2], a[0][1]],
        [a[0][2], 0, -a[0][0]],
        [-a[0][1], a[0][0], 0]
    ])


d_path = './data/data.mat'
d = io.loadmat(d_path)
K_0, K_1 = d['K_0'], d['K_1']
R_1, t_1 = d['R_1'], d['t_1']
cornersCam0, cornersCam1 = d['cornersCam0'], d['cornersCam1']

R_0, t_0 = np.identity(3), np.zeros((1, 3))

####################### 1. Feature matching using the epipolar constraint ################################

# a. compute F

# F = K'^-T [t]_x R K^-1
K_1_inv_t = np.linalg.inv(K_1.T)
t_skew_symmetric = skew(t_1)
F = K_1_inv_t @ t_skew_symmetric @ R_1 @ np.linalg.inv(K_0)

# b. compute the epipolar line in image camera01.jpg

# l' = Fx
# l = F^T x'
cornersCam0_homo = np.c_[cornersCam0, np.ones(cornersCam0.shape[0])]
l_1 = cornersCam0_homo @ F

# c. draw each epipolar line

path = './data/'
img = cv.imread(path + 'Camera01.jpg')

for l in l_1:
    start_point = (0, int((-l[0] * 0 - l[2]) / l[1]))
    end_point = (img.shape[1], int((-l[0] * img.shape[1] - l[2]) / l[1]))
    color = (0, 255, 0)
    img = cv.line(img, start_point, end_point, color, thickness=5)

# save image as epilines.jpg
cv.imwrite('./' + 'epilines.jpg', img)

# d. minimal algebraic distance to l_0
cornersCam1_homo = np.c_[cornersCam1, np.ones(cornersCam0.shape[0])]
l_0 = cornersCam1_homo @ F.T
alg_dist = abs(l_0[:, 0] * cornersCam1[:, 0] + l_0[:, 1] * cornersCam1[:, 1] + l_0[:, 2]) / np.sqrt(
    np.square(l_0[:, 0]) + np.square(l_0[:, 1]))

# e. connect corresponding points between images with lines

# combine two images
img0 = cv.imread(path + 'Camera00.jpg')
img1 = cv.imread(path + 'Camera01.jpg')
img_connected = np.concatenate((img0, img1), axis=0)

# draw lines
y_displacement = img0.shape[0]
for index in range(cornersCam0.shape[0]):
    start_point = (int(cornersCam0[index][0]), int(cornersCam0[index][1]))
    end_point = (int(cornersCam1[index][0]), int(cornersCam1[index][1]+y_displacement))
    color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
    img_connected = cv.line(img_connected, start_point, end_point, color, thickness=5)

cv.imwrite('./' + 'matches.jpg', img_connected)
