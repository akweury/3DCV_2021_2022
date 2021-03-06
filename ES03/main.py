import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io


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
Fu, Fs, Fvh = np.linalg.svd(F)
if Fs[2] > 0.0001:
    print('F is not rank 2, enforce F to rank 2.')
    Fs[2] = 0
    F = Fu @ np.diag(Fs) @ Fvh
print(F)
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
cornersCam1_homo = np.c_[cornersCam1, np.ones(cornersCam1.shape[0])]
cornersCam0_homo = cornersCam1_homo.T

matches_0 = []
for c0 in cornersCam0:
    l_1 = F @ np.array([[c0[0]], [c0[1]], [1]])
    distance = 100 * np.ones([len(cornersCam1_homo)])
    for index, c1 in enumerate(cornersCam1_homo):
        distance[index] = \
            abs(l_1[0] * c1[0] + l_1[1] * c1[1] + l_1[2]) / np.sqrt(
                np.square(l_1[0]) + np.square(l_1[1]))

    match = cornersCam1[np.argmin(distance)]
    matches_0.append(match)

# e. connect corresponding points between images with lines

# combine two images
img0 = cv.imread(path + 'Camera00.jpg')
img1 = cv.imread(path + 'Camera01.jpg')
img_connected = np.concatenate((img0, img1), axis=0)

# draw lines
y_displacement = img0.shape[0]
for index in range(cornersCam0.shape[0]):
    start_point = (int(cornersCam0[index][0]), int(cornersCam0[index][1]))
    end_point = (int(matches_0[index][0]), int(matches_0[index][1] + y_displacement))
    color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
    img_connected = cv.line(img_connected, start_point, end_point, color, thickness=5)

cv.imwrite('./' + 'matches.jpg', img_connected)

####################### 2. Structure Reconstruction ################################

# triangulate the 3D positions in the world c.s.
P_0 = K_0 @ np.c_[R_0, t_0.T]
P_1 = K_1 @ np.c_[R_1, t_1.T]

X_array = np.zeros((1, 3))
for index in range(cornersCam0.shape[0]):
    p_0_1, p_0_2, p_0_3 = P_0[0, :], P_0[1, :], P_0[2, :]
    p_1_1, p_1_2, p_1_3 = P_1[0, :], P_1[1, :], P_1[2, :]
    x0, y0 = cornersCam0[index]
    x1, y1 = matches_0[index]
    A = np.asarray([p_0_3 * x0 - p_0_1,
                    p_0_3 * y0 - p_0_2,
                    p_1_3 * x1 - p_1_1,
                    p_1_3 * y1 - p_1_2])
    """   A = np.asarray([
           cornersCam0[index][0] * P_0[2, :] - P_0[0, :],
           cornersCam0[index][1] * P_0[2, :] - P_0[1, :],
           matches_0[index][0] * P_1[2, :] - P_0[0, :],
           matches_0[index][1] * P_1[2, :] - P_0[1, :]
       ])
    """
    u, s, vh = np.linalg.svd(A)
    X_homo = vh.T[:, -1]  # last column of V is the minimizer of the problem
    X = X_homo / X_homo[-1]
    X_array = np.vstack([X_array, X[:-1]])

X_array = X_array[1:, :]

# Plot reconstructed chessboard points in 3D space
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_array[:, 0], X_array[:, 1], X_array[:, 2])

# visualize the optical center and optical axis

origin_0 = np.array([t_0[0][0], t_0[0][1], t_0[0][2]])

origin_1 = np.dot(np.linalg.inv(R_1), -t_1.reshape(3))
ax.scatter(origin_0[0], origin_0[1], origin_0[2], color='red')
ax.scatter(origin_1[0], origin_1[1], origin_1[2], color='green')

z_0 = R_0[2, :] * 100
z_1 = np.linalg.inv(R_1) @ np.array([[0], [0], [10]])
z_1 = z_1.reshape(3)
ax.plot([origin_0[0], z_0[0]], [origin_0[1], z_0[1]], zs=[origin_0[2], z_0[2]], color='red')
ax.plot([origin_1[0], z_1[0]], [origin_1[1], z_1[1]], zs=[origin_1[2], z_1[2]], color='green')

plt.show()
