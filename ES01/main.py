import glob
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import scipy.io as io


def project_points(X, K, R, T, distortion_flag=False, distortion_params=None):
    """Project points from 3d world coordinates to 2d image coordinates.
    Your code should work with considering distortion and without
    considering distortion parameters.
    """

    camera_points = T + np.matmul(R,X.T)
    ones = np.ones((1,camera_points.shape[1]))
    camera_points_homo = np.concatenate((camera_points, ones))

    I_3 = np.identity(3)
    zero_3 = np.zeros((3,1))
    I_3_zero_3 = np.concatenate((I_3,zero_3), axis=1)

    pixel_points = np.matmul(np.matmul(K, I_3_zero_3), camera_points_homo)
    X_homo = np.c_[X, np.ones((X.shape[0], X.shape[1], 1))]
    t = -np.matmul(R, T)
    P = np.matmul(K, np.concatenate((R, t), axis=2))
    x_2d_homo = np.matmul(P, X_homo.transpose(0, 2, 1))
    x_2d = f(X / Z)

    if distortion_flag:
        pass

    return x_2d


def project_and_draw(imgs, X_3d, K, R, T, distortion_flag, distortion_parameters):
    """
    call "project_points" function to project 3D points to camera coordinates
    draw the projected points on the image and save your output image here
    # save your results in a separate folder named "results"
    # Your implementation goes here!
    """

    # create folder
    Path("results").mkdir(parents=True, exist_ok=True)

    # clear folder contents
    files = glob.glob('./results/*')
    for f in files:
        os.remove(f)

    # call project_points function to project 3D points to camera coordinates
    for i, img in enumerate(imgs):
        projected_points = project_points(X_3d[i], K, R[i], T[i], distortion_flag=False, distortion_params=None)

        # draw projected points on the image

        # save image
        image_name = str(i) + '.jpg'


if __name__ == '__main__':
    base_folder = './data/'

    # Consider distorition
    dist_flag = True

    # Load the data
    # There are 25 views/or images/ and 40 3D points per view
    data = io.loadmat('data/ex_1_data.mat')

    # 3D points in the world coordinate system
    X_3D = data['x_3d_w']  # shape=[25, 40, 3]

    # Translation vector: as the world origin is seen from the camera coordinates
    TVecs = data['translation_vecs']  # shape=[25, 3, 1]

    # Rotation matrices: project from world to camera coordinate frame
    RMats = data['rot_mats']  # shape=[25, 3, 3]

    # five distortion parameters
    dist_params = data['distortion_params']

    # K matrix of the cameras
    Kintr = data['k_mat']  # shape 3,3

    imgs_list = [cv.imread(base_folder + str(i).zfill(5) + '.jpg') for i in range(TVecs.shape[0])]
    imgs = np.asarray(imgs_list)

    project_and_draw(imgs, X_3D, Kintr, RMats, TVecs, dist_flag, dist_params)
