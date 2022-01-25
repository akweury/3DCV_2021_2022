import math
import os

import cv2 as cv
import numpy as np
from sklearn.feature_extraction import image

# Define settings for your experiment

settings = {}
settings['dataset'] = 'kitti'  # choose from ['kitti', 'carla']

settings["patch_size"] = 9  # 7
if settings['dataset'] == 'kitti':
    settings["data_path"] = './data/kitti'
else:
    settings["data_path"] = './data/carla'

settings["results_directory"] = './results/' + settings['dataset']
experiment = 2
# We should down size the images, to see results quickly
# settings["width"] = 256
# settings["height"] = 128

settings["width"] = 512
settings["height"] = 256

# Num of depth proposals
settings["num_depths"] = 100
settings["min_depth"] = 2.0  # in meters
settings["max_depth"] = 20000.0

settings["similarity"] = "SSD"
os.makedirs(settings["results_directory"], exist_ok=True)


def get_depth_proposals(min_depth, max_depth, num_depths):
    '''
    return list of depth proposals
    you can sample the range [min_depth, max_depth] uniformly at num_depths points.
    Tip: linearly sampling depth range doesnot lead to a linear step along the epipolar line.
    Instead, linearly sample the inverse-depth [1/min_depth, 1/max_depth] then take its inverse to get depth values.
    This is practically more meaningful as it leads to linear step in pixel space.
    '''
    depth_proposals_inv = np.arange(1 / min_depth, 1 / max_depth, (1 / max_depth - 1 / min_depth) / num_depths)
    depth_proposals = 1 / depth_proposals_inv
    return depth_proposals


def depth_to_file(depth_map, filename):
    """
    Saves depth maps to as images
    feel free to modify it, it you want to get fancy pics,
    """
    depth_ = 1 / (depth_map + 0.00001)
    depth_ = 255.0 * depth_ / (np.percentile(depth_.max(), 95))
    cv.imwrite(filename, depth_)


def copy_make_border(img, patch_width):
    """
    This function applies cv.copyMakeBorder to extend the image by patch_width/2
    in top, bottom, left and right part of the image
    Patches/windows centered at the border of the image need additional padding of size patch_width/2
    """
    offset = np.int(patch_width / 2.0)
    return cv.copyMakeBorder(img,
                             top=offset, bottom=offset,
                             left=offset, right=offset,
                             borderType=cv.BORDER_REFLECT)


def extract_pathches(img, patch_width):
    '''
    Input:
        image: size[h,w,3]
    Return:
        patches: size[h, w, patch_width, patch_width, c]
    '''
    if img.ndim == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1
    img_padded = copy_make_border(img, patch_width)
    patches = image.extract_patches_2d(img_padded, (patch_width, patch_width))
    patches = patches.reshape(h, w, patch_width, patch_width, c)
    patches = patches.reshape(h, w, patch_width * patch_width * c)

    return patches


def read_kitti_calib_file():
    filename = os.path.join(settings["data_path"], 'calib.txt')
    data = np.fromfile(filename, sep=' ').reshape(3, 4)[0:3, 0:3]
    return data


def read_carla_calib_file():
    fov = 90.0
    height = 600
    width = 800
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / \
                        (2.0 * math.tan(fov * math.pi / 360.0))
    print(k)
    return k


def load_imgs_and_k_mats():
    img_0 = cv.imread(os.path.join(settings['data_path'], 'images', '0.png'))
    img_h, img_w, c = img_0.shape
    # Load and Downsize the images, for faster computation
    height, width = settings['height'], settings['width']
    # width, height = settings['height'], settings['width']
    imgs = [cv.resize(cv.imread(os.path.join(settings["data_path"], 'images', str(ii) + '.png')),
                      (settings['width'], settings['height'])) for ii in range(5)]
    source_img = imgs[2]
    input_imgs = imgs
    if settings['dataset'] == 'kitti':
        k_matrix = read_kitti_calib_file()
    else:
        k_matrix = read_carla_calib_file()
    k_matrix[0, :] = k_matrix[0, :] * float(width) / float(img_w)
    k_matrix[1, :] = k_matrix[1, :] * float(height) / float(img_h)
    return input_imgs, source_img, k_matrix


def load_camera_pose():
    if settings['dataset'] == 'kitti':
        filename = os.path.join(settings["data_path"], 'cam_pose.txt')
        data = np.fromfile(filename, sep=' ').reshape(5, 3, 4)
        RMats = data[:, 0:3, 0:3]
        TVecs = data[:, :, 3]
        # We should make the middle view as our source view.
        mid = 2
        ref_R = RMats[mid]
        ref_T = TVecs[mid]
        rot_mat_list = []
        t_vec_list = []
        for ii in range(5):
            R, T = RMats[ii], TVecs[ii]
            R_ii = np.dot(np.linalg.inv(R), ref_R)
            T_ii = np.dot(np.linalg.inv(R), (ref_T - T)).reshape(3, 1)
            rot_mat_list.append(R_ii)
            t_vec_list.append(T_ii)
    else:
        # This is for carla dataset
        # do not change this function
        t_vec_list = [np.array([0, 0, (i - 2) * 0.54], dtype=np.float).reshape(3, 1) for i in range(5)]
        rot_mat_list = [np.eye(3) for i in range(5)]
    return rot_mat_list, t_vec_list


def ssd(feature_1, feature_2):
    '''
    Compute the sum of square difference between the input features
    '''
    ssd_scores = np.sum(np.square(feature_1 - feature_2), axis=2)
    return ssd_scores


def ssd_2(features):
    feature_avg = (features[0] + features[1] + features[2] + features[3]) / 4

    feature_avg = np.concatenate((feature_avg,feature_avg,feature_avg,feature_avg), axis=2)
    features_all = np.concatenate((features[0], features[1], features[2], features[3]), axis=2)
    ssd_scores = np.sum(np.square(features_all - feature_avg), axis=2)

    # ssd_scores = np.sum(np.square(features[0] - feature_avg), axis=2) + \
    #              np.sum(np.square(features[1] - feature_avg), axis=2) + \
    #              np.sum(np.square(features[2] - feature_avg), axis=2) + \
    #              np.sum(np.square(features[3] - feature_avg), axis=2)
    # ssd_scores = np.sum(np.sum(np.square(features - feature_avg), axis=3), axis=0)
    return ssd_scores


def camera_to_world(camera_points, K_inv, d):
    prod = np.stack([K_inv @ cp for cp in camera_points])
    world_points = (d / np.linalg.norm(prod, axis=1, keepdims=True)).reshape(4, 1) * prod

    return world_points


def world_to_camera(world_points, K, R, t):
    P_matrix = K @ np.c_[R, t]
    world_points_homo = np.c_[world_points, np.ones(4)]
    camera_points = np.stack([P_matrix @ wph for wph in world_points_homo])
    camera_points[:, 0] = camera_points[:, 0] / camera_points[:, 2]
    camera_points[:, 1] = camera_points[:, 1] / camera_points[:, 2]

    return camera_points[:, :2].astype(np.int32)


def homography(four_corners, K, R, t, d):
    K_inv = np.linalg.inv(K)
    # project to 3D
    world_points = camera_to_world(four_corners, K_inv, d)
    # backproject to input view
    four_corners_input_view = world_to_camera(world_points, K, R, t)
    [H, mask] = cv.findHomography(four_corners, four_corners_input_view)
    return H

# Load Images, K-Matrix and Camera Pose
# Except K matric which is 3x3 array, other parameters are lists
input_imgs, source_img, k_matrix = load_imgs_and_k_mats()
# input_imgs is a list of 4 images while source_img is a single image
r_matrices, t_vectors = load_camera_pose()
# r_matrices and t_vectors are lists of length 5
# values at index 2 are for the middle camera which is the reference camera
# print(r_matrices[2]) # identity
# print(t_vectors[2]) # zeros
# print(k_matrix) # a 3x3 matrix

if __name__ == '__main__':
    # Your Code goes here
    img_h, img_w, ch = source_img.shape
    depth_map_scores = np.zeros((img_h, img_w, 1))
    # get depth proposals
    depth_proposals = get_depth_proposals(settings["min_depth"], settings["max_depth"], settings["num_depths"])
    four_corners = np.array([[0, 0, 1],
                             [0, img_w - 1, 1],
                             [img_h - 1, 0, 1],
                             [img_h - 1, img_w - 1, 1]
                             ])
    p_ref_patches = extract_pathches(source_img, settings["patch_size"])
    for i in range(len(depth_proposals)):
        scores = np.zeros((img_h, img_w, 1))
        if experiment == 1:
            for j in [0, 1, 3, 4]:
                H_j = homography(four_corners, k_matrix, r_matrices[j], t_vectors[j], depth_proposals[i])
                warpped_input_img_j = cv.warpPerspective(input_imgs[j], H_j,
                                                         (input_imgs[j].shape[1], input_imgs[j].shape[0]))
                p_patches = extract_pathches(warpped_input_img_j, settings["patch_size"])
                score = ssd(p_ref_patches, p_patches)
                scores = np.append(scores, np.expand_dims(score, axis=2), axis=2)
        else:
            patches = []
            # warpped_input_imgs = []
            for j in [0, 1, 3, 4]:
                H_j = homography(four_corners, k_matrix, r_matrices[j], t_vectors[j], depth_proposals[i])
                warpped_input_img_j = cv.warpPerspective(input_imgs[j], H_j,
                                                         (input_imgs[j].shape[1], input_imgs[j].shape[0]))
                # warpped_input_imgs.append(warpped_input_img_j)
                p_patches = extract_pathches(warpped_input_img_j, settings["patch_size"])
                patches.append(p_patches)
            # warpped_input_imgs_avg.append(img_avg(warpped_input_imgs))
            # warpped_input_imgs_median.append(img_median(warpped_input_imgs))
            score = ssd_2(patches)
            scores = np.append(scores, np.expand_dims(score, axis=2), axis=2)

        depth_map_score = np.sum(scores, axis=2)
        depth_map_scores = np.append(depth_map_scores, np.expand_dims(depth_map_score, axis=2), axis=2)
        print(f"depth_proposal at {i}/{len(depth_proposals)}")
    depth_map_index = np.argmax(depth_map_scores[:, :, 1:], axis=2)
    if experiment == 2:
        warpped_input_img_avg = (input_imgs[0] + input_imgs[1] + input_imgs[3] + input_imgs[4]) // 4
        warpped_input_img_median = np.zeros(shape=(img_h, img_w, 3))
        imgs = [input_imgs[0], input_imgs[1], input_imgs[3], input_imgs[4]]
        for i in range(img_h):
            for j in range(img_w):
                warpped_input_img_median[i, j] = np.median(np.array(imgs)[:, i, j, :], axis=0).astype(np.int32)

    depth_map = depth_proposals[depth_map_index]
    # store the depth map to the file
    filename = f"{settings['dataset']}_{settings['similarity']}_p_{settings['patch_size']}_{experiment}.png"
    depth_to_file(depth_map, filename)
    print(f'{filename} has been saved.')

    if experiment == 2:
        cv.imwrite('synthesis_avg.png', warpped_input_img_avg)
        cv.imwrite('synthesis_median.png', warpped_input_img_median)

    print('synthesis pictures have been saved.')
