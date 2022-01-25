import os

import cv2 as cv
import numpy as np
import scipy.io as io
from sklearn.feature_extraction import image

# Select the dataset
dataset = 'medieval_port'
# dataset = 'kitti'

experiment = 'medieval_port_exp_one'

os.makedirs(f'./{experiment}', exist_ok=True)

# While experimenting it is better to work with a lower resolution version of the image
# Since the dataset is of high resolution we will work with down-scaled version of the image.
# You can choose the reduction factor using the scale_factor variable.
scale_factor = 2

# Choose similarity metric by uncommenting you choice below

# similarity_metric = 'pixel'
similarity_metric = 'ncc'
# similarity_metric = 'ssd'

# Outlier Filtering Threshold. You can test other values, too.
# This is a parameter which you have to select carefully for each dataset
outlier_threshold = 3
outlier_filter = False

# Patch Size
# Experiment with other values like 3, 5, 7,9,11,15,13,17 and observe the result

patch_width = 7
if dataset == 'kitti':
    # Minimum and maximum disparies
    min_disparity = 0 // scale_factor
    max_disparity = 150 // scale_factor
    # Focal length
    calib = io.loadmat('./data/kitti/pose_and_K.mat')
    kmat = calib['K']
    # cam_pose = calib['Pose']
    baseline = calib['Baseline']
    kmat[0:2, 0:2] /= scale_factor
    focal_length = kmat[0, 0]
    left_img_path = './data/kitti/left.png'
    right_img_path = './data/kitti/right.png'

elif dataset == 'medieval_port':
    # Minimum and maximum disparies
    min_disparity = 0 // scale_factor
    max_disparity = 80 // scale_factor

    # Focal length
    kmat = np.array([[700.0000, 0.0000, 320.0000],
                     [0.0000, 933.3333, 240.0000],
                     [0.0000, 0.0000, 1.0000]], dtype=np.float32)
    kmat[:2, :] = kmat[:2, :] / scale_factor
    focal_length = kmat[0, 0]
    baseline = 0.5
    left_img_path = './data/medieval_port/left.jpg'
    right_img_path = './data/medieval_port/right.jpg'
else:
    assert False, 'Dataset Error'

# Read Images
l_im = cv.imread(left_img_path, 1)
h, w, c = l_im.shape
resized_l_img = cv.resize(l_im, (w // scale_factor, h // scale_factor))
r_im = cv.imread(right_img_path, 1)
resized_r_img = cv.resize(r_im, (w // scale_factor, h // scale_factor))

# plot left and right images
stacked_imgs = np.concatenate([resized_l_img, resized_r_img], axis=1)


# print(stacked_imgs.shape)


def ply_creator(input_3d, rgb_data=None, filename='dummy'):
    ''' Creates colored point cloud that you can visualise using meshlab.
    Inputs:
        input_3d: it sould have shape=[Nx3], each row is 3D coordinate of each point
        rgb_data(optional): it sould have shape=[Nx3], each row is rgb color value of each point
        filename: file name for the .ply file to be created
    Note: All 3D points whose Z value is set 0 are ignored.
    '''
    assert (input_3d.ndim == 2), "Pass 3d points as NumPointsX3 array "
    pre_text1 = """ply
    format ascii 1.0"""
    pre_text2 = "element vertex "
    pre_text3 = """property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header"""
    valid_points = input_3d.shape[0] - np.sum(input_3d[:, 2] == 0)
    pre_text22 = pre_text2 + str(valid_points)
    pre_text11 = pre_text1
    pre_text33 = pre_text3
    fid = open(filename + '.ply', 'w')
    fid.write(pre_text11)
    fid.write('\n')
    fid.write(pre_text22)
    fid.write('\n')
    fid.write(pre_text33)
    fid.write('\n')
    for i in range(input_3d.shape[0]):
        # Check if the depth is not set to zero
        if input_3d[i, 2] != 0:
            for c in range(3):
                fid.write(str(input_3d[i, c]) + ' ')
            if not rgb_data is None:
                for c in range(3):
                    fid.write(str(rgb_data[i, c]) + ' ')
            # fid.write(str(input_3d[i,2]))
            if i != input_3d.shape[0]:
                fid.write('\n')
    fid.close()
    return True


def disparity_to_depth(disparity, baseline):
    """
    Converts disparity to depth.
    """
    inv_depth = (disparity + 10e-5) / (baseline * focal_length)
    return 1 / inv_depth


def write_depth_to_file(depth, f_name):
    """
    This function writes depth map as an image
    You can modify it, if you think of a better way to visualise depth/disparity
    You can also use it to save disparities
    """
    assert (depth.ndim == 2), "Depth map should be a 2D array "

    depth = depth + 0.0001
    depth_norm = 255 * ((depth - np.min(depth)) / np.max(depth) * 0.9)
    cv.imwrite(f_name, depth_norm)


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
        patches: size[h, w, patch_width*patch_width*c]
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


def ssd(feature_1, feature_2):
    '''
    Compute the sum of square difference between the input features
    '''
    return np.sum(np.square(feature_1 - feature_2), axis=2)


def ncc(feature_1, feature_2):
    '''
    Normalised cross correlation.
    '''
    f1_norm = feature_1 / np.linalg.norm(feature_1, axis=2, keepdims=True) + 10e-06
    f2_norm = feature_2 / np.linalg.norm(feature_2, axis=2, keepdims=True) + 10e-06

    ncc_scores = (f1_norm * f2_norm).sum(axis=2)
    return ncc_scores


def mask_outliers(similiarity_scores, sim_metric, threshold):
    '''
    Details are given in the exercise sheet.
    '''
    img_h, img_w, ch = similiarity_scores.shape
    similiarity_scores_filtered = np.zeros((img_h, img_w, ch))
    if sim_metric == 'pixel':
        return similiarity_scores
    elif sim_metric == 'ssd':
        min_ssd_scores_mask = np.min(similiarity_scores, axis=2) * 1.5
        for i in range(img_h):
            for j in range(img_w):
                similiarity_scores_filtered[i, j] = similiarity_scores[i, j]
                if (similiarity_scores[i, j] < min_ssd_scores_mask[i, j]).sum() >= threshold:
                    similiarity_scores_filtered[i, j] = 0
    elif sim_metric == 'ncc':
        max_ncc_scores_mask = np.max(similiarity_scores, axis=2) * 0.8
        for i in range(img_h):
            for j in range(img_w):
                similiarity_scores_filtered[i, j] = similiarity_scores[i, j]
                if (similiarity_scores[i, j] > max_ncc_scores_mask[i, j]).sum() >= threshold:
                    similiarity_scores_filtered[i, j] = 0
    else:
        raise ValueError

    return similiarity_scores_filtered


def depth_map_calculate(img_left, img_right, patch_width):
    # The number of pixels
    num_rows, num_cols = img_right.shape[:2]

    # pixel matching
    if similarity_metric == 'pixel':
        pixel_scores = np.zeros((num_rows, num_cols, 1))
        for i in range(max_disparity):
            translation_matrix = np.float32([[1, 0, i], [0, 1, 0]])
            shifted_img_right = cv.warpAffine(img_right, translation_matrix, (num_cols, num_rows))
            pixel_score = np.sum(np.abs(img_left - shifted_img_right), axis=2)
            pixel_scores = np.append(pixel_scores, np.expand_dims(pixel_score, axis=2), axis=2)
        pixel_scores = np.array(pixel_scores[:, :, 1:])
        disparity = np.argmin(pixel_scores, axis=2)
    else:
        #  window matching
        window_scores = np.zeros((num_rows, num_cols, 1))
        left_patches = extract_pathches(img_left, patch_width)
        right_patches = extract_pathches(img_right, patch_width)
        for i in range(max_disparity):
            translation_matrix = np.float32([[1, 0, i], [0, 1, 0]])
            shifted_img_right = cv.warpAffine(img_right, translation_matrix, (num_cols, num_rows))
            shifted_right_patches = extract_pathches(shifted_img_right, patch_width)
            if similarity_metric == 'pixel':
                break
            elif similarity_metric == 'ssd':
                # ssd
                ssd_score = ssd(left_patches, shifted_right_patches)
                window_scores = np.append(window_scores, np.expand_dims(ssd_score, axis=2), axis=2)
            elif similarity_metric == 'ncc':
                # ncc
                ncc_score = ncc(left_patches, shifted_right_patches)
                window_scores = np.append(window_scores, np.expand_dims(ncc_score, axis=2), axis=2)
            else:
                raise ValueError
        window_scores = np.array(window_scores[:, :, 1:])

        # outlier filtering
        if outlier_filter:
            window_scores = mask_outliers(window_scores, similarity_metric, outlier_threshold)

        # 1. estimate disparity for every pixel in the left image

        if similarity_metric == 'ssd':
            disparity = np.argmin(window_scores, axis=2)
        elif similarity_metric == 'ncc':
            disparity = np.argmax(window_scores, axis=2)
        else:
            raise ValueError
    # 2. convert estimated disparity to depth, save it to file
    depth_map = disparity_to_depth(disparity, baseline)

    if similarity_metric == 'ncc':
        depth_filter = 100
    else:
        depth_filter = 30
    depth_map = np.where(depth_map < depth_filter, depth_map, 0)

    file_name = f'{dataset}_{similarity_metric}_N_{outlier_threshold}_Disparity{patch_width}x{patch_width}.jpg'
    write_depth_to_file(disparity, file_name)
    print(f'{file_name} has been saved!')

    return depth_map


def depth_to_3d(depth_map, kmat):
    """
    Input:
        depth_map: per pixel depth value, shape [h,w]
        kmat= marix of camera intrinsics, shape [3x3]
    Return: 3D coordinates, with shape [h, w, 3]
    1. First back-project the point from homogeneous image space to 3D,
    by multiplying it with inverse of the camera intrinsic matrix, inv(K)
    2. Then scale it by its depth.
    """
    num_rows, num_cols = depth_map.shape[:2]
    # calculate image homogeneous coordinates
    pixel_plane_x = np.mgrid[0:resized_l_img.shape[0]:1, 0:resized_l_img.shape[1]:1][1]
    pixel_plane_y = np.mgrid[resized_l_img.shape[0] - 1:-1:-1, resized_l_img.shape[1] - 1:-1:-1][0]
    pixel_plane_z = np.ones(shape=resized_l_img.shape[:2])
    pixel_plane_homo = np.dstack((pixel_plane_x, pixel_plane_y, pixel_plane_z))
    inv_K = np.linalg.inv(kmat)  # inverse intrinsic matrix
    # backproject from homogeneous to 3D
    points_3d_scaled = np.zeros((num_rows, num_cols, 3))
    for i in range(num_rows):
        for j in range(num_cols):
            points_3d_scaled[i, j] = (inv_K @ pixel_plane_homo[i, j].reshape(3, 1) * depth_map[i, j]).reshape(3)

    return points_3d_scaled.reshape((num_rows * num_cols, 3))


def stereo_matching(img_left, img_right, patch_width):
    '''
    This is tha main function for your implementation.
    It takes two rectified stereo pairs and window(or patch) size, and performs dense reconstruction
    '''
    # This is the main function for your implementation
    # make sure you do the following tasks
    num_rows, num_cols = img_left.shape[:2]
    depth_map = depth_map_calculate(img_left, img_right, patch_width)
    # 3. convert depth to 3D points and save it as colored point cloud using the ply_creator function
    points_3d_flatten = depth_to_3d(depth_map, kmat)
    colors = img_left.reshape((num_rows * num_cols, 3))
    ply_creator(points_3d_flatten, colors)
    # 4. visualize the estimted 3D point cloud using meshlab
    # For reference we have added a sample result .ply file, you can visualise it with meshlab
    # file located at "data/medieval_sample.ply"


if __name__ == '__main__':
    stereo_matching(resized_l_img, resized_r_img, patch_width)
