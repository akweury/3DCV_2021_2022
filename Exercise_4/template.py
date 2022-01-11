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
# Since the dataset is of high resolution we will work with resized version.
# You can choose the reduction factor using the scale_factor variable.
scale_factor = 2

# Choose similarity metric
# similarity_metric = 'ncc'
similarity_metric = 'ssd'

# Outlier Filtering Threshold. You can test other values, too.
# This is a parameter which you have to select carefully for each dataset
outlier_threshold = 4

# Patch Size
# Experiment with other values like 3, 5, 7,9,11,15,13,17 and observe the result

patch_width = 17

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


# **************************************** Utilitiy Functions ************************************#
def ply_creator(input_3d, rgb_data=None, filename='dummy'):
    ''' Creates colored point cloud that you can visualise using meshlab.
    Inputs:
        input_3d: it sould have shape=[Nx3], each row is 3D coordinate of each point
        rgb_data: it sould have shape=[Nx3], each row is rgb color value of each point
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
    return patches


# **************************************** Utilitiy Functions ************************************#

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
    xl = np.arange(0, resized_l_img.shape[1], 1)
    xl = np.reshape(xl, (1, xl.shape[0], 1))
    xl = np.repeat(xl, resized_l_img.shape[0], axis=0)

    yl = np.arange(0, resized_l_img.shape[0], 1)
    yl = np.reshape(yl, (yl.shape[0], 1, 1))
    yl = np.repeat(yl, resized_l_img.shape[1], axis=1)

    z = depth_map.reshape(depth_map.shape[0], depth_map.shape[1], 1)
    x = xl * z / focal_length
    y = yl * z / focal_length

    points_3d = np.concatenate((x, y, z), axis=2)
    return points_3d


def mask_outliers(similiarity_scores, sim_metric, threshold):
    '''
    Details are given in the exercise sheet.
    '''

    raise NotImplementedError


def ssd(feature_1, feature_2):
    '''
    Compute the sum of square difference between the input features
    '''
    [w, h, fw, fh, ch] = feature_1.shape
    vector_feature_1 = feature_1.reshape(w, h, fw * fh * ch)
    vector_feature_2 = feature_2.reshape(w, h, fw * fh * ch)

    cost = np.sum(np.square(vector_feature_1 - vector_feature_2), axis=-1)
    return cost


def ncc(feature_1, feature_2):
    '''
    Normalised cross correlation.
    '''
    [w, h, fw, fh, ch] = feature_1.shape
    vector_feature_1 = feature_1.reshape(w, h, fw * fh * ch)
    vector_feature_2 = feature_2.reshape(w, h, fw * fh * ch)

    vector_feature_1_sum = np.sum(vector_feature_1, axis=-1)
    vector_feature_2_sum = np.sum(vector_feature_2, axis=-1)

    cost = vector_feature_1_sum * vector_feature_2_sum / (
            np.linalg.norm(vector_feature_1, axis=2) * np.linalg.norm(vector_feature_2, axis=2) + 1e-5
    )
    return cost


def pixel_matching(img_left, img_right):
    costs = []
    for d in range(min_disparity, max_disparity):
        M = np.array([[1, 0, d], [0, 1, 0]], dtype='float64')
        img_right_shifted = cv.warpAffine(img_right, M, (img_right.shape[1], img_right.shape[0]))
        cost = np.sum(img_left - img_right_shifted, axis=2)
        costs.append(cost)
    disparity = np.argmin(np.array(costs), axis=0)
    return disparity


def window_matching(img_left, img_right, method):
    left_patches = extract_pathches(img_left, patch_width)
    # right_patches = extract_pathches(img_right, patch_width)
    costs = []
    for d in range(min_disparity, max_disparity):
        # shift the right_patches matrix by d pixels
        M = np.array([[1, 0, d], [0, 1, 0]], dtype='float64')
        img_right_shifted = cv.warpAffine(img_right, M, (img_right.shape[1], img_right.shape[0]))
        #     shifted_right_patches --> will be of shape [H, W, feat_size]
        #     left_patches --> will be of shape [H, W, feat_size]
        shifted_right_patches = extract_pathches(img_right_shifted, patch_width)
        cost = method(left_patches, shifted_right_patches)
        costs.append(cost)
    if method is ncc:
        disparity = np.argmax(np.array(costs), axis=0)
    else:
        disparity = np.argmin(np.array(costs), axis=0)
    return disparity


def stereo_matching(img_left, img_right, patch_width):
    '''
    This is tha main function for your implementation.
    '''
    # This is the main function for your implementation
    # make sure you do the following tasks
    # 1. estimate disparity for every pixel in the left image
    disparity = pixel_matching(img_left, img_right)
    disparity = window_matching(img_left, img_right, ncc)
    # disparity = window_matching(img_left, img_right, ssd)

    # 2. convert estimated disparity to depth, save it to file
    depth = disparity_to_depth(disparity, baseline)
    write_depth_to_file(depth, 'pixelDisparity.jpg')

    # 3. convert depth to 3D points and save it as colored point cloud using the ply_creator function
    points_3d = depth_to_3d(depth, kmat)
    points_3d_flatten = points_3d.reshape((points_3d.shape[0] * points_3d.shape[1], 3))
    ply_creator(points_3d_flatten)
    # 4. visualize the estimted 3D point cloud using meshlab


if __name__ == '__main__':
    stereo_matching(resized_l_img, resized_r_img, patch_width)
    # Practical Tip:
    # Use smaller image resolutions untill you get the solution
    # A naive appraoch would use three levels of for-loop
    #     for x in range(0, width):
    #         for y in range(0, height):
    #             for d in range(0, d_max):
    # Such methods might get prohibitively slow,
    # Therefore try to avoid for loops(as much as you can)
    # instead try to think of solutions that use multi-dimensional array operations,
    # for example you can have only one for loop: going over the disparities

    # First extract patches, as follows
    # left_patches = extract_pathches(img_left, patch_width)
    # right_patches = extract_pathches(img_right, patch_width)
    # for d in range(d_min, d_max):
    #     shifted_right_patches = shift the right_patches matrix by d pixels
    #     shifted_right_patches --> will be of shape [H, W, feat_size]
    #     left_patches --> will be of shape [H, W, feat_size]
    #     compute the ssd and ncc on multi-dimensional arrays of left_patches and shifted_right_patches
