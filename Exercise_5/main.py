import os, math
import time
import numpy as np
import cv2 as cv
import scipy.io as io
import sklearn
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

# We should down size the images, to see results quickly
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
    source_img = imgs.pop(2)
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


def ssd(feature_1, feature_2, coordinates):
    '''
    Compute the sum of square difference between the input features
    '''
    img_h, img_w, ch = feature_1.shape
    ssd_scores = np.zeros((img_h, img_w))
    for i in range(img_h):
        for j in range(img_w):
            x, y = coordinates[i, j]
            if x in range(0, feature_1.shape[1]) and y in range(0, feature_1.shape[0]):
                ssd_scores[i, j] = np.sum(np.square(feature_1[i, j] - feature_2[y, x]))
    return ssd_scores


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
    # get the pixel homogeneous coordinates
    x = np.mgrid[0:img_h:1, 0:img_w:1][1]
    y = np.mgrid[0:img_h:1, 0:img_w:1][0]
    z = np.ones(shape=(img_h, img_w))
    points_pixel_homo = np.dstack((x, y, z))
    # inverse K
    K_inv = np.linalg.inv(k_matrix)
    depth_map = np.zeros((img_h, img_w))
    depth_map_scores = np.zeros((img_h, img_w, 1))
    # get depth proposals
    depth_proposals = get_depth_proposals(settings["min_depth"], settings["max_depth"], settings["num_depths"])

    for depth_proposal in depth_proposals:
        # a. compute the corresponding 3D points
        p_ref = np.expand_dims(depth_proposal / (np.linalg.norm(points_pixel_homo @ K_inv.T, ord=2, axis=2)),
                               axis=2) * (points_pixel_homo @ K_inv.T)
        # b. project the 3D points  onto all other camera images (C0, C1, C3, C4). Let’s call these pro-
        # jected image locations {p0, p1, p3, p4}.
        p0 = p_ref @ r_matrices[0].T + t_vectors[0].reshape(3)
        p1 = p_ref @ r_matrices[1].T + t_vectors[1].reshape(3)
        p3 = p_ref @ r_matrices[3].T + t_vectors[3].reshape(3)
        p4 = p_ref @ r_matrices[4].T + t_vectors[4].reshape(3)
        p = [p0, p1, p3, p4]
        # c. Use SSD to determine a similarity score (also referred as photometric consistency)
        # between the source pixel pref and the pixel values at {p0,p1,p3,p4}.
        # Remember, pixel-wise comparison is not accurate and instead
        # apply SSD matching on a patch centered around pref against the patches centered around {p0, p1, p3, p4}.
        p_ref_patches = extract_pathches(source_img, settings["patch_size"])
        scores = np.zeros((img_h, img_w, 1))
        for i in range(4):
            img_coordinates_p_homo = p[i] @ k_matrix.T
            img_coordinates_p = np.dstack((img_coordinates_p_homo[:, :, 0] / img_coordinates_p_homo[:, :, 2],
                                           img_coordinates_p_homo[:, :, 1] / img_coordinates_p_homo[:, :, 2]))
            img_coordinates_p = img_coordinates_p.astype(np.int32)
            p_patches = extract_pathches(input_imgs[i], settings["patch_size"])
            score = ssd(p_ref_patches, p_patches, img_coordinates_p)
            scores = np.append(scores, np.expand_dims(score, axis=2), axis=2)
            # Store the similarity computed above in a list
            # (this list will have the same size as the number of depth proposals).

        depth_map_score = np.sum(scores, axis=2)
        depth_map_scores = np.append(depth_map_scores, np.expand_dims(depth_map_score, axis=2), axis=2)
        print(f"depth_proposal at {depth_proposal}/{depth_proposals}")
    depth_map = np.max(depth_map_scores, axis=2)
    # store the depth map to the file
    filename = f"{settings['similarity']}_p_{settings['patch_size']}.png"
    depth_to_file(depth_map, filename)
