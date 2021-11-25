import numpy as np
import scipy.io as io


def is_rotation_matrix(R):
    product = np.dot(R, R.T)
    np.fill_diagonal(product, 0)
    product = np.abs(product)
    return np.all(product < 0.00001)


def compute_relative_rotation(H, K, name):
    R_rel = None

    K_inv = np.linalg.inv(K)

    h1 = H.T[0]
    h2 = H.T[1]
    h3 = H.T[2]

    denominator = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = denominator * np.dot(K_inv, h1)
    r2 = denominator * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    R_rel = np.array((r1, r2, r3))

    print(f'{name}: {R_rel}')

    # checks whether R_rel fulfills the properties of a rotation matrix

    if not is_rotation_matrix(R_rel):
        # if necessary, corrects R_rel and prints the new rotation matrix to the console
        u, s, vh = np.linalg.svd(R_rel)
        R_rel_corrected = np.dot(u, vh)
        print(f'Corrected {name}: {R_rel_corrected}')
        if is_rotation_matrix(R_rel_corrected):
            print(f'{name} corrected is a rotation matrix.')
        else:
            print(f'{name} corrected is NOT a rotation matrix. ')
    else:
        print(f'{name} is a rotation matrix.\n\n')

    return R_rel


def compute_pose(H, K):
    R, t = None, None

    K_inv = np.linalg.inv(K)

    h1 = H.T[0]
    h2 = H.T[1]
    h3 = H.T[2]

    denominator = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = denominator * np.dot(K_inv, h1)
    r2 = denominator * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    R = np.array((r1, r2, r3))

    l = (np.dot(K_inv, h1) / r1)[0]
    t = (np.dot(K_inv, h3) / l)

    print(f'R:{R}')
    print(f't:{t}')


    return R, t


if __name__ == '__main__':
    # Load the data
    data = io.loadmat('data/ex2.mat')

    # data
    H1 = data['H1']  # shape=[3, 3]
    H2 = data['H2']  # shape=[3, 3]
    H3 = data['H3']  # shape=[3, 3]
    x_0 = data['x_0']  # shape=[1,1]
    y_0 = data['y_0']  # shape=[1,1]
    s = data['s']  # shape=[1,1]
    alpha_x = data['alpha_x']  # shape=[1,1]
    alpha_y = data['alpha_y']  # shape=[1,1]

    K = np.array([[alpha_x[0][0], s[0][0], x_0[0][0]],
                  [0, alpha_y[0][0], y_0[0][0]],
                  [0, 0, 1]])

    # compute R_rel and prints it to the console
    R1_rel = compute_relative_rotation(H1, K, 'H1')  # relative rotation between two images

    R2_rel = compute_relative_rotation(H2, K, 'H2')  # relative rotation between two images

    R, t = compute_pose(H3, K)
