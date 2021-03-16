import numpy as np
import os, imageio
from pathlib import Path
import cv2

def load_dtu_data(path):
    imgdir = os.path.join(path, 'image')
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, 0)
    num = imgs.shape[0]

    cam_path = os.path.join(path, "cameras.npz")
    all_cam = np.load(cam_path)

    focal = 0

    coord_trans_world = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
    coord_trans_cam = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )

    poses = []
    for i in range(num):
        P = all_cam["world_mat_" + str(i)]
        P = P[:3]

        K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
        K = K / K[2, 2]

        focal += (K[0,0] + K[1,1]) / 2

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        scale_mtx = all_cam.get("scale_mat_" + str(i))
        if scale_mtx is not None:
            norm_trans = scale_mtx[:3, 3:]
            norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

            pose[:3, 3:] -= norm_trans
            pose[:3, 3:] /= norm_scale

        pose = (
                coord_trans_world
                @ pose
                @ coord_trans_cam
            )
        poses.append(pose[:3,:4])
    
    poses = np.stack(poses)
    print('poses shape:', poses.shape)


    focal = focal / num
    H, W = imgs[0].shape[:2]
    print("HWF", H, W, focal)

    return imgs, poses, [H, W, focal]