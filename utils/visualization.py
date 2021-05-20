import numpy as np
import os
import imageio
from pathlib import Path
from read_write_model import *


def visualize_keypoints(basedir, factor=8, bd_factor=.75):
    imgdir = imgdir = os.path.join(basedir, 'images')
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    imgs = [imageio.imread(f)[...,:3] for f in imgfiles]
    

    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        # for i in range(len(images[id_im].xys)):
        #     point2D = images[id_im].xys[i]
        #     id_3D = images[id_im].point3D_ids[i]
        #     if id_3D == -1:
        #         continue
        #     point3D = points[id_3D].xyz
        #     depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
        #     if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
        #         continue
        #     err = points[id_3D].error
        #     weight = 2 * np.exp(-(err/Err_mean)**2)
        #     depth_list.append(depth)
        #     coord_list.append(point2D/factor)
        #     weight_list.append(weight)
        # print(id_im, np.min(depth_list), np.max(depth_list), np.mean(depth_list))
        # data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list)})
        imageio.imwrite(os.path.join(basedir,'visual', '{:03d}_visual.png'.format(id_im-1)), imgs[id_im-1])
   
    return 


def main():
    visualize_keypoints("/data2/kangled/datasets/DSNeRF/dtu/scan21_all")

if __name__ == "__main__":
    main()