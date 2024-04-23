import pandas as pd
import numpy as np
import struct
from util import imread,imshow,imsave
import cv2
from predict import get_predict_func_detectron
import open3d as o3d
import tqdm
from hdmap_ext import process_image
import os

# use left camera (cam2)
kitti_path = '/home/hrz/project/kitti/'
point_cloud_file = 'dataset/sequences/%02d/velodyne/%06d.bin'
image_file = 'data_odometry_color/dataset/sequences/%02d/image_2/%06d.png'
calib_file = 'data_odometry_calib/dataset/sequences/%02d/calib.txt'
pose_file = 'data_odometry_labels/dataset/sequences/%02d/poses.txt'
label_file = 'data_odometry_labels/dataset/sequences/%02d/labels/%06d.label'
sem_file = 'data_odometry_color_sem/dataset/sequences/%02d/semimage/%06d.png'

# hardware synchronization
# no motion distortion correction

# segmentation
model_config="imseg/mask2former/config/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"
model_file="imseg/mask2former/model/model.pkl"
predict = get_predict_func_detectron(model_config,model_file)


vistas2kitti_dict = {
     0: 0,
    55: 10,  # Car
    52: 11,  # Bicycle
    54: 13,  # Bus
    57: 15,  # Motorcycle
    58: 16,  # On Rails
    61: 18,  # Truck
    59: 20,  # Other Vehicle
    19: 30,  # Person
    20: 31,  # Bicyclist
    21: 32,  # Motorcyclist
    13: 40,  # Road
    10: 44,  # Parking
    15: 48,  # Sidewalk
     7: 49,  # Other Ground #######
     8: 49,
     9: 49,
    11: 49,
    12: 49,
    14: 49,
    17: 50,  # Building
    3:  51,   # Fence
    16: 52,   # Other Structure ######
    18: 52,
     2: 52,
     4: 52,
     5: 52,
     6: 52,
    24: 60,  # Lane Marking
    30: 70,  # Vegetation
    29: 72,  # Terrain
    45: 80,  # Pole
    46: 81,  # Traffic Sign
    49: 81,
    50: 81
}

def vistas2kitti(cls):
    if cls in vistas2kitti_dict:
        return vistas2kitti_dict[cls]
    else:
        return 99

def miou(predictions_labels):
    """
    Calculate the mean Intersection over Union (mIOU) from an array of predictions and labels.

    :param predictions_labels: n x 2 numpy array where the first column is predictions and the second is labels
    :return: mean IOU across all classes
    """
    predictions = predictions_labels[:, 0]
    labels = predictions_labels[:, 1]

    # Find all unique classes in the predictions and labels
    unique_classes = np.unique(np.concatenate([predictions, labels]))

    ious = []  # List to store the IOU for each class

    for cls in unique_classes:
        # Calculate intersection and union for this class
        intersection = np.sum((predictions == cls) & (labels == cls))
        union = np.sum((predictions == cls) | (labels == cls))

        # Calculate IOU for this class
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        ious.append(iou)

    # Calculate mean IOU across all classes
    miou = np.mean(ious)
    return miou

def read_kitti_point_cloud(bin_path):
    """
    读取KITTI点云数据文件
    :param bin_path: 点云数据的路径
    :return: point cloud
    """
    # KITTI点云数据是以float32格式保存的，每个点4个值(x, y, z, reflectance)
    pc = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pc

def read_kitti_image(img_path):
    return imread(img_path)

def read_labels(label_filename):
    """ 读取并返回 SemanticKITTI 数据集的标签文件 """
    label = np.fromfile(label_filename, dtype=np.uint32)
    # SemanticKITTI 使用 32位无符号整数存储标签，高16位为实例标签，低16位为语义标签
    semantic_label = label & 0xFFFF  # 提取低16位的语义标签
    return semantic_label

def read_seq_poses(pose_path):
    data = np.loadtxt(pose_path)
    
    # 如果数据只有一行，需要确保其形状是(1, 12)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    
    # 将每一行数据reshape成3x4的矩阵
    transformed_data = data.reshape(-1, 3, 4)
    
    # 初始化一个4x4的矩阵，其中最后一行是[0, 0, 0, 1]
    num_matrices = transformed_data.shape[0]
    final_matrices = np.zeros((num_matrices, 4, 4))
    final_matrices[:, :3, :] = transformed_data
    final_matrices[:, 3, :] = np.array([0, 0, 0, 1])
    
    return final_matrices

def read_calib(calib_path):
    calibration_matrices = {}
    
    with open(calib_path, 'r') as file:
        for line in file:
            # 分割行，获取矩阵标签和数据
            parts = line.split(':')
            label = parts[0].strip()
            values = np.array(parts[1].strip().split(), dtype=float)
            matrix = values.reshape((3, 4))
            # 存储矩阵
            calibration_matrices[label] = matrix

    return calibration_matrices


def proj(ps,label,ex,K,semimg):
    '''
    ps (n*3) tr(4*4) K(3*4)
    '''
    ps = np.hstack((ps,label.reshape(-1,1)))
    ps = ps[(ps[:,0]>0.2)&(ps[:,0]<20)&(ps[:,1]>-15)&(ps[:,1]<15)]
    cls = ps[:,4]
    ps = ps.T[:3,:]
    l = ps.shape[1]
    e = np.ones((1,l),dtype=float)
    ps = np.concatenate((ps,e))
    tmp = K.dot(ex.dot(ps))
    depths = tmp[2]
    res = tmp/tmp[2]
    x = res[0].astype(int)
    y = res[1].astype(int)
    img = np.full(semimg.shape, np.inf)
    valid = (x >= 0) & (x < img.shape[1]) & (y >= 0) & (y < img.shape[0])
    # Use C++ to accelearte the loop, it will be faster two times, see in extension/ext.cpp
    imgp = process_image(x[valid],y[valid],depths[valid],cls[valid],img,semimg)
    # The following code is the equal implementation in python
    # imgp = {}
    # img = np.full(semimg.shape, np.inf)
    # for xi, yi, di, ci in zip(x[valid], y[valid], depths[valid], cls[valid]):
    #     if di < img[yi, xi]:
    #         img[yi, xi] = di
    #         imgp[(yi,xi)] = [vistas2kitti(semimg[yi,xi]),int(ci)]
    return imgp

def pcshow(bin_path):
    """
    读取KITTI点云数据文件
    :param bin_path: 点云数据的路径
    :return: Open3D的PointCloud对象
    """
    # KITTI点云数据是以float32格式保存的，每个点4个值(x, y, z, reflectance)
    ps = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    ps = ps[(ps[:,0]>0.2)&(ps[:,1]>-15)&(ps[:,1]<15)]
    
    # 创建Open3D的PointCloud对象
    pc = o3d.geometry.PointCloud()
    # 只取x, y, z，忽略reflectance
    pc.points = o3d.utility.Vector3dVector(ps[:, :3])
    o3d.visualization.draw_geometries([pc])


folder_index = range(21)
#folder_index = range(11,21)
folder_index = range(11)

file_index = 0

total_evas = []

for foi in folder_index:
    os.makedirs(kitti_path+'data_odometry_color_sem/dataset/sequences/%02d/semimage/'%foi,exist_ok=True)
    evas = []
    pose = read_seq_poses(kitti_path+pose_file%(foi))
    calib = read_calib(kitti_path+calib_file%(foi))
    ex = calib['Tr']
    ex = np.concatenate((ex,np.array([[0,0,0,1]])))
    K = calib['P2']
    count = len(pose)
    mious = 0
    with tqdm.tqdm(range(count)) as c:
        for i in c:
            pc = read_kitti_point_cloud(kitti_path+point_cloud_file%(foi,i))
            img = read_kitti_image(kitti_path+image_file%(foi,i))
            label = read_labels(kitti_path+label_file%(foi,i))
            try:
                semimg = imread(kitti_path+sem_file%(foi,i),0)
                semimg[0]
            except:
                semimg = predict(img)
                imsave(kitti_path+sem_file%(foi,i),semimg)
            limgp = proj(pc,label,ex,K,semimg)
            eva = np.array(list(limgp.values()))
            miou_v = miou(eva)
            mious += miou_v
            c.set_postfix(miou=miou_v)
            #print(f"epoch {i} miou: {miou(eva)}")
            evas.append(eva)
            total_evas.append(eva)
    evas = np.vstack(evas)
    print(f"sequece {foi} miou: {miou(evas)}")
    #print(f"sequece {foi} miou 2: {mious/count:.3f}")
total_evas = np.vstack(total_evas)
print(f"total miou: {miou(total_evas)}")

# calib = read_calib("/home/hrz/project/kitti/data_odometry_calib/dataset/sequences/15/calib.txt")