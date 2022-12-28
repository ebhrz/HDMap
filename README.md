# IPNL - HDMap Builder
## Introduction
This is a tool for HDMap construction developed by IPNL. It's a whole pipeline, which can build a HD semantic map or HD vector map. A docker version will coming soon.

<div align="center">
    <div style="display:inline; float:left;margin:5px">
            <img height=350px src="img/garage.gif"/>
            <p><strong>The progress of indoor map construction</strong></p>
    </div>
    <div style="display:inline; float:left;margin:5px">
            <img height=350px src="img/garage.png"/>
            <p><strong>The 3D semantic map of the indoor scenario</strong></p>
    </div>
</div>
<div style="clear:both"></div>
<div align="center">
    <div style="display:inline; float:left;margin:5px">
            <img height=350px src="img/outdoor.png"/>
            <p><strong>The 3D semantic map of the outdoor scenario</strong></p>
    </div>
    <div style="display:inline;float:left ;margin:5px;text-align:center">
            <img height=350px src="img/vector.png"/>
            <p><strong>The 3D vector map of the outdoor scenario</strong></p>
    </div>
</div>
<div style="clear:both"></div>

## Oveview
* [Features](#key-features)
* [Pipeline](#pipeline-overview)
* [Requirements](#requirements)
* [Preprocess](#preprocess)
* [Usage](#usage)
    * [Folder Structure](#folder-structure)
    * [Bag File Structure](#bag-file-structure)

## Key Features
Our tool supports below features:
* **Soft time synchronization**
* **Image undistortion**
* **LiDAR motion distortion correction**
* **Image semantic segmentation**
* **Lane lines and poles vectorization**
* **...**

For more details, please refer to our [paper](blank)

## Pipeline Overview

<p align="center">
    <img src = "img/pipeline.png"/>
</p>

## Requirements
This tool is built with Python. To use this tool, there are several dependencies to be installed. Here are the lists:
* ***ROS***
Our tool utilizes ROS to store the recording data and do the visualization. ROS installation refers to [here](http://wiki.ros.org/noetic/Installation)
    * ***Novatel Msgs***
    Map construction requires accurate localization. In our sensorkit, a span-CPT is employed to be the ground truth. Please install its message type via 
    ```bash
        sudo apt install ros-(your ROS version)-novatel-gps-msgs
    ```
* ***Pandas*** and ***Numpy***
Please use the commands below to install:
```bash
pip install numpy
pip install pandas
```
* ***OpenCV***
Our tool also requires OpenCV. Install it via:
```bash
pip install opencv-python
```
* ***PyTorch***
Please refer to PyTorch installation guide at [here](https://pytorch.org/get-started/locally/)
    * ***Detectron2***
    This is a image segmentation framework by Facebook using PyTorch. Our tool directly uses the pretrained Swin2Transformer model. Please refer to its installation guide at [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
* ***Scikit-learn***
This is a powerful tool for machine learning. Our tool uses DBSCAN from it to do the cluster. Install it via:
```bash
pip install scikit-learn
```
* ***Pclpy***
Since we'll process the point cloud, we choose this python wrapper for PCL which uses pybind11 to maintain the same API. Installation refers to [here](https://github.com/davidcaron/pclpy)

## Folder Structure
```
.
├── LICENSE
├── README.md
├── data
│   ├── indoor
│   │   ├── indoor.bag  # place your indoor bag file here
│   │   └── pose6d.csv  # the prior made pose estimation result
│   └── outdoor
│       └── outdoor.bag # place your outdoor bag file here
├── result # this folder will be gererated automatically
├── mask2former
│   ├── class.json
│   ├── config
│   ├── config.py
│   ├── model
│   │   └── model.pkl # place the pretrainied swin2transformer model weights
│   └── mask2former
├── indoor.py
├── outdoor.py
├── vector.py
├── make_pcl.py
├── predict.py
└── util.py
```
If you want to run an example, please click the link to download the data.
* [indoor.bag]()
* [outdoor.bag]()
* [model.pkl]()
## Bag File Structure
For the outdoor bag file, here's an example:
```
path:        outdoor.bag
version:     2.0
duration:    34:38s (2078s)
start:       Dec 23 2021 11:43:52.57 (1640231032.57)
end:         Dec 23 2021 12:18:31.01 (1640233111.01)
size:        64.5 GB
messages:    327941
compression: none [41569/41569 chunks]
types:       novatel_msgs/INSPVAX        [b5d66747957184042a6cca9b7368742f]
             sensor_msgs/CompressedImage [8f7a12909da2c9d3332d540a0977563f]
             sensor_msgs/PointCloud2     [1158d486dd51d683ce2f1be655c3c181]
topics:      /acc_pose             207936 msgs    : novatel_msgs/INSPVAX          
             /image                 57732 msgs    : sensor_msgs/CompressedImage
             /origin_cloud          41515 msgs    : sensor_msgs/PointCloud2
```
And for the indoor bag file, here's an example:
```
path:        garage.bag
version:     2.0
duration:    6:59s (419s)
start:       Feb 04 2022 18:04:18.39 (1643969058.39)
end:         Feb 04 2022 18:11:18.08 (1643969478.08)
size:        70.1 GB
messages:    342273
compression: none [29148/29148 chunks]
types:       sensor_msgs/Image           [060021388200f6f0f447d0fcd9c64743]
             sensor_msgs/PointCloud2     [1158d486dd51d683ce2f1be655c3c181]
topics:      /velodyne_points                   4193 msgs    : sensor_msgs/PointCloud2        
             /zed2/camera/left/image_raw       12477 msgs    : sensor_msgs/Image          
```
So far, we should configure the ros topics in [outdoor.py]() and [indoor.py]() manually, but as the config file is supported, it can be defined in the config file.
## Preprocess
```
The tool now needs to be configured manually, config file support will be avalibable soon.
```
### Intrinsic Parameters and Distortion Parameters

The intrinsic parameters should be changed to your own's in file [util.py](https://github.com/ebhrz/HDMap/blob/f16c8624c30adf9079fd35350e69ca2989fc204f/util.py#L35)
```python
#Modify the intrinsic parameters to your camera's
K = np.array([
    [543.5046, 0, 630.7183], 
    [0, 540.5383, 350.9063], 
    [0, 0, 1]
])
#And the distortion parameters
dismatrix = np.array([-1.05873889e-01,  1.32265629e-01, -8.55667814e-05,-1.04098281e-03, -7.01241428e-02])
```

### Extrinsic Parameters
The extrinsic parameters should be changed to your own's in file [util.py](https://github.com/ebhrz/HDMap/blob/f16c8624c30adf9079fd35350e69ca2989fc204f/util.py#L26)
```python
#Modify the extrinsic parameters from LiDAR to camera to your own's, 
extrinsic = np.matrix(
    [
         [ 1.0102, -0.0026, -0.0087,  0.1135],
         [-0.0033, -0.0030, -0.9963, -0.1617],
         [ 0.0049,  0.9962, -0.0287,  0.0516],
         [ 0.0000,  0.0000,  0.0000,  1.0000]
    ]
)
```


### Indoor Pose Estimation
For indoor scenario, there is no GNSS signal to provide global localization. Thus, pose estimation should be performed at first. Also, We recommond you to use LiDAR SLAM algorithm such as A-LOAM, LIO-SAM, etc. to estimate the pose. [Here](TODO) is an implementation of LIO-SAM with several changes to generate the pose for all the LiDAR messages.
## Usage
* `outdoor.py`
```
  -h, --help                                    show this help message and exit
  -b BAG, --bag BAG                             The recorded ros bag file
  -f FASTFOWARD, --fastfoward FASTFOWARD        Start to play at the nth seconds
  -d DURATION, --duration DURATION              Time to play
  -u UNDISTORTION, --undistortion UNDISTORTION  Do LiDAR points undistortion
```
* `indoor.py`
```
  -h, --help                                    show this help message and exit
  -b BAG, --bag BAG                             The recorded ros bag
  -f FASTFOWARD, --fastfoward FASTFOWARD        Start to play at the nth seconds
  -d DURATION, --duration DURATION              Time to play
  -p POSE, --pose POSE                          Pose file for the construction
```
* `make_pcl.py`
```
  -h, --help                                    show this help message and exit
  -i INPUT, --input INPUT
  -m {indoor,outdoor}, --mode {indoor,outdoor}  Depend on the way to store the pickle file
  -f FILTERS [FILTERS ...], --filters FILTERS [FILTERS ...] Default to show all the classes, the meaning of each class refers to class.json
  -s SAVE, --save SAVE                          Save to pcd file
  -t TRAJECTORY, --trajectory TRAJECTORY        Trajectory file, use to follow the camera
  --semantic SEMANTIC                           Semantic photos folder
  --origin ORIGIN                               Origin photos folder
```
* `vector.py`
```
  -h, --help                                    show this help message and exit
  -i INPUT, --input INPUT
  -m {outdoor,indoor}, --mode {outdoor,indoor}  Depend on the way to store the pickle file
  -f FILTERS [FILTERS ...], --filters FILTERS [FILTERS ...] Default to show all the classes, the meaning of each class refers to class.json
  -s SAVE, --save SAVE                          Save to pcd file
  -t TRAJECTORY, --trajectory TRAJECTORY        Trajectory file, use to follow the camera
  --semantic SEMANTIC                           Semantic photos folder
  --origin ORIGIN                               Origin photos folder
  --vector                                      Do the vectorization, only available when filters are accepted
```
### Example
To run the map builder, please first start the ros core
```
$ roscore
```
And you can open the rviz to see the visualization
```
rviz -d vis.rviz
```
#### Indoor Example
1. First synchronize and segment.
```bash
python3 indoor.py -p data/indoor/pose6d.csv -b data/indoor/garage.bag -f 10 -d 5
```
The command below will ignore the first 10 seconds and process the next 5 seconds. Attention to provide the pose file via `-p`. And this command will generate result folder and files automatically. The result contains origin images (result/indoor/originpics), semantic images (result/indoor/sempics), the processing pose file (result/indoor/pose.csv), and the semantic local point cloud (result/indoor/indoor.pkl)

2. Make the global 3D semantic map. 
```
python3 make_pcl.py --semantic result/indoor/sempics --origin result/indoor/originpics -i result/indoor/indoor.pkl -t result/indoor/pose.csv -s result/indoor/result.pcd -m indoor
```
This command is used for combine the local point cloud and do the visualization. `-i` option is **compulsory**. `-s` will determin where to save point cloud map.

3. Make the global 3D vector map.
```
python3 vector.py --semantic result/indoor/sempics --origin result/indoor/originpics -i result/indoor/indoor.pkl -t result/indoor/pose.csv -s result/indoor/result.pcd -m indoor --vector
```
This command is similar to `make_pcl.py` but the option `--vector` can vectorize the lane lines and poles.

#### Outdoor Example
1. First synchronize and segment.
```bash
python3 outdoor.py -b data/outdoor/outdoor.bag -u -f 120 -d 5 -u
```
The command below will ignore the first 120 seconds and process the next 5 seconds. And during the process, LiDAR motion distortion correction will be performed via option `-u`. And this command will generate result folder and files automatically. The result contains origin images (result/outdoor/originpics), semantic images (result/outdoor/sempics), the processing pose file (result/outdoor/pose.csv), and the semantic local point cloud (result/outdoor/indoor.pkl)

2. Make the global 3D semantic map. 
```
python3 make_pcl.py --semantic result/outdoor/sempics --origin result/outdoor/originpics -i result/outdoor/outdoor.pkl -t result/outdoor/pose.csv -s result/outdoor/result.pcd -m outdoor
```
This command is used for combine the local point cloud and do the visualization. `-i` option is **compulsory**. `-s` will determin where to save point cloud map.

3. Make the global 3D vector map.
```
python3 vector.py --semantic result/outdoor/sempics --origin result/outdoor/originpics -i result/outdoor/outdoor.pkl -t result/outdoor/pose.csv -s result/outdoor/result.pcd -m outdoor --vector
```
This command is similar to `make_pcl.py` but the option `--vector` can vectorize the lane lines and poles.

-----

## Example Sensor Kit Setup
We use part of the sensor kit in **[UrbanNavDataset](https://github.com/IPNL-POLYU/UrbanNavDataset/blob/master/README.md)**, they are:
  - 3D LiDAR sensor: ([HDL 32E Velodyne](https://velodynelidar.com/products/hdl-32e/)): (360 HFOV, +10~-30 VFOV, 80m range, 10Hz)，
  - Camera: [ZED2](https://www.stereolabs.com/zed-2/) Stereo (30 Hz)
  - [SPAN-CPT](https://www.novatel.com/products/span-gnss-inertial-systems/span-combined-systems/span-cpt/): (RTK GNSS/INS, RMSE: 5cm, 100Hz)

In outdoor scenario, all the three sensors are used. And in indoor scenario, GNSS is not available, thus only LiDAR and camera are used. For the detail of the sensor kit, please refer to our **[UrbanNavDataset](https://github.com/IPNL-POLYU/UrbanNavDataset/blob/master/README.md)**
<p align="center">
  <img width="712pix" src="img/hongkong_sensor2.png">
</p>
<p align="center">
  <img width="712pix" src="img/calibration_sensors.png">
</p>