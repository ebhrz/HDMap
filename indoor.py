#!/usr/bin/python3

global INIT
global path


import pickle
import sys
import time
import rospy
import genpy
import predict
from predict import get_colors
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image,CompressedImage
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from util import *
import tf
import tf2_ros as tf2
from tf.transformations import quaternion_from_euler as qfe
from tf.transformations import euler_from_quaternion as efq
from tf.transformations import quaternion_slerp
import pymap3d as pm
import geometry_msgs
import math
from tf import transformations
from queue import Queue
import copy
import pclpy
from pclpy import pcl
import os
from rosbag import Bag
import multiprocessing as mp
from threading import Thread
import argparse
from tf.transformations import quaternion_from_euler as qfe
import json


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def cmkdir(path):
    path = path.split('/')
    cur = ''
    for p in path:
        try:
            cur = '/'.join([cur,p])
            os.mkdir(cur[1:])
        except:
            pass

def class2color(cls,alpha = False):
    c = color_classes[cls]
    if not alpha:
        return np.array(c).astype(np.uint8)
    else:
        return np.array([*c, 255]).astype(np.uint8)

def quit(signum, frame):
    print('')
    print('stop function')
    sys.exit()


def gtInterp(itime,gt1,gt2):
    if abs(gt2[0] - gt1[0]) < rospy.Duration(0.0001):
        return (itime,gt1[1],gt1[2])
    t1=gt1[1]
    t2=gt2[1]
    r1=gt1[2]
    r2=gt2[2]
    k = (itime-gt1[0])/(gt2[0]-gt1[0])
    t = k*(t2-t1)+t1
    r = quaternion_slerp(r1, r2, k)
    return (itime,t,r)

def getGt(itime, gtQ):
    flag = False
    for i,v in enumerate(gtQ):
        if itime <= v[0]:
            flag = True
            break
    if flag:
        return gtInterp(itime, gtQ[i-1], gtQ[i])
    else:
        return None


def qmul(q1, q0):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array([x1*w0 + y1*z0 - z1*y0 + w1*x0,-x1*z0 + y1*w0 + z1*x0 + w1*y0,x1*y0 - y1*x0 + z1*w0 + w1*z0,-x1*x0 - y1*y0 - z1*z0 + w1*w0], dtype='float64')


def qstar(q):
    return np.array((-q[0],-q[1],-q[2],q[3]))


def fix_points(lps,lpsgt,gtQ):
    fixlp = []
    for lp in lps:
        lpts = lpsgt[0] + rospy.Duration(lp[5])
        lpgt = getGt(lpts, gtQ)
        dt = (lpgt[1] - lpsgt[1])
        tmp_x = 0
        tmp_y = np.sqrt(dt[0]**2+dt[1]**2)
        tmp_z = dt[2]
        dt = np.array((tmp_x,tmp_y,tmp_z))
        lpq = lpgt[2]
        lpsq = lpsgt[2]
        dr = qmul(qstar(lpsq),lpq)
        mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
        lp[:3] = np.dot(mat44, np.array((lp[0], lp[1], lp[2], 1)))[:3]
        fixlp.append(pc2Point(*lp[:3], lp[3], int(lp[4]), lp[5]))

    return fixlp

def get_semantic_pcd(img,pcd):
    rimg = cv2.undistort(img, K, dismatrix)
    src = pcl2image(pcd, img.shape[1], img.shape[0], extrinsic)
    # segmentation
    cimg = predict(rimg)
    src[:, :, 2] = cimg
    # recover pcd from depth img
    sem_pcdata = img2pcl(src)
    # filter noisy points
    if len(sem_pcdata)==0:
        return np.array([]).reshape((0,4)),cimg
    return sem_pcdata,cimg


def pcd_trans(pcd,dt,dr):
    length = len(pcd)
    pcd = pcd.T
    pcd_xyz = pcd[:3]
    ones = np.ones((1, length))
    transpcd = np.vstack((pcd_xyz, ones))
    mat44 = np.dot(transformations.translation_matrix(dt), transformations.euler_matrix(*dr))
    pcd[:3] = np.dot(mat44, transpcd)[:3]
    transedpcd = pcd.T
    return transedpcd

def get_pose(e,n,u,q,ts):
    #u = 0
    pose = PoseStamped()
    pose.header.stamp = ts
    pose.pose.position.x = e
    pose.pose.position.y = n
    pose.pose.position.z = u
    pose.pose.orientation = Quaternion(*q)
    return pose

class myqueue(list):
    def __init__(self, cnt=-1):
        self.cnt = cnt

    def append(self, obj):
        if len(self) >= self.cnt and self.cnt != -1:
            self.remove(self[0])
        super().append(obj)

    def is_empty(self):
        if len(self) == 0:
            return True
        else:
            return False

def getPose(lmsg,poses):
    try:
        return poses[abs(poses[:,-1]-lmsg.header.stamp.to_sec())<0.001][0]
    except Exception:
        return []

def getImg(lmsg,IQ):
    i_last = None
    for imsg in IQ:
        if lmsg.header.stamp < imsg.header.stamp:
            if i_last is None:
                return imsg
            if lmsg.header.stamp - i_last.header.stamp > imsg.header.stamp - lmsg.header.stamp:
                return imsg
            else:
                return i_last
        else:
            i_last = imsg
    # search fail in the queue
    return None



parser = argparse.ArgumentParser(description='Semantic point cloud builder, due to the large computation, map construction is devided into several steps to avoid interrupting in case')
parser.add_argument('-c','--config',help='The config file path, recommand use this method to start the tool')
parser.add_argument('-b','--bag',help='The recorded ros bag path')
parser.add_argument('-f','--fastfoward',help='Start to play at the nth seconds', default=0,type = float)
parser.add_argument('-d','--duration',help='Time to play', default=None,type = float)
parser.add_argument('-p','--pose',help='Pose file for the construction')
args = parser.parse_args()


with open((args.config or 'config/indoor_config.json'),'r') as f:
    config = json.load(f)
args.bag = (args.bag or config['bag_file'])
args.pose = (args.pose or config['pose_file'])
args.fastfoward = (args.fastfoward or config['start_time'])
args.duration = (args.duration or config['play_time'])

color_classes = get_colors(config['cmap'])
K = config['intrinsic'] or K
extrinsic = config['extrinsic'] or extrinsic
dismatrix = config['distortion_matrix'] or dismatrix
K = np.matrix(K)
extrinsic = np.matrix(extrinsic)
dismatrix = np.matrix(dismatrix)

colors = color_classes.astype('uint8')
rospy.init_node('fix_distortion', anonymous=False, log_level=rospy.FATAL)
fixCloudPubHandle = rospy.Publisher('dedistortion_cloud', PointCloud2, queue_size=5)
originCloudPubHandle = rospy.Publisher('origin_cloud', PointCloud2, queue_size=5)
semanticCloudPubHandle = rospy.Publisher('SemanticCloud', PointCloud2, queue_size=5)
vecCloudPubHandle = rospy.Publisher('vec_cloud', PointCloud2, queue_size=5)
imgPubHandle = rospy.Publisher('Img', Image, queue_size=5)
semimgPubHandle = rospy.Publisher('SemanticImg', Image, queue_size=5)
groundTruthPubHandle = rospy.Publisher('ground_truth', Path, queue_size=0)
print('ros ready')

labels = get_colors()
predict = getattr(predict,config['predict_func'])(config['model_config'],config['model_file'])
print('torch ready')

bag = Bag(args.bag)
start = bag.get_start_time()
start = start+args.fastfoward
if args.duration != -1:
    end = start+args.duration
    end = genpy.Time(end)
else:
    end = None
start = genpy.Time(start)

bagread = bag.read_messages(start_time=start,end_time = end)
print('bag ready')


cmkdir(config['save_folder']+"/originpics")
cmkdir(config['save_folder']+"/sempics")
briconvert = config['image_compressed'] and bri.compressed_imgmsg_to_cv2 or bri.imgmsg_to_cv2

tnow = None
QSIZE=20
#queue flag
QFLAG=False
#lidar ready flag
LFLAG=False
#image ready flag
IFLAG=False
gtQ = []
# LQ = []
# IQ = []
LOQ = []

sem_world = []
world = []

qcnt = 0
path = Path()
path.header.frame_id = 'world'


lidartopicmsg = None
imgtopicmsg = None


poses = np.loadtxt(args.pose,delimiter=',')

IQ = myqueue(40)
LQ = myqueue(5)


index = 0
simgs = []
pose_save = []
save_step = 2
for msg in bagread:
    if msg.topic == config['camera_topic']:
        imsg = msg.message
        IQ.append(imsg)
        # IFLAG = True
    elif msg.topic == config['LiDAR_topic']:
        lmsg = msg.message
        LQ.append(lmsg)
        removeQ = []
        for lmsg in LQ:
            pose = getPose(lmsg,poses)
            if len(pose) == 0:
                continue
            imgmsg = getImg(lmsg,IQ)
            if not imgmsg:
                continue    
            if index <= -1:
                print('jump this frame')
                removeQ.append(lmsg)
            else:
                #img = bri.compressed_imgmsg_to_cv2(imgmsg)
                img = briconvert(imgmsg)

                xyz = pose[:3]
                rot = qfe(*pose[4:7])
                pose_save.append(np.array([*xyz,*rot]))
                rimg = cv2.undistort(img, K, dismatrix)
                cv2.imwrite(config['save_folder']+"/originpics/%06d.png"%(index),rimg)
                lps = np.array(list(pc2.read_points(lmsg)))
                lps = lps[lps[:, 1] > 0.2]
                sem_pcd, semimg = get_semantic_pcd(img, lps)
                cv2.imwrite(config['save_folder']+"/sempics/%06d.png" % (index), semimg)
                semimg = colors[semimg.flatten()].reshape((*semimg.shape, 3))
                semimgPubHandle.publish(bri.cv2_to_imgmsg(semimg,'bgr8'))
                if len(sem_pcd) != 0:
                    sem_world_pcd = pcd_trans(sem_pcd, pose[:3], pose[4:7])
                    sem_world.append(sem_world_pcd)
                    sem_msg = get_rgba_pcd_msg(sem_world_pcd)
                    sem_msg.header.frame_id = 'world'
                    sem_msg.header.stamp = lmsg.header.stamp
                    semanticCloudPubHandle.publish(sem_msg)
                    imgPubHandle.publish(bri.cv2_to_imgmsg(img,'bgr8'))
                    print('semantic point publish')
                    removeQ.append(lmsg)
                else:
                    print('no semantic info')
                if index%200 == 0:
                    with open(config['save_folder']+'/indoor.pkl','wb') as f:
                        pickle.dump(sem_world,f)
                    print('saved epoch %d'%index)
            index+=1
        for lmsg in removeQ:
            LQ.remove(lmsg)
pose_save = np.stack(pose_save)
#Now save the middle-ware
np.savetxt(config['save_folder']+'/pose.csv',pose_save,delimiter=',')
with open(config['save_folder']+'/indoor.pkl','wb') as f:
    pickle.dump(sem_world,f)
print('Done')
