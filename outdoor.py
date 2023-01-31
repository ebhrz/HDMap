#!/usr/bin/python3
global INIT
global path

import pickle
import time
import rospy
import genpy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image,CompressedImage
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from novatel_oem7_msgs.msg import INSPVAX
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
import predict
from util import *
import tf2_ros as tf2
from tf.transformations import quaternion_from_euler as qfe
from tf.transformations import euler_from_quaternion as efq
from tf.transformations import quaternion_slerp
import pymap3d as pm
import pandas as pd
import geometry_msgs
import math
from tf import transformations
import os
from rosbag import Bag
import multiprocessing as mp
from threading import Thread
import argparse
import json


def cmkdir(path):
    path = path.split('/')
    cur = ''
    for p in path:
        try:
            cur = '/'.join([cur,p])
            os.mkdir(cur[1:])
        except:
            pass

def Handle(msg):
    if msg.topic == '/acc_pose':
        groundTruth(msg)
    elif msg.topic == '/velodyne_points':
        lidarHandle(msg)
    elif msg.topic == '/zed2/camera/left/image_raw':
        imageHandle(msg)

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


def pcd_trans(pcd,dt,dr,inverse=False):
    length = len(pcd)
    pcd = pcd.T
    pcd_xyz = pcd[:3]
    ones = np.ones((1, length))
    transpcd = np.vstack((pcd_xyz, ones))
    mat44 = np.dot(transformations.translation_matrix(dt), transformations.quaternion_matrix(dr))
    if inverse:
        mat44 = np.matrix(mat44).I
    pcd[:3] = np.dot(mat44, transpcd)[:3]
    transedpcd = pcd.T
    return transedpcd

def get_pose(e,n,u,q,ts):
    pose = PoseStamped()
    pose.header.stamp = ts
    pose.pose.position.x = e
    pose.pose.position.y = n
    pose.pose.position.z = u
    pose.pose.orientation = Quaternion(*q)
    return pose

def class2color(cls,alpha = False):
    c = color_classes[cls]
    if not alpha:
        return np.array(c).astype(np.uint8)
    else:
        return np.array([*c, 255]).astype(np.uint8)




cmkdir("result/outdoor/originpics")
cmkdir("result/outdoor/sempics")

parser = argparse.ArgumentParser(description='Semantic point cloud builder, due to the large computation, map construction is devided into several steps to avoid interrupting in case')
parser.add_argument('-c','--config',help='The config file path, recommand use this method to start the tool')
parser.add_argument('-b','--bag',help='The recorded ros bag')
parser.add_argument('-f','--fastfoward',help='Start to play at the nth seconds', default=0,type = float)
parser.add_argument('-d','--duration',help='Time to play', default=None,type = float)
parser.add_argument('-u','--undistortion',help='do LiDAR points undistortion',type=bool)
args = parser.parse_args()

with open((args.config or 'config/outdoor_config.json'),'r') as f:
    config = json.load(f)
args.bag = (args.bag or config['bag_file'])
args.fastfoward = (args.fastfoward or config['start_time'])
args.duration = (args.duration or config['play_time'])
args.undistortion = (args.undistortion or config['cloud_distortion'])

color_classes = get_colors(config['cmap'])
K = config['intrinsic'] or K
extrinsic = config['extrinsic'] or extrinsic
dismatrix = config['distortion_matrix'] or dismatrix
K = np.matrix(K)
extrinsic = np.matrix(extrinsic)
dismatrix = np.matrix(dismatrix)


colors = color_classes.astype('uint8')
rospy.init_node('fix_distortion', anonymous=False, log_level=rospy.DEBUG)
fixCloudPubHandle = rospy.Publisher('dedistortion_cloud', PointCloud2, queue_size=5)
originCloudPubHandle = rospy.Publisher('origin_cloud', PointCloud2, queue_size=5)
semanticCloudPubHandle = rospy.Publisher('SemanticCloud', PointCloud2, queue_size=5)
vecCloudPubHandle = rospy.Publisher('vec_cloud', PointCloud2, queue_size=5)
imgPubHandle = rospy.Publisher('Img', Image, queue_size=5)
groundTruthPubHandle = rospy.Publisher('ground_truth', Path, queue_size=0)
semimgPubHandle = rospy.Publisher('SemanticImg', Image, queue_size=5)
print('ros ready')

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

tnow = None

QSIZE=20
#queue flag
QFLAG=False
#lidar ready flag
LFLAG=False
#image ready flag
IFLAG=False
gtQ = []
LQ = []
IQ = []
LOQ = []

sem_world = []


qcnt = 0
path = Path()
path.header.frame_id = 'world'


lidartopicmsg = None
imgtopicmsg = None


# undistortion switch
UNDISTORTION = args.undistortion

briconvert = config['image_compressed'] and bri.compressed_imgmsg_to_cv2 or bri.imgmsg_to_cv2


store_file = open(config['save_folder']+'/outdoor.pkl','wb')
index = 0
pose_save = []

for msg in bagread:
    #Handle(msg)
    if len(gtQ) == 0 and msg.topic == config['LiDAR_topic']:
        continue
    if not rospy.is_shutdown():
        try:
            if msg.topic == config['GNSS_topic']:
                gnss = msg.message
                if not INIT:
                    INIT = [gnss.latitude, gnss.longitude, gnss.altitude, gnss.azimuth, 0, 0, 0]
                    print('init:',gnss.latitude,gnss.longitude,gnss.altitude)
                else:
                    e, n, u = pm.geodetic2enu(gnss.latitude, gnss.longitude, gnss.altitude, INIT[0], INIT[1], INIT[2])
                    t = np.array((e,n,u))
                    q = qfe(r(gnss.roll), r(gnss.pitch), -r(gnss.azimuth))
                    br.sendTransform((e, n, u), q, msg.timestamp, 'velodyne', 'world')
                    pose = get_pose(e,n,u,q,msg.timestamp)
                    path.header.stamp = msg.timestamp
                    path.poses.append(pose)
                    groundTruthPubHandle.publish(path)
                    # maintain 15 gts in the queue
                    if not QFLAG:
                        gtQ.append([msg.timestamp,t,q])
                        qcnt+=1
                        if qcnt == 30:
                            QFLAG = True
                    else:
                        gtQ.remove(gtQ[0])
                        gtQ.append([msg.timestamp,t,q])
                # process lidar data in high  frequency messages

                lidar_remove = []
                for lidartopicmsg in LOQ:
                    lmsg = lidartopicmsg.message
                    lps = np.array(list(pc2.read_points(lmsg)))
                    gtts = lmsg.header.stamp
                    gt = getGt(gtts, gtQ)
                    perf_time_start = time.time()
                    perf_all_start = perf_time_start
                    if UNDISTORTION:
                        gttssec = gtts.to_sec()
                        tsmax = genpy.Time(gttssec + lps[:, 5].max())
                        gtmax = getGt(tsmax,gtQ)
                        if not gtmax:
                            break
                        lidar_remove.append(lidartopicmsg)
                        gt = getGt(gtts, gtQ)
                        lps_bak = lps.copy()
                        # TODO
                        # filter self car
                        #lps = lps[(lps[:,0]<-0.7)|(lps[:,0]>0.7)|(lps[:,1]<-1.1)|(lps[:,1]>2.2)]
                        # only process far points, maybe > 10?
                        lps = lps[lps[:,1]>5]
                        fixlp = []
                        pool = mp.Pool(24)
                        per_epoch = int(len(lps)/24)
                        lps_a = []
                        for k in range(24):
                            lps_a.append(lps[k*per_epoch:(k+1)*per_epoch])
                        results = [pool.apply_async(fix_points, args=(perlps, gt, gtQ)) for perlps in lps_a]
                        pool.close()
                        pool.join()
                        for res in results:
                            fixlp.extend(res.get())
                        fixcloud = pc2.create_cloud(lmsg.header, lmsg.fields, fixlp)
                        fixCloudPubHandle.publish(fixcloud)
                        originCloudPubHandle.publish(lmsg)
                        pp = fixlp
                        perf_time_end = time.time()
                        print('undistortion used %.2f'%(perf_time_end-perf_time_start))
                    else:
                        lidar_remove.append(lidartopicmsg)
                        pp = lps
                        fixCloudPubHandle.publish(lmsg)
                    LQ.append((gt, pp))
                    image_remove = []
                    for imgtopicmsg in IQ:
                        imgmsg = imgtopicmsg.message
                        imts = imgmsg.header.stamp
                        imgt = getGt(imts,gtQ)
                        l_next = None
                        l_last = None
                        for l in LQ[::-1]:
                            if l[0][0] < imts:
                                l_last = l
                                break
                            else:
                                l_next = l
                        if l_next is None or l_last is None:
                            continue
                        # the nearest lidar frame
                        if imts-l_last[0][0] > l_next[0][0]-imts:
                            l = l_next
                        else:
                            l = l_last
                        index += 1
                        pose_save.append(np.array([*imgt[1], *imgt[2]]))
                        img = briconvert(imgmsg)
                        imgPubHandle.publish(bri.cv2_to_imgmsg(img))
                        cv2.imwrite(config['save_folder']+'/originpics/%06d.png'%index,img)
                        #del img
                        pcd = np.array(l[1])
                        pcd = pcd_trans(pcd,l[0][1],l[0][2])
                        align_pcd = pcd_trans(pcd,imgt[1],imgt[2],True)
                        perf_time_start = time.time()
                        sem_pcd, semimg = get_semantic_pcd(img,align_pcd)
                        perf_time_end = time.time()
                        print('image segmentation used %.2f'% (perf_time_end - perf_time_start))
                        cv2.imwrite(config['save_folder']+"/sempics/%06d.png"%index,semimg)
                        semimg = colors[semimg.flatten()].reshape((*semimg.shape, 3))
                        semimgPubHandle.publish(bri.cv2_to_imgmsg(semimg, 'bgr8'))
                        if len(sem_pcd) != 0:
                            sem_world_pcd = pcd_trans(sem_pcd, imgt[1], imgt[2])
                            #save result
                            pickle.dump(sem_world_pcd,store_file)
                            store_file.flush()
                            perf_time_start = time.time()
                            sem_msg = get_rgba_pcd_msg(sem_world_pcd)
                            perf_time_end = time.time()
                            print('rgb pc generation used %.2f\nepoch total used %2.f' % (perf_time_end - perf_time_start,perf_time_end - perf_all_start))
                            sem_msg.header.frame_id = 'world'
                            semanticCloudPubHandle.publish(sem_msg)
                            print('semantic point publish')
                    # queue out
                        image_remove.append(imgtopicmsg)
                    for tmp in image_remove:
                        IQ.remove(tmp)
                for tmp in lidar_remove:
                    LOQ.remove(tmp)
            elif msg.topic == config['LiDAR_topic']:
                LOQ.append(msg)
            elif msg.topic == config['camera_topic']:
                IQ.append(msg)
        except rospy.ROSInterruptException:
            break
        except KeyboardInterrupt:
            print('break')
            break
pose_save=np.stack(pose_save)
np.savetxt(config['save_folder']+'/pose.csv',pose_save,delimiter=',')
store_file.close()