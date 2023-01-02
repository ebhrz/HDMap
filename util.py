import copy
import tf
from tf.transformations import quaternion_from_euler as qfe
from tf.transformations import quaternion_multiply as qmul
from tf.transformations import quaternion_inverse as qi
from tf.transformations import euler_from_quaternion as efq
import numpy as np
import pandas as pd
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2,PointField
from PIL import Image as Im
import cv2
from sklearn.cluster import DBSCAN
from cv_bridge import CvBridge
import collections
import pclpy
from pclpy import pcl
import multiprocessing as mp
from predict import get_colors


#Modify the extrinsic parameters from LiDAR to camera to your own's, 
extrinsic = np.matrix(
    [
         [ 1.0102, -0.0026, -0.0087,  0.1135],
         [-0.0033, -0.0030, -0.9963, -0.1617],
         [ 0.0049,  0.9962, -0.0287,  0.0516],
         [ 0.0000,  0.0000,  0.0000,  1.0000]
    ]
)
#Modify the intrinsic parameters to your camera's
K = np.array([
    [543.5046, 0, 630.7183], 
    [0, 540.5383, 350.9063], 
    [0, 0, 1]
])
#And the distortion parameters
dismatrix = np.array([-1.05873889e-01,  1.32265629e-01, -8.55667814e-05,-1.04098281e-03, -7.01241428e-02])


global dbs
dbs = DBSCAN()
bri = CvBridge()
INIT = None
tmp = None
pc2Point = collections.namedtuple('pc2Point', ['x', 'y', 'z', 'intensity', 'ring', 'time'])
pc2PointRGB = collections.namedtuple('pc2PointRGB', ['x', 'y', 'z', 'rgba'])
pcfilter = pclpy.pcl.filters.StatisticalOutlierRemoval.PointXYZRGBA()
br = tf.TransformBroadcaster()

e_i = extrinsic.I
K_i = np.matrix(K).I
color_classes = get_colors()
color={
    (255,0,255,0):4278255360,
    (255,0,0,255):4294901760
}


def color2int32(tup):
    return np.array([*tup[1:], 255]).astype(np.uint8).view('uint32')[0]

def color_convert(cls):
    c = color_classes[cls]
    return np.array([*c, 255]).astype(np.uint8).view('uint32')[0]

def r(d):
    return d * 3.1415926 / 180

def r2d(r):
    return r/3.1415926 * 180

def quaterRot(q1,q2):
    e1 = np.array(efq(q1))
    e2 = np.array(efq(q2))
    q = np.array(qfe(*(e2-e1)))
    return q

def imshow(img):
    Im.fromarray(img).show()

def imread(name,type = 1):
    if type:
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        return cv2.imread(name,0)


def transformPointCloud(Tf,target_frame, point_cloud):
    """
    :param target_frame: the tf target frame, a string
    :param ps: the sensor_msgs.msg.PointCloud message
    :return: new sensor_msgs.msg.PointCloud message, in frame target_frame
    :raises: any of the exceptions that :meth:`~tf.Transformer.lookupTransform` can raise

    Transforms a geometry_msgs PoseStamped message to frame target_frame, returns a new PoseStamped message.
    """
    r = sensor_msgs.msg.PointCloud2()
    r.header.stamp = point_cloud.header.stamp
    r.header.frame_id = target_frame
    r.channels = point_cloud.channels

    mat44 = Tf.asMatrix(target_frame, point_cloud.header)
    def xf(p):
        xyz = tuple(np.dot(mat44, np.array([p.x, p.y, p.z, 1.0])))[:3]
        return geometry_msgs.msg.Point(*xyz)
    r.points = [xf(p) for p in point_cloud.points]
    return r

def save_pc(pcmsg, fname):
    data = pc2.read_points(pcmsg)
    tmp = pcl.PointCloud(np.array(data).astype(np.float32)[:,:3])
    pcl.save(tmp, fname)

def save_nppc(nparr,fname):
    s = nparr.shape
    if s[1] == 4:#rgb
        tmp = pcl.PointCloud.PointXYZRGBA(nparr[:,:3],np.array([color_classes[int(i)] for i in nparr[:,3]]))
    else:
        tmp = pcl.PointCloud.PointXYZ(nparr)
    pcl.io.save(fname,tmp)
    return tmp

def cam2pixel(p,K):
    return p[0]*K[0,0]+K[0,2],p[1]*K[1,1]+K[1,2]

def getMsgImg(imgmsg):
    img = np.frombuffer(imgmsg.data, np.uint8).reshape((imgmsg.height, imgmsg.width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def interp(x):
    i=0
    s={0:-1,1:0}
    length=len(x)
    while i < length:
        if x[i] == 0:
            i+=1
            continue
        #initial and move next
        if s[0] != -1 and i - s[0] != 1 and ( i != length-1 or x[length-1] !=0):
            l = i-s[0]
            step = (x[i] - s[1])/l
            for j in range(s[0]+1,i):
                x[j] = s[1]+step*(j-s[0])
        s[0] = i
        s[1] = x[i]
        i+=1
    return x


def pcd_trans_44(pcd,tfmatrix44):
    pcd = pcd.astype('float32')
    length = len(pcd)
    pcd = pcd.T
    pcd_xyz = pcd[:3]
    ones = np.ones((1, length))
    transpcd = np.vstack((pcd_xyz, ones))
    pcd[:3] = np.dot(tfmatrix44, transpcd)[:3]
    transedpcd = pcd.T
    return transedpcd


def pcl2image(pc,w,h,ex):
    #pc is a np array
    #Transform to camera frame
    pc = pcd_trans_44(pc,ex)
    #Only process left and right side within 10 meters and front side out of 0.2 meters
    d=pc[(pc[:,2]>0.2)&(pc[:,0]>-10)&(pc[:,0]<10)]
    b=d[:,3]
    e=d[:,:3]
    #Project to image
    r=K.dot(e.T)
    scale = copy.deepcopy(r[2])
    r=r/r[2]
    t = r.T[:,:2]
    p = np.concatenate([t,b.reshape(-1,1),scale.reshape(-1,1)],axis=1)
    p = p[(p[:,0]>=0)&(p[:,1]>=1)&(p[:,0]<w)&(p[:,1]<h)]
    img = np.zeros((h,w,3))#intensity depth class
    #Next will use pybind11 to accelerate this part
    for i in p:
        try:
            img[int(i[1]),int(i[0])]=np.array([i[2],i[3],0])
        except Exception as e:
            print(e,i[1],i[0],int(i[1]))
    return img
#Temporarily stop using interpolation, due to multi classes of the segmentation
    #interpolation
    interp_img = copy.deepcopy(img)
    pool = mp.Pool(24)
    per_epoch = int(img.shape[1] / 24)
    img_a = []
    for k in range(24):
        img_a.append((img[:,k * per_epoch:(k + 1) * per_epoch],k * per_epoch,(k + 1) * per_epoch))
    results = [pool.apply_async(interp_wrapper, args=(perimg[0],perimg[1],perimg[2])) for perimg in img_a]
    pool.close()
    pool.join()
    for res in results:
        perimg,start,end = res.get()
        interp_img[:,start:end]=perimg
    # for i in range(img.shape[1]):
    #     interp(img[:,i,1])
    return interp_img

def interp_wrapper(perimg,start,end):
    for i in range(perimg.shape[1]):
        perimg[:,i,1] = interp(perimg[:,i,1])
    return perimg,start,end

def pcl2image1(pc,w,h,ox,oy,oz):
    #pc is a np array
    #discard back

    pc[:,0]=pc[:,0]+ox
    pc[:,1]=pc[:,1]+oy
    pc[:,2]=pc[:,2]+oz
    d=pc[pc[:,1]>2.25]
    b=d[:,3]
    #convert to camera frame
    d=tf_L2C(d)
    e=d[:,:3]
    #project to image
    r=K.dot(e.T)
    r=r/r[2]
    t = r.T[:,:2]
    p = np.concatenate([t,b.reshape(-1,1)],axis=1)
    p = p[(p[:,0]>=0)&(p[:,1]>=1)&(p[:,0]<w)&(p[:,1]<h)]
    img = np.zeros((h,w))#intensity depth class
    for i in p:
        try:
            img[h-int(i[1]),int(i[0])]=i[2]
        except Exception as e:
            pass
    return img

# pimg=pcl2image(z,672,376)
# imshow(pimg)



def img2pcl(d_img,msg = False):
    res = []
    sem = np.array(np.nonzero(d_img[:,:,1])).T
    for i in sem:
        col = d_img[tuple(i)]
        z=col[1]
        c=col[2]
        x=i[1]
        y = i[0]
        # y=d_img.shape[0]-i[0]
        res.append((x*z,y*z,z,c))
    if len(res) == 0:
        return np.array([]).reshape((0,4))
    res = np.array(res)
    src = K_i.dot(res[:,:3].T).T

    xyz = pcd_trans_44(src,e_i)
    res[:,:3]=xyz
    res=np.unique(res, axis=0)
    if not msg:
        return res
    else:
        ret = np.zeros( (res.shape[0], 1), dtype={"names": ( "x", "y", "z", "rgba" ),"formats": ( "f4", "f4", "f4", "u4" )} )
        ret['x']=res[:,0].reshape(-1,1)
        ret['y']=res[:,1].reshape(-1,1)
        ret['z']=res[:,2].reshape(-1,1)
        rgba = np.array([color_classes(i) for i in res[:,3].reshape(-1,1)])
        ret['rgba'] = rgba
        msg = PointCloud2()
        msg.fields = [
            PointField('x',  0, PointField.FLOAT32, 1),
            PointField('y',  4, PointField.FLOAT32, 1),
            PointField('z',  8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]

        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * res.shape[0]
        msg.is_dense = True
        msg.height = 1
        msg.width = res.shape[0]
        msg.data = ret.tostring()
        return msg


def get_i_pcd_msg(pcd):
    ret = np.zeros((pcd.shape[0], 1), dtype={"names": ("x", "y", "z", "intensity"), "formats": ("f4", "f4", "f4", "f4")})
    ret['x'] = pcd[:, 0].reshape(-1, 1)
    ret['y'] = pcd[:, 1].reshape(-1, 1)
    ret['z'] = pcd[:, 2].reshape(-1, 1)
    ret['intensity'] = pcd[:,3].reshape(-1,1)
    msg = PointCloud2()
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * pcd.shape[0]
    msg.is_dense = True
    msg.height = 1
    msg.width = pcd.shape[0]
    msg.data = ret.tostring()
    return msg

def  get_rgba_pcd_msg(pcd,pcdcolor=color[(255,0,0,255)],frame = 'world'):
    ret = np.zeros((pcd.shape[0], 1), dtype={"names": ("x", "y", "z", "rgba"), "formats": ("f4", "f4", "f4", "u4")})
    ret['x'] = pcd[:, 0].reshape(-1, 1)
    ret['y'] = pcd[:, 1].reshape(-1, 1)
    ret['z'] = pcd[:, 2].reshape(-1, 1)
    try:
        ret['rgba'] = np.array([color_convert(int(i)) for i in pcd[:, 3].reshape(-1, 1)]).reshape(-1,1)
    except Exception as e:
        #print(e)
        ret['rgba'] = pcdcolor
    msg = PointCloud2()
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]

    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * pcd.shape[0]
    msg.is_dense = True
    msg.height = 1
    msg.width = pcd.shape[0]
    msg.data = ret.tostring()
    msg.header.frame_id = frame
    return msg


def tf_C2L():
    pass

def base_rotate(arr,theta):
    rarr = np.zeros(arr.shape)
    if len(rarr.shape) == 2:
        rarr[:,0] = arr[:,0]*np.cos(theta) + arr[:,1]*np.sin(theta)
        rarr[:,1] = arr[:,1]*np.cos(theta) - arr[:,0]*np.sin(theta)
    else:
        for i in range(len(rarr)):
            rarr[i,:,0] = arr[i,:,0]*np.cos(theta) + arr[i,:,1]*np.sin(theta)
            rarr[i,:,1] = arr[i,:,1]*np.cos(theta) - arr[i,:,0]*np.sin(theta)
    return rarr


def draw_line(p1,p2):
    assert isinstance(p1,np.ndarray) or isinstance(p1,set)
    assert isinstance(p2,np.ndarray) or isinstance(p2,set)
    assert p1.shape == p2.shape
    if len(p1.shape) == 2 or p1.shape[0]== 2:
        d = np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2)
        n = int(d/0.01)
        x = np.linspace(p1[0],p2[0],n)
        y = np.linspace(p1[1],p2[1],n)
        line = np.stack((x,y),axis=1)
    else:
        d = np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2+(p1[2]-p2[2])**2)
        n = int(d/0.01)
        x = np.linspace(p1[0],p2[0],n)
        y = np.linspace(p1[1],p2[1],n)
        z = np.linspace(p1[2],p2[2],n)
        line = np.stack((x,y,z),axis=1)
    return line

def draw_box(lb,lt,rb,rt):
    b_line = draw_line(lb,rb)
    t_line = draw_line(lt,rt)
    l_line = draw_line(lb,lt)
    r_line = draw_line(rb,rt)
    return (b_line,t_line,l_line,r_line)


def vectorize(pc):
    vpc = np.array(()).reshape(-1,3)
    dbs.fit(pc)
    labels = dbs.fit_predict(pc)#label
    cluster = list(set(labels))
    n = len(cluster)
    boxes = []
    vector = []
    for i in range(n):
        c = pc[labels==i]#each cluster
        if len(c) == 0:
            continue
        xmin = c[:,0].min()
        xmax = c[:,0].max()
        ymin = c[:,1].min()
        ymax = c[:,1].max()
        w = xmax-xmin
        l = ymax-ymin
        if w/l > 0.2:
            #discard the part not like a lane
            cluster.remove(i)
            continue
        #geometry centroid i4 i5  mass centroid i6 i7
        boxes.append(((xmin,ymin,0),(xmin,ymax,0),(xmax,ymin,0),(xmax,ymax,0),((xmax-xmin)/2+xmin,ymin,0),((xmax-xmin)/2+xmin,ymax,0),(c[:,0].sum()/len(c),ymin,0),(c[:,0].sum()/len(c),ymax,0)))


        #print('point:',(xmin,(ymax-ymin)/2+ymin),(xmax,(ymax-ymin)/2+ymin))
    boxes = np.array(boxes)

    for i in boxes:
        #print('after resume rotate:',i[4],i[5])
        c_line = draw_line(i[6],i[7])
        box = draw_box(i[0],i[1],i[2],i[3])
        #vpc = np.concatenate((vpc,c_line,*box),axis=0)
        vpc = np.concatenate((vpc,c_line),axis=0)

    return vpc, boxes