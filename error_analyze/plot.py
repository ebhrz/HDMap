from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import pickle

# if __name__ == "__main__":
# 	with open('no_offset.pkl','rb') as f:
# 		r=np.array(pickle.load(f)).astype('float')

# 	with open('no_offset_t.pkl','rb') as f:
# 		t=np.array(pickle.load(f)).astype('float')


def data_load(fn):
    with open(fn,'rb') as f:
        r=np.array(pickle.load(f)).astype('float')
    return r

def plot_r(r,title=""):
	r=np.array(r).astype(float)
	tt=r[:,3].copy()
	# tt[tt>5]=5
	cmr = cm.ScalarMappable(cmap=cm.viridis)
	# color_map.set_array(res[:,3])
	fig=plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111,projection='3d')
	img = ax.scatter(r[:,0],r[:,1],r[:,2],c=cmr.to_rgba(tt,alpha=1),s=10)
	ax.set_xlabel(r'$\alpha$ offset(°)')
	ax.set_ylabel(r'$\beta$ offset(°)')
	ax.set_zlabel(r'$\gamma$ offset(°)')
	ax.set_title(title)
	plt.colorbar(cmr)
	plt.show()


def plot_t(t,title=""):
	t = np.array(t).astype(float)
	cmt = cm.ScalarMappable(cmap=cm.viridis)
	cmt.set_array(t[:,3])
	fig=plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111,projection='3d')
	img = ax.scatter(t[:,0],t[:,1],t[:,2],c=cmt.to_rgba(t[:,3],alpha=1),s=10)
	ax.set_xlabel(u'X offset(cm)')
	ax.set_ylabel(u'Y offset(cm)')
	ax.set_zlabel(u'Z offset(cm)')
	ax.set_title(title)
	plt.colorbar(cmt)
	plt.show()

def plot_i(i,title=""):
	i = np.array(i).astype(float)
	cmt = cm.ScalarMappable(cmap=cm.viridis)
	cmt.set_array(i[:,2])
	fig=plt.figure(figsize=(10,5))
	ax = fig.add_subplot(111,projection='3d')
	img = ax.scatter(i[:,0],i[:,1],i[:,2],c=cmt.to_rgba(i[:,2],alpha=0.8),s=10)
	ax.set_xlabel(r'$\Delta{u}$')
	ax.set_ylabel(r'$\Delta{v}$')
	ax.set_zlabel('error(m)')
	ax.set_title(title)
	plt.colorbar(cmt)
	plt.show()
