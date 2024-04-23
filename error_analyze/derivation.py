from sympy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import sys

a,b,r,x,y,z=symbols('\\alpha \\beta \gamma x y z')
a1,b1,r1,x1,y1,z1=symbols('a1 b1 r1 x1 y1 z1')
da,db,dr,dx,dy,dz=symbols('\Delta{\\alpha} \Delta{\\beta} \Delta{\gamma} \Delta{x} \Delta{y} \Delta{z}')
i,j,k=symbols('i j k')
di,dj,dk = symbols('\Delta{i} \Delta{j} \Delta{k}')
k1,k2,p1,p2=symbols('k1 k2 p1 p2')
u,v,d=symbols('u v d')
du,dv,dd=symbols('\Delta{u} \Delta{v} \Delta{d}')
g = symbols('g')
T_gt, T_e = symbols('T_gt T_e')

target_p={x:1,y:3,g:-1.1}
idealT={a:pi/2,b:0,r:0,i:0,j:0,k:0}
gtT={a:1.59528686,b:-9.82013573e-04,r:-4.82066200e-03,i:0.1135,j:-0.1617,k:0.0516}
offset={da:0,db:0,dr:0,di:0,dj:0,dk:0}
gtK={k1:543.5046,p1:630.7183,k2:540.5383,p2:350.9063}
p=Matrix([x,y,g,1])

#v=Matrix([a,b,r,x,y,z])


T_gt=Matrix([
	[                       cos(b)*cos(r),                        -sin(r)*cos(b),         sin(b),i],
	[sin(a)*sin(b)*cos(r) + sin(r)*cos(a), -sin(a)*sin(b)*sin(r) + cos(a)*cos(r), -sin(a)*cos(b),j],
	[sin(a)*sin(r) - sin(b)*cos(a)*cos(r),  sin(a)*cos(r) + sin(b)*sin(r)*cos(a),  cos(a)*cos(b),k],
	[0,0,0,1]
])
T=T_gt
T_e=T_gt.subs({a: a+da, b: b+db, r: r+dr, i: i+di, j: j+dj, k: k+dk})
T_e_inv = T_e.inv()
T_inv = T.inv()

K=Matrix([
[k1,0,p1],
[0,k2,p2],
[0,0,1]
])

def dT():
	d=(K*Matrix((T*p)[0:3]))[2]
	temp=(T_e_inv*T*p)[2]
	d1 = g*d/temp
	p3=d1/d*T_e_inv*T*p
	p3=simplify(p3)
	return p3


def dI():
	u,v,d=K*Matrix((T*p)[:3])
	u = u/d
	v = v/d
	i1=Matrix([u+du,v+dv,1])
	d1 = symbols('d1')
	xx,yy,zz,_=T_inv*Matrix([*(K.inv()*(d1*i1)),1])
	d1=solve(Eq(zz,g),d1)[0]
	xx,yy,zz,_=T_inv*Matrix([*(K.inv()*(d1*i1)),1])
	p3 = Matrix([xx,yy,zz,1])
	#p3=simplify(p3)
	return p3


def ddi():
	u1,v1,d1 = K*Matrix((T_e*p)[:3])
	u1 = u1/d1
	v1 =v1/d1
	return Matrix([u1,v1])


try:
	task=sys.argv[1]
except Exception:
	task='dT'

if task == 'dT':
	p3=dT()
	temp = Matrix((p-p3)[:3])
	dis = sqrt(temp.dot(temp))
	deg=pi/180
elif task == 'dI':
	p3=dI()
	temp = Matrix((p-p3)[:3])
	dis = sqrt(temp.dot(temp))
	deg=pi/180
elif task == 'di':
	i1 = ddi()
	i1 = i1.subs({**gtK})
	i = ddi()
	i = i.subs({da:0,db:0,dr:0,di:0,dj:0,dk:0,**gtK})
	temp = Matrix(i1-i)
	dis = sqrt(temp.dot(temp))
	deg = pi/180
else:
	print('error input')




def task1():
	res=[]
	dis1=dis.subs({**target_p,**idealT,di:0,dj:0,dk:0})
	for ii in range(-11,11):
		for jj in range(-11,11):
			for kk in range(-11,11):
				res.append([ii,jj,kk,dis1.evalf(subs={da:deg*ii,db:deg*jj,dr:deg*kk})])
				print(ii,jj,kk)
	print('task1')
	return res


def task2():
	res=[]
	dis1=dis.subs({**target_p,**idealT,da:0,db:0,dr:0})
	for ii in range(-11,11):
		for jj in range(-11,11):
			for kk in range(-11,11):
				res.append([ii,jj,kk,dis1.evalf(subs={di:0.01*ii,dj:0.01*jj,dk:0.01*kk,})])
				print(ii,jj,kk)
	print('task2')
	return res

def task3():
	res = []
	dis1=dis.subs({**target_p,**gtT,di:0,dj:0,dk:0})
	for ii in range(-11, 11):
		for jj in range(-11, 11):
			for kk in range(-11, 11):
				res.append([ii, jj, kk, dis1.evalf(
					subs={da: deg * ii, db: deg * jj, dr: deg * kk})])
				print(ii, jj, kk)
	print('task3')
	return res

def task4():
	res=[]
	dis1=dis.subs({**target_p,**gtT,da:0,db:0,dr:0})
	for ii in range(-11,11):
		for jj in range(-11,11):
			for kk in range(-11,11):
				res.append([ii,jj,kk,dis1.evalf(subs={di:0.01*ii,dj:0.01*jj,dk:0.01*kk})])
				print(ii,jj,kk)
	print('task4')
	return res

def task5():
	res=[]
	dis1=dis.subs({**target_p,**idealT,**gtK})
	for uu in range(-30,30):
		for vv in range(-30,30):
			res.append([uu,vv,dis1.evalf(subs={du:uu,dv:vv})])
			print(uu,vv)
	print('task5')
	return res

def task6():
	res=[]
	dis1=dis.subs({**target_p,**gtT,**gtK})
	for uu in range(-30,30):
		for vv in range(-30,30):
			res.append([uu,vv,dis1.evalf(subs={du:uu,dv:vv})])
			print(uu,vv)
	print('task6')
	return res


# p_1=simplify(T_e.inv()*T_gt*p)
# dp_t = simplify(p_1-p)



# proj=K*(T_gt*p)[:3,:]

# u=(proj/proj[2])[0]
# v=(proj/proj[2])[1]
# d=proj[2]

# p_i = Matrix([u,v,1])
# p_l = T.inv()*Matrix([*(K.inv()*(p_i*d)),1])
#
# p_i1 = Matrix([u+du,v+dv,1])
# p_l1 = T.inv()*Matrix([*(K.inv()*(p_i1*(d+du))),1])
# dp_i = simplify(p_l1-p_l)

# p.jacobin(v)

# Matrix([
# [ i*(-sin(a)*cos(r) - sin(r)*cos(a)*cos(b)) + j*(sin(a)*sin(r) - cos(a)*cos(b)*cos(r)) + k*sin(b)*cos(a),  i*sin(a)*sin(b)*sin(r) + j*sin(a)*sin(b)*cos(r) + k*sin(a)*cos(b),  i*(-sin(a)*cos(b)*cos(r) - sin(r)*cos(a)) + j*(sin(a)*sin(r)*cos(b) - cos(a)*cos(r)), 1, 0, 0],
# [i*(-sin(a)*sin(r)*cos(b) + cos(a)*cos(r)) + j*(-sin(a)*cos(b)*cos(r) - sin(r)*cos(a)) + k*sin(a)*sin(b), -i*sin(b)*sin(r)*cos(a) - j*sin(b)*cos(a)*cos(r) - k*cos(a)*cos(b), i*(-sin(a)*sin(r) + cos(a)*cos(b)*cos(r)) + j*(-sin(a)*cos(r) - sin(r)*cos(a)*cos(b)), 0, 1, 0],
# [                                                                                                      0,                       i*sin(r)*cos(b) + j*cos(b)*cos(r) - k*sin(b),                                                     i*sin(b)*cos(r) - j*sin(b)*sin(r), 0, 0, 1],
# [                                                                                                      0,                                                                  0,                                                                                     0, 0, 0, 0]])

# T_e=Matrix([
# 	[                       cos(b+db)*cos(r+dr),                        -sin(r+dr)*cos(b+db),         sin(b+db)],
# 	[sin(a+da)*sin(b+db)*cos(r+dr) + sin(r+dr)*cos(a+da), -sin(a+da)*sin(b+db)*sin(r+dr) + cos(a+da)*cos(r+dr), -sin(a+da)*cos(b+db)],
# 	[sin(a+da)*sin(r+dr) - sin(b+db)*cos(a+da)*cos(r+dr),  sin(a+da)*cos(r+dr) + sin(b+db)*sin(r+dr)*cos(a+da),  cos(a+da)*cos(b+db)]
# ])

# T=Matrix([
# 	[cos(a)*cos(r)-cos(b)*sin(a)*sin(r),-cos(b)*cos(r)*sin(a)-cos(a)*sin(r), sin(a)*sin(b),i],
# 	[cos(r)*sin(a)+cos(a)*cos(b)*sin(r), cos(a)*cos(b)*cos(r)-sin(a)*sin(r),-cos(a)*sin(b),j],
# 	[sin(b)*sin(r),cos(r)*sin(b),cos(b),k],
# 	[0,0,0,1]
# ])

# T_e=Matrix([
# 	[cos(a+da)*cos(r+dr)-cos(b+db)*sin(a+da)*sin(r+dr),-cos(b+db)*cos(r+dr)*sin(a+da)-cos(a+da)*sin(r+dr), sin(a+da)*sin(b+db),i+di],
# 	[cos(r+dr)*sin(a+da)+cos(a+da)*cos(b+db)*sin(r+dr), cos(a+da)*cos(b+db)*cos(r+dr)-sin(a+da)*sin(r+dr),-cos(a+da)*sin(b+db),j+dj],
# 	[sin(b+db)*sin(r+dr),cos(r+dr)*sin(b+db),cos(b+db),k+dk],
# 	[0,0,0,1]
# ])

# T_e_inv=Matrix([
# [-sin(da + a)*sin(dr + r)*cos(db + b) + cos(da + a)*cos(dr + r),  sin(da + a)*cos(dr + r) + sin(dr + r)*cos(da + a)*cos(db + b), sin(db + b)*sin(dr + r), di*sin(da + a)*sin(dr + r)*cos(db + b) - di*cos(da + a)*cos(dr + r) - dj*sin(da + a)*cos(dr + r) - dj*sin(dr + r)*cos(da + a)*cos(db + b) - dk*sin(db + b)*sin(dr + r) + i*sin(da + a)*sin(dr + r)*cos(db + b) - i*cos(da + a)*cos(dr + r) - j*sin(da + a)*cos(dr + r) - j*sin(dr + r)*cos(da + a)*cos(db + b) - k*sin(db + b)*sin(dr + r)],
# [-sin(da + a)*cos(db + b)*cos(dr + r) - sin(dr + r)*cos(da + a), -sin(da + a)*sin(dr + r) + cos(da + a)*cos(db + b)*cos(dr + r), sin(db + b)*cos(dr + r), di*sin(da + a)*cos(db + b)*cos(dr + r) + di*sin(dr + r)*cos(da + a) + dj*sin(da + a)*sin(dr + r) - dj*cos(da + a)*cos(db + b)*cos(dr + r) - dk*sin(db + b)*cos(dr + r) + i*sin(da + a)*cos(db + b)*cos(dr + r) + i*sin(dr + r)*cos(da + a) + j*sin(da + a)*sin(dr + r) - j*cos(da + a)*cos(db + b)*cos(dr + r) - k*sin(db + b)*cos(dr + r)],
# [                                                                                          sin(da + a)*sin(db + b),                                                                                          -sin(db + b)*cos(da + a),                              cos(db + b),                                                                   -di*sin(da + a)*sin(db + b) + dj*sin(db + b)*cos(da + a) - dk*cos(db + b) - i*sin(da + a)*sin(db + b) + j*sin(db + b)*cos(da + a) - k*cos(db + b)],
# [                                                                                                                                                0,                                                                                                                                                 0,                            0,                                                                                                                                                               1]
# ])
