import os 
import numpy as np
import numpy
import glob
import pandas as pd
#height and width of grid
h = int(input("Enter the grid height"))
W = int(input("Enter the grid width"))
#no of iteration
n_iter = int(input("Enter total iterations"))
#image to be segmented
img = input("Enter the name of image with extension")

#Converting png images to numpy array and creating dataset for training the network
from PIL import Image
image = Image.open(img)
ff = image.split()
shape = np.array(ff[0]).shape
size = np.array(ff[0]).size
r1 = np.double(np.ndarray.flatten(numpy.array(ff[0])))/255
g1 = np.ndarray.flatten(numpy.array(ff[1]))/255
b1 = np.ndarray.flatten(numpy.array(ff[2]))/255

def decay_radius(init_radius,i,time_constant):
    return init_radius*np.exp(-i/time_constant)
def decay_learning_rate(init_learning_rate,i,n_iter):
    return init_learning_rate* np.exp(-i/n_iter)
def cal_influence(distance,radius):
    return np.exp(-distance/(2*(radius**2)))

def eucl_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
def find_bmu(t,net):
    min_dist=1000000
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            unit=net[x,y].reshape(1,-1)
            t=t.reshape(1,-1)
            euc_dist=eucl_dist(unit,t)
            if euc_dist < min_dist:
                min_dist=euc_dist
                bmu=net[x,y]
                bmu_idx=np.array([x,y])
    return (bmu,bmu_idx)

#neightbourhood func
def cal_influence(distance,radius):
    return np.exp(-distance/(2*(radius**2)))
def eucl_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

init_learning_rate=0.1
m = size
n = 3
network_dim=np.array([h,W])
net=np.random.random((network_dim[0],network_dim[1],n))
init_radius=max(network_dim[0],network_dim[1])/2
time_constant=n_iter/np.log(init_radius)

# Commented out IPython magic to ensure Python compatibility.
#training prcess
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib import patches
for i in range(n_iter):
 nn = np.random.randint(0,size)
 t = np.array([r1[nn],b1[nn],g1[nn]])
 rad = decay_radius(init_radius,i,time_constant)
 l = decay_learning_rate(init_learning_rate,i,n_iter)
 bmu,bmu_idx=find_bmu(t,net)
 for x in range(net.shape[0]):
    for y in range(net.shape[1]):
        w=net[x,y].reshape(1,n)
        w_dist=eucl_dist(np.array([[x,y]]),bmu_idx.reshape(1,2))
        if w_dist<=rad:
            influence=cal_influence(w_dist,rad)
            new_w=w+(l*influence*(t.reshape(1,-1)-w))
            net[x,y]=new_w.reshape(1,n)

 if(i%50 == 0):
    c = i/n_iter
    print("Progress "+str(c)+" %")

s_r =[]
s_b =[]
s_g = []
for j in range(size):
  bmu,bmu_idx=find_bmu(np.array([r1[j],b1[j],g1[j]]),net)
  r = net[bmu_idx[0],bmu_idx[1],0]
  g = net[bmu_idx[0],bmu_idx[1],1]
  b = net[bmu_idx[0],bmu_idx[1],2]
  s_r.append(r)
  s_b.append(b)
  s_g.append(g)
  c = j/size
  print("Progress "+str(c)+" %")

r11 =  np.array(s_r).reshape(shape)
b11 =  np.array(s_b).reshape(shape)
g11 =  np.array(s_g).reshape(shape)

rgb11 = np.array([r11,b11,g11]).transpose(1, 2, 0)

plt.figure(figsize = (20,10))
plt.imshow(rgb11,interpolation='nearest')
plt.show()
